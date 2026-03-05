#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from .grpo_config import GRPOConfig
from .grpo_dataset import GRPOGroupedDataset
from .grpo_collate import grpo_collate_fn
from .grpo_ops import (
    forward_with_response_stats,
    masked_token_kl_from_logps,
    compute_group_action_entropy,
)


class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.acc = Accelerator(
            gradient_accumulation_steps=cfg.grad_accum_steps,
            mixed_precision=("bf16" if cfg.bf16 else ("fp16" if cfg.fp16 else "no")),
        )
        set_seed(cfg.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = (
            torch.bfloat16 if cfg.bf16
            else (torch.float16 if cfg.fp16 else None)
        )

        # ---------- Trainable policy ----------
        base_policy = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch_dtype,
        )

        if cfg.lora_target_modules:
            target_modules = [s.strip() for s in cfg.lora_target_modules.split(",") if s.strip()]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        if cfg.init_adapter:
            self.model = PeftModel.from_pretrained(base_policy, cfg.init_adapter, is_trainable=True)
        else:
            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            self.model = get_peft_model(base_policy, lora_cfg)

        # 
        ref_base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if cfg.init_adapter:
            self.ref_model = PeftModel.from_pretrained(ref_base, cfg.init_adapter, is_trainable=False)
        else:
            lora_cfg_ref = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            self.ref_model = get_peft_model(ref_base, lora_cfg_ref)

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.dataset = GRPOGroupedDataset(cfg.train_jsonl)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: grpo_collate_fn(
                b,
                tokenizer=self.tokenizer,
                max_prompt_tokens=cfg.max_prompt_tokens,
                max_completion_tokens=cfg.max_completion_tokens,
                use_chat_template=cfg.use_chat_template,
            ),
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        steps_per_epoch = math.ceil(len(self.dataloader) / cfg.grad_accum_steps)
        total_steps = max(1, steps_per_epoch * cfg.epochs)
        warmup_steps = max(1, int(0.03 * total_steps))

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.model, self.ref_model, self.optimizer, self.dataloader, self.scheduler = self.acc.prepare(
            self.model, self.ref_model, self.optimizer, self.dataloader, self.scheduler
        )

        self.output_dir = Path(cfg.output_dir)
        if self.acc.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "config.json").write_text(
                json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        self.global_step = 0

    # -----------------------------------------------------
    # Advantage
    # -----------------------------------------------------

    def _compute_advantages(self, rewards: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg

        rewards = rewards * cfg.reward_scale
        if cfg.reward_power != 1.0:
            rewards = torch.sign(rewards) * (rewards.abs() ** cfg.reward_power)

        adv_list: List[torch.Tensor] = []
        offset = 0
        for k in group_sizes.tolist():
            rg = rewards[offset: offset + k]
            adv = rg - rg.mean()
            if cfg.adv_normalize:
                adv = adv / (rg.std(unbiased=False) + cfg.adv_eps)
            adv_list.append(adv)
            offset += k

        return torch.cat(adv_list, dim=0)

    # -----------------------------------------------------
    # GRPO/PPO-style clipped policy objective
    # -----------------------------------------------------

    def _policy_loss_grpo(
        self,
        seq_logp_pi: torch.Tensor,    # [B_flat]
        seq_logp_old: torch.Tensor,   # [B_flat]
        advantages: torch.Tensor,     # [B_flat]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg

        log_ratio = seq_logp_pi - seq_logp_old
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages

        # maximize min(surr1, surr2)  <=> minimize negative of it
        policy_loss = -torch.min(surr1, surr2).mean()

        clipped = (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean()

        aux = {
            "ratio_mean": float(ratio.mean().detach().cpu()),
            "ratio_min": float(ratio.min().detach().cpu()),
            "ratio_max": float(ratio.max().detach().cpu()),
            "clip_frac": float(clipped.detach().cpu()),
        }
        return policy_loss, aux

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor,
        group_sizes: torch.Tensor,
    ):
        cfg = self.cfg

        out_pi = forward_with_response_stats(self.model, input_ids, attention_mask, labels)
        seq_logp_pi = out_pi["seq_logp_mean"]        # use mean to reduce length bias
        logp_all_pi = out_pi["logp_all"]
        response_mask = out_pi["response_mask"]

        with torch.no_grad():
            out_old = forward_with_response_stats(self.ref_model, input_ids, attention_mask, labels)
            seq_logp_old = out_old["seq_logp_mean"]
            logp_all_old = out_old["logp_all"]

        advantages = self._compute_advantages(rewards, group_sizes).to(seq_logp_pi.device)

        policy_loss, policy_aux = self._policy_loss_grpo(
            seq_logp_pi=seq_logp_pi,
            seq_logp_old=seq_logp_old,
            advantages=advantages,
        )

        # KL to frozen initial policy
        kl = masked_token_kl_from_logps(logp_all_pi, logp_all_old, response_mask)

        # group-level action entropy
        group_entropy = compute_group_action_entropy(seq_logp_pi, group_sizes)

        loss = policy_loss + cfg.kl_beta * kl - cfg.entropy_beta * group_entropy

        # diagnostics
        score_gaps: List[torch.Tensor] = []
        offset = 0
        for k in group_sizes.tolist():
            s = seq_logp_pi[offset: offset + k]
            score_gaps.append(s.max() - s.min())
            offset += k
        score_gap = float(torch.stack(score_gaps).mean().detach().cpu()) if score_gaps else 0.0

        stats = {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "group_entropy": float(group_entropy.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "adv_mean": float(advantages.mean().detach().cpu()),
            "adv_abs_mean": float(advantages.abs().mean().detach().cpu()),
            "adv_std": float(advantages.std(unbiased=False).detach().cpu()),
            "logp_mean": float(seq_logp_pi.mean().detach().cpu()),
            "old_logp_mean": float(seq_logp_old.mean().detach().cpu()),
            "reward_mean": float(rewards.mean().detach().cpu()),
            "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
            "score_gap": score_gap,
            "ratio_mean": policy_aux["ratio_mean"],
            "ratio_min": policy_aux["ratio_min"],
            "ratio_max": policy_aux["ratio_max"],
            "clip_frac": policy_aux["clip_frac"],
        }
        return loss, stats

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    def save(self, tag: str = "latest") -> None:
        if not self.acc.is_main_process:
            return
        out = self.output_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        unwrapped = self.acc.unwrap_model(self.model)
        unwrapped.save_pretrained(out)
        self.tokenizer.save_pretrained(out)

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg
        self.model.train()

        # current optimizer update
        running = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "group_entropy": 0.0,
            "kl": 0.0,
            "adv_abs_mean": 0.0,
            "logp_mean": 0.0,
            "old_logp_mean": 0.0,
            "ratio_mean": 0.0,
            "clip_frac": 0.0,
            "score_gap": 0.0,
        }
        micro_count = 0

        # log window 
        log_running = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "group_entropy": 0.0,
            "kl": 0.0,
            "adv_abs_mean": 0.0,
            "logp_mean": 0.0,
            "old_logp_mean": 0.0,
            "ratio_mean": 0.0,
            "clip_frac": 0.0,
            "score_gap": 0.0,
        }
        log_step_count = 0

        keys = list(running.keys())

        for epoch in range(cfg.epochs):
            for batch in self.dataloader:
                with self.acc.accumulate(self.model):
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    rewards = batch["rewards"]
                    group_sizes = batch["group_sizes"]

                    loss, st = self._compute_loss(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        rewards=rewards,
                        group_sizes=group_sizes,
                    )

                    self.acc.backward(loss)

                    if self.acc.sync_gradients and cfg.grad_clip and cfg.grad_clip > 0:
                        self.acc.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # accumulate
                for k in keys:
                    running[k] += float(st[k])
                micro_count += 1

                # optimizer update
                if self.acc.sync_gradients:
                    self.global_step += 1
                    step_stats = {k: running[k] / max(1, micro_count) for k in keys}

                    for k in keys:
                        log_running[k] += step_stats[k]
                    log_step_count += 1

                    # log
                    if self.acc.is_main_process and (self.global_step % cfg.log_every == 0):
                        lr = self.scheduler.get_last_lr()[0]
                        avg_stats = {k: log_running[k] / max(1, log_step_count) for k in keys}

                        print(
                            f"[GRPO epoch {epoch+1}/{cfg.epochs} step {self.global_step}] "
                            f"step_loss={step_stats['loss']:.4f} "
                            f"step_pg={step_stats['policy_loss']:.4f} "
                            f"step_ent={step_stats['group_entropy']:.4f} "
                            f"step_kl={step_stats['kl']:.4f} "
                            f"step_|adv|={step_stats['adv_abs_mean']:.4f} "
                            f"step_logp={step_stats['logp_mean']:.4f} "
                            f"step_old_logp={step_stats['old_logp_mean']:.4f} "
                            f"step_ratio={step_stats['ratio_mean']:.4f} "
                            f"step_clip={step_stats['clip_frac']:.4f} "
                            f"step_gap={step_stats['score_gap']:.4f} | "
                            f"loss={avg_stats['loss']:.4f} "
                            f"pg={avg_stats['policy_loss']:.4f} "
                            f"ent={avg_stats['group_entropy']:.4f} "
                            f"kl={avg_stats['kl']:.4f} "
                            f"|adv|={avg_stats['adv_abs_mean']:.4f} "
                            f"logp={avg_stats['logp_mean']:.4f} "
                            f"old_logp={avg_stats['old_logp_mean']:.4f} "
                            f"ratio={avg_stats['ratio_mean']:.4f} "
                            f"clip={avg_stats['clip_frac']:.4f} "
                            f"gap={avg_stats['score_gap']:.4f} "
                            f"lr={lr:.2e}",
                            flush=True,
                        )

                        for k in keys:
                            log_running[k] = 0.0
                        log_step_count = 0

                    if self.global_step % cfg.save_every == 0:
                        self.save(tag=f"step-{self.global_step}")

                    # clean
                    for k in keys:
                        running[k] = 0.0
                    micro_count = 0

        self.save(tag="final")