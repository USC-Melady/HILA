#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

from .sft_config import SFTConfig
from .sft_dataset import SFTJsonlDataset
from .sft_collate import sft_collate_fn
from .sft_ops import masked_token_kl_from_logits


class SFTTrainer:
    def __init__(self, cfg: SFTConfig):
        self.cfg = cfg
        self.acc = Accelerator(
            gradient_accumulation_steps=cfg.grad_accum_steps,
            mixed_precision=("bf16" if cfg.bf16 else ("fp16" if cfg.fp16 else "no")),
        )
        set_seed(cfg.seed)

        #  Tokenizer 
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

        if cfg.lora_target_modules:
            target_modules = [s.strip() for s in cfg.lora_target_modules.split(",") if s.strip()]
        else:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        #  Trainable model 
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

        if cfg.init_adapter:
            self.model = PeftModel.from_pretrained(
                base_model,
                cfg.init_adapter,
                is_trainable=True,
            )
        else:
            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            self.model = get_peft_model(base_model, lora_cfg)

        #  Frozen reference model (only if KL enabled) 
        self.ref_model: Optional[nn.Module] = None
        if cfg.kl_beta > 0:
            ref_base = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                trust_remote_code=cfg.trust_remote_code,
                torch_dtype=torch_dtype,
            )
            if hasattr(ref_base.config, "use_cache"):
                ref_base.config.use_cache = False

            if cfg.init_adapter:
                self.ref_model = PeftModel.from_pretrained(
                    ref_base,
                    cfg.init_adapter,
                    is_trainable=False,
                )
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

            # Make ref exactly match the initial trainable model state
            self.ref_model.load_state_dict(self.model.state_dict(), strict=True)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        #  Dataset / Dataloader 
        self.dataset = SFTJsonlDataset(cfg.train_jsonl)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: sft_collate_fn(
                b,
                tokenizer=self.tokenizer,
                max_prompt_tokens=cfg.max_prompt_tokens,
                max_completion_tokens=cfg.max_completion_tokens,
                use_chat_template=cfg.use_chat_template,
            ),
        )

        #  Optimizer 
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

        #  Prepare 
        if self.ref_model is not None:
            self.model, self.ref_model, self.optimizer, self.dataloader, self.scheduler = self.acc.prepare(
                self.model, self.ref_model, self.optimizer, self.dataloader, self.scheduler
            )
        else:
            self.model, self.optimizer, self.dataloader, self.scheduler = self.acc.prepare(
                self.model, self.optimizer, self.dataloader, self.scheduler
            )

        #  Output 
        self.output_dir = Path(cfg.output_dir)
        if self.acc.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "config.json").write_text(
                json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        self.global_step = 0

    # ---
    # Save
    # ---

    def save(self, tag: str = "latest") -> None:
        if not self.acc.is_main_process:
            return

        out = self.output_dir / tag
        out.mkdir(parents=True, exist_ok=True)

        unwrapped = self.acc.unwrap_model(self.model)
        unwrapped.save_pretrained(out)
        self.tokenizer.save_pretrained(out)

    # ---
    # Train
    # ---
    def train(self) -> None:
        cfg = self.cfg
        self.model.train()

        # optimizer update
        running_loss = 0.0
        running_ce = 0.0
        running_kl = 0.0
        running_resp = 0.0
        micro_count = 0

        # log window
        log_running_loss = 0.0
        log_running_ce = 0.0
        log_running_kl = 0.0
        log_running_resp = 0.0
        log_step_count = 0

        for epoch in range(cfg.epochs):
            for batch in self.dataloader:
                with self.acc.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    ce_loss = outputs.loss

                    if cfg.kl_beta > 0:
                        if self.ref_model is None:
                            raise RuntimeError("kl_beta > 0 but ref_model is None")

                        with torch.no_grad():
                            ref_outputs = self.ref_model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                            )

                        kl_loss = masked_token_kl_from_logits(
                            logits_pi=outputs.logits,
                            logits_ref=ref_outputs.logits,
                            labels=batch["labels"],
                        )
                        loss = ce_loss + cfg.kl_beta * kl_loss
                    else:
                        kl_loss = ce_loss.new_zeros(())
                        loss = ce_loss

                    self.acc.backward(loss)

                    if self.acc.sync_gradients and cfg.grad_clip and cfg.grad_clip > 0:
                        self.acc.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # accumulate
                running_loss += float(loss.detach().cpu())
                running_ce += float(ce_loss.detach().cpu())
                running_kl += float(kl_loss.detach().cpu())
                running_resp += float(batch["response_lengths"].float().mean().cpu())
                micro_count += 1

                if self.acc.sync_gradients:
                    self.global_step += 1
                    step_loss = running_loss / max(1, micro_count)
                    step_ce = running_ce / max(1, micro_count)
                    step_kl = running_kl / max(1, micro_count)
                    step_resp = running_resp / max(1, micro_count)

                    log_running_loss += step_loss
                    log_running_ce += step_ce
                    log_running_kl += step_kl
                    log_running_resp += step_resp
                    log_step_count += 1

                    if self.acc.is_main_process and (self.global_step % cfg.log_every == 0):
                        lr = self.scheduler.get_last_lr()[0]

                        avg_loss = log_running_loss / max(1, log_step_count)
                        avg_ce = log_running_ce / max(1, log_step_count)
                        avg_kl = log_running_kl / max(1, log_step_count)
                        avg_resp = log_running_resp / max(1, log_step_count)

                        print(
                            f"[SFT epoch {epoch+1}/{cfg.epochs} step {self.global_step}] "
                            f"step_loss={step_loss:.4f} "
                            f"step_ce={step_ce:.4f} "
                            f"step_kl={step_kl:.4f} "
                            f"step_resp_len={step_resp:.1f} | "
                            f"loss={avg_loss:.4f} "
                            f"ce={avg_ce:.4f} "
                            f"kl={avg_kl:.4f} "
                            f"resp_len={avg_resp:.1f} "
                            f"lr={lr:.2e}",
                            flush=True,
                        )
                        log_running_loss = 0.0
                        log_running_ce = 0.0
                        log_running_kl = 0.0
                        log_running_resp = 0.0
                        log_step_count = 0

                    if self.global_step % cfg.save_every == 0:
                        self.save(tag=f"step-{self.global_step}")

                    running_loss = 0.0
                    running_ce = 0.0
                    running_kl = 0.0
                    running_resp = 0.0
                    micro_count = 0

        self.save(tag="final")

    def trainv0(self) -> None:
        cfg = self.cfg
        self.model.train()

        running_loss = 0.0
        running_ce = 0.0
        running_kl = 0.0
        running_resp = 0.0
        micro_count = 0

        for epoch in range(cfg.epochs):
            for batch in self.dataloader:
                with self.acc.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    ce_loss = outputs.loss

                    if cfg.kl_beta > 0:
                        if self.ref_model is None:
                            raise RuntimeError("kl_beta > 0 but ref_model is None")

                        with torch.no_grad():
                            ref_outputs = self.ref_model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                            )

                        kl_loss = masked_token_kl_from_logits(
                            logits_pi=outputs.logits,
                            logits_ref=ref_outputs.logits,
                            labels=batch["labels"],
                        )
                        loss = ce_loss + cfg.kl_beta * kl_loss
                    else:
                        kl_loss = ce_loss.new_zeros(())
                        loss = ce_loss

                    self.acc.backward(loss)

                    if self.acc.sync_gradients and cfg.grad_clip and cfg.grad_clip > 0:
                        self.acc.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                running_loss += float(loss.detach().cpu())
                running_ce += float(ce_loss.detach().cpu())
                running_kl += float(kl_loss.detach().cpu())
                running_resp += float(batch["response_lengths"].float().mean().cpu())
                micro_count += 1

                if self.acc.sync_gradients:
                    self.global_step += 1

                    avg_loss = running_loss / max(1, micro_count)
                    avg_ce = running_ce / max(1, micro_count)
                    avg_kl = running_kl / max(1, micro_count)
                    avg_resp = running_resp / max(1, micro_count)

                    if self.acc.is_main_process and (self.global_step % cfg.log_every == 0):
                        lr = self.scheduler.get_last_lr()[0]
                        print(
                            f"[SFT epoch {epoch+1}/{cfg.epochs} step {self.global_step}] "
                            f"loss={avg_loss:.4f} "
                            f"ce={avg_ce:.4f} "
                            f"kl={avg_kl:.4f} "
                            f"resp_len={avg_resp:.1f} "
                            f"lr={lr:.2e}",
                            flush=True,
                        )

                    if self.global_step % cfg.save_every == 0:
                        self.save(tag=f"step-{self.global_step}")

                    running_loss = 0.0
                    running_ce = 0.0
                    running_kl = 0.0
                    running_resp = 0.0
                    micro_count = 0

        self.save(tag="final")