#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_main.py
~~~~~~~~~~~~~
Unified entry for trainers: GRPO and SFT (LoRA).

Examples:
1) GRPO from scratch:
  accelerate launch train_main.py --trainer grpo --train_jsonl grpo_train.jsonl --model <base> --output_dir outputs/grpo1

2) GRPO continue from existing LoRA:
  accelerate launch train_main.py --trainer grpo --train_jsonl grpo_train.jsonl --model <base> --init_adapter outputs/grpo1/final --output_dir outputs/grpo2

3) SFT warm-start from GRPO LoRA:
  accelerate launch train_main.py --trainer sft --train_jsonl sft_train.jsonl --model <base> --init_adapter outputs/grpo1/final --output_dir outputs/sft_from_grpo
"""

from __future__ import annotations

import argparse

from src.trainer import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig

DEFAULT_MODEL_ID: str = ("meta-llama/Llama-3.1-8B-Instruct")

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("trainer entry")

    ap.add_argument("--trainer", type=str, required=True, choices=["grpo", "sft"])

    # shared: data
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--max_completion_tokens", type=int, default=None)  # trainer-specific default if None
    ap.add_argument("--use_chat_template", action="store_true", default=True)
    ap.add_argument("--no_chat_template", action="store_false", dest="use_chat_template")
    
    # shared: model
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--init_adapter", type=str, default=None, help="Path to an existing LoRA adapter to load before training")
    ap.add_argument("--ref_model", type=str, default=None, help="(GRPO only) reference model base path")

    # shared: lora
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default=None)

    # shared: optimization/training
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--fp16", action="store_true", default=False)

    ap.add_argument("--num_workers", type=int, default=2)

    # GRPO-specific
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--kl_beta", type=float, default=0.02)
    ap.add_argument("--adv_normalize", action="store_true", default=True)
    ap.add_argument("--no_adv_normalize", action="store_false", dest="adv_normalize")
    ap.add_argument("--reward_scale", type=float, default=1.0)

    return ap


def main():
    args = build_argparser().parse_args()

    if args.trainer == "grpo":
        max_comp = args.max_completion_tokens if args.max_completion_tokens is not None else 32
        cfg = GRPOConfig(
            train_jsonl=args.train_jsonl,
            max_prompt_tokens=args.max_prompt_tokens,
            max_completion_tokens=max_comp,
            use_chat_template=args.use_chat_template,

            model_name_or_path=args.model,
            init_adapter=args.init_adapter,

            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,

            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,

            log_every=args.log_every,
            save_every=args.save_every,
            output_dir=args.output_dir,

            seed=args.seed,
            bf16=args.bf16,
            fp16=args.fp16,

            #clip_eps=args.clip_eps,
            kl_beta=args.kl_beta,
            adv_normalize=args.adv_normalize,
            reward_scale=args.reward_scale,

            num_workers=args.num_workers,
        )
        trainer = GRPOTrainer(cfg)
        trainer.train()
        return

    if args.trainer == "sft":
        max_comp = args.max_completion_tokens if args.max_completion_tokens is not None else 1024
        cfg = SFTConfig(
            train_jsonl=args.train_jsonl,
            max_prompt_tokens=args.max_prompt_tokens,
            max_completion_tokens=max_comp,
            use_chat_template=args.use_chat_template,

            model_name_or_path=args.model,
            init_adapter=args.init_adapter,

            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,

            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,

            log_every=args.log_every,
            save_every=args.save_every,
            output_dir=args.output_dir,

            kl_beta=args.kl_beta,

            seed=args.seed,
            bf16=args.bf16,
            fp16=args.fp16,

            num_workers=args.num_workers,
        )
        trainer = SFTTrainer(cfg)
        trainer.train()
        return

    raise ValueError(f"Unknown trainer: {args.trainer}")


if __name__ == "__main__":
    main()