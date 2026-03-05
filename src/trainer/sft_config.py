#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SFTConfig:
    # data
    train_jsonl: str
    max_prompt_tokens: int = 2048
    max_completion_tokens: int = 1024
    use_chat_template: bool = True

    # model
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    trust_remote_code: bool = True
    init_adapter: Optional[str] = None

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[str] = None

    # optimization
    lr: float = 2e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0

    # training
    epochs: int = 1
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    log_every: int = 20
    save_every: int = 500
    output_dir: str = "outputs/sft"
    seed: int = 0
    bf16: bool = True
    fp16: bool = False

    # optional KL regularization
    kl_beta: float = 0.0  # set > 0 to enable KL anchoring

    # misc
    num_workers: int = 2