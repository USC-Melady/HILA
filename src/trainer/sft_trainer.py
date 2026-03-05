#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainer/sft_trainer.py
~~~~~~~~~~~~~~~~~~~~~~
Compatibility facade for the modularized SFT trainer.

This file preserves the original import path:
    from trainer.sft_trainer import SFTTrainer, SFTConfig
"""

from __future__ import annotations

from .sft_config import SFTConfig
from .sft_trainer_core import SFTTrainer

__all__ = ["SFTConfig", "SFTTrainer"]