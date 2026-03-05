#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainer/grpo.py
~~~~~~~~~~~~~~~
Compatibility facade for the modularized GRPO trainer.

This file preserves the original import path:
    from trainer.grpo import GRPOTrainer, GRPOConfig
"""

from __future__ import annotations

from .grpo_config import GRPOConfig
from .grpo_trainer import GRPOTrainer

__all__ = ["GRPOConfig", "GRPOTrainer"]