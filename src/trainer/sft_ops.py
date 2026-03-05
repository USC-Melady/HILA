#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch


def masked_token_kl_from_logits(
    logits_pi: torch.Tensor,   # [B, T, V]
    logits_ref: torch.Tensor,  # [B, T, V]
    labels: torch.Tensor,      # [B, T]
) -> torch.Tensor:
    """
    Mean token-level KL: KL(pi || ref), averaged only over response tokens.
    We use labels to determine the response mask:
      labels == -100  => prompt tokens (excluded)
      labels != -100  => completion tokens (included)
    """
    shift_logits_pi = logits_pi[:, :-1, :]
    shift_logits_ref = logits_ref[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = (shift_labels != -100)  # [B, T-1]

    logp_pi = torch.log_softmax(shift_logits_pi, dim=-1)
    logp_ref = torch.log_softmax(shift_logits_ref, dim=-1)
    p_pi = torch.exp(logp_pi)

    kl_tok = (p_pi * (logp_pi - logp_ref)).sum(dim=-1)  # [B, T-1]
    kl_tok = kl_tok * mask

    denom = mask.sum().clamp(min=1)
    return (kl_tok.sum() / denom).to(torch.float32)