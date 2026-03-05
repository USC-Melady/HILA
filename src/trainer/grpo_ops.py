#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def forward_with_response_stats(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B, T, V]

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    response_mask = (shift_labels != -100)

    logp_all = torch.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]

    tgt = shift_labels.clamp(min=0)
    token_logp = torch.gather(logp_all, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    token_logp = token_logp * response_mask

    token_count = response_mask.sum(dim=-1).clamp(min=1)  # [B]
    seq_logp_sum = token_logp.sum(dim=-1).to(torch.float32)    # [B]
    seq_logp_mean = (seq_logp_sum / token_count).to(torch.float32)  # [B]

    return {
        "seq_logp_sum": seq_logp_sum,
        "seq_logp_mean": seq_logp_mean,
        "logp_all": logp_all,
        "response_mask": response_mask,
    }


def masked_token_kl_from_logps(
    logp_p: torch.Tensor,
    logp_q: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean token-level KL: KL(p || q), averaged over response tokens.
    """
    p = torch.exp(logp_p)
    kl_tok = p * (logp_p - logp_q)     # [B, T-1, V]
    kl_tok = kl_tok.sum(dim=-1)        # [B, T-1]
    kl_tok = kl_tok * mask
    denom = mask.sum().clamp(min=1)
    return (kl_tok.sum() / denom).to(torch.float32)


def compute_group_action_entropy(
    scores: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Entropy over candidate actions within each group, then average across groups.
    scores: [B_flat]
    """
    entropies: List[torch.Tensor] = []
    offset = 0
    for k in group_sizes.tolist():
        s = scores[offset: offset + k]
        log_probs = torch.log_softmax(s, dim=0)
        probs = torch.exp(log_probs)
        ent = -(probs * log_probs).sum()
        entropies.append(ent)
        offset += k

    if not entropies:
        return scores.new_zeros(())
    return torch.stack(entropies).mean().to(torch.float32)