#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer


def _apply_chat_template_if_needed(
    tokenizer: AutoTokenizer,
    prompt: str,
    use_chat_template: bool,
) -> str:
    if not use_chat_template:
        return prompt

    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


def _build_inputs_for_pair(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    completion_text: str,
    max_prompt_tokens: int,
    max_completion_tokens: int,
    use_chat_template: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_text = _apply_chat_template_if_needed(tokenizer, prompt_text, use_chat_template)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_tokens,
        return_tensors=None,
    )["input_ids"]

    max_comp_body = max(1, max_completion_tokens - 1)
    comp_ids = tokenizer(
        completion_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_comp_body,
        return_tensors=None,
    )["input_ids"]

    if tokenizer.eos_token_id is not None:
        comp_ids = comp_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + comp_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + comp_ids

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def grpo_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_prompt_tokens: int,
    max_completion_tokens: int,
    use_chat_template: bool,
) -> Dict[str, Any]:
    input_ids_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    rewards_list: List[float] = []
    group_sizes: List[int] = []

    for g in batch:
        prompt = g["prompt"]
        completions = g["completions"]
        rewards = g["rewards"]

        k = len(completions)
        group_sizes.append(k)

        for c, r in zip(completions, rewards):
            ids, am, lab = _build_inputs_for_pair(
                tokenizer=tokenizer,
                prompt_text=prompt,
                completion_text=str(c),
                max_prompt_tokens=max_prompt_tokens,
                max_completion_tokens=max_completion_tokens,
                use_chat_template=use_chat_template,
            )
            input_ids_list.append(ids)
            attn_list.append(am)
            labels_list.append(lab)
            rewards_list.append(float(r))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    max_len = max(x.size(0) for x in input_ids_list)

    def _pad(x: torch.Tensor, pad_value: int) -> torch.Tensor:
        if x.size(0) == max_len:
            return x
        pad = torch.full((max_len - x.size(0),), pad_value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    input_ids = torch.stack([_pad(x, pad_id) for x in input_ids_list], dim=0)
    attention_mask = torch.stack([_pad(x, 0) for x in attn_list], dim=0)
    labels = torch.stack([_pad(x, -100) for x in labels_list], dim=0)
    rewards_t = torch.tensor(rewards_list, dtype=torch.float32)
    group_sizes_t = torch.tensor(group_sizes, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "rewards": rewards_t,
        "group_sizes": group_sizes_t,
    }