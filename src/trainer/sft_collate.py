#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer


def _apply_chat_template_if_needed(
    tokenizer: AutoTokenizer,
    prompt: str,
    use_chat_template: bool,
) -> str:
    """
    Wrap prompt as a user message and add generation prompt for assistant.
    This matches your inference-time behavior:
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    """
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


def _build_sft_inputs(
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    max_prompt_tokens: int,
    max_completion_tokens: int,
    use_chat_template: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build:
      input_ids  = prompt_ids + completion_ids(+eos)
      labels     = [-100]*len(prompt_ids) + completion_ids(+eos)
    Returns response token count too for diagnostics.
    """
    prompt_text = _apply_chat_template_if_needed(tokenizer, prompt, use_chat_template)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_tokens,
        return_tensors=None,
    )["input_ids"]

    has_eos = tokenizer.eos_token_id is not None
    max_comp_body = max(1, max_completion_tokens - 1) if has_eos else max_completion_tokens

    completion_ids = tokenizer(
        completion,
        add_special_tokens=False,
        truncation=True,
        max_length=max_comp_body,
        return_tensors=None,
    )["input_ids"]

    if has_eos:
        if len(completion_ids) == 0 or completion_ids[-1] != tokenizer.eos_token_id:
            completion_ids = completion_ids + [tokenizer.eos_token_id]

    if len(completion_ids) == 0:
        raise ValueError("Encountered empty completion_ids after tokenization.")

    input_ids = prompt_ids + completion_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + completion_ids

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        len(completion_ids),
    )


def sft_collate_fn(
    batch: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    max_prompt_tokens: int,
    max_completion_tokens: int,
    use_chat_template: bool,
) -> Dict[str, torch.Tensor]:
    input_ids_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    resp_lens: List[int] = []

    for ex in batch:
        ids, am, lab, resp_len = _build_sft_inputs(
            tokenizer=tokenizer,
            prompt=ex["prompt"],
            completion=ex["completion"],
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens,
            use_chat_template=use_chat_template,
        )
        input_ids_list.append(ids)
        attn_list.append(am)
        labels_list.append(lab)
        resp_lens.append(resp_len)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
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
    response_lengths = torch.tensor(resp_lens, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_lengths": response_lengths,
    }