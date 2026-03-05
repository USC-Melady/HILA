# -*- coding: utf-8 -*-

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from .prompt_builders import wrap_chat
from .structured_signals import StructuredDecisionSignalsBuilder


def build_policy_prompt(
    task_type: str,
    base_prompt: str,
    self_history: List[str],
    others_histories: List[List[str]],
    agents: int,
    self_idx: int,
    use_chat_template: bool,
    tokenizer: Optional[AutoTokenizer] = None,
    sample_meta: Optional[Dict[str, Any]] = None,
    sds_builder: Optional[StructuredDecisionSignalsBuilder] = None,
) -> str:
    """
    Stage A policy prompt. Must output ONLY one of:
      - "EVAL <idx>"
      - "CREATE"
      - "DEFER"
    """
    if task_type in {"code_unit_test", "math_numeric", "math_symbolic", "mcq"}:
        self_history = self_history[-1:]
        others_histories = [h[-1:] for h in others_histories]

    self_last = self_history[-1] if self_history else "(none)"
    others_last = [h[-1] if h else "(none)" for h in others_histories]
    others_block = "\n\n".join(others_last) if others_last else "(none)"

    structured_block = ""
    if sds_builder is not None:
        structured_block = sds_builder.build(
            task_type=task_type,
            self_history=self_history,
            others_histories=others_histories,
            self_idx=self_idx,
            agents=agents,
            sample_meta=sample_meta or {},
        )

    action_lines = [
        "- DEFER       (ask a human expert)\n",
        f"- EVAL <idx>   (copy Agent idx; idx in 0..{agents-1})\n",
        "- CREATE      (write a new solution yourself)\n",
    ]
    random.shuffle(action_lines)
    actions_block = "".join(action_lines)

    user_content = (
        "You are a meta-policy controller for a multi-agent system.\n"
        "Choose ONE action and output ONLY the action line.\n\n"
        "Valid actions (no extra text):\n"
        f"{actions_block}\n"
        f"{structured_block}"
        "=== Problem ===\n"
        f"{base_prompt}\n\n"
        "=== Your Latest Solution ===\n"
        f"{self_last}\n\n"
        "=== Other Agents' Latest Solutions ===\n"
        f"{others_block}\n\n"
        "Now output ONLY one action line.\n"
    )

    if use_chat_template:
        if tokenizer is None:
            raise ValueError("tokenizer is required when use_chat_template=True")
        return wrap_chat(tokenizer, user_content)
    return user_content


def parse_policy(raw: str, self_idx: int, agents: int) -> Tuple[str, Optional[int]]:
    """
    Parse Stage A output into (act, idx).
    act ∈ {"EVAL","CREATE","DEFER"}.
    """
    s = "" if raw is None else str(raw)
    s = s.strip()
    if not s:
        return ("EVAL", 0 if self_idx != 0 else min(1, agents - 1))

    line = s.splitlines()[0].strip()
    line_norm = re.sub(r"[^\w\s]", " ", line)
    line_norm = re.sub(r"\s+", " ", line_norm).strip().upper()

    if line_norm == "CREATE":
        return ("CREATE", None)
    if line_norm == "DEFER":
        return ("DEFER", None)

    m = re.match(r"^EVAL(?:\s+IDX)?\s+(\d+)$", line_norm)
    if m:
        idx = int(m.group(1))
        idx = max(0, min(idx, agents - 1))
        if idx == self_idx and agents > 1:
            idx = 0 if self_idx != 0 else 1
        return ("EVAL", idx)

    m2 = re.search(r"(\d+)", line_norm)
    if m2:
        idx = int(m2.group(1))
        idx = max(0, min(idx, agents - 1))
        if idx == self_idx and agents > 1:
            idx = 0 if self_idx != 0 else 1
        return ("EVAL", idx)

    idx = 0 if self_idx != 0 else (1 if agents > 1 else 0)
    return ("EVAL", idx)