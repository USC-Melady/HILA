# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List


def build_mmlu_prompt(question: str, choices: List[str]) -> str:
    """
    Deterministic MCQ prompt format.
    Expect the model to answer with A/B/C/D (preferred) or option text.
    """
    letters = ["A", "B", "C", "D"]
    lines = []
    lines.append(f"Question: {question}".strip())
    lines.append("Choices:")
    for i, c in enumerate(choices):
        prefix = letters[i] if i < len(letters) else f"Option{i}"
        lines.append(f"{prefix}. {c}")
    lines.append("")
    lines.append("Answer (A, B, C, or D):")
    return "\n".join(lines)
