# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Dict, Optional, Type


@dataclass
class EvalResult:
    correct: bool
    gold_norm: Optional[str] = None
    pred_norm: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class BaseEvaluator:
    task_type: str

    def evaluate(self, pred: str, gold: Any, meta: Dict[str, Any]) -> EvalResult:
        raise NotImplementedError


class EvaluatorFactory:
    _registry: Dict[str, Type[BaseEvaluator]] = {}

    @classmethod
    def register(cls, task_type: str, evaluator_cls: Type[BaseEvaluator]) -> None:
        cls._registry[task_type] = evaluator_cls

    @classmethod
    def create(cls, task_type: str) -> BaseEvaluator:
        if task_type not in cls._registry:
            raise KeyError(
                f"No evaluator registered for task_type='{task_type}'. "
                f"Registered: {sorted(cls._registry.keys())}"
            )
        return cls._registry[task_type]()


# -----------------------------
# Normalization helpers
# -----------------------------

_NUM_CLEAN_RE = re.compile(r"[,\s$]")  # remove commas, spaces, $
_NUM_KEEP_RE = re.compile(r"[^0-9+\-eE\.\/]")  # allowed: digits, signs, dot, exponent, slash


def normalize_numeric_str(x: Any) -> Optional[str]:
    """
    Normalize numeric-ish strings:
    - strip
    - remove commas, spaces, $
    - keep only digits/sign/dot/e/E/slash
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = _NUM_CLEAN_RE.sub("", s)
    # remove trailing period
    if s.endswith("."):
        s = s[:-1]
    # keep only allowed chars
    s = _NUM_KEEP_RE.sub("", s)
    return s.strip() or None


def try_parse_decimal(x: Optional[str]) -> Optional[Decimal]:
    if x is None:
        return None
    # handle simple fraction like "3/4"
    if "/" in x and "e" not in x.lower():
        parts = x.split("/")
        if len(parts) == 2 and parts[0] and parts[1]:
            try:
                return Decimal(parts[0]) / Decimal(parts[1])
            except (InvalidOperation, ZeroDivisionError):
                return None
    try:
        return Decimal(x)
    except InvalidOperation:
        return None


_LATEX_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_LATEX_LEFT_RIGHT_RE = re.compile(r"\\left|\\right")
_LATEX_WHITESPACE_RE = re.compile(r"\s+")


def normalize_latexish(x: Any) -> Optional[str]:
    """
    Baseline normalization for LaTeX-ish answers:
    - strip
    - remove \boxed{...} wrapper if present
    - remove \left / \right
    - collapse whitespace
    - strip outer $...$ if present
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    # strip surrounding $...$
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()

    m = _LATEX_BOXED_RE.search(s)
    if m:
        s = m.group(1).strip()

    s = _LATEX_LEFT_RIGHT_RE.sub("", s)
    s = _LATEX_WHITESPACE_RE.sub(" ", s).strip()
    return s or None


_MCQ_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_MCQ_DIGIT_RE = re.compile(r"\b([0-3])\b")


def parse_mcq_choice(pred: str) -> Optional[int]:
    """
    Parse model output into choice index 0-3.
    Accepts:
      - 'A'/'B'/'C'/'D' (case-insensitive)
      - '0'/'1'/'2'/'3'
      - phrases containing those tokens (e.g. "Answer: C")
    """
    if pred is None:
        return None
    s = str(pred).strip()
    if not s:
        return None

    m = _MCQ_LETTER_RE.search(s)
    if m:
        ch = m.group(1).upper()
        return {"A": 0, "B": 1, "C": 2, "D": 3}.get(ch)

    m = _MCQ_DIGIT_RE.search(s)
    if m:
        return int(m.group(1))

    return None
