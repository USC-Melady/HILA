# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Optional

from src.eval import (
    normalize_numeric_str,
    try_parse_decimal,
    normalize_latexish,
    parse_mcq_choice,
)
from src.dataset import (
    TASK_MATH_NUMERIC,
    TASK_MATH_SYMBOLIC,
    TASK_MCQ,
    TASK_CODE_UNIT_TEST,
)

# --------------- regex helpers ---------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE | re.DOTALL)
_HASH_RE = re.compile(r"####\s*(.*)")
# number-ish: supports commas, decimals, exponent, simple fraction a/b
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:e[+-]?\d+)?(?:/\d+)?", re.IGNORECASE)


@dataclass
class ParsedPrediction:
    """
    The unified parsed prediction from a raw model output.
    - pred_str: the string you feed to evaluator
    - vote_key: stable representation for voting
    - ok: whether parsing extracted a meaningful answer
    - method: which rule matched (boxed/hash/regex/fallback/...)
    - debug: extra info for failure analysis
    """
    pred_str: str
    vote_key: str
    ok: bool
    method: str
    debug: Dict[str, Any] = field(default_factory=dict)

# remove _BOXED_RE entirely (or keep but don't use)

def _extract_braced_content(s: str, lbrace_pos: int) -> Optional[tuple[str, int]]:
    """
    Given s and position of '{' (lbrace_pos), return (content_inside, end_pos_of_matching_rbrace).
    Supports nested braces. Returns None if unbalanced.
    """
    if lbrace_pos < 0 or lbrace_pos >= len(s) or s[lbrace_pos] != "{":
        return None

    depth = 0
    i = lbrace_pos
    start = lbrace_pos + 1
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (s[start:i], i)
        i += 1
    return None

_BOXED_START_RE = re.compile(r"(\\)?boxed\s*\{", re.IGNORECASE)

def _find_last_boxed(text: str) -> Optional[str]:
    """
    Find the last (\\)?boxed{...} (case-insensitive, allows spaces before '{')
    and return its full (possibly nested) braced content.
    """
    if not text:
        return None
    s = str(text)

    last = None
    for m in _BOXED_START_RE.finditer(s):
        last = m
    if last is None:
        return None

    # m.group(0) ends with '{' due to regex, so the '{' position is:
    lbrace_pos = last.end() - 1
    got = _extract_braced_content(s, lbrace_pos)
    if not got:
        return None

    content, _end = got
    content = content.strip()
    return content if content else None

def _find_last_hash(text: str) -> Optional[str]:
    if not text:
        return None
    ms = _HASH_RE.findall(text)
    if not ms:
        return None
    cand = ms[-1].strip()
    return cand if cand else None


def _find_last_numlike(text: str) -> Optional[str]:
    if not text:
        return None
    ms = _NUM_RE.findall(text)
    if not ms:
        return None
    return ms[-1].strip()


def _alpha_from_index(idx: int) -> str:
    return ["A", "B", "C", "D"][idx] if 0 <= idx <= 3 else ""

def _parse_math_numeric(raw: str) -> ParsedPrediction:
    s = "" if raw is None else str(raw)

    boxed = _find_last_boxed(s)
    if boxed is None:
        return ParsedPrediction(
            pred_str="",          
            vote_key="",          
            ok=False,
            method="none",       
            debug={"boxed": None, "raw_tail": s[-200:]},
        )

    # boxed 
    cand = boxed
    method = "boxed"

    pred_norm = normalize_numeric_str(cand)
    pred_str = "" if pred_norm is None else pred_norm

    dec = try_parse_decimal(pred_norm)
    vote_key = str(dec) if dec is not None else pred_str

    ok = bool(pred_str)
    return ParsedPrediction(
        pred_str=pred_str,
        vote_key=vote_key,
        ok=ok,
        method=method,
        debug={"boxed": boxed, "raw_tail": s[-200:]},
    )

def _parse_math_numeric(raw: str) -> ParsedPrediction:
    s = "" if raw is None else str(raw)

    boxed = _find_last_boxed(s)
    if boxed is not None:
        cand = boxed
        method = "boxed"
    else:
        hashed = _find_last_hash(s)
        if hashed is not None:
            cand = hashed
            method = "hash"
        else:
            num = _find_last_numlike(s)
            cand = num if num is not None else ""
            method = "num_fallback" if num is not None else "empty"

    # normalize numeric string for evaluator input
    pred_norm = normalize_numeric_str(cand)
    pred_str = "" if pred_norm is None else pred_norm

    # vote_key: canonical Decimal string when possible, else pred_str
    dec = try_parse_decimal(pred_norm)
    vote_key = str(dec) if dec is not None else pred_str

    ok = bool(pred_str)
    return ParsedPrediction(
        pred_str=pred_str,
        vote_key=vote_key,
        ok=ok,
        method=method,
        debug={"boxed": boxed, "raw_tail": s[-200:]},
    )


def _parse_math_symbolic(raw: str) -> ParsedPrediction:
    s = "" if raw is None else str(raw)

    boxed = _find_last_boxed(s)
    if boxed is not None:
        cand = boxed
        method = "boxed"
    else:
        # fallback: take the last non-empty line
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        cand = lines[-1] if lines else ""
        method = "last_line_fallback" if cand else "empty"

    # pred_str: keep as string (light normalization optional)
    pred_norm = normalize_latexish(cand)
    pred_str = "" if pred_norm is None else pred_norm

    vote_key = pred_str  # already normalized enough for voting
    ok = bool(pred_str)

    return ParsedPrediction(
        pred_str=pred_str,
        vote_key=vote_key,
        ok=ok,
        method=method,
        debug={"boxed": boxed, "raw_tail": s[-200:]},
    )


def _parse_mcq(raw: str) -> ParsedPrediction:
    s = "" if raw is None else str(raw)

    boxed = _find_last_boxed(s)
    if boxed is not None:
        # boxed content could be "C" or "2"
        idx = parse_mcq_choice(boxed)
        method = "boxed"
    else:
        idx = parse_mcq_choice(s)
        method = "regex"

    if idx is None:
        return ParsedPrediction(
            pred_str="",
            vote_key="",
            ok=False,
            method="empty",
            debug={"raw_tail": s[-200:]},
        )

    pred_str = _alpha_from_index(idx)  # standardized human-readable
    vote_key = str(idx)                # stable for voting
    return ParsedPrediction(
        pred_str=pred_str,
        vote_key=vote_key,
        ok=True,
        method=method,
        debug={"idx": idx, "raw_tail": s[-200:]},
    )


def _parse_code_unit_test(raw: str, meta: Dict[str, Any]) -> ParsedPrediction:
    """
    For HumanEval: pred should be "completion code" appended to prompt.
    We intentionally do NOT parse boxed.
    Minimal heuristic:
      - return raw as-is (strip leading spaces only)
    You can improve later: strip markdown fences, etc.
    """
    s = "" if raw is None else str(raw)

    # strip common markdown code fences
    # ```python ... ```
    fence = re.search(r"```(?:python)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s2 = fence.group(1).strip("\n")
        method = "code_fence"
    else:
        s2 = s
        method = "raw"

    ok = bool(s2.strip())
    return ParsedPrediction(
        pred_str=s2,
        vote_key="",  # voting not recommended for code; leave empty
        ok=ok,
        method=method,
        debug={"raw_tail": s[-200:]},
    )


def parse_prediction(raw_text: str, task_type: str, meta: Optional[Dict[str, Any]] = None) -> ParsedPrediction:
    """
    Unified entry point.
    """
    meta = meta or {}

    if task_type == TASK_MATH_NUMERIC:
        return _parse_math_numeric(raw_text)
    if task_type == TASK_MATH_SYMBOLIC:
        return _parse_math_symbolic(raw_text)
    if task_type == TASK_MCQ:
        return _parse_mcq(raw_text)
    if task_type == TASK_CODE_UNIT_TEST:
        return _parse_code_unit_test(raw_text, meta)

    # default fallback: return last boxed if any else last line
    s = "" if raw_text is None else str(raw_text)
    boxed = _find_last_boxed(s)
    if boxed is not None:
        pred = boxed.strip()
        method = "boxed_default"
    else:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        pred = lines[-1] if lines else ""
        method = "last_line_default" if pred else "empty_default"

    ok = bool(pred)
    return ParsedPrediction(pred_str=pred, vote_key=pred, ok=ok, method=method, debug={"raw_tail": s[-200:]})
