# -*- coding: utf-8 -*-
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.eval import BaseEvaluator, EvalResult, EvaluatorFactory
from src.dataset import TASK_CODE_UNIT_TEST


@dataclass
class _RunResult:
    ok: bool
    error: Optional[str] = None


def _worker_exec(code: str, test_code: str, entry_point: str, q: mp.Queue) -> None:
    """
    Run untrusted-ish code in a child process.
    If everything passes, put ok=True else ok=False with error.
    """
    try:
        g: Dict[str, Any] = {}
        exec(code, g, g)

        if entry_point not in g or not callable(g.get(entry_point)):
            q.put(_RunResult(False, f"entry_point '{entry_point}' not found or not callable"))
            return

        # HumanEval tests typically call check(candidate)
        g["candidate"] = g[entry_point]

        exec(test_code, g, g)
        q.put(_RunResult(True, None))
    except BaseException as e:
        q.put(_RunResult(False, f"{type(e).__name__}: {e}"))


def run_with_timeout(code: str, test_code: str, entry_point: str, timeout_s: float) -> _RunResult:
    # Prefer fork on POSIX/Linux to avoid __main__ re-import issues with spawn.
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        # e.g., on Windows fork is unavailable
        ctx = mp.get_context("spawn")

    q: mp.Queue = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_worker_exec, args=(code, test_code, entry_point, q), daemon=True)
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return _RunResult(False, f"Timeout after {timeout_s:.2f}s")

    if not q.empty():
        return q.get()

    return _RunResult(False, "No result returned from worker process")


import re

def _defines_entry_point(code: str, entry_point: str) -> bool:
    # 匹配顶层函数定义：def entry_point(
    # (允许前导空白，避免 code block 内缩进)
    pat = rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\("
    return re.search(pat, code) is not None

class CodeUnitTestEvaluator(BaseEvaluator):
    """
    HumanEval-style evaluator:
    - pred: model completion string (expected to be appended to prompt)
    - meta must include: prompt, test, entry_point
    """
    task_type = TASK_CODE_UNIT_TEST

    def evaluate(self, pred: str, gold: Any, meta: Dict[str, Any]) -> EvalResult:
        prompt = meta.get("prompt", "")
        test_code = meta.get("test", None)
        entry_point = meta.get("entry_point", None)

        timeout_s = float(meta.get("timeout_s", 3.0))

        if not isinstance(test_code, str) or not test_code.strip():
            return EvalResult(
                correct=False,
                details={"error": "Missing test code in meta['test']"},
            )
        if not isinstance(entry_point, str) or not entry_point.strip():
            return EvalResult(
                correct=False,
                details={"error": "Missing entry point in meta['entry_point']"},
            )
        '''
        pred_str = "" if pred is None else str(pred)
        if "def " in pred_str and entry_point in pred_str and prompt.strip() == "":
            full_code = pred_str
        else:
            full_code = str(prompt) + pred_str'''

        pred_str = "" if pred is None else str(pred)
        # 
        fence = re.search(r"```(?:python)?\s*(.*?)```", pred_str, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            pred_str = fence.group(1)
        
        if _defines_entry_point(pred_str, entry_point):
            full_code = pred_str                      # full functions
        else:
            full_code = str(prompt) + pred_str        # completion

        rr = run_with_timeout(full_code, test_code, entry_point, timeout_s=timeout_s)

        return EvalResult(
            correct=rr.ok,
            gold_norm=None,
            pred_norm=None,
            details={
                "ok": rr.ok,
                "error": rr.error,
                "entry_point": entry_point,
                "timeout_s": timeout_s,
            },
        )


EvaluatorFactory.register(TASK_CODE_UNIT_TEST, CodeUnitTestEvaluator)
