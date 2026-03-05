# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

from src.eval import BaseEvaluator, EvalResult, EvaluatorFactory, normalize_latexish
from src.dataset import TASK_MATH_SYMBOLIC


class MathSymbolicEvaluator(BaseEvaluator):
    task_type = TASK_MATH_SYMBOLIC

    def evaluate(self, pred: str, gold: Any, meta: Dict[str, Any]) -> EvalResult:
        gold_s = normalize_latexish(gold)
        pred_s = normalize_latexish(pred)

        # Baseline: exact match after normalization
        correct = (gold_s is not None and pred_s is not None and gold_s == pred_s)

        return EvalResult(
            correct=correct,
            gold_norm=gold_s,
            pred_norm=pred_s,
            details={},
        )


EvaluatorFactory.register(TASK_MATH_SYMBOLIC, MathSymbolicEvaluator)
