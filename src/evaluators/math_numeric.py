# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

from src.eval import BaseEvaluator, EvalResult, EvaluatorFactory, normalize_numeric_str, try_parse_decimal
from src.dataset import TASK_MATH_NUMERIC


class MathNumericEvaluator(BaseEvaluator):
    task_type = TASK_MATH_NUMERIC

    def evaluate(self, pred: str, gold: Any, meta: Dict[str, Any]) -> EvalResult:
        gold_s = normalize_numeric_str(gold)
        pred_s = normalize_numeric_str(pred)

        gold_d = try_parse_decimal(gold_s)
        pred_d = try_parse_decimal(pred_s)

        correct = (gold_d is not None and pred_d is not None and gold_d == pred_d)

        return EvalResult(
            correct=correct,
            gold_norm=gold_s,
            pred_norm=pred_s,
            details={
                "gold_decimal": str(gold_d) if gold_d is not None else None,
                "pred_decimal": str(pred_d) if pred_d is not None else None,
            },
        )


EvaluatorFactory.register(TASK_MATH_NUMERIC, MathNumericEvaluator)
