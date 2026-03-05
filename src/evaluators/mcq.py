# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

from src.eval import BaseEvaluator, EvalResult, EvaluatorFactory, parse_mcq_choice
from src.dataset import TASK_MCQ


class MCQEvaluator(BaseEvaluator):
    task_type = TASK_MCQ

    def evaluate(self, pred: str, gold: Any, meta: Dict[str, Any]) -> EvalResult:
        pred_idx = parse_mcq_choice(pred)

        # gold in MMLU is typically int index 0-3 (keep as-is)
        try:
            gold_idx = int(gold) if gold is not None else None
        except (TypeError, ValueError):
            gold_idx = None

        correct = (pred_idx is not None and gold_idx is not None and pred_idx == gold_idx)

        return EvalResult(
            correct=correct,
            gold_norm=str(gold_idx) if gold_idx is not None else None,
            pred_norm=str(pred_idx) if pred_idx is not None else None,
            details={
                "gold_idx": gold_idx,
                "pred_idx": pred_idx,
                "subject": meta.get("subject"),
            },
        )


EvaluatorFactory.register(TASK_MCQ, MCQEvaluator)
