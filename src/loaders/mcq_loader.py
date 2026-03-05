# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.dataset import (
    BaseLoader,
    DatasetSpec,
    Sample,
    LoaderFactory,
    TASK_MCQ,
    iter_jsonl,
)
from src.prompts import build_mmlu_prompt


class MCQLoader(BaseLoader):
    """
    For multiple-choice QA (e.g., MMLU).
    - prompt: question + formatted choices
    - gold: keep as-is (MMLU uses int index)
    - meta: keep choices list + subject + optional extras
    """
    task_type = TASK_MCQ

    def load(self, path: Path, spec: DatasetSpec) -> Iterable[Sample]:
        q_key = spec.field_map.get("question", "question")
        c_key = spec.field_map.get("choices", "choices")
        a_key = spec.field_map.get("answer", "answer")
        subj_key = spec.field_map.get("subject", "subject")
        human_idea_key = spec.field_map.get("human_idea", "human_idea")
        human_reasoning_key = spec.field_map.get("human_reasoning", "human_reasoning")

        for idx, row in enumerate(iter_jsonl(path)):
            question = row.get(q_key, "")
            choices = row.get(c_key, [])

            # Ensure choices is a list of strings
            if choices is None:
                choices = []
            if not isinstance(choices, list):
                raw_choices = choices
                choices = [str(raw_choices)]
            else:
                choices = [str(x) for x in choices]

            prompt = build_mmlu_prompt(str(question), choices)
            gold = row.get(a_key, None)

            meta: Dict[str, Any] = {
                "choices": choices,
                "_row_index": idx,
            }

            if subj_key in row and row[subj_key] is not None:
                meta["subject"] = row[subj_key]

            # standardized keys in meta
            if human_idea_key in row and row[human_idea_key] is not None:
                meta["human_idea"] = row[human_idea_key]
            if human_reasoning_key in row and row[human_reasoning_key] is not None:
                meta["human_reasoning"] = row[human_reasoning_key]

            uid = f"{spec.name}:{path.name}:{idx}"
            yield Sample(
                uid=uid,
                task_type=spec.task_type,
                prompt=prompt,
                gold=gold,
                meta=meta,
            )


# Register into factory
LoaderFactory.register(TASK_MCQ, MCQLoader)