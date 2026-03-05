# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from src.dataset import (
    BaseLoader,
    DatasetSpec,
    Sample,
    LoaderFactory,
    TASK_MATH_SYMBOLIC,
    iter_jsonl,
)


class Math500Loader(BaseLoader):
    """
    For MATH-500 style problems:
    - prompt: question/problem
    - gold: answer (string, may be latex/fractions/tuples)
    - meta: reasoning/solution, subject, level, id, url, etc.
    """
    task_type = TASK_MATH_SYMBOLIC

    def load(self, path: Path, spec: DatasetSpec) -> Iterable[Sample]:
        q_key = spec.field_map.get("question", "question")
        a_key = spec.field_map.get("answer", "answer")

        reasoning_key = spec.field_map.get("reasoning", "reasoning")
        solution_key = spec.field_map.get("solution", "solution")
        subject_key = spec.field_map.get("subject", "subject")
        level_key = spec.field_map.get("level", "level")
        id_key = spec.field_map.get("id", "id")
        unique_id_key = spec.field_map.get("unique_id", "unique_id")
        url_key = spec.field_map.get("url", "url")
        source_key = spec.field_map.get("source", "source")
        human_idea_key = spec.field_map.get("human_idea", "human_idea")
        human_reasoning_key = spec.field_map.get("human_reasoning", "human_reasoning")

        for idx, row in enumerate(iter_jsonl(path)):
            question = row.get(q_key, "")
            prompt = (
                spec.prompt_builder(row, spec)
                if spec.prompt_builder is not None
                else str(question)
            )

            gold = row.get(a_key, None)

            meta: Dict[str, Any] = {"_row_index": idx}

            if reasoning_key in row and row[reasoning_key] is not None:
                meta["reasoning"] = row[reasoning_key]
            if solution_key in row and row[solution_key] is not None:
                meta["solution"] = row[solution_key]
            if subject_key in row and row[subject_key] is not None:
                meta["subject"] = row[subject_key]
            if level_key in row and row[level_key] is not None:
                meta["level"] = row[level_key]
            if id_key in row and row[id_key] is not None:
                meta["id"] = row[id_key]
            if unique_id_key in row and row[unique_id_key] is not None:
                meta["unique_id"] = row[unique_id_key]
            if url_key in row and row[url_key] is not None:
                meta["url"] = row[url_key]
            if source_key in row and row[source_key] is not None:
                meta["source"] = row[source_key]

            # standardized keys in meta
            if human_idea_key in row and row[human_idea_key] is not None:
                meta["human_idea"] = row[human_idea_key]
            if human_reasoning_key in row and row[human_reasoning_key] is not None:
                meta["human_reasoning"] = row[human_reasoning_key]

            uid = f"{spec.name}:{path.name}:{idx}"
            yield Sample(
                uid=uid,
                task_type=spec.task_type,
                prompt=str(prompt),
                gold=gold,
                meta=meta,
            )


# Register into factory
LoaderFactory.register(TASK_MATH_SYMBOLIC, Math500Loader)