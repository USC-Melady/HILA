# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from src.dataset import (
    BaseLoader,
    DatasetSpec,
    Sample,
    LoaderFactory,
    TASK_CODE_UNIT_TEST,
    iter_jsonl,
)


class CodeUnitTestLoader(BaseLoader):
    """
    For HumanEval-like datasets:
    - prompt: code prompt (function signature + docstring)
    - gold: keep canonical_solution as-is (mainly for debugging/sanity)
    - meta: keep task_id, entry_point, test, canonical_solution, etc.
    """
    task_type = TASK_CODE_UNIT_TEST

    def load(self, path: Path, spec: DatasetSpec) -> Iterable[Sample]:
        prompt_key = spec.field_map.get("prompt", "prompt")
        test_key = spec.field_map.get("test", "test")
        entry_key = spec.field_map.get("entry_point", "entry_point")
        task_id_key = spec.field_map.get("task_id", "task_id")
        canon_key = spec.field_map.get("canonical_solution", "canonical_solution")
        human_idea_key = spec.field_map.get("human_idea", "human_idea")
        human_reasoning_key = spec.field_map.get("human_reasoning", "human_reasoning")

        for idx, row in enumerate(iter_jsonl(path)):
            prompt = row.get(prompt_key, "")
            test_code = row.get(test_key, None)
            entry_point = row.get(entry_key, None)
            task_id = row.get(task_id_key, None)
            canonical_solution = row.get(canon_key, None)

            # gold kept raw; evaluator will not use it for unit tests, but it's useful for debugging.
            gold = canonical_solution

            meta: Dict[str, Any] = {
                "_row_index": idx,
                "task_id": task_id,
                "entry_point": entry_point,
                "test": test_code,
                "prompt": str(prompt),
            }

            # Keep canonical solution + full reference program (optional but handy)
            if canonical_solution is not None:
                meta["canonical_solution"] = canonical_solution
                meta["reference_code"] = str(prompt) + str(canonical_solution)

            # standardized keys in meta
            if human_idea_key in row and row[human_idea_key] is not None:
                meta["human_idea"] = row[human_idea_key]
            if human_reasoning_key in row and row[human_reasoning_key] is not None:
                meta["human_reasoning"] = row[human_reasoning_key]

            uid = f"{spec.name}:{path.name}:{idx}" if task_id is None else str(task_id)
            yield Sample(
                uid=uid,
                task_type=spec.task_type,
                prompt=str(prompt),
                gold=gold,
                meta=meta,
            )


# Register into factory
LoaderFactory.register(TASK_CODE_UNIT_TEST, CodeUnitTestLoader)