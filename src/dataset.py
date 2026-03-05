# -*- coding: utf-8 -*-
"""
Dataset loading core:
- unified Sample schema: uid, task_type, prompt, gold, meta
- DatasetSpec registry in one place
- load_samples() uses task_type -> loader routing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Type
import json


# -----------------------------
# Task types (router keys)
# -----------------------------
TASK_MATH_NUMERIC = "math_numeric"
TASK_MATH_SYMBOLIC = "math_symbolic"
TASK_MCQ = "mcq"
TASK_CODE_UNIT_TEST = "code_unit_test"

ALL_TASK_TYPES = {
    TASK_MATH_NUMERIC,
    TASK_MATH_SYMBOLIC,
    TASK_MCQ,
    TASK_CODE_UNIT_TEST,
}


# -----------------------------
# Unified in-memory sample
# -----------------------------
@dataclass(frozen=True)
class Sample:
    uid: str
    task_type: str
    prompt: str
    gold: Any  # keep original; evaluator will normalize/parse
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Dataset spec (per dataset)
# -----------------------------
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_type: str
    splits: Dict[str, str]  # split -> relative path from data_root
    # map standardized keys -> json keys in file row
    # e.g. {"question": "question", "answer": "answer"} or {"question": "problem", "answer": "answer"}
    field_map: Dict[str, str] = field(default_factory=dict)

    # optional: custom prompt builder; signature (row, spec) -> prompt string
    prompt_builder: Optional[Callable[[Dict[str, Any], "DatasetSpec"], str]] = None

    def resolve_path(self, split: str, data_root: str | Path) -> Path:
        if split not in self.splits:
            raise KeyError(
                f"Split '{split}' not found for dataset '{self.name}'. "
                f"Available: {sorted(self.splits.keys())}"
            )
        return Path(data_root) / self.splits[split]


# -----------------------------
# Loader interface + factory
# -----------------------------
class BaseLoader:
    """Load a jsonl file and yield Sample objects."""
    task_type: str  # must match one of ALL_TASK_TYPES

    def load(self, path: Path, spec: DatasetSpec) -> Iterable[Sample]:
        raise NotImplementedError


class LoaderFactory:
    _registry: Dict[str, Type[BaseLoader]] = {}

    @classmethod
    def register(cls, task_type: str, loader_cls: Type[BaseLoader]) -> None:
        if task_type not in ALL_TASK_TYPES:
            raise ValueError(f"Unknown task_type '{task_type}'. Known: {sorted(ALL_TASK_TYPES)}")
        cls._registry[task_type] = loader_cls

    @classmethod
    def create(cls, task_type: str) -> BaseLoader:
        if task_type not in cls._registry:
            raise KeyError(
                f"No loader registered for task_type='{task_type}'. "
                f"Registered: {sorted(cls._registry.keys())}"
            )
        return cls._registry[task_type]()


# -----------------------------
# JSONL utilities
# -----------------------------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {path} at line {line_no}: {e}") from e


# -----------------------------
# Prompt builder defaults
# -----------------------------
def default_qa_prompt_builder(row: Dict[str, Any], spec: DatasetSpec) -> str:
    # "question" is the standardized key; look up actual key via field_map
    q_key = spec.field_map.get("question", "question")
    q = row.get(q_key, "")
    return str(q)


# -----------------------------
# Central dataset registry
# -----------------------------
def get_dataset_specs(data_root: str | Path = "data") -> Dict[str, DatasetSpec]:
    """
    Keep all dataset specs in one place.
    Paths are relative to data_root.
    You can adjust filenames to match what you actually saved.
    """
    specs: Dict[str, DatasetSpec] = {
        # GSM8K
        "gsm8k": DatasetSpec(
            name="gsm8k",
            task_type=TASK_MATH_NUMERIC,
            splits={
                "train": "gsm8k/gsm8k_train.jsonl",
                "test": "gsm8k/gsm8k_test.jsonl",
                "human": "gsm8k/gsm8k_real_human.jsonl",
            },
            field_map={
                "question": "question",
                "answer": "answer",
                "reasoning": "reasoning",
                "human_idea": "human_idea",
                "human_reasoning": "human_reasoning",
            },
            prompt_builder=default_qa_prompt_builder,
        ),

        # AMC
        "amc": DatasetSpec(
            name="amc",
            task_type=TASK_MATH_NUMERIC,
            splits={
                "test": "amc/amc_test.jsonl",
                "human": "amc/amc_real_human.jsonl",
            },
            field_map={
                "question": "question",
                "answer": "answer",
                "reasoning": "reasoning",
                "url": "url",
                "id": "id",
                "human_idea": "human_idea",
                "human_reasoning": "human_reasoning",
            },
            prompt_builder=default_qa_prompt_builder,
        ),

        # AIME24
        "aime24": DatasetSpec(
            name="aime24",
            task_type=TASK_MATH_NUMERIC,
            splits={
                "test": "aime24/aime24_test.jsonl",
            },
            field_map={
                "question": "question",
                "answer": "answer",
                "reasoning": "reasoning",
                "url": "url",
                "id": "id",
                "human_idea": "human_idea",
                "human_reasoning": "human_reasoning",
            },
            prompt_builder=default_qa_prompt_builder,
        ),

        # AIME25
        "aime25": DatasetSpec(
            name="aime25",
            task_type=TASK_MATH_NUMERIC,
            splits={
                "test": "aime25/aime25_test.jsonl",
            },
            field_map={
                "question": "question",
                "answer": "answer",
                "reasoning": "reasoning",
                "source": "source",
                "human_idea": "human_idea",
                "human_reasoning": "human_reasoning",
            },
            prompt_builder=default_qa_prompt_builder,
        ),

        # MATH-500
        "math500": DatasetSpec(
            name="math500",
            task_type=TASK_MATH_SYMBOLIC,
            splits={
                "test": "math500/math500_test.jsonl",
                "human": "math500/math500_human.jsonl",
            },
            field_map={
                "question": "question",
                "answer": "answer",
                "reasoning": "reasoning",
                "subject": "subject",
                "level": "level",
                "id": "id",
                "human_idea": "human_idea",
                "human_reasoning": "human_reasoning",
            },
            prompt_builder=default_qa_prompt_builder,
        ),

        # MMLU
        "mmlu": DatasetSpec(
            name="mmlu",
            task_type=TASK_MCQ,
            splits={
                "test": "mmlu/mmlu_test.jsonl",
                "human": "mmlu/mmlu_real_human.jsonl",
            },
            field_map={
                "question": "question",
                "choices": "choices",
                "answer": "answer",
                "subject": "subject",
                "human_idea": "human_idea",
                "human_reasoning": "human_reasoning",
            },
            prompt_builder=default_qa_prompt_builder,
        ),

        # HumanEval
        "humaneval": DatasetSpec(
            name="humaneval",
            task_type=TASK_CODE_UNIT_TEST,
            splits={
                "test": "humaneval/humaneval_test.jsonl",
            },
            field_map={
                "task_id": "task_id",
                "prompt": "prompt",
                "canonical_solution": "canonical_solution",
                "test": "test",
                "entry_point": "entry_point",
                "human_idea": "human_idea",
                "human_reasoning": "human_reasoning",
            },
            prompt_builder=None,
        ),
    }

    # sanity check: ensure paths are relative
    for spec in specs.values():
        if spec.task_type not in ALL_TASK_TYPES:
            raise ValueError(f"Spec '{spec.name}' has unknown task_type '{spec.task_type}'.")
        for split, relpath in spec.splits.items():
            if Path(relpath).is_absolute():
                raise ValueError(f"Spec '{spec.name}' split '{split}' uses absolute path: {relpath}")

    return specs


# -----------------------------
# Public API
# -----------------------------
def load_samples(
    dataset_name: str,
    split: str,
    data_root: str | Path = "data",
    *,
    limit: Optional[int] = None,
) -> List[Sample]:
    """
    Load samples for a dataset split and return a list[Sample].
    loader does NOT transform answers; gold is kept as-is.
    """
    specs = get_dataset_specs(data_root=data_root)
    if dataset_name not in specs:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {sorted(specs.keys())}")

    spec = specs[dataset_name]
    path = spec.resolve_path(split=split, data_root=data_root)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    loader = LoaderFactory.create(spec.task_type)
    samples: List[Sample] = []
    for s in loader.load(path, spec):
        samples.append(s)
        if limit is not None and len(samples) >= limit:
            break
    return samples