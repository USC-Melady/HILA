# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---- bootstrap
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import loaders  # noqa: F401
from src import evaluators  # noqa: F401

from src.dataset import get_dataset_specs, load_samples, TASK_MCQ, TASK_CODE_UNIT_TEST
from src.eval import EvaluatorFactory
from src.parsing import parse_prediction


def _idx_to_letter(idx: int) -> str:
    return ["A", "B", "C", "D"][idx]


def build_mock_model_output(sample) -> str:
    """
    Simulate "ideal" model output to validate parser + evaluator consistency.
    """
    if sample.task_type == TASK_CODE_UNIT_TEST:
        # for humaneval, pretend model outputs canonical solution completion
        return str(sample.meta.get("canonical_solution", ""))

    if sample.task_type == TASK_MCQ:
        # gold is index 0-3
        try:
            idx = int(sample.gold)
        except Exception:
            idx = 0
        return f"Reasoning...\n\\boxed{{{_idx_to_letter(idx)}}}\n"

    # math numeric/symbolic etc.
    return f"Reasoning...\n\\boxed{{{sample.gold}}}\n"


def main():
    data_root = "data"
    n = 500
    seed = 42

    specs = get_dataset_specs(data_root=data_root)
    rng = random.Random(seed)

    print(f"[dryrun_parse] data_root={data_root} n={n} seed={seed}\n")

    overall_eval_ok = 0
    overall_total = 0
    overall_parse_ok = 0

    for name, spec in specs.items():
        for split in spec.splits.keys():
            samples = load_samples(name, split, data_root=data_root)
            if not samples:
                continue

            picked = samples if len(samples) <= n else rng.sample(samples, n)

            ev = EvaluatorFactory.create(picked[0].task_type)

            parse_ok = 0
            eval_ok = 0

            for s in picked:
                raw = build_mock_model_output(s)
                parsed = parse_prediction(raw, s.task_type, s.meta)

                if parsed.ok:
                    parse_ok += 1

                r = ev.evaluate(pred=parsed.pred_str, gold=s.gold, meta=s.meta)
                if r.correct:
                    eval_ok += 1

            overall_parse_ok += parse_ok
            overall_eval_ok += eval_ok
            overall_total += len(picked)

            parse_rate = parse_ok / len(picked) * 100.0
            acc = eval_ok / len(picked) * 100.0

            print(f"{name:<10} {split:<8} task={spec.task_type:<15} parse_ok={parse_ok}/{len(picked)} ({parse_rate:.1f}%)  eval={eval_ok}/{len(picked)} ({acc:.1f}%)")

        print()

    if overall_total:
        print(f"[dryrun_parse] overall parse_ok={overall_parse_ok}/{overall_total} ({overall_parse_ok/overall_total*100:.1f}%)")
        print(f"[dryrun_parse] overall eval    ={overall_eval_ok}/{overall_total} ({overall_eval_ok/overall_total*100:.1f}%)")


if __name__ == "__main__":
    main()
