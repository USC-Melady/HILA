# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# add project root to sys.path so `import src...` works even when running by file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../HumanAgent
sys.path.insert(0, str(PROJECT_ROOT))


#from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from src import loaders  # noqa: F401  # trigger loader registration
from src import evaluators  # noqa: F401  # trigger evaluator registration

from src.dataset import get_dataset_specs, load_samples, TASK_CODE_UNIT_TEST
from src.eval import EvaluatorFactory


def _pick_pred(sample) -> str:
    # For most tasks, using gold as pred should pass
    if sample.task_type != TASK_CODE_UNIT_TEST:
        return "" if sample.gold is None else str(sample.gold)

    # For HumanEval, pred should be the completion appended to prompt.
    # canonical_solution is usually the function body; we keep it in meta.
    return str(sample.meta.get("canonical_solution", ""))


def dryrun_one_dataset(dataset_name: str, split: str, data_root: str, n: int, seed: int) -> Tuple[int, int, List[Tuple[str, str]]]:
    samples = load_samples(dataset_name, split, data_root=data_root)
    if not samples:
        return 0, 0, []

    rng = random.Random(seed)
    picked = samples if len(samples) <= n else rng.sample(samples, n)

    ev = EvaluatorFactory.create(picked[0].task_type)

    ok = 0
    failures: List[Tuple[str, str]] = []
    for s in picked:
        pred = _pick_pred(s)
        r = ev.evaluate(pred=pred, gold=s.gold, meta=s.meta)
        if r.correct:
            ok += 1
        else:
            # keep short failure summary
            err = r.details.get("error") if isinstance(r.details, dict) else None
            summary = err or f"gold_norm={r.gold_norm} pred_norm={r.pred_norm}"
            failures.append((s.uid, summary))

    return ok, len(picked), failures


def main():
    data_root = "data"
    n = 5
    seed = 42
    max_fail_print = 3

    specs = get_dataset_specs(data_root=data_root)

    print(f"[dryrun] data_root={data_root} n={n} seed={seed}\n")

    total_ok, total_n = 0, 0

    for name, spec in specs.items():
        for split in spec.splits.keys():
            try:
                ok, nn, failures = dryrun_one_dataset(name, split, data_root, n, seed)
                total_ok += ok
                total_n += nn
                rate = (ok / nn * 100.0) if nn > 0 else 0.0
                print(f"{name:<10} {split:<8} task={spec.task_type:<15}  {ok}/{nn}  ({rate:.1f}%)")

                if failures:
                    for uid, msg in failures[:max_fail_print]:
                        print(f"  - FAIL {uid}: {msg}")
            except Exception as e:
                print(f"{name:<10} {split:<8} task={spec.task_type:<15}  ERROR: {type(e).__name__}: {e}")

        print()

    overall = (total_ok / total_n * 100.0) if total_n > 0 else 0.0
    print(f"[dryrun] overall: {total_ok}/{total_n} ({overall:.1f}%)")


if __name__ == "__main__":
    main()
