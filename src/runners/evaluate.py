# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---- bootstrap so `import src...` works when running by file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import loaders  # noqa: F401
from src import evaluators  # noqa: F401

from src.dataset import load_samples, TASK_MCQ, TASK_CODE_UNIT_TEST
from src.eval import EvaluatorFactory


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_predictions_file(pred_path: Path) -> Dict[str, str]:
    """
    Prediction file format (jsonl), one per line:
      {"uid": "...", "pred": "..."}
    Returns uid -> pred
    """
    uid2pred: Dict[str, str] = {}
    for obj in iter_jsonl(pred_path):
        uid = obj.get("uid")
        pred = obj.get("pred")
        if uid is None:
            continue
        uid2pred[str(uid)] = "" if pred is None else str(pred)
    return uid2pred


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g., gsm8k, mmlu, humaneval, math500, ...")
    ap.add_argument("--split", required=True, help="e.g., test, train")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)  # reserved for future sampling
    ap.add_argument("--pred_path", type=str, default=None, help="jsonl with {uid, pred}. If omitted, uses gold-as-pred (debug).")
    ap.add_argument("--out_path", type=str, default=None, help="where to save detailed results jsonl")

    args = ap.parse_args()

    samples = load_samples(args.dataset, args.split, data_root=args.data_root, limit=args.limit)
    if not samples:
        print("No samples loaded.")
        return

    task_type = samples[0].task_type
    ev = EvaluatorFactory.create(task_type)

    uid2pred: Optional[Dict[str, str]] = None
    if args.pred_path:
        uid2pred = load_predictions_file(Path(args.pred_path))

    correct = 0
    total = 0

    # For breakdowns
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})

    results_rows: List[Dict[str, Any]] = []

    for s in samples:
        if uid2pred is None:
            # debug fallback: use gold as pred
            if task_type == TASK_CODE_UNIT_TEST:
                pred = str(s.meta.get("canonical_solution", ""))  # will be appended to prompt in evaluator
            else:
                pred = "" if s.gold is None else str(s.gold)
        else:
            pred = uid2pred.get(s.uid, "")

        r = ev.evaluate(pred=pred, gold=s.gold, meta=s.meta)
        total += 1
        if r.correct:
            correct += 1

        # subject breakdown for MMLU (and any dataset providing meta["subject"])
        subj = s.meta.get("subject")
        if subj is not None:
            by_subject[str(subj)]["total"] += 1
            if r.correct:
                by_subject[str(subj)]["correct"] += 1

        results_rows.append(
            {
                "uid": s.uid,
                "dataset": args.dataset,
                "split": args.split,
                "task_type": s.task_type,
                "prompt": s.prompt,
                "gold": s.gold,
                "pred": pred,
                "correct": r.correct,
                "gold_norm": r.gold_norm,
                "pred_norm": r.pred_norm,
                "details": r.details,
                "meta": s.meta,
            }
        )

    acc = correct / total if total else 0.0
    print(f"[evaluate] dataset={args.dataset} split={args.split} task_type={task_type}")
    print(f"[evaluate] correct={correct}/{total}  ({acc*100:.2f}%)")

    # Print per-subject top lines (useful for mmlu)
    if by_subject:
        print("\n[breakdown] by subject:")
        # sort by total desc
        items = sorted(by_subject.items(), key=lambda x: x[1]["total"], reverse=True)
        for subj, st in items[:30]:
            a = st["correct"] / st["total"] if st["total"] else 0.0
            print(f"  {subj:<30} {st['correct']}/{st['total']} ({a*100:.1f}%)")
        if len(items) > 30:
            print(f"  ... ({len(items)-30} more)")

    if args.out_path:
        out_path = Path(args.out_path)
        save_jsonl(out_path, results_rows)
        print(f"\n[evaluate] saved detailed results to: {out_path}")


if __name__ == "__main__":
    main()
