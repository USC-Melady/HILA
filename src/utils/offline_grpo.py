#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_grpo_reward.py
~~~~~~~~~~~~~~~~~~~~~~
Offline reward builder for GRPO grouped dataset.

Input (your saved file):
  - grpo_groups.jsonl (one group per line)
    Each group has:
      - policy_prompt (state x)
      - candidates: list of { action_str, answer_text, (optional) answer_parsed, ... }
      - task_type, gold, meta, ...

Output (training-ready GRPO jsonl):
  Each line is one group:
    {
      "prompt": <policy_prompt>,
      "completions": [ "EVAL 0", "EVAL 1", "CREATE", "DEFER", ... ],
      "rewards": [ r0, r1, ... ],
      "meta": {...},
      "debug": {... optional ...}
    }

Reward design:
  r = correct_scale * 1[correct]
      - lambda_defer  * 1[action==DEFER]
      - lambda_create * 1[action==CREATE]
      - lambda_len    * norm_len(answer_text)     (optional)
      - lambda_parse_fail * 1[parse_ok==False]    (optional)

Notes:
- correctness is computed via EvaluatorFactory + parse_prediction(pred_str)
- We DO NOT require adding new tokens for actions.

New:
- You can now control the kept ratio between:
    1) groups where all actions produce the same answer
    2) groups where at least one action produces a different answer

Example:
  --same_answer_ratio 1 --mixed_answer_ratio 3
means keep roughly:
  same-answer : mixed-answer = 1 : 3
while keeping as many samples as possible under that ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
# trigger registration (if you rely on side-effect registries)
from src import loaders, evaluators  # noqa: F401

from src.parsing import parse_prediction
from src.eval import EvaluatorFactory


# -----------------------------
# Utilities
# -----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON on line {ln} in {path}: {e}")
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def action_type(action_str: str) -> str:
    s = (action_str or "").strip().upper()
    if s.startswith("EVAL"):
        return "EVAL"
    if s == "CREATE":
        return "CREATE"
    if s == "DEFER":
        return "DEFER"
    # tolerate variants
    if "CREATE" in s:
        return "CREATE"
    if "DEFER" in s or "HUMAN" in s:
        return "DEFER"
    if any(ch.isdigit() for ch in s):
        return "EVAL"
    return "UNKNOWN"


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalized_length_penalty(text: str, denom: int = 512) -> float:
    """
    A gentle length penalty: len(text)/denom, capped at 2.0
    Using characters instead of tokens keeps this script dependency-light.
    You can swap to tokenizer-based length if you want.
    """
    n = len(text or "")
    x = n / max(1, denom)
    return float(min(2.0, x))


def sanitize_policy_prompt(prompt: str) -> str:
    """
    Remove specific instruction text from the saved policy prompt
    while keeping all other content unchanged.

    Removed:
    1) "Think step by step, show your reasoning, and be careful with arithmetic."
       + following boxed-answer line
    2) The whole Guidance block:
       "Guidance:
        - Avoid choosing yourself for EVAL unless necessary.
        - If you are not confident which is correct, prefer DEFER."
    """
    if not prompt:
        return prompt

    # Remove the math reasoning / boxed-answer instruction block
    prompt = prompt.replace(
        "Think step by step, show your reasoning, and be careful with arithmetic.\n"
        "Must give the final answer in the form \\boxed{...}.",
        ""
    )

    # Remove the meta-policy guidance block
    prompt = prompt.replace(
        "Guidance:\n"
        "- Avoid choosing yourself for EVAL unless necessary.\n"
        "- If you are not confident which is correct, prefer DEFER.",
        ""
    )

    return prompt


# -----------------------------
# Reward config
# -----------------------------
@dataclass
class RewardConfig:
    correct_scale: float = 1.0
    lambda_defer: float = 0.3
    lambda_create: float = 0.1

    # optional shaping
    lambda_len: float = 0.0
    len_denom: int = 512

    lambda_parse_fail: float = 0.0

    # clamp reward range to stabilize RL
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None


# -----------------------------
# Core: compute reward for one candidate
# -----------------------------
def compute_candidate_reward(
    cand: Dict[str, Any],
    task_type: str,
    gold: Any,
    meta: Dict[str, Any],
    ev,
    cfg: RewardConfig,
    use_cached_parsed: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns: (reward, debug_dict)
    """
    act_str = cand.get("action_str", "")
    act = action_type(act_str)
    answer_text = cand.get("answer_text", "") or ""

    # Parse: prefer cached answer_parsed if present & requested
    parsed_pred_str: str = ""
    parsed_ok: bool = False
    parsed_vote_key: str = ""

    if use_cached_parsed:
        ap = cand.get("answer_parsed")
        if isinstance(ap, dict):
            parsed_pred_str = str(ap.get("pred_str", "") or "")
            parsed_vote_key = str(ap.get("vote_key", "") or "")
            parsed_ok = bool(ap.get("ok", False))

    if not parsed_pred_str and not parsed_vote_key:
        parsed = parse_prediction(answer_text, task_type, meta)
        parsed_pred_str = parsed.pred_str
        parsed_vote_key = parsed.vote_key
        parsed_ok = bool(parsed.ok)

    # Correctness via evaluator
    r_eval = ev.evaluate(parsed_pred_str, gold, meta)
    correct = bool(getattr(r_eval, "correct", False))

    # Base reward
    reward = cfg.correct_scale * (1.0 if correct else 0.0)

    # Action cost penalties
    if act == "DEFER":
        reward -= cfg.lambda_defer
    elif act == "CREATE":
        reward -= cfg.lambda_create

    # Optional shaping
    if cfg.lambda_len > 0:
        reward -= cfg.lambda_len * normalized_length_penalty(answer_text, denom=cfg.len_denom)

    if cfg.lambda_parse_fail > 0 and (not parsed_ok):
        reward -= cfg.lambda_parse_fail

    # Clamp
    if cfg.clamp_min is not None:
        reward = max(cfg.clamp_min, reward)
    if cfg.clamp_max is not None:
        reward = min(cfg.clamp_max, reward)

    debug = {
        "act": act,
        "action_str": act_str,
        "correct": correct,
        "parsed_ok": parsed_ok,
        "pred_str": parsed_pred_str,
        "vote_key": parsed_vote_key,
    }
    return float(reward), debug


# -----------------------------
# Build training groups
# -----------------------------
def build_training_group(
    g: Dict[str, Any],
    ev_cache: Dict[str, Any],
    cfg: RewardConfig,
    use_cached_parsed: bool = True,
    keep_debug: bool = True,
) -> Dict[str, Any]:
    """
    Convert one saved group into one training-ready group:
      - prompt
      - completions (actions)
      - rewards
      - meta
      - optional debug info
    """
    policy_prompt = g.get("policy_prompt") or g.get("prompt") or ""
    policy_prompt = sanitize_policy_prompt(policy_prompt)

    candidates = g.get("candidates", [])
    if not isinstance(candidates, list) or not policy_prompt:
        raise ValueError("Bad group format: missing policy_prompt or candidates list")

    task_type = g.get("task_type", "") or safe_get(g, "meta", "task_type", default="") or ""
    gold = g.get("gold", None)
    meta = g.get("meta", {}) if isinstance(g.get("meta", {}), dict) else {}

    if task_type not in ev_cache:
        ev_cache[task_type] = EvaluatorFactory.create(task_type)
    ev = ev_cache[task_type]

    completions: List[str] = []
    rewards: List[float] = []
    cand_debug: List[Dict[str, Any]] = []

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        act_str = str(cand.get("action_str", "") or "").strip()
        if not act_str:
            continue

        r, dbg = compute_candidate_reward(
            cand=cand,
            task_type=task_type,
            gold=gold,
            meta=meta,
            ev=ev,
            cfg=cfg,
            use_cached_parsed=use_cached_parsed,
        )
        completions.append(act_str)
        rewards.append(r)
        if keep_debug:
            cand_debug.append(dbg)

    out: Dict[str, Any] = {
        "prompt": policy_prompt,
        "completions": completions,
        "rewards": rewards,
        "meta": {
            "sample_id": g.get("sample_id", meta.get("sample_id")),
            "round": g.get("round", meta.get("round")),
            "agent_idx": g.get("agent_idx", meta.get("agent_idx")),
            "task_type": task_type,
        },
    }

    if keep_debug:
        out["debug"] = {
            "gold": gold,
            "candidates": cand_debug,
            "rollout_chosen_action": g.get("rollout_chosen_action"),
            "rollout_chosen_candidate_idx": g.get("rollout_chosen_candidate_idx"),
        }

    return out


# -----------------------------
# Reporting
# -----------------------------
def summarize(training_groups: List[Dict[str, Any]]) -> None:
    """
    Print high-signal stats:
    - total groups, total candidates
    - action distribution
    - average reward per action
    - correctness rate per action (from debug)
    """
    total_groups = len(training_groups)
    total_cands = sum(len(g.get("completions", [])) for g in training_groups)

    action_cnt = Counter()
    reward_sum = defaultdict(float)
    reward_cnt = Counter()

    correct_cnt = Counter()
    correct_tot = Counter()

    for g in training_groups:
        comps = g.get("completions", [])
        rews = g.get("rewards", [])
        dbg = safe_get(g, "debug", "candidates", default=[])
        for i, (c, r) in enumerate(zip(comps, rews)):
            at = action_type(c)
            action_cnt[at] += 1
            reward_sum[at] += float(r)
            reward_cnt[at] += 1

            # correctness from debug if available
            if isinstance(dbg, list) and i < len(dbg) and isinstance(dbg[i], dict):
                correct = bool(dbg[i].get("correct", False))
                correct_tot[at] += 1
                if correct:
                    correct_cnt[at] += 1

    print("\n===== GRPO Offline Reward Summary =====")
    print(f"Groups: {total_groups}")
    print(f"Candidates: {total_cands}")

    if total_cands > 0:
        print("\nAction distribution:")
        for k in ["EVAL", "CREATE", "DEFER", "UNKNOWN"]:
            if action_cnt[k] == 0:
                continue
            print(f"  - {k:7s}: {action_cnt[k]:8d}  ({action_cnt[k]/total_cands*100:6.2f}%)")

        print("\nAverage reward by action:")
        for k in ["EVAL", "CREATE", "DEFER", "UNKNOWN"]:
            if reward_cnt[k] == 0:
                continue
            print(f"  - {k:7s}: {reward_sum[k]/reward_cnt[k]: .4f}")

        if sum(correct_tot.values()) > 0:
            print("\nCorrectness rate by action (from debug):")
            for k in ["EVAL", "CREATE", "DEFER", "UNKNOWN"]:
                if correct_tot[k] == 0:
                    continue
                print(f"  - {k:7s}: {correct_cnt[k]/correct_tot[k]: .4f}  (✓ {correct_cnt[k]} / {correct_tot[k]})")


def should_filter_same_answer_group(
    raw_group: Dict[str, Any],
    use_parsed_answer: bool = True,
) -> bool:
    """
    Return True if all valid candidate answers in this group are the same.

    Priority:
    1) If use_parsed_answer and candidate.answer_parsed.pred_str exists, compare pred_str
    2) Otherwise compare raw answer_text

    Only filter groups with at least 2 valid candidate answers.
    """
    candidates = raw_group.get("candidates", [])
    if not isinstance(candidates, list):
        return False

    answers: List[str] = []

    for cand in candidates:
        if not isinstance(cand, dict):
            continue

        act_str = str(cand.get("action_str", "") or "").strip()
        if not act_str:
            continue

        ans = ""
        if use_parsed_answer:
            ap = cand.get("answer_parsed")
            if isinstance(ap, dict):
                ans = str(ap.get("pred_str", "") or "").strip()

        if not ans:
            ans = str(cand.get("answer_text", "") or "").strip()

        if ans:
            answers.append(ans)

    if len(answers) < 2:
        return False

    first = answers[0]
    return all(a == first for a in answers[1:])


def rebalance_groups_by_answer_ratio(
    rows: List[Dict[str, Any]],
    same_answer_ratio: float,
    mixed_answer_ratio: float,
    use_parsed_answer: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Re-sample groups so that:
      kept_same_answer : kept_mixed_answer ~= same_answer_ratio : mixed_answer_ratio

    Strategy:
    - Split rows into two buckets:
        1) same-answer groups
        2) mixed-answer groups
    - Keep as many as possible while matching the desired ratio.
    - If one bucket is the bottleneck, downsample the other bucket.

    Returns:
      selected_rows, stats
    """
    if same_answer_ratio < 0 or mixed_answer_ratio < 0:
        raise ValueError("same_answer_ratio and mixed_answer_ratio must be >= 0")
    if same_answer_ratio == 0 and mixed_answer_ratio == 0:
        raise ValueError("same_answer_ratio and mixed_answer_ratio cannot both be 0")

    same_groups: List[Dict[str, Any]] = []
    mixed_groups: List[Dict[str, Any]] = []

    for g in rows:
        if should_filter_same_answer_group(g, use_parsed_answer=use_parsed_answer):
            same_groups.append(g)
        else:
            mixed_groups.append(g)

    rng = random.Random(seed)
    rng.shuffle(same_groups)
    rng.shuffle(mixed_groups)

    n_same_avail = len(same_groups)
    n_mixed_avail = len(mixed_groups)

    # Edge cases: user wants only one type
    if same_answer_ratio == 0:
        selected_same = []
        selected_mixed = mixed_groups
    elif mixed_answer_ratio == 0:
        selected_same = same_groups
        selected_mixed = []
    else:
        # Find the largest scale k such that:
        #   keep_same  = k * same_answer_ratio
        #   keep_mixed = k * mixed_answer_ratio
        # and both do not exceed available counts.
        k = min(
            n_same_avail / same_answer_ratio if same_answer_ratio > 0 else float("inf"),
            n_mixed_avail / mixed_answer_ratio if mixed_answer_ratio > 0 else float("inf"),
        )

        keep_same = int(math.floor(k * same_answer_ratio))
        keep_mixed = int(math.floor(k * mixed_answer_ratio))

        # avoid pathological all-zero when both buckets exist
        if keep_same == 0 and n_same_avail > 0 and same_answer_ratio > 0 and n_mixed_avail > 0:
            keep_same = 1
        if keep_mixed == 0 and n_mixed_avail > 0 and mixed_answer_ratio > 0 and n_same_avail > 0:
            keep_mixed = 1

        # After fallback, re-check against available counts
        keep_same = min(keep_same, n_same_avail)
        keep_mixed = min(keep_mixed, n_mixed_avail)

        selected_same = same_groups[:keep_same]
        selected_mixed = mixed_groups[:keep_mixed]

    selected_rows = selected_same + selected_mixed
    rng.shuffle(selected_rows)

    stats = {
        "same_available": n_same_avail,
        "mixed_available": n_mixed_avail,
        "same_kept": len(selected_same),
        "mixed_kept": len(selected_mixed),
        "total_kept": len(selected_rows),
        "same_dropped": n_same_avail - len(selected_same),
        "mixed_dropped": n_mixed_avail - len(selected_mixed),
    }
    return selected_rows, stats


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Offline GRPO reward builder")

    ap.add_argument("--in_jsonl", type=str, default="./offline_data/data.jsonl", help="Input groups jsonl (saved from mode=grpo_data)")
    ap.add_argument("--out_jsonl", type=str, default="./offline_data/data_processed.jsonl", help="Output training jsonl (GRPO-ready grouped format)")

    # Reward hyperparameters 
    ap.add_argument("--correct_scale", type=float, default=1.0)
    ap.add_argument("--lambda_defer", type=float, default=0.3)
    ap.add_argument("--lambda_create", type=float, default=0.03)

    # Optional shaping
    ap.add_argument("--lambda_len", type=float, default=0.0, help="Length penalty weight (0 disables)")
    ap.add_argument("--len_denom", type=int, default=512, help="Denominator for normalized length penalty")

    ap.add_argument("--lambda_parse_fail", type=float, default=0.0, help="Penalty if parse ok==False (0 disables)")

    # Reward clamp
    ap.add_argument("--clamp_min", type=float, default=None)
    ap.add_argument("--clamp_max", type=float, default=None)

    # Behavior switches
    ap.add_argument("--no_cached_parsed", action="store_true", help="Ignore candidate.answer_parsed; re-parse from answer_text")
    ap.add_argument("--no_debug", action="store_true", help="Do not store debug section in output")

    ap.add_argument(
        "--filter_same_answer_groups",
        action="store_true",
        help="Filter out groups where all actions produce the same answer"
    )

    # New: ratio-based rebalancing
    ap.add_argument(
        "--same_answer_ratio",
        type=float,
        default=0,
        help="Target ratio weight for groups where all actions produce the same answer"
    )
    ap.add_argument(
        "--mixed_answer_ratio",
        type=float,
        default=1,
        help="Target ratio weight for groups where at least one action produces a different answer"
    )
    ap.add_argument(
        "--ratio_seed",
        type=int,
        default=42,
        help="Random seed for ratio-based sampling"
    )

    args = ap.parse_args()

    if args.filter_same_answer_groups and (
        args.same_answer_ratio is not None or args.mixed_answer_ratio is not None
    ):
        raise ValueError(
            "Do not use --filter_same_answer_groups together with "
            "--same_answer_ratio/--mixed_answer_ratio. "
            "Use ratio 0:1 if you want to fully remove same-answer groups."
        )

    cfg = RewardConfig(
        correct_scale=args.correct_scale,
        lambda_defer=args.lambda_defer,
        lambda_create=args.lambda_create,
        lambda_len=args.lambda_len,
        len_denom=args.len_denom,
        lambda_parse_fail=args.lambda_parse_fail,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
    )

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)

    rows = read_jsonl(in_path)
    if not rows:
        raise RuntimeError(f"No rows found in {in_path}")

    ratio_stats = None
    use_parsed_for_group_check = (not args.no_cached_parsed)

    # Ratio-based rebalance first
    if args.same_answer_ratio is not None or args.mixed_answer_ratio is not None:
        if args.same_answer_ratio is None or args.mixed_answer_ratio is None:
            raise ValueError("Please provide both --same_answer_ratio and --mixed_answer_ratio together")

        rows, ratio_stats = rebalance_groups_by_answer_ratio(
            rows=rows,
            same_answer_ratio=args.same_answer_ratio,
            mixed_answer_ratio=args.mixed_answer_ratio,
            use_parsed_answer=use_parsed_for_group_check,
            seed=args.ratio_seed,
        )

    ev_cache: Dict[str, Any] = {}
    out_rows: List[Dict[str, Any]] = []

    filtered_same_answer = 0

    for g in tqdm(rows, desc="Scoring groups"):
        if args.filter_same_answer_groups and should_filter_same_answer_group(
            g,
            use_parsed_answer=(not args.no_cached_parsed),
        ):
            filtered_same_answer += 1
            continue

        out_rows.append(
            build_training_group(
                g=g,
                ev_cache=ev_cache,
                cfg=cfg,
                use_cached_parsed=(not args.no_cached_parsed),
                keep_debug=(not args.no_debug),
            )
        )

    write_jsonl(out_path, out_rows)
    print(f"\n✅ Wrote training groups to: {out_path}")

    if ratio_stats is not None:
        print("\nRatio-based sampling stats:")
        print(json.dumps(ratio_stats, indent=2, ensure_ascii=False))
        if ratio_stats["same_kept"] > 0 and ratio_stats["mixed_kept"] > 0:
            print(f"Kept ratio (same:mixed) = {ratio_stats['same_kept']}:{ratio_stats['mixed_kept']}")
        elif ratio_stats["same_kept"] > 0:
            print("Kept ratio (same:mixed) = same only")
        elif ratio_stats["mixed_kept"] > 0:
            print("Kept ratio (same:mixed) = mixed only")

    if args.filter_same_answer_groups:
        print(f"Filtered same-answer groups: {filtered_same_answer}")

    # Report
    summarize(out_rows)

    # Print current hyperparams for reproducibility
    print("\nReward hyperparams:")
    print(json.dumps(cfg.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()