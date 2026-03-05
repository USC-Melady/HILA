#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from vllm.lora.request import LoRARequest
except Exception:
    LoRARequest = None

from src import loaders, evaluators  # noqa: F401
from src.dataset import load_samples
from src.parsing import parse_prediction
from src.eval import EvaluatorFactory


# -----------------------------
# Prompt helpers
# -----------------------------
def wrap_chat(tokenizer: AutoTokenizer, user_content: str) -> str:
    msgs = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _get_question(sample: Any) -> str:
    return (getattr(sample, "prompt", "") or "").strip()


def _answer_format_text(task_type: str, force_boxed: bool) -> str:
    if task_type == "code_unit_test":
        return (
            "Output ONLY Python code.\n"
            "Do NOT include explanations, markdown, or extra text.\n"
            "Keep the original function signature exactly as given.\n"
            "Do not write test code. Do not use input()/print().\n"
            "You may use the Python standard library.\n"
        )

    if force_boxed:
        return "Must give the final answer in the form \\boxed{...}.\n"

    return (
        "End your response with a single line that clearly states the final answer.\n"
        "If the answer is a number, output only the number on that final line.\n"
    )


def build_base_prompt(sample: Any, force_boxed: bool = False) -> str:
    question = _get_question(sample)
    task_type = getattr(sample, "task_type", "") or ""
    answer_format = _answer_format_text(task_type, force_boxed)

    if task_type == "mcq":
        return (
            "Answer the following multiple-choice question. Choose the single best option.\n\n"
            f"Solve the problem:\n\n{question}\n\n"
            "Think step by step, show your reasoning, "
            f"{answer_format}"
        )
    elif task_type in {"math_numeric", "math_symbolic"}:
        return (
            f"Solve the following problem:\n\n{question}\n\n"
            "Think step by step, show your reasoning, and be careful with arithmetic.\n"
            f"{answer_format}"
        )
    elif task_type == "code_unit_test":
        return (
            "You are given a Python programming task.\n"
            "Write a correct and efficient solution that passes all unit tests.\n\n"
            "Rules:\n"
            f"{answer_format}\n"
            f"{question}\n"
        )
    else:
        return (
            f"Solve the following problem:\n\n{question}\n\n"
            "Think step by step.\n"
            f"{answer_format}"
        )


def build_revision_prompt(
    sample: Any,
    self_history: str,
    others_history: str,
    force_boxed: bool = False,
) -> str:
    question = _get_question(sample)
    task_type = getattr(sample, "task_type", "") or ""

    if task_type == "code_unit_test":
        return (
            "You are revising your previous Python solution after seeing other agents' solutions.\n\n"
            f"Original programming task:\n\n{question}\n\n"
            f"### Your previous solutions across rounds:\n{self_history}\n\n"
            f"### Other agents' solutions across rounds:\n{others_history}\n\n"
            "Compare your previous solution with the others. "
            "Produce a revised solution that is most likely to pass all unit tests.\n\n"
            "Rules:\n"
            "- Output ONLY Python code.\n"
            "- Do NOT include explanations, markdown, or extra text.\n"
            "- Keep the original function signature exactly as given.\n"
            "- Do not write test code. Do not use input()/print().\n"
            "- You may use the Python standard library.\n"
        )

    answer_format = _answer_format_text(task_type, force_boxed)

    return (
        f"The original problem:\n\n{question}\n\n"
        f"### Your previous reasoning across rounds:\n{self_history}\n\n"
        f"### Other agents' responses across rounds:\n{others_history}\n\n"
        "Compare your previous reasoning with the others. "
        "If you find an error, correct it. If you are confident, keep the stronger solution. "
        "Then provide your updated final response.\n"
        f"{answer_format}"
    )


# -----------------------------
# Exact raw-sample loader
# -----------------------------
def parse_uid(uid: str) -> Tuple[str, str, int]:
    parts = uid.split(":")
    if len(parts) < 3:
        raise ValueError(f"Bad uid format: {uid}")
    dataset_name = parts[0]
    file_name = parts[1]
    row_idx = int(parts[2])
    return dataset_name, file_name, row_idx


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"JSON file is not a list: {path}")


def load_raw_records_file(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        return read_json(path)
    raise ValueError(f"Unsupported raw file format: {path}")


def resolve_raw_file(data_root: str, dataset_name: str, file_name: str) -> Path:
    candidates = [
        Path(data_root) / dataset_name / file_name,
        Path(data_root) / file_name,
        Path(file_name),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find raw data file '{file_name}'. Tried: "
        + ", ".join(str(x) for x in candidates)
    )


def build_raw_lookup(samples: List[Any], data_root: str) -> Dict[str, List[Dict[str, Any]]]:
    cache: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        uid = getattr(s, "uid", None)
        if not uid:
            raise RuntimeError("Sample has no uid. Cannot recover exact original raw sample.")
        dataset_name, file_name, _row_idx = parse_uid(uid)
        if file_name not in cache:
            raw_path = resolve_raw_file(data_root, dataset_name, file_name)
            cache[file_name] = load_raw_records_file(raw_path)
    return cache


def get_exact_raw_sample(sample: Any, raw_cache: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    uid = getattr(sample, "uid", None)
    if not uid:
        raise RuntimeError("Sample has no uid, cannot map back to exact raw record.")

    _dataset_name, file_name, row_idx = parse_uid(uid)
    rows = raw_cache[file_name]

    if row_idx < 0 or row_idx >= len(rows):
        raise IndexError(f"row_idx out of range: {row_idx} for file {file_name}")

    return rows[row_idx]


# -----------------------------
# Backend
# -----------------------------
class LLMBackend(ABC):
    @abstractmethod
    def generate_batch(self, prompts: List[str], show_tqdm: bool = False) -> List[str]:
        raise NotImplementedError


class VLLMBackend(LLMBackend):
    def __init__(
        self,
        llm: LLM,
        sampling: SamplingParams,
        lora_request: Optional[Any] = None,
    ):
        self.llm = llm
        self.sampling = sampling
        self.lora_request = lora_request

    def generate_batch(self, prompts: List[str], show_tqdm: bool = False) -> List[str]:
        gen_kwargs = {"use_tqdm": show_tqdm}
        if self.lora_request is not None:
            gen_kwargs["lora_request"] = self.lora_request
        outs = self.llm.generate(prompts, self.sampling, **gen_kwargs)
        return [o.outputs[0].text for o in outs]


# -----------------------------
# Debate
# -----------------------------
def solve_multi_round_debate_batch(
    samples: List[Any],
    tokenizer: AutoTokenizer,
    backend: VLLMBackend,
    agents: int,
    rounds: int,
    force_boxed: bool,
    show_tqdm_round0: bool = False,
    show_tqdm_later: bool = False,
) -> List[List[List[str]]]:
    """
    Returns:
        history[sample_idx][agent_idx][round_idx] = raw generated text
    """
    n = len(samples)
    history: List[List[List[str]]] = [[[] for _ in range(agents)] for _ in range(n)]

    # Round 0
    prompts: List[str] = []
    index_map: List[Tuple[int, int]] = []

    for i, sample in enumerate(samples):
        base_prompt = build_base_prompt(sample, force_boxed=force_boxed)
        for a in range(agents):
            prompts.append(wrap_chat(tokenizer, base_prompt))
            index_map.append((i, a))

    outs = backend.generate_batch(prompts, show_tqdm=show_tqdm_round0)
    for (i, a), text in zip(index_map, outs):
        history[i][a].append(text)

    # Debate rounds
    for r in range(1, rounds):
        prompts = []
        index_map = []

        for i, sample in enumerate(samples):
            for a in range(agents):
                self_history_blocks = [
                    f"Round {k}:\n{history[i][a][k]}" for k in range(len(history[i][a]))
                ]
                self_history = "\n\n".join(self_history_blocks)

                others_blocks = []
                for j in range(agents):
                    if j == a:
                        continue
                    other_agent_rounds = [
                        f"Round {k} from agent {j}:\n{history[i][j][k]}"
                        for k in range(len(history[i][j]))
                    ]
                    others_blocks.append("\n\n".join(other_agent_rounds))
                others_history = "\n\n".join(others_blocks)

                revise_prompt = build_revision_prompt(
                    sample=sample,
                    self_history=self_history,
                    others_history=others_history,
                    force_boxed=force_boxed,
                )

                prompts.append(wrap_chat(tokenizer, revise_prompt))
                index_map.append((i, a))

        outs = backend.generate_batch(prompts, show_tqdm=show_tqdm_later)
        for (i, a), text in zip(index_map, outs):
            history[i][a].append(text)

    return history


# -----------------------------
# Selection logic
# -----------------------------
CAT1 = "all_same_correct"
CAT2 = "majority_same_correct"
CAT3 = "majority_same_wrong"
CAT4 = "all_different_all_wrong"

CATEGORY_ORDER = [CAT1, CAT2, CAT3, CAT4]


def normalize_pred(pred: Any) -> Optional[str]:
    if pred is None:
        return None
    s = str(pred).strip()
    return s if s else None


def classify_sample_by_final_agents(
    sample: Any,
    final_agent_outputs: List[str],
    ev: Any,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Classify into one of 4 categories or return None.

    Returns:
        (category_name_or_None, debug_info)
    """
    task_type = getattr(sample, "task_type", "") or ""

    agent_preds: List[Optional[str]] = []
    agent_corrects: List[bool] = []

    for raw_output in final_agent_outputs:
        parsed = parse_prediction(raw_output, task_type, sample.meta)
        pred = normalize_pred(parsed.pred_str)
        agent_preds.append(pred)

        r_eval = ev.evaluate(pred, sample.gold, sample.meta)
        agent_corrects.append(bool(getattr(r_eval, "correct", False)))

    cnt = Counter(agent_preds)
    unique_n = len(cnt)
    max_count = max(cnt.values()) if cnt else 0

    # unique majority: exactly one answer has the maximum frequency
    top_items = cnt.most_common()
    unique_majority = False
    majority_answer = None
    if top_items:
        if len(top_items) == 1:
            unique_majority = top_items[0][1] > (len(agent_preds) // 2)
            majority_answer = top_items[0][0]
        else:
            unique_majority = (
                top_items[0][1] > top_items[1][1]
                and top_items[0][1] > (len(agent_preds) // 2)
            )
            majority_answer = top_items[0][0] if unique_majority else None

    unanimous = (unique_n == 1)
    all_different = (unique_n == len(agent_preds))

    # majority answer correctness (evaluate the majority answer once)
    majority_correct = False
    if majority_answer is not None:
        r_major = ev.evaluate(majority_answer, sample.gold, sample.meta)
        majority_correct = bool(getattr(r_major, "correct", False))

    category = None

    # 1) all same and correct
    if unanimous and max_count == len(agent_preds) and all(agent_corrects):
        category = CAT1

    # 2) majority same and correct, but NOT unanimous
    elif (not unanimous) and unique_majority and majority_correct:
        category = CAT2

    # 3) majority same and wrong, but NOT unanimous
    elif (not unanimous) and unique_majority and (not majority_correct):
        category = CAT3

    # 4) all different and all wrong
    elif all_different and (not any(agent_corrects)):
        category = CAT4

    info = {
        "uid": getattr(sample, "uid", None),
        "id": getattr(sample, "id", None),
        "agent_preds": agent_preds,
        "agent_corrects": agent_corrects,
        "counts": dict(cnt),
        "majority_answer": majority_answer,
        "majority_correct": majority_correct,
        "category": category,
    }
    return category, info


def quotas_met(selected_raw: Dict[str, List[Dict[str, Any]]], target_per_cat: Dict[str, int]) -> bool:
    return all(len(selected_raw[c]) >= target_per_cat[c] for c in CATEGORY_ORDER)


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# Main
# -----------------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    ap = argparse.ArgumentParser("vllm debate + select samples by final multi-agent answer patterns")

    ap.add_argument("--dataset", type=str, default="mmlu")
    ap.add_argument("--split", type=str, default="test_2")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--limit", type=int, default=10000)

    # debate
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--sample_batch_size", type=int, default=64)

    # targets
    ap.add_argument("--need_cat1", type=int, default=30)
    ap.add_argument("--need_cat2", type=int, default=30)
    ap.add_argument("--need_cat3", type=int, default=30)
    ap.add_argument("--need_cat4", type=int, default=30)

    # vLLM only
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--base_model", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--lora_path", type=str, default=None)
    ap.add_argument("--lora_name", type=str, default="default_lora")
    ap.add_argument("--lora_id", type=int, default=1)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )

    # decoding
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)

    # prompt
    ap.add_argument("--force_boxed", action="store_true")
    ap.add_argument("--no_force_boxed", dest="force_boxed", action="store_false")
    ap.set_defaults(force_boxed=True)

    # tokenizer
    ap.add_argument("--tokenizer_model", type=str, default=None)

    # output
    ap.add_argument("--out_dir", type=str, default="selected_debate_samples")
    ap.add_argument("--out_prefix", type=str, default="selected")

    args = ap.parse_args()

    if args.agents < 2:
        raise ValueError("--agents must be >= 2")
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.sample_batch_size < 1:
        raise ValueError("--sample_batch_size must be >= 1")

    samples = load_samples(args.dataset, args.split, data_root=args.data_root, limit=args.limit)
    if not samples:
        raise RuntimeError("No samples loaded. Check dataset/split/data_root/limit.")

    raw_cache = build_raw_lookup(samples, args.data_root)

    tok_src = args.tokenizer_model or (args.base_model if args.lora_path else args.model)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    use_lora = bool(args.lora_path)
    if use_lora and LoRARequest is None:
        raise RuntimeError(
            "Current vLLM installation does not expose LoRARequest. "
            "Please upgrade vLLM to a version with LoRA support."
        )

    vllm_model_path = args.base_model if use_lora else args.model

    llm_kwargs = {
        "model": vllm_model_path,
        "dtype": args.dtype,
    }
    if use_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank

    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        stop=None,
        ignore_eos=False,
    )

    lora_request = None
    if use_lora:
        lora_request = LoRARequest(args.lora_name, args.lora_id, args.lora_path)

    backend = VLLMBackend(llm=llm, sampling=sampling, lora_request=lora_request)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_per_cat = {
        CAT1: args.need_cat1,
        CAT2: args.need_cat2,
        CAT3: args.need_cat3,
        CAT4: args.need_cat4,
    }

    file_map = {
        CAT1: out_dir / f"{args.out_prefix}_{CAT1}.jsonl",
        CAT2: out_dir / f"{args.out_prefix}_{CAT2}.jsonl",
        CAT3: out_dir / f"{args.out_prefix}_{CAT3}.jsonl",
        CAT4: out_dir / f"{args.out_prefix}_{CAT4}.jsonl",
    }

    # clear old files
    for p in file_map.values():
        if p.exists():
            p.unlink()

    selected_raw: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORY_ORDER}
    selected_info: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORY_ORDER}

    ev_cache: Dict[str, Any] = {}
    total_seen = 0
    total_debated = 0
    t0 = time.time()

    num_batches = math.ceil(len(samples) / args.sample_batch_size)

    for batch_idx in range(num_batches):
        if quotas_met(selected_raw, target_per_cat):
            break

        start = batch_idx * args.sample_batch_size
        end = min(len(samples), start + args.sample_batch_size)
        batch_samples = samples[start:end]
        total_debated += len(batch_samples)

        print(
            f"\n[Batch {batch_idx + 1}/{num_batches}] "
            f"processing samples {start}:{end} "
            f"(selected: "
            f"{CAT1}={len(selected_raw[CAT1])}/{target_per_cat[CAT1]}, "
            f"{CAT2}={len(selected_raw[CAT2])}/{target_per_cat[CAT2]}, "
            f"{CAT3}={len(selected_raw[CAT3])}/{target_per_cat[CAT3]}, "
            f"{CAT4}={len(selected_raw[CAT4])}/{target_per_cat[CAT4]})"
        )

        history = solve_multi_round_debate_batch(
            samples=batch_samples,
            tokenizer=tokenizer,
            backend=backend,
            agents=args.agents,
            rounds=args.rounds,
            force_boxed=args.force_boxed,
            show_tqdm_round0=True,
            show_tqdm_later=False,
        )

        staged_new_raw: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORY_ORDER}
        staged_new_info: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORY_ORDER}

        for local_i, sample in enumerate(batch_samples):
            total_seen += 1

            task_type = sample.task_type
            if task_type not in ev_cache:
                ev_cache[task_type] = EvaluatorFactory.create(task_type)
            ev = ev_cache[task_type]

            final_agent_outputs = [history[local_i][a][-1] for a in range(args.agents)]

            category, info = classify_sample_by_final_agents(
                sample=sample,
                final_agent_outputs=final_agent_outputs,
                ev=ev,
            )

            if category is None:
                continue

            already_have = len(selected_raw[category]) + len(staged_new_raw[category])
            if already_have >= target_per_cat[category]:
                continue

            raw_sample = get_exact_raw_sample(sample, raw_cache)

            staged_new_raw[category].append(raw_sample)
            staged_new_info[category].append(info)

            # early break within current batch once all quotas satisfied
            ready = True
            for c in CATEGORY_ORDER:
                if len(selected_raw[c]) + len(staged_new_raw[c]) < target_per_cat[c]:
                    ready = False
                    break
            if ready:
                break

        # flush staged rows to disk
        for c in CATEGORY_ORDER:
            if staged_new_raw[c]:
                append_jsonl(file_map[c], staged_new_raw[c])
                selected_raw[c].extend(staged_new_raw[c])
                selected_info[c].extend(staged_new_info[c])

        if quotas_met(selected_raw, target_per_cat):
            print("✅ Target quotas reached. Stopping early.")
            break

    dur = time.time() - t0

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "requested": target_per_cat,
        "collected": {c: len(selected_raw[c]) for c in CATEGORY_ORDER},
        "agents": args.agents,
        "rounds": args.rounds,
        "sample_batch_size": args.sample_batch_size,
        "total_loaded": len(samples),
        "total_debated": total_debated,
        "total_seen_for_selection": total_seen,
        "duration_sec": round(dur, 2),
        "files": {c: str(file_map[c]) for c in CATEGORY_ORDER},
        "selected_info": selected_info,
    }

    summary_path = out_dir / f"{args.out_prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==================== Done ====================")
    print(f"⏱️  Duration: {dur:.1f}s")
    print(f"📦 Total loaded samples   : {len(samples)}")
    print(f"🧠 Total debated samples  : {total_debated}")
    print(f"🔎 Total checked samples  : {total_seen}")
    for c in CATEGORY_ORDER:
        print(f"  - {c}: {len(selected_raw[c])} / {target_per_cat[c]}")
        print(f"    saved to: {file_map[c]}")
    print(f"📝 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()