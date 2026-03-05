#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from vllm.lora.request import LoRARequest
except Exception:
    LoRARequest = None

from openai import AsyncOpenAI

from src import loaders, evaluators  # noqa: F401
from src.dataset import load_samples
from src.parsing import parse_prediction
from src.eval import EvaluatorFactory


# -----------------------------
# Prompt
# -----------------------------
def wrap_chat(tokenizer: AutoTokenizer, user_content: str) -> str:
    msgs = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_base_prompt(sample: Any, force_boxed: bool = False) -> str:
    question = (getattr(sample, "prompt", "") or "").strip()
    task_type = getattr(sample, "task_type", "") or ""

    if force_boxed:
        answer_format = "Must give the final answer in the form \\boxed{...}.\n"
    else:
        answer_format = (
            "End your response with a single line that clearly states the final answer.\n"
            "If the answer is a number, output only the number on that final line.\n"
        )

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
            "- Output ONLY Python code.\n"
            "- Do NOT include explanations, comments outside the given prompt, or additional text.\n"
            "- Keep the original function signature exactly as given.\n"
            "- Do not write any test code. Do not use input()/print().\n"
            "- You may use the Python standard library.\n\n"
            f"{question}\n"
        )
    else:
        return (
            f"Solve the following problem:\n\n{question}\n\n"
            "Think step by step.\n"
            f"{answer_format}"
        )


# -----------------------------
# Exact raw-sample loader
# -----------------------------
def parse_uid(uid: str) -> Tuple[str, str, int]:
    """
    Example:
      gsm8k:gsm8k_train.jsonl:11
    -> ("gsm8k", "gsm8k_train.jsonl", 11)
    """
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
    """
    Try a few common locations.
    """
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
    """
    Cache raw file contents by filename.
    """
    cache: Dict[str, List[Dict[str, Any]]] = {}

    for s in samples:
        uid = getattr(s, "uid", None)
        if not uid:
            raise RuntimeError(
                "Sample has no uid. Cannot recover exact original raw sample."
            )
        dataset_name, file_name, _row_idx = parse_uid(uid)
        if file_name not in cache:
            raw_path = resolve_raw_file(data_root, dataset_name, file_name)
            cache[file_name] = load_raw_records_file(raw_path)

    return cache


def get_exact_raw_sample(sample: Any, raw_cache: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Return the exact original raw record from the source dataset file.
    """
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


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        request_timeout: int = 60,
        max_concurrency: int = 32,
        retries: int = 2,
    ):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set. Please use env var or --openai_api_key.")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.request_timeout = request_timeout
        self.max_conc = max_concurrency
        self.retries = retries

    def generate_batch(self, prompts: List[str], show_tqdm: bool = False) -> List[str]:
        async def _runner() -> List[str]:
            sem = asyncio.Semaphore(self.max_conc)
            client = AsyncOpenAI()
            outs: List[str] = [""] * len(prompts)

            async def _one(i: int, p: str):
                async with sem:
                    last_err = None
                    for attempt in range(self.retries + 1):
                        try:
                            r = await client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": p}],
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                timeout=self.request_timeout,
                            )
                            outs[i] = (r.choices[0].message.content or "").strip()
                            return
                        except Exception as e:
                            last_err = e
                            if attempt < self.retries:
                                await asyncio.sleep(0.5 * (2 ** attempt))
                    raise last_err

            tasks = [asyncio.create_task(_one(i, p)) for i, p in enumerate(prompts)]

            pbar = tqdm(total=len(tasks), desc="OpenAI completed", leave=False) if show_tqdm else None
            try:
                for fut in asyncio.as_completed(tasks):
                    await fut
                    if pbar:
                        pbar.update(1)
                return outs
            finally:
                if pbar:
                    pbar.close()
                await client.close()

        return asyncio.run(_runner())


# -----------------------------
# Main
# -----------------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
API_KEY = ""

def main():
    ap = argparse.ArgumentParser("single-pass eval, save exact original wrong samples")

    ap.add_argument("--dataset", type=str, default="gsm8k")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--limit", type=int, default=10000)

    ap.add_argument("--agent_backend", type=str, default="openai", choices=["vllm", "openai"])

    # vLLM 
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--base_model", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--lora_path", type=str, default="./outputs/checkpoints/")
    ap.add_argument("--lora_name", type=str, default="default_lora")
    ap.add_argument("--lora_id", type=int, default=1)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])

    # decoding
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)

    # prompt
    ap.add_argument("--force_boxed", action="store_true")
    ap.add_argument("--no_force_boxed", dest="force_boxed", action="store_false")
    ap.set_defaults(force_boxed=True)

    # OpenAI
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")  # gpt-3.5-turbo gpt-4o-mini gpt-4.1-nano
    ap.add_argument("--openai_api_key", type=str, default=API_KEY)
    ap.add_argument("--openai_concurrency", type=int, default=32)
    ap.add_argument("--openai_timeout", type=int, default=60)

    # tokenizer
    ap.add_argument("--tokenizer_model", type=str, default=None)

    # output
    ap.add_argument("--out_json", type=str, default="output_mmlu.json")

    args = ap.parse_args()

    samples = load_samples(args.dataset, args.split, data_root=args.data_root, limit=args.limit)
    if not samples:
        raise RuntimeError("No samples loaded. Check dataset/split/data_root/limit.")

    # Build exact raw-record cache FIRST
    raw_cache = build_raw_lookup(samples, args.data_root)

    if args.agent_backend == "vllm" and args.lora_path:
        tok_src = args.tokenizer_model or args.base_model
    else:
        tok_src = args.tokenizer_model or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    if args.agent_backend == "vllm":
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
            print(">>>>[Using Lora]: ", args.lora_path)

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
        use_chat_template = True
    else:
        backend = OpenAIBackend(
            model=args.openai_model,
            api_key=args.openai_api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            request_timeout=args.openai_timeout,
            max_concurrency=args.openai_concurrency,
        )
        use_chat_template = False

    base_prompts = [build_base_prompt(s, force_boxed=args.force_boxed) for s in samples]
    prompts = [wrap_chat(tokenizer, p) if use_chat_template else p for p in base_prompts]

    t0 = time.time()
    raw_outputs = backend.generate_batch(prompts, show_tqdm=True)
    dur = time.time() - t0

    ev_cache: Dict[str, Any] = {}
    wrong_raw_samples: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    for i, (sample, raw_output) in enumerate(zip(samples, raw_outputs)):
        task_type = sample.task_type

        if task_type not in ev_cache:
            ev_cache[task_type] = EvaluatorFactory.create(task_type)
        ev = ev_cache[task_type]

        parsed = parse_prediction(raw_output, task_type, sample.meta)
        r_eval = ev.evaluate(parsed.pred_str, sample.gold, sample.meta)
        correct = bool(getattr(r_eval, "correct", False))

        all_results.append(
            {
                "id": getattr(sample, "id", i),
                "uid": getattr(sample, "uid", None),
                "correct": correct,
                "pred": parsed.pred_str,
            }
        )

        if not correct:
            wrong_raw_samples.append(get_exact_raw_sample(sample, raw_cache))

    total = len(samples)
    wrong = len(wrong_raw_samples)
    correct_n = total - wrong
    acc = correct_n / total if total else 0.0

    print(f"\n✅ Finished {total} samples in {dur:.1f}s")
    print(f"🎯 Accuracy: {acc:.4f} (✓ {correct_n} / ✗ {wrong})")
    print(f"📝 Wrong exact raw samples: {wrong}")

if __name__ == "__main__":
    main()