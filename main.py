#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
~~~~~~~
Multi-agent, multi-round multi-agent collaboration evaluation with pluggable backend:
- vLLM (local HF model via vllm)
- OpenAI ChatCompletions (API)

Supports:
1) mode=mas_collaboration
2) mode=grpo_data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from vllm.lora.request import LoRARequest
except Exception:
    LoRARequest = None

from src import loaders, evaluators  # noqa: F401  # trigger registration
from src.dataset import load_samples

from src.models import (
    DEFAULT_MODEL_ID,
    API_KEY,
    ConsoleHumanBackend,
    DualLLM,
    OpenAIBackend,
    VLLMBackend,
    build_grpo_dataset,
    run_mas_collaboration,
)

def main():
    ap = argparse.ArgumentParser("mas_collaboration eval / grpo dataset generator (vLLM/OpenAI backends)")

    ap.add_argument("--mode", type=str, default="mas_collaboration", choices=["mas_collaboration", "grpo_data"])

    ap.add_argument("--dataset", type=str, default="gsm8k")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--limit", type=int, default=50)

    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)

    # Agent backend selection
    ap.add_argument("--agent_backend", type=str, default="vllm", choices=["vllm", "openai"])

    # vLLM config (agent)
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HF model id or local path for vLLM (base model or merged model)",
    )
    ap.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Base model path/id for vLLM when using LoRA adapter",
    )
    ap.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to PEFT LoRA adapter directory (adapter_config.json + adapter weights)",
    )
    ap.add_argument("--lora_name", type=str, default="default_lora", help="Logical LoRA adapter name for vLLM")
    ap.add_argument("--lora_id", type=int, default=1, help="Logical LoRA adapter id for vLLM")
    ap.add_argument("--max_lora_rank", type=int, default=64, help="Max supported LoRA rank for vLLM")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])

    # Shared decoding-ish params (OpenAI + vLLM) for AGENT generation
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)

    # Force boxed control
    ap.add_argument("--force_boxed", action="store_true", help="Ask model to end with \\boxed{...} format")
    ap.add_argument("--no_force_boxed", dest="force_boxed", action="store_false")
    ap.set_defaults(force_boxed=True)

    # Outputs
    ap.add_argument("--out_jsonl", type=str, default="output.jsonl", help="[mas_collaboration] save per-sample results JSONL")
    ap.add_argument(
        "--out_grpo_jsonl",
        type=str,
        default="grpo_groups.jsonl",
        help="[grpo_data] save GRPO groups JSONL",
    )

    # OpenAI config for AGENT backend if agent_backend=openai
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument(
        "--openai_api_key",
        type=str,
        default=API_KEY,
        help="If not set, use env OPENAI_API_KEY",
    )
    ap.add_argument("--openai_concurrency", type=int, default=32)
    ap.add_argument("--openai_timeout", type=int, default=60)

    # Human expert
    ap.add_argument("--human_openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument(
        "--human_openai_api_key",
        type=str,
        default=API_KEY,
        help="If not set, use env OPENAI_API_KEY",
    )
    ap.add_argument("--human_openai_concurrency", type=int, default=32)
    ap.add_argument("--human_openai_timeout", type=int, default=60)
    ap.add_argument("--human_max_tokens", type=int, default=1024)
    ap.add_argument("--human_temperature", type=float, default=0.3)
    ap.add_argument("--human_top_p", type=float, default=0.95)

    ap.add_argument(
        "--human_passive_flag",
        action="store_true",
        help=(
            "If set, when an agent chooses DEFER, directly use the sample's stored "
            "human_reasoning instead of calling the human LLM. "
            "If human_reasoning is missing for a sample, it will fall back to the original human LLM logic."
        ),
    )
    ap.add_argument(
        "--human_active_flag",
        action="store_true",
        help="If set, inject auxiliary human information into the round-0 initialization prompt only.",
    )
    ap.add_argument(
        "--active_source",
        type=str,
        default="human_idea",
        choices=["human_idea", "human_reasoning"],
        help="Which standardized sample.meta field to inject when --human_active_flag is enabled.",
    )

    # Interaction mode
    ap.add_argument(
        "--interaction_mode",
        type=str,
        default="openai",
        choices=["openai", "interactive"],
        help=(
            "How DEFER requests are handled when not using --human_passive_flag. "
            "'openai' = ask the configured human OpenAI model; "
            "'interactive' = print prompt in terminal and wait for real human input."
        ),
    )
    ap.add_argument(
        "--interactive_multiline",
        action="store_true",
        help=(
            "If set together with --interaction_mode interactive, "
            "allow multi-line terminal input. Input ends on an empty line."
        ),
    )
    ap.add_argument(
        "--no_interactive_multiline",
        dest="interactive_multiline",
        action="store_false",
    )
    ap.set_defaults(interactive_multiline=True)

    ap.add_argument(
        "--interactive_end_marker",
        type=str,
        default="",
        help=(
            "Optional end marker for multi-line interactive input. "
            "Default empty string means: finish on an empty line."
        ),
    )
    
    ap.add_argument(
        "--save_sft_data",
        action="store_true",
        help=(
            "If set, save SFT training pairs for each real GPT defer event. "
            "Each record contains the base initialization prompt as 'question' "
            "and the human OpenAI reply as 'response'."
        ),
    )
    ap.add_argument(
        "--sft_out_jsonl",
        type=str,
        default="_test_sft_data.jsonl",
        help="Where to save SFT JSONL when --save_sft_data is enabled.",
    )

    # Tokenizer
    ap.add_argument(
        "--tokenizer_model",
        type=str,
        default=None,
        help="HF tokenizer id/path (defaults to --model)",
    )

    # GRPO options
    ap.add_argument("--exclude_self_eval", action="store_true", help="[grpo_data] exclude EVAL self_idx")
    ap.add_argument("--include_self_eval", dest="exclude_self_eval", action="store_false")
    ap.set_defaults(exclude_self_eval=True)

    ap.add_argument("--store_parsed", action="store_true", help="[grpo_data] store parse_prediction outputs per candidate")
    ap.add_argument("--no_store_parsed", dest="store_parsed", action="store_false")
    ap.set_defaults(store_parsed=True)

    ap.add_argument("--store_eval_result", action="store_true", help="[grpo_data] store evaluator result per candidate (slower)")
    ap.add_argument("--no_store_eval_result", dest="store_eval_result", action="store_false")
    ap.set_defaults(store_eval_result=False)

    ap.add_argument(
        "--rollout_strategy",
        type=str,
        default="random",
        choices=["prefer_no_defer", "prefer_eval", "prefer_create", "random"],
        help="[grpo_data] which candidate to append into history to reach later rounds",
    )
    ap.add_argument("--rollout_seed", type=int, default=0, help="[grpo_data] seed for rollout choice")

    args = ap.parse_args()

    # -----------------------------
    # Load samples
    # -----------------------------
    samples = load_samples(args.dataset, args.split, data_root=args.data_root, limit=args.limit)
    if not samples:
        raise RuntimeError("No samples loaded. Check dataset/split/data_root/limit.")

    # -----------------------------
    # Build tokenizer
    # -----------------------------
    if args.agent_backend == "vllm" and args.lora_path:
        tok_src = args.tokenizer_model or args.base_model
    else:
        tok_src = args.tokenizer_model or args.model

    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    # -----------------------------
    # Build agent backend
    # -----------------------------
    if args.agent_backend == "vllm":
        use_lora = bool(args.lora_path)

        if use_lora:
            if not args.base_model:
                raise RuntimeError("--lora_path is set, but --base_model is missing.")
            if LoRARequest is None:
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
            lora_request = LoRARequest(
                args.lora_name,
                args.lora_id,
                args.lora_path,
            )

        agent_backend = VLLMBackend(
            llm=llm,
            sampling=sampling,
            lora_request=lora_request,
        )
        use_chat_template_agent = True
    else:
        agent_backend = OpenAIBackend(
            model=args.openai_model,
            api_key=args.openai_api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            request_timeout=args.openai_timeout,
            max_concurrency=args.openai_concurrency,
        )
        use_chat_template_agent = False

    # -----------------------------
    # Build human backend
    # -----------------------------
    if args.interaction_mode == "interactive":
        human_backend = ConsoleHumanBackend(
            multiline=args.interactive_multiline,
            end_marker=args.interactive_end_marker,
            show_prompt_separator=True,
        )
    else:
        human_backend = OpenAIBackend(
            model=args.human_openai_model,
            api_key=args.human_openai_api_key,
            max_tokens=args.human_max_tokens,
            temperature=args.human_temperature,
            top_p=args.human_top_p,
            request_timeout=args.human_openai_timeout,
            max_concurrency=args.human_openai_concurrency,
        )

    llms = DualLLM(agent=agent_backend, human=human_backend)

    sft_records: Optional[List[Dict[str, object]]] = [] if args.save_sft_data else None
    t0 = time.time()

    # -----------------------------
    # Run mode
    # -----------------------------
    if args.mode == "mas_collaboration":
        results, stats = run_mas_collaboration(
            samples=samples,
            llms=llms,
            tokenizer=tokenizer,
            agents=args.agents,
            rounds=args.rounds,
            force_boxed=args.force_boxed,
            use_chat_template_agent=use_chat_template_agent,
            show_tqdm=True,
            human_passive_flag=args.human_passive_flag,
            human_active_flag=args.human_active_flag,
            active_source=args.active_source,
            sft_records=sft_records,
        )
        dur = time.time() - t0

        correct = sum(1 for r in results if r["correct"])
        acc = correct / len(results) if results else 0.0

        print(f"\n✅ Finished [collaboration] {len(results)} samples ×{args.rounds} rounds ×{args.agents} agents in {dur:.1f}s")
        print(f"🎯 Accuracy: {acc:.4f} (✓ {correct} / ✗ {len(results) - correct})")
        print(f"📊 Total input tokens : {stats.total_in_tokens}")
        print(f"📈 Total output tokens: {stats.total_out_tokens}")

        total_actions = sum(stats.action_counts.values())
        if total_actions > 0:
            print("\n🧭 Stage-A action distribution (overall, rounds>=1):")
            for k in ["EVAL", "CREATE", "DEFER"]:
                c = stats.action_counts[k]
                print(f"  - {k:6s}: {c:8d}  ({c / total_actions * 100:6.2f}%)")

            print("\n🧭 Stage-A action distribution (by round):")
            for rr, d in enumerate(stats.action_counts_by_round, start=1):
                tot = sum(d.values()) or 1
                print(
                    f"  Round {rr}: "
                    + ", ".join([f"{k} {d[k]} ({d[k] / tot * 100:.1f}%)" for k in ["EVAL", "CREATE", "DEFER"]])
                )

        if stats.human_defer_total > 0:
            acc_h = stats.human_defer_correct / stats.human_defer_total
            print(
                f"\n🧑‍🏫 Human(DEFER) correctness over defer events: "
                f"{acc_h:.4f} (✓ {stats.human_defer_correct} / {stats.human_defer_total})"
            )
        else:
            print("\n🧑‍🏫 Human(DEFER) correctness: (no defer events)")

        if args.out_jsonl:
            out_path = Path(args.out_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"📝 Saved collaboration results to: {out_path}")

        if args.save_sft_data and sft_records is not None:
            sft_path = Path(args.sft_out_jsonl)
            sft_path.parent.mkdir(parents=True, exist_ok=True)
            with sft_path.open("w", encoding="utf-8") as f:
                for x in sft_records:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            print(f"🧠 Saved SFT data to: {sft_path}  (records={len(sft_records)})")

    else:
        groups, grpo_stats = build_grpo_dataset(
            samples=samples,
            llms=llms,
            tokenizer=tokenizer,
            agents=args.agents,
            rounds=args.rounds,
            force_boxed=args.force_boxed,
            use_chat_template_agent=use_chat_template_agent,
            exclude_self_eval=args.exclude_self_eval,
            store_parsed=args.store_parsed,
            store_eval_result=args.store_eval_result,
            rollout_strategy=args.rollout_strategy,
            rollout_seed=args.rollout_seed,
            show_tqdm=True,
        )
        dur = time.time() - t0

        print(f"\n✅ Finished [grpo_data] in {dur:.1f}s")
        print(f"📦 Groups: {grpo_stats['groups']}  (expected ~ {len(samples) * args.agents * max(0, args.rounds - 1)})")
        print(
            f"🧩 Candidates: {grpo_stats['candidates']}  "
            f"(EVAL {grpo_stats['candidates_eval']}, CREATE {grpo_stats['candidates_create']}, DEFER {grpo_stats['candidates_defer']})"
        )
        print(f"📊 Total input tokens : {grpo_stats['total_in_tokens']}")
        print(f"📈 Total output tokens: {grpo_stats['total_out_tokens']}")
        print(f"🎲 Rollout strategy: {grpo_stats['rollout_strategy']} (seed={args.rollout_seed})")

        if args.out_grpo_jsonl:
            out_path = Path(args.out_grpo_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for g in groups:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"📝 Saved GRPO groups to: {out_path}")


if __name__ == "__main__":
    main()