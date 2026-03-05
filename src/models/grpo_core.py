# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

from src.eval import EvaluatorFactory
from src.parsing import parse_prediction

from .backends import DualLLM
from .policy_utils import build_policy_prompt
from .prompt_builders import (
    build_base_prompt,
    build_collaboration_prompt,
    build_human_defer_prompt,
    wrap_chat,
)
from .token_utils import count_tokens


def _rollout_choose_candidate(
    candidates: List[Dict[str, Any]],
    strategy: str,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Choose one candidate to append into history for next round (rollout only).
    """
    if not candidates:
        raise ValueError("Empty candidates")

    strat = (strategy or "").lower()

    if strat == "prefer_eval":
        evals = [c for c in candidates if str(c.get("action_str", "")).startswith("EVAL")]
        return rng.choice(evals) if evals else rng.choice(candidates)

    if strat == "prefer_create":
        creates = [c for c in candidates if c.get("action_str") == "CREATE"]
        return creates[0] if creates else rng.choice(candidates)

    if strat == "prefer_no_defer":
        non_defer = [c for c in candidates if c.get("action_str") != "DEFER"]
        return rng.choice(non_defer) if non_defer else rng.choice(candidates)

    if strat == "random":
        return rng.choice(candidates)

    non_defer = [c for c in candidates if c.get("action_str") != "DEFER"]
    return rng.choice(non_defer) if non_defer else rng.choice(candidates)


def build_grpo_dataset(
    samples: List[Any],
    llms: DualLLM,
    tokenizer: AutoTokenizer,
    agents: int,
    rounds: int,
    force_boxed: bool,
    use_chat_template_agent: bool,
    exclude_self_eval: bool = True,
    store_parsed: bool = True,
    store_eval_result: bool = False,
    rollout_strategy: str = "prefer_no_defer",
    rollout_seed: int = 0,
    show_tqdm: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    N = len(samples)
    rng = random.Random(rollout_seed)

    stats: Dict[str, Any] = {
        "total_in_tokens": 0,
        "total_out_tokens": 0,
        "groups": 0,
        "candidates": 0,
        "candidates_eval": 0,
        "candidates_create": 0,
        "candidates_defer": 0,
        "rollout_strategy": rollout_strategy,
        "exclude_self_eval": exclude_self_eval,
        "store_parsed": store_parsed,
        "store_eval_result": store_eval_result,
    }

    history: List[List[List[str]]] = [[[] for _ in range(agents)] for _ in range(N)]
    base_prompts: List[str] = [build_base_prompt(s, force_boxed=force_boxed) for s in samples]

    ev_cache: Dict[str, Any] = {}

    # stage initial
    prompts: List[str] = []
    index_map: List[Tuple[int, int]] = []

    for i in range(N):
        base_plain = base_prompts[i]
        base = wrap_chat(tokenizer, base_plain) if use_chat_template_agent else base_plain
        for a in range(agents):
            prompts.append(base)
            index_map.append((i, a))
            stats["total_in_tokens"] += count_tokens(tokenizer, base)

    outs = llms.agent.generate_batch(prompts, show_tqdm=show_tqdm)
    for (i, a), text in zip(index_map, outs):
        history[i][a].append(text)
        stats["total_out_tokens"] += count_tokens(tokenizer, text)

    groups: List[Dict[str, Any]] = []

    # stage rounds
    for rr in range(1, rounds):
        group_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

        create_prompts: List[str] = []
        create_keys: List[Tuple[int, int]] = []

        defer_prompts: List[str] = []
        defer_keys: List[Tuple[int, int]] = []

        for i in range(N):
            s = samples[i]
            task_type = s.task_type
            base_plain = base_prompts[i]

            for a in range(agents):
                others_histories: List[List[str]] = []
                for j in range(agents):
                    if j == a:
                        continue
                    hj = history[i][j]
                    if hj:
                        pref_last = f"[Agent {j}] {hj[-1]}"
                        others_histories.append([pref_last])
                    else:
                        others_histories.append([f"[Agent {j}] (none)"])

                policy_prompt = build_policy_prompt(
                    task_type=task_type,
                    base_prompt=base_plain,
                    self_history=history[i][a],
                    others_histories=others_histories,
                    agents=agents,
                    self_idx=a,
                    use_chat_template=use_chat_template_agent,
                    tokenizer=tokenizer if use_chat_template_agent else None,
                )
                stats["total_in_tokens"] += count_tokens(tokenizer, policy_prompt)

                group: Dict[str, Any] = {
                    "sample_id": getattr(s, "id", i),
                    "round": rr,
                    "agent_idx": a,
                    "task_type": task_type,
                    "question": s.prompt,
                    "base_prompt": base_plain,
                    "policy_prompt": policy_prompt,
                    "gold": s.gold,
                    "meta": {
                        "sample_id": getattr(s, "id", i),
                        "round": rr,
                        "agent_idx": a,
                        "agents": agents,
                        "exclude_self_eval": exclude_self_eval,
                    },
                    "candidates": [],
                    "rollout_chosen_action": None,
                    "rollout_chosen_candidate_idx": None,
                }

                # (A) EVAL candidates
                for idx in range(agents):
                    if exclude_self_eval and idx == a:
                        continue
                    answer_text = history[i][idx][-1] if history[i][idx] else ""
                    cand: Dict[str, Any] = {
                        "action_str": f"EVAL {idx}",
                        "source": "copy",
                        "answer_text": answer_text,
                    }

                    if store_parsed:
                        parsed = parse_prediction(answer_text, task_type, s.meta)
                        cand["answer_parsed"] = {
                            "pred_str": parsed.pred_str,
                            "vote_key": parsed.vote_key,
                            "ok": bool(parsed.ok),
                            "method": getattr(parsed, "method", None),
                        }

                    if store_eval_result:
                        if task_type not in ev_cache:
                            ev_cache[task_type] = EvaluatorFactory.create(task_type)
                        ev = ev_cache[task_type]
                        pred_for_eval = (
                            cand.get("answer_parsed", {}).get("pred_str")
                            if store_parsed
                            else parse_prediction(answer_text, task_type, s.meta).pred_str
                        )
                        r_eval = ev.evaluate(pred_for_eval, s.gold, s.meta)
                        cand["eval"] = r_eval.__dict__ if hasattr(r_eval, "__dict__") else str(r_eval)

                    group["candidates"].append(cand)
                    stats["candidates"] += 1
                    stats["candidates_eval"] += 1

                # (B) CREATE candidate
                self_hist = history[i][a]
                others_histories_full = [history[i][j] for j in range(agents) if j != a]
                dp = build_collaboration_prompt(
                    task_type=task_type,
                    base_prompt=base_plain,
                    self_history=self_hist,
                    others_histories=others_histories_full,
                    use_chat_template=use_chat_template_agent,
                    tokenizer=tokenizer if use_chat_template_agent else None,
                )
                create_prompts.append(dp)
                create_keys.append((i, a))
                stats["total_in_tokens"] += count_tokens(tokenizer, dp)

                # (C) DEFER candidate
                latest = "\n\n".join([f"[Agent {j}]\n{history[i][j][-1]}" for j in range(agents)])
                hp = build_human_defer_prompt(task_type, base_plain, latest)
                defer_prompts.append(hp)
                defer_keys.append((i, a))
                stats["total_in_tokens"] += count_tokens(tokenizer, hp)

                group_map[(i, a)] = group

        create_outs = llms.agent.generate_batch(create_prompts, show_tqdm=show_tqdm) if create_prompts else []
        defer_outs = llms.human.generate_batch(defer_prompts, show_tqdm=show_tqdm) if defer_prompts else []

        # Fill CREATE
        for (i, a), text in zip(create_keys, create_outs):
            s = samples[i]
            task_type = s.task_type

            cand: Dict[str, Any] = {
                "action_str": "CREATE",
                "source": "agent_gen",
                "answer_text": text,
            }
            stats["total_out_tokens"] += count_tokens(tokenizer, text)

            if store_parsed:
                parsed = parse_prediction(text, task_type, s.meta)
                cand["answer_parsed"] = {
                    "pred_str": parsed.pred_str,
                    "vote_key": parsed.vote_key,
                    "ok": bool(parsed.ok),
                    "method": getattr(parsed, "method", None),
                }

            if store_eval_result:
                if task_type not in ev_cache:
                    ev_cache[task_type] = EvaluatorFactory.create(task_type)
                ev = ev_cache[task_type]
                pred_for_eval = (
                    cand.get("answer_parsed", {}).get("pred_str")
                    if store_parsed
                    else parse_prediction(text, task_type, s.meta).pred_str
                )
                r_eval = ev.evaluate(pred_for_eval, s.gold, s.meta)
                cand["eval"] = r_eval.__dict__ if hasattr(r_eval, "__dict__") else str(r_eval)

            group_map[(i, a)]["candidates"].append(cand)
            stats["candidates"] += 1
            stats["candidates_create"] += 1

        # Fill DEFER
        for (i, a), text in zip(defer_keys, defer_outs):
            s = samples[i]
            task_type = s.task_type

            cand: Dict[str, Any] = {
                "action_str": "DEFER",
                "source": "human_gen",
                "answer_text": text,
            }
            stats["total_out_tokens"] += count_tokens(tokenizer, text)

            if store_parsed:
                parsed = parse_prediction(text, task_type, s.meta)
                cand["answer_parsed"] = {
                    "pred_str": parsed.pred_str,
                    "vote_key": parsed.vote_key,
                    "ok": bool(parsed.ok),
                    "method": getattr(parsed, "method", None),
                }

            if store_eval_result:
                if task_type not in ev_cache:
                    ev_cache[task_type] = EvaluatorFactory.create(task_type)
                ev = ev_cache[task_type]
                pred_for_eval = (
                    cand.get("answer_parsed", {}).get("pred_str")
                    if store_parsed
                    else parse_prediction(text, task_type, s.meta).pred_str
                )
                r_eval = ev.evaluate(pred_for_eval, s.gold, s.meta)
                cand["eval"] = r_eval.__dict__ if hasattr(r_eval, "__dict__") else str(r_eval)

            group_map[(i, a)]["candidates"].append(cand)
            stats["candidates"] += 1
            stats["candidates_defer"] += 1

        # Rollout
        for i in range(N):
            for a in range(agents):
                g = group_map[(i, a)]
                stats["groups"] += 1

                chosen = _rollout_choose_candidate(g["candidates"], rollout_strategy, rng)
                g["rollout_chosen_action"] = chosen.get("action_str")
                g["rollout_chosen_candidate_idx"] = g["candidates"].index(chosen)

                history[i][a].append(chosen.get("answer_text", ""))
                groups.append(g)

    return groups, stats