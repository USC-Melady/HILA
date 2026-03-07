# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import SamplingParams

from src.eval import EvaluatorFactory
from src.parsing import parse_prediction
from src.voting import majority_vote

from .backends import DualLLM, OpenAIBackend, VLLMBackend
from .policy_utils import build_policy_prompt, parse_policy
from .prompt_builders import (
    build_base_prompt,
    build_collaboration_prompt,
    build_human_defer_prompt,
    build_initial_prompt,
    get_human_passive_reasoning,
    wrap_chat,
)
from .structured_signals import StructuredDecisionSignalsBuilder
from .token_utils import count_tokens


@dataclass
class CollaborateStats:
    total_in_tokens: int = 0
    total_out_tokens: int = 0

    action_counts: Dict[str, int] = field(
        default_factory=lambda: {"EVAL": 0, "CREATE": 0, "DEFER": 0}
    )
    action_counts_by_round: List[Dict[str, int]] = field(default_factory=list)

    human_defer_total: int = 0
    human_defer_correct: int = 0


def run_mas_collaboration(
    samples: List[Any],
    llms: DualLLM,
    tokenizer: AutoTokenizer,
    agents: int,
    rounds: int,
    force_boxed: bool,
    use_chat_template_agent: bool,
    show_tqdm: bool = True,
    human_passive_flag: bool = False,
    human_active_flag: bool = False,
    active_source: str = "human_idea",
    sft_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], CollaborateStats]:
    N = len(samples)
    stats = CollaborateStats()
    stats.action_counts_by_round = [
        {"EVAL": 0, "CREATE": 0, "DEFER": 0} for _ in range(max(0, rounds - 1))
    ]
    ev_cache_for_human: Dict[str, Any] = {}

    sds_builder = StructuredDecisionSignalsBuilder(tokenizer=tokenizer)

    policy_sampling = SamplingParams(
        max_tokens=4,
        temperature=1.0,
        top_p=1.0,
        stop=["\n"],
    )

    history: List[List[List[str]]] = [[[] for _ in range(agents)] for _ in range(N)]

    base_prompts: List[str] = [build_base_prompt(s, force_boxed=force_boxed) for s in samples]
    init_prompts: List[str] = [
        build_initial_prompt(
            s,
            force_boxed=force_boxed,
            human_active_flag=human_active_flag,
            active_source=active_source,
        )
        for s in samples
    ]

    # initial
    prompts: List[str] = []
    index_map: List[Tuple[int, int]] = []
    for i in range(N):
        init_plain = init_prompts[i]
        init_prompt = wrap_chat(tokenizer, init_plain) if use_chat_template_agent else init_plain
        for a in range(agents):
            prompts.append(init_prompt)
            index_map.append((i, a))
            stats.total_in_tokens += count_tokens(tokenizer, init_prompt)

    outs = llms.agent.generate_batch(prompts, show_tqdm=show_tqdm)
    for (i, a), text in zip(index_map, outs):
        history[i][a].append(text)
        stats.total_out_tokens += count_tokens(tokenizer, text)

    # Rounds
    for _r in range(1, rounds):
        # (1) Stage A
        policy_prompts: List[str] = []
        policy_map: List[Tuple[int, int]] = []

        for i in range(N):
            base_plain = base_prompts[i]
            task_type = samples[i].task_type

            for a in range(agents):
                self_hist = history[i][a]

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

                pp = build_policy_prompt(
                    task_type=task_type,
                    base_prompt=base_plain,
                    self_history=self_hist,
                    others_histories=others_histories,
                    agents=agents,
                    self_idx=a,
                    use_chat_template=use_chat_template_agent,
                    tokenizer=tokenizer if use_chat_template_agent else None,
                    sample_meta=getattr(samples[i], "meta", {}),
                    sds_builder=sds_builder,
                )
                policy_prompts.append(pp)
                policy_map.append((i, a))
                stats.total_in_tokens += count_tokens(tokenizer, pp)

        if isinstance(llms.agent, VLLMBackend):
            policy_outs = llms.agent.generate_batch(
                policy_prompts,
                show_tqdm=show_tqdm,
                sampling_override=policy_sampling,
            )
        else:
            policy_outs = llms.agent.generate_batch(policy_prompts, show_tqdm=show_tqdm)

        decisions: Dict[Tuple[int, int], Tuple[str, Optional[int], str]] = {}
        for (i, a), t in zip(policy_map, policy_outs):
            act, idx = parse_policy(t, self_idx=a, agents=agents)
            decisions[(i, a)] = (act, idx, t)
            stats.total_out_tokens += count_tokens(tokenizer, t)
            stats.action_counts[act] += 1
            stats.action_counts_by_round[_r - 1][act] += 1

        # (2) Stage B Execute
        # 2.1 EVAL -> copy
        for i in range(N):
            for a in range(agents):
                act, idx, _raw = decisions[(i, a)]
                if act != "EVAL":
                    continue
                chosen = idx if idx is not None else (0 if a != 0 else (1 if agents > 1 else 0))
                chosen = max(0, min(agents - 1, chosen))
                if chosen == a and agents > 1:
                    chosen = 0 if a != 0 else 1
                history[i][a].append(history[i][chosen][-1])

        # 2.2 CREATE -> generate updated
        create_prompts: List[str] = []
        create_map: List[Tuple[int, int]] = []
        for i in range(N):
            base_plain = base_prompts[i]
            task_type = samples[i].task_type
            for a in range(agents):
                act, _idx, _raw = decisions[(i, a)]
                if act != "CREATE":
                    continue
                self_hist = history[i][a]
                others_histories = [history[i][j] for j in range(agents) if j != a]

                dp = build_collaboration_prompt(
                    task_type=task_type,
                    base_prompt=base_plain,
                    self_history=self_hist,
                    others_histories=others_histories,
                    use_chat_template=use_chat_template_agent,
                    tokenizer=tokenizer if use_chat_template_agent else None,
                )
                create_prompts.append(dp)
                create_map.append((i, a))
                stats.total_in_tokens += count_tokens(tokenizer, dp)

        create_outs = llms.agent.generate_batch(create_prompts, show_tqdm=show_tqdm) if create_prompts else []
        for (i, a), text in zip(create_map, create_outs):
            history[i][a].append(text)
            stats.total_out_tokens += count_tokens(tokenizer, text)

        # 2.3 DEFER
        defer_prompts: List[str] = []
        defer_map: List[Tuple[int, int]] = []
        passive_defer_items: List[Tuple[int, int, str]] = []

        for i in range(N):
            base_plain = base_prompts[i]
            task_type = samples[i].task_type

            for a in range(agents):
                act, _idx, _raw = decisions[(i, a)]
                if act != "DEFER":
                    continue

                if human_passive_flag:
                    stored_text = get_human_passive_reasoning(samples[i])
                    if stored_text:
                        passive_defer_items.append((i, a, stored_text))
                        continue

                latest = "\n\n".join([f"[Agent {j}]\n{history[i][j][-1]}" for j in range(agents)])
                hp = build_human_defer_prompt(task_type, base_plain, latest)

                defer_prompts.append(hp)
                defer_map.append((i, a))
                stats.total_in_tokens += count_tokens(tokenizer, hp)

        # Passive DEFER
        for i, a, text in passive_defer_items:
            history[i][a].append(text)

            s = samples[i]
            task_type = s.task_type
            if task_type not in ev_cache_for_human:
                ev_cache_for_human[task_type] = EvaluatorFactory.create(task_type)
            evh = ev_cache_for_human[task_type]

            parsed_h = parse_prediction(text, task_type, s.meta)
            r_h = evh.evaluate(parsed_h.pred_str, s.gold, s.meta)

            stats.human_defer_total += 1
            if getattr(r_h, "correct", False):
                stats.human_defer_correct += 1

        # Real DEFER
        defer_outs = llms.human.generate_batch(defer_prompts, show_tqdm=show_tqdm) if defer_prompts else []
        for (i, a), text in zip(defer_map, defer_outs):
            history[i][a].append(text)
            stats.total_out_tokens += count_tokens(tokenizer, text)

            if sft_records is not None and isinstance(llms.human, OpenAIBackend):
                sft_records.append(
                    {
                        "sample_id": getattr(samples[i], "id", i),
                        "task_type": samples[i].task_type,
                        "agent_idx": a,
                        "question": samples[i].prompt,
                        "prompt": base_prompts[i],
                        "completion": text,
                    }
                )

            s = samples[i]
            task_type = s.task_type
            if task_type not in ev_cache_for_human:
                ev_cache_for_human[task_type] = EvaluatorFactory.create(task_type)
            evh = ev_cache_for_human[task_type]

            parsed_h = parse_prediction(text, task_type, s.meta)
            r_h = evh.evaluate(parsed_h.pred_str, s.gold, s.meta)

            stats.human_defer_total += 1
            if getattr(r_h, "correct", False):
                stats.human_defer_correct += 1

    # Final: parse + eval
    ev_cache: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []

    for i, s in enumerate(samples):
        task_type = s.task_type
        if task_type not in ev_cache:
            ev_cache[task_type] = EvaluatorFactory.create(task_type)
        ev = ev_cache[task_type]

        final_raws = [history[i][a][-1] for a in range(agents)]
        parseds = [parse_prediction(t, task_type, s.meta) for t in final_raws]

        if task_type == "code_unit_test":
            agent_evals = []
            best_idx = None
            best_eval = None

            for a, p in enumerate(parseds):
                r = ev.evaluate(p.pred_str, s.gold, s.meta)
                agent_evals.append(r)
                if getattr(r, "correct", False) and best_idx is None:
                    best_idx = a
                    best_eval = r

            correct_any = any(getattr(r, "correct", False) for r in agent_evals)
            if best_idx is None:
                best_idx = 0
                best_eval = agent_evals[0] if agent_evals else ev.evaluate("", s.gold, s.meta)

            pred = parseds[best_idx].pred_str
            results.append(
                {
                    "id": getattr(s, "id", i),
                    "task_type": task_type,
                    "question": s.prompt,
                    "base_prompt": base_prompts[i],
                    "gold": s.gold,
                    "pred": pred,
                    "passed_agent_idx": best_idx if correct_any else None,
                    "agent_pred_strs": [p.pred_str for p in parseds],
                    "agent_final_raws": final_raws,
                    "correct": bool(correct_any),
                    "eval": best_eval.__dict__ if hasattr(best_eval, "__dict__") else str(best_eval),
                }
            )
            continue

        vote = majority_vote([p.vote_key for p in parseds])
        pred = vote.chosen_key if getattr(vote, "ok", False) else ""
        r_eval = ev.evaluate(pred, s.gold, s.meta)

        results.append(
            {
                "id": getattr(s, "id", i),
                "task_type": task_type,
                "question": s.prompt,
                "base_prompt": base_prompts[i],
                "gold": s.gold,
                "pred": pred,
                "vote_ok": bool(getattr(vote, "ok", False)),
                "agent_vote_keys": [p.vote_key for p in parseds],
                "agent_pred_strs": [p.pred_str for p in parseds],
                "agent_final_raws": final_raws,
                "correct": bool(getattr(r_eval, "correct", False)),
                "eval": r_eval.__dict__ if hasattr(r_eval, "__dict__") else str(r_eval),
            }
        )

    return results, stats