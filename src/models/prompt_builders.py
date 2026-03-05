# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, List, Optional

from transformers import AutoTokenizer


def wrap_chat(tokenizer: AutoTokenizer, user_content: str) -> str:
    msgs = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_base_prompt(sample: Any, force_boxed: bool = False) -> str:
    """
    Turn a bare question string into a full instruction prompt.
    - sample.prompt: assumed to be ONLY the question text.
    """
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
        instruction = (
            "Answer the following multiple-choice question. Choose the single best option.\n\n"
            f"Solve the problem:\n\n{question}\n\n"
            "Think step by step, show your reasoning, "
            f"{answer_format}"
        )
    elif task_type in {"math_numeric", "math_symbolic"}:
        instruction = (
            f"Solve the following problem:\n\n{question}\n\n"
            "Think step by step, show your reasoning, and be careful with arithmetic.\n"
            f"{answer_format}"
        )
    elif task_type == "code_unit_test":
        instruction = (
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
        instruction = (
            f"Solve the following problem:\n\n{question}\n\n"
            "Think step by step.\n"
            f"{answer_format}"
        )

    return instruction


def _get_sample_meta_field(sample: Any, key: str) -> str:
    """
    Read a standardized field from sample.meta and return a stripped string.
    Missing / None -> "".
    """
    meta = getattr(sample, "meta", {}) or {}
    v = meta.get(key, "")
    return "" if v is None else str(v).strip()


def get_human_passive_reasoning(sample: Any) -> str:
    """
    Used when human_passive_flag=True and an agent chooses DEFER.
    Returns the stored human_reasoning from sample.meta.
    """
    return _get_sample_meta_field(sample, "human_reasoning")


def get_human_active_text(sample: Any, active_source: str) -> str:
    """
    active_source must be one of:
      - "human_idea"
      - "human_reasoning"

    Returns the corresponding standardized field from sample.meta.
    """
    if active_source not in {"human_idea", "human_reasoning"}:
        raise ValueError(
            f"Unsupported active_source='{active_source}'. "
            f"Must be one of: human_idea, human_reasoning"
        )
    return _get_sample_meta_field(sample, active_source)


def build_initial_prompt(
    sample: Any,
    force_boxed: bool = False,
    human_active_flag: bool = False,
    active_source: str = "human_idea",
) -> str:
    """
    Round-0 initialization prompt only.

    - If human_active_flag=False:
        identical to build_base_prompt(sample, force_boxed)
    - If human_active_flag=True:
        prepend auxiliary human info from sample.meta[active_source]
        as additional context for the initial generation only.

    This function is intentionally separate from build_base_prompt()
    so later policy/collaborate/defer prompts remain unchanged.
    """
    base_prompt = build_base_prompt(sample, force_boxed=force_boxed)

    if not human_active_flag:
        return base_prompt

    aux_text = get_human_active_text(sample, active_source)
    if not aux_text:
        return base_prompt

    task_type = getattr(sample, "task_type", "") or ""

    if task_type == "code_unit_test":
        return (
            "You are given the following auxiliary hint from a human collaborator.\n"
            "Use it as optional guidance, but still solve the task independently.\n\n"
            "=== Human Auxiliary Hint ===\n"
            f"{aux_text}\n\n"
            "=== Task ===\n"
            f"{base_prompt}"
        )

    return (
        "You are given the following auxiliary hint from a human collaborator.\n"
        "Use it as additional context if helpful, but still reason carefully and solve the problem yourself.\n\n"
        "=== Human Auxiliary Hint ===\n"
        f"{aux_text}\n\n"
        "=== Problem ===\n"
        f"{base_prompt}"
    )


def build_collaboration_prompt(
    task_type: str,
    base_prompt: str,
    self_history: List[str],
    others_histories: List[List[str]],
    use_chat_template: bool,
    tokenizer: Optional[AutoTokenizer] = None,
) -> str:
    """
    Collaborate prompt: show original prompt + self previous responses + other agents responses,
    then ask for an updated final answer.
    """
    if task_type in {"code_unit_test", "math_numeric", "math_symbolic", "mcq"}:
        self_history = self_history[-1:]
        others_histories = [h[-1:] for h in others_histories]

    if self_history:
        self_block = "\n\n".join(f"[Round {k}]\n{t}" for k, t in enumerate(self_history))
    else:
        self_block = "(none)"

    other_blocks: List[str] = []
    for j, hist in enumerate(others_histories):
        h = "\n\n".join(f"[Round {k}]\n{t}" for k, t in enumerate(hist))
        other_blocks.append(f"=== Other Agent {j} ===\n{h}")
    others_block = "\n\n".join(other_blocks) if other_blocks else "(none)"

    user_content = (
        "You are in a multi-agent debate.\n\n"
        "=== Original Prompt ===\n"
        f"{base_prompt}\n\n"
        "=== Your Previous Responses ===\n"
        f"{self_block}\n\n"
        "=== Other Agents' Responses ===\n"
        f"{others_block}\n\n"
        "Now compare the solutions, resolve disagreements, and provide an UPDATED final answer.\n"
        "Keep reasoning concise but correct. Finish with a clear final answer line.\n"
    )

    if use_chat_template:
        if tokenizer is None:
            raise ValueError("tokenizer is required when use_chat_template=True")
        return wrap_chat(tokenizer, user_content)
    return user_content


def build_human_defer_prompt(
    task_type: str,
    base_plain: str,
    agents_latest: str,
) -> str:
    if task_type == "code_unit_test":
        ask = (
            "Provide ONLY Python code for a correct solution that passes all unit tests.\n"
            "Do NOT include explanations.\n"
        )
    else:
        ask = (
            "Provide the best corrected solution and a clear final answer line.\n"
            "Keep reasoning concise but correct.\n"
        )
    hp = (
        "You are a human expert helping a multi-agent system.\n\n"
        "=== Problem ===\n"
        f"{base_plain}\n\n"
        "=== Agents' Latest Solutions ===\n"
        f"{agents_latest}\n\n"
        f"{ask}"
    )
    return hp