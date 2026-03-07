# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.parsing import parse_prediction


class StructuredDecisionSignalsBuilder:
    """
    Build a compact, rule-based structured summary for meta-policy prompting.

    We organize signals into three cognitive-science-aligned levels:
      (1) Social consensus cues (group consistency)
      (2) Metacognitive monitoring cues (self reliability)
      (3) Metacognitive control & cognitive offloading cues (progress / intervention)
    """

    def __init__(self, tokenizer: Optional[Any] = None):
        self.tokenizer = tokenizer

    def build(
        self,
        task_type: str,
        self_history: List[str],
        others_histories: List[List[str]],
        self_idx: int,
        agents: int,
        sample_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        sample_meta = sample_meta or {}

        self_last = self_history[-1] if self_history else "(none)"
        self_prev = self_history[-2] if len(self_history) >= 2 else None
        others_last = [h[-1] if h else "(none)" for h in others_histories]

        # --- parse all latest solutions ---
        self_parsed = self._safe_parse(self_last, task_type, sample_meta)
        others_parsed = [self._safe_parse(t, task_type, sample_meta) for t in others_last]

        # map to global agent indices (others_histories excludes self)
        global_latest: Dict[int, str] = {}
        global_parsed: Dict[int, Dict[str, Any]] = {}

        other_global_indices = [j for j in range(agents) if j != self_idx]
        global_latest[self_idx] = self_last
        global_parsed[self_idx] = self_parsed

        for j, txt, parsed in zip(other_global_indices, others_last, others_parsed):
            global_latest[j] = txt
            global_parsed[j] = parsed

        # collect vote keys across agents
        all_vote_keys: List[str] = []
        for j in range(agents):
            vk = global_parsed[j]["vote_key"]
            if vk:
                all_vote_keys.append(vk)

        # Social consensus cues
        self_vote_key = self_parsed["vote_key"]

        agree_count = 0
        if self_vote_key:
            for j in other_global_indices:
                if global_parsed[j]["vote_key"] == self_vote_key:
                    agree_count += 1
        max_others = max(0, agents - 1)

        distinct_answers = len(set(all_vote_keys)) if all_vote_keys else 0
        majority_key, majority_count, second_count = self._majority_info(all_vote_keys)
        majority_margin = max(0, majority_count - second_count)

        # Metacognitive monitoring cues
        self_parsed_ok = bool(self_parsed["ok"])
        self_has_final_answer = bool(self_parsed["pred_str"])

        same_as_prev = self._same_as_previous_round(self_prev, self_last, task_type, sample_meta)
        completeness = self._reasoning_completeness(self_last)

        # Metacognitive control & offloading cues
        help_level = self._external_help_level(
            self_parsed_ok=self_parsed_ok,
            self_has_final_answer=self_has_final_answer,
            completeness=completeness,
            distinct_answers=distinct_answers,
            majority_margin=majority_margin,
            agree_count=agree_count,
            max_others=max_others,
            same_as_prev=same_as_prev,
        )

        internal_progress = self._internal_progress_potential(
            self_parsed_ok=self_parsed_ok,
            self_has_final_answer=self_has_final_answer,
            completeness=completeness,
            distinct_answers=distinct_answers,
            majority_margin=majority_margin,
            agree_count=agree_count,
            max_others=max_others,
            same_as_prev=same_as_prev,
        )

        trusted_idx, best_other_idx = self._find_best_other_candidate(
            self_vote_key=self_vote_key,
            majority_key=majority_key,
            global_latest=global_latest,
            global_parsed=global_parsed,
            other_global_indices=other_global_indices,
        )
        _ = trusted_idx, best_other_idx  #

        # self_vs_best_other = self._compare_self_vs_best_other(...)
        # overlap_label = self._text_overlap_label(...)

        # --- render labels ---
        agreement_desc = self._agreement_desc(agree_count, max_others)
        diversity_desc = self._diversity_desc(distinct_answers, agents)
        margin_desc = self._majority_margin_desc(majority_margin)
        same_prev_desc = "yes" if same_as_prev else "no"
        parsed_desc = "yes" if (self_parsed_ok and self_has_final_answer) else "no"
        guide_desc = "Guidance:\n- Avoid choosing yourself for EVAL unless necessary."

        lines = [
            "=== Structured Decision Signals ===",
            f"- Parsed answer: {parsed_desc}",
            f"- Agreement with others: {agree_count}/{max_others} ({agreement_desc})",
            f"- Answer diversity: {distinct_answers} ({diversity_desc})",
            f"- Majority margin: {majority_margin} ({margin_desc})",
            f"- Same as previous round: {same_prev_desc}",
            f"- Reasoning completeness: {completeness}",
            f"- External-help signal: {help_level}",
            f"- Internal progress potential: {internal_progress}",
        ]
        lines.append(guide_desc)
        return "\n".join(lines) + "\n\n"

    # -------------------------
    def _safe_parse(self, text: str, task_type: str, sample_meta: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = parse_prediction(text, task_type, sample_meta)
            return {
                "ok": bool(getattr(p, "ok", False)),
                "pred_str": (getattr(p, "pred_str", "") or "").strip(),
                "vote_key": (getattr(p, "vote_key", "") or "").strip(),
            }
        except Exception:
            return {"ok": False, "pred_str": "", "vote_key": ""}

    def _majority_info(self, vote_keys: List[str]) -> Tuple[str, int, int]:
        if not vote_keys:
            return "", 0, 0
        cnt = Counter(vote_keys).most_common()
        majority_key, majority_count = cnt[0]
        second_count = cnt[1][1] if len(cnt) >= 2 else 0
        return majority_key, majority_count, second_count

    def _same_as_previous_round(
        self,
        self_prev: Optional[str],
        self_last: str,
        task_type: str,
        sample_meta: Dict[str, Any],
    ) -> bool:
        if not self_prev:
            return False
        prev_p = self._safe_parse(self_prev, task_type, sample_meta)
        last_p = self._safe_parse(self_last, task_type, sample_meta)
        if prev_p["vote_key"] and last_p["vote_key"]:
            return prev_p["vote_key"] == last_p["vote_key"]
        return self._normalize_text(self_prev) == self._normalize_text(self_last)

    def _reasoning_completeness(self, text: str) -> str:
        if not text or text == "(none)":
            return "low"

        t = text.strip()
        t_norm = t.lower()

        char_len = len(t)
        tok_len = self._approx_token_len(t)

        has_conclusion = any(
            k in t_norm for k in [
                "therefore", "thus", "so the answer", "final answer", "\\boxed", "the answer is"
            ]
        )
        has_steps = bool(re.search(r"(^|\n)\s*(\d+[\.\)]|\- |\* )", t))
        has_math_work = bool(re.search(r"[=+\-*/]", t))

        score = 0
        if char_len >= 80 or tok_len >= 20:
            score += 1
        if has_steps or has_math_work:
            score += 1
        if has_conclusion:
            score += 1

        if score <= 1:
            return "low"
        elif score == 2:
            return "medium"
        return "high"

    def _find_best_other_candidate(
        self,
        self_vote_key: str,
        majority_key: str,
        global_latest: Dict[int, str],
        global_parsed: Dict[int, Dict[str, Any]],
        other_global_indices: List[int],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Kept for compatibility / future analysis
        """
        scored: List[Tuple[int, int]] = []

        for j in other_global_indices:
            parsed = global_parsed[j]
            txt = global_latest[j]
            score = 0

            if parsed["ok"] and parsed["pred_str"]:
                score += 2
            if majority_key and parsed["vote_key"] == majority_key:
                score += 2
            comp = self._reasoning_completeness(txt)
            if comp == "high":
                score += 2
            elif comp == "medium":
                score += 1
            if parsed["vote_key"] and parsed["vote_key"] == self_vote_key:
                score += 1

            scored.append((j, score))

        if not scored:
            return None, None

        scored.sort(key=lambda x: x[1], reverse=True)
        best_other_idx = scored[0][0]
        best_score = scored[0][1]

        trusted_idx = best_other_idx if best_score >= 4 else None
        return trusted_idx, best_other_idx

    def _compare_self_vs_best_other(
        self,
        self_last: str,
        self_parsed: Dict[str, Any],
        best_other_idx: Optional[int],
        global_latest: Dict[int, str],
        global_parsed: Dict[int, Dict[str, Any]],
        majority_key: str,
    ) -> str:
        """
        Kept for compatibility / future analysis
        """
        if best_other_idx is None:
            return "n/a"

        other_txt = global_latest[best_other_idx]
        other_parsed = global_parsed[best_other_idx]

        self_score = 0
        other_score = 0

        if self_parsed["ok"] and self_parsed["pred_str"]:
            self_score += 2
        if other_parsed["ok"] and other_parsed["pred_str"]:
            other_score += 2

        if majority_key and self_parsed["vote_key"] == majority_key:
            self_score += 2
        if majority_key and other_parsed["vote_key"] == majority_key:
            other_score += 2

        self_comp = self._reasoning_completeness(self_last)
        other_comp = self._reasoning_completeness(other_txt)

        self_score += {"low": 0, "medium": 1, "high": 2}[self_comp]
        other_score += {"low": 0, "medium": 1, "high": 2}[other_comp]

        if self_score >= other_score + 2:
            return "better"
        elif other_score >= self_score + 2:
            return "weaker"
        return "similar"

    def _internal_progress_potential(
        self,
        self_parsed_ok: bool,
        self_has_final_answer: bool,
        completeness: str,
        distinct_answers: int,
        majority_margin: int,
        agree_count: int,
        max_others: int,
        same_as_prev: bool,
    ) -> str:
        """
        Heuristic proxy for whether internal actions (EVAL/CREATE) are likely to be fruitful.
        Output: low / medium / high
        """
        # if group has a clear majority, internal path likely exists
        if majority_margin >= 2:
            return "high"
        if majority_margin == 1 and distinct_answers <= max(2, max_others):  # weak majority + limited diversity
            return "medium"
        self_reliable = self_parsed_ok and self_has_final_answer and completeness != "low"
        if self_reliable and same_as_prev and distinct_answers >= 2:
            return "medium"
        if majority_margin == 0 and (not self_reliable):
            return "low"

        # fallback
        if majority_margin == 0 and distinct_answers >= 2:
            return "low"

        return "medium"

    # control/offloading cues
    def _external_help_level(
        self,
        self_parsed_ok: bool,
        self_has_final_answer: bool,
        completeness: str,
        distinct_answers: int,
        majority_margin: int,
        agree_count: int,
        max_others: int,
        same_as_prev: bool,
    ) -> str:
        """
        Heuristic proxy for metacognitive control / cognitive offloading.
        Output: low / medium / high
        """
        # strong red flags on self usability
        self_unreliable = (not self_parsed_ok) or (not self_has_final_answer) or (completeness == "low")
        group_unresolved = (distinct_answers >= max(2, max_others + 1)) or (majority_margin == 0)
        isolated = (max_others > 0 and agree_count == 0 and distinct_answers >= 2)
        # stable but wrong? instability can be a mild signal
        unstable = (not same_as_prev) and (distinct_answers >= 2)

        if self_unreliable and (group_unresolved or isolated):
            return "high"
        if group_unresolved and isolated:
            return "high"
        if self_unreliable and unstable:
            return "high"

        if self_unreliable or group_unresolved or isolated:
            return "medium"

        return "low"


    def _text_overlap_label(self, a: str, b: str) -> str:
        """
        Kept for compatibility / future analysis
        """
        sa = self._simple_token_set(a)
        sb = self._simple_token_set(b)
        if not sa or not sb:
            return "low"
        j = len(sa & sb) / max(1, len(sa | sb))
        if j < 0.20:
            return "low"
        elif j < 0.50:
            return "medium"
        return "high"

    def _simple_token_set(self, text: str) -> set:
        text = self._normalize_text(text)
        toks = re.findall(r"[a-z0-9]+", text)
        stop = {
            "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on",
            "we", "need", "find", "answer", "final", "therefore", "thus", "so"
        }
        return {t for t in toks if t not in stop}

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _approx_token_len(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, len((text or "").split()))

    def _agreement_desc(self, agree_count: int, max_others: int) -> str:
        if max_others <= 0:
            return "n/a"
        if agree_count == 0:
            return "isolated"
        if agree_count == max_others:
            return "fully aligned"
        return "partial agreement"

    def _diversity_desc(self, distinct_answers: int, agents: int) -> str:
        if distinct_answers <= 0:
            return "unknown"
        if distinct_answers == 1:
            return "all agree"
        if distinct_answers >= agents:
            return "all different"
        return "partial disagreement"

    def _majority_margin_desc(self, margin: int) -> str:
        if margin <= 0:
            return "no clear majority"
        if margin == 1:
            return "weak majority"
        return "strong majority"