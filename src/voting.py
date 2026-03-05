# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VoteResult:
    chosen_key: str
    ok: bool
    method: str
    stats: Dict[str, Any]


def majority_vote(keys: List[Optional[str]]) -> VoteResult:
    """
    Majority vote on vote_key (strings).
    Ignores None/empty.
    """
    xs = [str(x) for x in keys if x is not None and str(x).strip() != ""]
    if not xs:
        return VoteResult(chosen_key="", ok=False, method="majority", stats={"counts": {}})

    cnt = Counter(xs)
    chosen, n = cnt.most_common(1)[0]
    return VoteResult(
        chosen_key=chosen,
        ok=True,
        method="majority",
        stats={"counts": dict(cnt), "winner_count": n, "total": len(xs)},
    )


def first_nonempty(keys: List[Optional[str]]) -> VoteResult:
    """
    Deterministic fallback vote.
    """
    for k in keys:
        if k is not None and str(k).strip() != "":
            return VoteResult(chosen_key=str(k), ok=True, method="first_nonempty", stats={})
    return VoteResult(chosen_key="", ok=False, method="first_nonempty", stats={})
