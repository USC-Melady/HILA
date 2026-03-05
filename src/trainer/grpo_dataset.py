#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset


class GRPOGroupedDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Train jsonl not found: {self.path}")

        self.groups: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                completions = obj.get("completions", [])
                rewards = obj.get("rewards", [])

                if not prompt or not isinstance(completions, list) or not isinstance(rewards, list):
                    raise ValueError(f"Invalid group format on line {ln}: need prompt/completions/rewards")

                if len(completions) != len(rewards) or len(completions) == 0:
                    raise ValueError(f"Invalid group lengths on line {ln}: completions vs rewards mismatch/empty")

                self.groups.append(obj)

        if not self.groups:
            raise RuntimeError(f"No groups loaded from {self.path}")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.groups[idx]