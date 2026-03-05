#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset


class SFTJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Train jsonl not found: {self.path}")

        self.rows: List[Dict[str, str]] = []

        with self.path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                # Strictly prefer prompt/completion (your main format)
                prompt = obj.get("prompt", None)
                completion = obj.get("completion", None)

                # Fallbacks only if primary fields are absent
                if prompt is None:
                    prompt = obj.get("question", None)
                if completion is None:
                    completion = obj.get("output", None)
                if prompt is None:
                    prompt = obj.get("input", None)

                # Last-resort fallback: question + gold only if gold is already a string
                if completion is None and isinstance(obj.get("gold", None), str):
                    completion = obj["gold"]

                if not isinstance(prompt, str) or not isinstance(completion, str):
                    raise ValueError(
                        f"Bad row on line {ln}: need string prompt/completion. "
                        f"Got prompt={type(prompt)}, completion={type(completion)}"
                    )

                if len(prompt) == 0:
                    raise ValueError(f"Bad row on line {ln}: empty prompt")
                if len(completion) == 0:
                    raise ValueError(f"Bad row on line {ln}: empty completion")

                self.rows.append({
                    "prompt": prompt,
                    "completion": completion,
                })

        if not self.rows:
            raise RuntimeError(f"No rows loaded from {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.rows[idx]