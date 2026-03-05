#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path


def normalize_to_jsonl(input_path: str, output_path: str) -> None:
    """
    Normalize a messy JSON/JSONL file into standard JSONL:
    - one JSON object per line
    - supports multiple objects on the same line
    - ignores extra blank lines / whitespace
    - also supports a top-level JSON array: [ {...}, {...} ]
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    text = input_file.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()

    objs = []
    i = 0
    n = len(text)

    while i < n:
        # Skip whitespace
        while i < n and text[i].isspace():
            i += 1

        if i >= n:
            break

        # If the whole file (or remaining content) starts with a JSON array
        if text[i] == "[":
            arr, end = decoder.raw_decode(text, i)
            if not isinstance(arr, list):
                raise ValueError(f"Expected a JSON array at position {i}, got {type(arr)}")
            for item in arr:
                if not isinstance(item, dict):
                    raise ValueError(f"Top-level array contains non-object item: {type(item)}")
                objs.append(item)
            i = end
            continue

        # Skip stray commas between objects if they exist
        if text[i] == ",":
            i += 1
            continue

        # Parse one JSON value
        obj, end = decoder.raw_decode(text, i)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected a JSON object at position {i}, got {type(obj)}")
        objs.append(obj)
        i = end

    with output_file.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. Parsed {len(objs)} samples.")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize messy JSON/JSONL into standard JSONL.")
    parser.add_argument("input", help="Path to the input file")
    parser.add_argument("output", help="Path to the normalized output JSONL file")
    args = parser.parse_args()

    normalize_to_jsonl(args.input, args.output)