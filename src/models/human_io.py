# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List


def read_multiline_human_input(
    end_marker: str = "",
    prompt_prefix: str = "human> ",
) -> str:
    """
    Read multi-line human input from terminal.

    Rules:
    - If end_marker == "": finish when user enters an empty line.
    - Otherwise: finish when user enters a line exactly equal to end_marker.

    Returns the joined text.
    """
    lines: List[str] = []

    if end_marker == "":
        print("(Enter your response below. Submit an empty line to finish.)")
    else:
        print(f"(Enter your response below. Type '{end_marker}' on a new line to finish.)")

    while True:
        try:
            line = input(prompt_prefix)
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n[Interrupted by user]")
            break

        if end_marker == "":
            if line == "":
                break
        else:
            if line == end_marker:
                break

        lines.append(line)

    return "\n".join(lines).strip()