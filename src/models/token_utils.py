# -*- coding: utf-8 -*-

from transformers import AutoTokenizer


def count_tokens(tokenizer: AutoTokenizer, txt: str) -> int:
    """
    Approximate token count with HF tokenizer.
    Also used as a rough proxy for OpenAI prompts/outputs.
    """
    return len(tokenizer.encode(txt, add_special_tokens=False))