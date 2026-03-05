# -*- coding: utf-8 -*-

from .constants import DEFAULT_MODEL_ID, API_KEY
from .token_utils import count_tokens
from .prompt_builders import (
    wrap_chat,
    build_base_prompt,
    get_human_passive_reasoning,
    get_human_active_text,
    build_initial_prompt,
    build_collaboration_prompt,
    build_human_defer_prompt,
)
from .structured_signals import StructuredDecisionSignalsBuilder
from .policy_utils import build_policy_prompt, parse_policy
from .human_io import read_multiline_human_input
from .backends import (
    LLMBackend,
    ConsoleHumanBackend,
    VLLMBackend,
    OpenAIBackend,
    DualLLM,
)
from .mas_collaboration_core import CollaborateStats, run_mas_collaboration
from .grpo_core import build_grpo_dataset