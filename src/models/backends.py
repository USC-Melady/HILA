# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from openai import AsyncOpenAI
from tqdm import tqdm
from vllm import LLM, SamplingParams

from .human_io import read_multiline_human_input


class LLMBackend(ABC):
    @abstractmethod
    def generate_batch(self, prompts: List[str], show_tqdm: bool = False) -> List[str]:
        raise NotImplementedError


class ConsoleHumanBackend(LLMBackend):
    """
    A human backend that prints each prompt to the terminal and waits for real user input.
    """

    def __init__(
        self,
        multiline: bool = True,
        end_marker: str = "",
        show_prompt_separator: bool = True,
    ):
        self.multiline = multiline
        self.end_marker = end_marker
        self.show_prompt_separator = show_prompt_separator

    def generate_batch(self, prompts: List[str], show_tqdm: bool = False) -> List[str]:
        outs: List[str] = []
        total = len(prompts)

        for k, p in enumerate(prompts, start=1):
            if self.show_prompt_separator:
                print("\n" + "=" * 100)
                print(f"[INTERACTIVE HUMAN INPUT] Request {k}/{total}")
                print("=" * 100)
                print(p)
                print("=" * 100)

            if self.multiline:
                text = read_multiline_human_input(
                    end_marker=self.end_marker,
                    prompt_prefix="human> ",
                )
            else:
                try:
                    text = input("human> ").strip()
                except EOFError:
                    print()
                    text = ""
                except KeyboardInterrupt:
                    print("\n[Interrupted by user]")
                    text = ""

            outs.append(text)

        return outs


class VLLMBackend(LLMBackend):
    def __init__(
        self,
        llm: LLM,
        sampling: SamplingParams,
        lora_request: Optional[Any] = None,
    ):
        self.llm = llm
        self.sampling = sampling
        self.lora_request = lora_request

    def generate_batch(
        self,
        prompts: List[str],
        show_tqdm: bool = False,
        sampling_override: Optional[SamplingParams] = None,
    ) -> List[str]:
        sp = sampling_override or self.sampling

        gen_kwargs = {"use_tqdm": show_tqdm}
        if self.lora_request is not None:
            gen_kwargs["lora_request"] = self.lora_request

        outs = self.llm.generate(prompts, sp, **gen_kwargs)
        return [o.outputs[0].text for o in outs]


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        request_timeout: int = 60,
        max_concurrency: int = 32,
        retries: int = 2,
    ):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set (env var or --openai_api_key/--human_openai_api_key)"
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.request_timeout = request_timeout
        self.max_conc = max_concurrency
        self.retries = retries

    def generate_batch(self, prompts: List[str], show_tqdm: bool = False) -> List[str]:
        async def _runner() -> List[str]:
            sem = asyncio.Semaphore(self.max_conc)
            client = AsyncOpenAI()
            outs: List[str] = [""] * len(prompts)

            async def _one(i: int, p: str):
                async with sem:
                    last_err = None
                    for attempt in range(self.retries + 1):
                        try:
                            r = await client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": p}],
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                timeout=self.request_timeout,
                            )
                            outs[i] = (r.choices[0].message.content or "").strip()
                            return i
                        except Exception as e:
                            last_err = e
                            if attempt < self.retries:
                                await asyncio.sleep(0.5 * (2 ** attempt))
                    raise last_err

            tasks = [asyncio.create_task(_one(i, p)) for i, p in enumerate(prompts)]

            if show_tqdm:
                pbar = tqdm(total=len(tasks), desc="OpenAI completed", leave=False)
            else:
                pbar = None

            try:
                for fut in asyncio.as_completed(tasks):
                    await fut
                    if pbar:
                        pbar.update(1)
                return outs
            finally:
                if pbar:
                    pbar.close()
                await client.close()

        return asyncio.run(_runner())


@dataclass
class DualLLM:
    agent: LLMBackend
    human: LLMBackend