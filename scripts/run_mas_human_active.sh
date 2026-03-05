#!/usr/bin/env bash
#set -euo pipefail

# ====== Experiment ======
DATASET="gsm8k"
SPLIT="human"
DATA_ROOT="data"
LIMIT=10000

# ====== Agent (vLLM) ======
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TOKENIZER_MODEL="$MODEL"

ACTIVE_SOURCE="human_idea"  # human_reasoning

# ====== Human (OpenAI) ======
HUMAN_OPENAI_MODEL="gpt-4o-mini"

# ====== Output ======
OUT_JSONL="outputs/mas_collaboration_human_active_idea.jsonl"

python main.py \
  --mode mas_collaboration \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --data_root "$DATA_ROOT" \
  --limit "$LIMIT" \
  --agent_backend vllm \
  --model "$MODEL" \
  --tokenizer_model "$TOKENIZER_MODEL" \
  --human_openai_model "$HUMAN_OPENAI_MODEL" \
  --human_active_flag \
  --active_source "$ACTIVE_SOURCE" \
  --out_jsonl "$OUT_JSONL"