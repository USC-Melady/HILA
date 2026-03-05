#!/usr/bin/env bash
#set -euo pipefail

# ====== Experiment ======
DATASET="gsm8k"
SPLIT="test"
DATA_ROOT="data"
LIMIT=10000

# ====== Agent (vLLM) ======
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TOKENIZER_MODEL="$MODEL"

# ====== Human (OpenAI) ======
HUMAN_OPENAI_MODEL="gpt-4o-mini"

# ====== Output ======
OUT_JSONL="outputs/mas_collaboration_vllm_openai.jsonl"

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
  --out_jsonl "$OUT_JSONL"