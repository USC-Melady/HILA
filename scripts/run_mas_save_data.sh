#!/usr/bin/env bash
#set -euo pipefail

# ====== Experiment ======
DATASET="gsm8k"
SPLIT="train"
DATA_ROOT="data"
LIMIT=50

# ====== Agent (vLLM) ======
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TOKENIZER_MODEL="$MODEL"

# ====== Human (OpenAI) ======
HUMAN_OPENAI_MODEL="gpt-4o-mini"

# ====== Output ======
OUT_JSONL="outputs/mas_collaboration_save_sft.jsonl"
SFT_OUT_JSONL="outputs/mas_collaboration_sft_data.jsonl"

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
  --save_sft_data \
  --sft_out_jsonl "$SFT_OUT_JSONL" \
  --out_jsonl "$OUT_JSONL"