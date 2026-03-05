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

# ====== Output ======
OUT_JSONL="outputs/mas_collaboration_interactive.jsonl"

python main.py \
  --mode mas_collaboration \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --data_root "$DATA_ROOT" \
  --limit "$LIMIT" \
  --agent_backend vllm \
  --model "$MODEL" \
  --tokenizer_model "$TOKENIZER_MODEL" \
  --interaction_mode interactive \
  --interactive_multiline \
  --out_jsonl "$OUT_JSONL"