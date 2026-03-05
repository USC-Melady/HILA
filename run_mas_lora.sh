#!/usr/bin/env bash
#set -euo pipefail

# ====== Experiment ======
DATASET="mmlu"
SPLIT="test"
DATA_ROOT="data"
LIMIT=10000

# ====== Agent (vLLM + LoRA) ======
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TOKENIZER_MODEL="$BASE_MODEL"
LORA_PATH="./checkpoints/LLAMA-8B"

# ====== Human (OpenAI) ======
HUMAN_OPENAI_MODEL="gpt-4o-mini"

# ====== Output ======
OUT_JSONL="outputs/mas_collaboration_vllm_openai_lora.jsonl"

python main.py \
  --mode mas_collaboration \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --data_root "$DATA_ROOT" \
  --limit "$LIMIT" \
  --agent_backend vllm \
  --base_model "$BASE_MODEL" \
  --tokenizer_model "$TOKENIZER_MODEL" \
  --lora_path "$LORA_PATH" \
  --lora_name default_lora \
  --lora_id 1 \
  --human_openai_model "$HUMAN_OPENAI_MODEL" \
  --out_jsonl "$OUT_JSONL"