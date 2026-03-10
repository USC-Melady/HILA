<div align="center">

# Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning

### **ICLR 2026**

[![Conference](https://img.shields.io/badge/ICLR-2026-blueviolet?style=for-the-badge)](#)
[![Framework](https://img.shields.io/badge/Framework-HILA-success?style=for-the-badge)](#)
[![Training](https://img.shields.io/badge/Optimization-DLPO-orange?style=for-the-badge)](#)
[![Code](https://img.shields.io/badge/Code-Open%20Source-black?style=for-the-badge)](https://github.com/USC-Melady/HILA.git)

<p align="center">
  <b>HILA</b>: A principled human-in-the-loop multi-agent collaboration framework for adaptive deferral, metacognitive decision-making, and continual capability improvement.
</p>

<p align="center">
  <a href="https://github.com/USC-Melady/HILA.git"><b>Code</b></a> •
  <a href="#-overview"><b>Overview</b></a> •
  <a href="#-datasets"><b>Datasets</b></a> •
  <a href="#-checkpoints"><b>Checkpoints</b></a> •
  <a href="#-environment-setup"><b>Setup</b></a> •
  <a href="#-running-inference--data-generation"><b>Run</b></a> •
  <a href="#-training"><b>Training</b></a> •
  <a href="#-citation"><b>Citation</b></a>
</p>

</div>

---

## 📌 Overview

While scaling individual large language models has led to impressive progress, the next frontier lies in scaling **collaboration**. Existing multi-agent systems (MAS), however, are often fundamentally limited by a *closed-world assumption*: they can only reason within the static knowledge boundaries of their pre-trained parameters. This makes them brittle when facing problems that require external knowledge, corrective intervention, or adaptation beyond what the model already knows.

To address this limitation, we introduce **HILA** (**H**uman-**I**n-the-**L**oop Multi-**A**gent Collaboration), a principled framework that enables multi-agent LLM systems to **adaptively decide when to solve problems autonomously and when to defer to an external human expert**. Rather than treating human intervention as an ad-hoc patch, HILA formalizes it as a learnable metacognitive decision.

At the core of HILA is **Dual-Loop Policy Optimization (DLPO)**:

- **Inner Loop**: optimizes *immediate deferral decisions* using a cost-aware policy objective.
- **Outer Loop**: converts expert interventions into supervised learning signals, enabling *continual improvement* of the agents’ reasoning capabilities over time.

This repository contains the official implementation, datasets, checkpoints, and scripts for reproducing the HILA pipeline.

---

## 📝 Abstract

> While scaling individual Large Language Models (LLMs) has delivered remarkable progress, the next frontier lies in scaling collaboration through multi-agent systems (MAS). However, purely autonomous MAS remain ``closed-world'' systems, constrained by the static knowledge horizon of pre-trained models. This limitation makes them brittle on tasks requiring knowledge beyond training data, often leading to collective failure under novel challenges. To address this, we propose the **Human-In-the-Loop Multi-Agent Collaboration (HILA)** framework, a principled paradigm for human--agent collaboration. HILA trains agents to learn a metacognitive policy that governs when to solve problems autonomously and when to defer to a human expert. To operationalize this policy, we introduce **Dual-Loop Policy Optimization**, which disentangles immediate decision-making from long-term capability growth. The inner loop applies Group Relative Policy Optimization (GRPO) with a cost-aware reward to optimize deferral decisions, while the outer loop implements continual learning, transforming expert feedback into high-quality supervised signals that strengthen the agent's reasoning ability. Experiments on challenging mathematical and problem-solving benchmarks show that HILA, equipped with Dual-Loop Policy Optimization, consistently outperforms advanced MAS, establishing a principled foundation for collaborative and continually improving agentic systems. The code is available at [https://github.com/USC-Melady/HILA.git](https://github.com/USC-Melady/HILA.git).

---

## ✨ Key Features

- **Human-in-the-loop multi-agent collaboration** for adaptive expert deferral.
- **Metacognitive policy learning** that decides *when* to ask for help.
- **Dual-loop optimization** that separates short-term decision quality from long-term capability growth.
- **Continual learning from expert feedback**, turning interventions into future performance gains.
- Support for **multiple LLM backbones**, including **LLaMA** and **Qwen** families.
- Flexible support for both **local inference via vLLM** and **API-based inference via OpenAI backends**.
- Scripts for:
  - multi-agent debate inference,
  - GRPO-style data generation,
  - supervised fine-tuning (SFT),
  - checkpoint-based evaluation and experimentation.

---
## 📂 Datasets

### Public Benchmark Datasets

Due to licensing/copyright restrictions, we **cannot redistribute** some of the original benchmark datasets used in our experiments. Instead, we rely on the official releases hosted on **Hugging Face**, and you can download the required datasets directly from there:

- 🤗 **Hugging Face**: https://huggingface.co/

These include the standard reasoning / mathematical / problem-solving benchmarks used throughout our evaluation pipeline. Please fetch the relevant datasets from Hugging Face and place them under your local `data/` directory following the expected folder structure of the provided scripts.

To help you quickly verify that the pipeline runs end-to-end, we also provide a demo dataset on [Google Drive](https://drive.google.com/drive/folders/1XrzsCG8cg9xU0wTYhVm4eCQvPA6BO1YY?usp=drive_link) that can be used as a lightweight sanity-check.

### Human-Feedback-Based Dataset

In addition to public benchmarks, we construct a **human-feedback-based dataset** tailored to the HILA evaluation pipeline, designed to support **real human-in-the-loop** settings. This dataset contains expert intervention signals and annotations produced by **multiple trained PhD-level annotators** we hired for this project:

- 🧑‍🏫 **Human-feedback dataset**: **[Google Drive](https://drive.google.com/drive/folders/1divi-1z-ypH6FVgDYQMl9diaUtQOG4Bg?usp=drive_link)**

### Recommended Data Organization

A typical local structure may look like:

```bash
data/
├── gsm8k/
├── math/
├── mmlu/
└── ...
```

Please adapt the exact folder structure to match the arguments passed to `--data_root`, `--dataset`, and `--split`.

---

## 🧠 Checkpoints

We release trained checkpoints for multiple backbones to facilitate reproducibility and downstream experimentation.

### LLaMA Family

- **LLaMA 8B checkpoint**  
  [https://drive.google.com/drive/folders/1X-S--iJzUkyYeWJqpLgI7f2jan72beHI?usp=drive_link](https://drive.google.com/drive/folders/1X-S--iJzUkyYeWJqpLgI7f2jan72beHI?usp=drive_link)

- **LLaMA 3B checkpoint**  
  [https://drive.google.com/drive/folders/1_7fY8PLppgOjZ677ayrST4R7XBSkqzY-?usp=drive_link](https://drive.google.com/drive/folders/1_7fY8PLppgOjZ677ayrST4R7XBSkqzY-?usp=drive_link)

### Qwen Family

- **Qwen 7B checkpoint**  
  [https://drive.google.com/drive/folders/1RHjIk80gm9WySAnvDXatHaaKLcc3_j71?usp=drive_link](https://drive.google.com/drive/folders/1RHjIk80gm9WySAnvDXatHaaKLcc3_j71?usp=drive_link)

- **Qwen 3B checkpoint**  
  [https://drive.google.com/drive/folders/1GqwINRRAJWZ02Y9utT-NzSUCylLyprbO?usp=drive_link](https://drive.google.com/drive/folders/1GqwINRRAJWZ02Y9utT-NzSUCylLyprbO?usp=drive_link)

### Notes on Checkpoint Usage

Depending on your inference setup, you may use:

- a **fully merged model path**, or
- a **base model + LoRA adapter** pair.

This repository supports both workflows. If you are using LoRA adapters, make sure the adapter directory contains the expected PEFT files (e.g., `adapter_config.json` and adapter weights), and pass the corresponding `--lora_path` argument during inference.

---

## ⚙️ Environment Setup

We provide environment configuration files to simplify reproducibility:

- `environment.yaml`
- `environment.txt`

These are intended to support a clean and portable setup for both training and inference.

### Option 1: Conda (Recommended)

If `environment.yaml` is provided, the recommended setup is:

```bash
conda env create -f environment.yaml
conda activate hila
```

If your environment name in the YAML is different, please activate the corresponding name defined in the file.

### Option 2: Manual / Existing Conda Environment

If you already have a Python environment and want to install dependencies manually:

```bash
conda create -n hila python=3.10 -y
conda activate hila
pip install -r environment.txt
```

### Option 3: Update Existing Environment

If you already created the environment earlier and want to sync it with the latest YAML:

```bash
conda env update -f environment.yaml --prune
conda activate hila
```

### Suggested Dependency Stack

This project typically relies on the following ecosystem:

- **PyTorch**
- **Transformers**
- **vLLM**
- **PEFT / LoRA**
- **OpenAI Python SDK**
- **Accelerate**
- **Datasets**
- common scientific Python tooling (`numpy`, `tqdm`, etc.)

> We recommend using the provided environment files instead of manually pinning versions unless you have a strong reason to customize the stack.

---

### Practical Setup Tips

Before running the code, please double-check the following:

- Make sure your **CUDA** and driver versions are compatible with your installed **PyTorch** and **vLLM** versions.
- For **local multi-agent inference**, a **GPU-backed environment** is strongly recommended.
- For **OpenAI-backed runs**, please configure your API key in:

```python
# ./src/models/constants.py
API_KEY = ""
```

---

## 🚀 Running the Code

We provide **6 execution modes** to support different experimental settings, including basic MAS collaboration, LoRA-based evaluation, real-time interactive human intervention, and training.

> **Tip:** Run all scripts from the project root:
> ```bash
> bash <script_name>.sh
> ```

---

### ① 🧠 Basic MAS Collaboration (vLLM Agent + OpenAI Human)

This is the **default pipeline**: configure the vLLM base model and the OpenAI human expert model, then run the full MAS collaboration workflow.

- 📌 Script: `run_mas.sh`

```bash
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
```

### ② 🧩 MAS Collaboration with LoRA (vLLM + LoRA Adapter)

This mode runs the same pipeline, but loads a trained LoRA adapter on top of the base vLLM model (useful for evaluating fine-tuned checkpoints).

- 📌 Script: `run_mas_lora.sh`

```bash
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
```

### ③ 💬 Interactive Human-in-the-Loop (Real-Time Terminal Input)

In this mode, whenever an agent chooses `DEFER`, the system will print the request context and wait for real-time human input in the terminal.

- 📌 Script: `run_mas_interactive.sh`

```bash
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
```

### ④ 🧑‍🏫 Human Active Hints (Proactive Human Guidance at Initialization)

In this setting, real humans proactively provide helpful guidance at Round-0 initialization via `human_idea` or `human_reasoning`.

- 📌 Script: `run_mas_human_active.sh`
- 🔧 Key switch: `ACTIVE_SOURCE="human_idea"` (or `human_reasoning`)

```bash
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
```

### ⑤ 🧑‍💻 Human Passive Feedback (Use Stored Human Reasoning)

This mode enables passive human supervision: if an agent chooses `DEFER`, the system directly uses the stored field `human_reasoning` in `sample.meta` instead of calling OpenAI or waiting for interactive input.

- 📌 Script: `run_mas_human_passive.sh`

```bash
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

# ====== Output ======
OUT_JSONL="outputs/mas_collaboration_human_passive.jsonl"

python main.py \
  --mode mas_collaboration \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --data_root "$DATA_ROOT" \
  --limit "$LIMIT" \
  --agent_backend vllm \
  --model "$MODEL" \
  --tokenizer_model "$TOKENIZER_MODEL" \
  --human_passive_flag \
  --out_jsonl "$OUT_JSONL"
```

### ⑥ 🏋️ Training Mode

We also provide training scripts for running fine-tuning using the unified training entrypoint under `src/train.py`.

- 📌 Script: `run_train_sft_mode.sh`

```bash
python3 -m src.train \
  --trainer sft \
  --train_jsonl ./offline_data/data.jsonl \
  --init_adapter outputs/grpo/ \
  --output_dir outputs/sft_llama_8B \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --epochs 3 \
  --per_device_batch_size 1 \
  --grad_accum_steps 8 \
  --lr 1e-4 \
  --save_every 40 \
  --max_completion_tokens 768
```

---

### Training Notes

- Ensure your training JSONL is formatted correctly before launching.
- Make sure the chosen base model matches the tokenizer / adapter assumptions in your pipeline.
- For larger models, adjust:
  - batch size,
  - gradient accumulation,
  - precision,
  - available GPU memory.
- If resuming from prior adapters or intermediate checkpoints, verify the path passed to `--init_adapter`.

---

## 📁 Recommended Repository Contents

A clean repository layout may include files such as:

```bash
.
├── README.md
├── environment.yaml
├── environment.txt
├── run.sh
├── main.py
├── data/
├── outputs/
├── checkpoints/
└── ...
```

The exact structure may evolve, but the key scripts and configuration files above are the main entry points for most users.

---

## 🔬 Reproducibility Tips

To improve reproducibility of results:

- Fix random seeds when possible.
- Keep model, tokenizer, and adapter versions aligned.
- Record exact checkpoint paths used in each run.
- Use the same dataset split definitions across experiments.
- Store raw outputs in JSONL form for later re-evaluation.
- Prefer shell scripts (`run.sh`, `train_main.sh`) for stable experiment management.

For serious benchmarking, we strongly recommend logging:

- dataset name and split,
- number of agents and rounds,
- backend type (`vllm` vs `openai`),
- decoding parameters,
- checkpoint / adapter version,
- output file names.


---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{yangadaptive,
  title={Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning},
  author={Yang, Wei and Cao, Defu and Pang, Jiacheng and Weng, Muyan and Liu, Yan},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
