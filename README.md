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

### Practical Setup Tips

- Make sure your CUDA / driver version is compatible with your installed PyTorch and vLLM versions.
- For **local multi-agent inference**, a GPU-backed environment is strongly recommended.
- For **OpenAI-backed runs**, ensure your API key is exported or passed explicitly:
  
```bash
export OPENAI_API_KEY=your_key_here
```

- If you are running on a cluster or server, verify:
  - sufficient GPU memory,
  - network access (for API-based calls),
  - correct filesystem paths for datasets and checkpoints.

---

## 🚀 Running Inference / Data Generation

This repository supports two main runtime modes:

- **`debate`**: run the multi-agent debate / collaboration pipeline for inference
- **`grpo_data`**: generate grouped data for GRPO-style training or downstream policy optimization

At a high level, the runtime script exposes configurable options for:

- dataset and split selection,
- number of agents and debate rounds,
- local vs API-based agent backend,
- model / checkpoint / LoRA loading,
- decoding behavior,
- output file destinations,
- human expert backend configuration.

### Core Runtime Arguments

Below are the most important configurable arguments:

#### General mode and data
- `--mode`  
  Selects the execution mode. Typical choices:
  - `debate`
  - `grpo_data`

- `--dataset`  
  Specifies which dataset to run on (e.g., `gsm8k`).

- `--split`  
  Specifies the dataset split (e.g., `human`, `test`, etc.).

- `--data_root`  
  Root directory for local dataset files.

- `--limit`  
  Maximum number of examples to process.

#### Multi-agent collaboration
- `--agents`  
  Number of collaborating agents.

- `--rounds`  
  Number of discussion / debate rounds.

#### Agent backend
- `--agent_backend`  
  Selects the backend used by the agents:
  - `vllm` for local model serving
  - `openai` for API-based model inference

#### Model loading (vLLM backend)
- `--model`  
  Model identifier or local model path. Can point to a merged model or a directly loadable checkpoint.

- `--base_model`  
  Base model path / identifier, especially useful when using LoRA adapters.

- `--lora_path`  
  Path to the PEFT LoRA adapter directory.

- `--lora_name`  
  Logical name for the loaded LoRA adapter.

- `--lora_id`  
  Logical numeric identifier for the LoRA adapter.

- `--max_lora_rank`  
  Maximum LoRA rank supported by the runtime.

- `--dtype`  
  Runtime precision (e.g., `bfloat16`, `float16`, `float32`, or `auto`).

#### Decoding parameters
- `--max_tokens`  
  Maximum number of generated tokens per response.

- `--temperature`  
  Sampling temperature for agent generation.

- `--top_p`  
  Nucleus sampling threshold.

- `--seed`  
  Random seed for reproducibility.

#### Output formatting
- `--force_boxed` / `--no_force_boxed`  
  Controls whether final answers are forced into `\boxed{...}` format.

#### Output files
- `--out_jsonl`  
  Output JSONL file for per-sample debate results.

- `--out_grpo_jsonl`  
  Output JSONL file for grouped GRPO data.

#### OpenAI backend (agent)
- `--openai_model`
- `--openai_api_key`
- `--openai_concurrency`
- `--openai_timeout`

These control API-based inference when `--agent_backend openai` is used.

#### Human expert backend
- `--human_openai_model`
- `--human_openai_api_key`
- `--human_openai_concurrency`
- `--human_openai_timeout`
- `--human_max_tokens`
- `--human_temperature`
- `--human_top_p`

These configure the external “human expert” proxy used in the HILA pipeline (typically implemented with a stronger API model).

---

### Quick Start

Once your paths and environment are configured, the simplest way to run is:

```bash
sh run.sh
```

This is the recommended entry point for standard experiments.

### Typical Workflow

1. Download / prepare the dataset.
2. Download the appropriate checkpoint (or base model + LoRA adapter).
3. Update paths and runtime settings in your shell script or command line.
4. Launch the run:
   ```bash
   sh run.sh
   ```
5. Inspect the generated JSONL outputs.

---

## 🏋️ Training

The repository also provides a training entry point for supervised fine-tuning and related optimization workflows.

A typical command structure looks like:

```bash
python3 train_main.py \
  --trainer sft \
  --init_adapter "" \
  --output_dir outputs/sft_llama_3B \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --epochs 4 \
  --per_device_batch_size 1 \
  --grad_accum_steps 8 \
  --lr 1e-4 \
  --kl_beta 0.0 \
  --save_every 40
```

### Important Training Arguments

- `--trainer`  
  Selects the training routine (e.g., SFT).

- `--train_jsonl`  
  Path to the training data in JSONL format.

- `--init_adapter`  
  Optional path to an initial adapter checkpoint.

- `--output_dir`  
  Directory for saving trained checkpoints.

- `--model`  
  Base model identifier or path.

- `--epochs`  
  Number of training epochs.

- `--per_device_batch_size`  
  Batch size per device.

- `--grad_accum_steps`  
  Gradient accumulation steps.

- `--lr`  
  Learning rate.

- `--kl_beta`  
  KL regularization coefficient (if applicable).

- `--save_every`  
  Checkpoint save interval.

### Recommended Launch Method

For the standard training pipeline:

```bash
sh train_main.sh
```

This keeps the training workflow more reproducible and easier to manage than repeatedly typing long commands manually.

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
├── train_main.sh
├── train_main.py
├── data/
├── outputs/
├── sft_data/
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
