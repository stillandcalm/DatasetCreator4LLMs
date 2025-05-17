# Crawler and Dataset Creator for LLaMA Finetuning

This repository provides a complete end‑to‑end pipeline for:

1. **Data acquisition & preprocessing** via multi‑threaded web crawling and text cleaning
2. **Dataset tokenization & packing** into train/test sequences
3. **Distributed fine‑tuning** of LLaMA‑3 (8B) using DeepSpeed
4. **Inference CLI** for rapid model querying

---

## Table of Contents

* [Prerequisites](#prerequisites)
* [Directory Structure](#directory-structure)
* [Setup (Multi‑Node SSH & Environment)](#setup-multi-node-ssh--environment)
* [Data Pipeline](#data-pipeline)
* [Fine‑Tuning](#fine-tuning)
* [Inference](#inference)
* [Commands & Examples](#commands--examples)
* [License](#license)

---

## Prerequisites

* Two Ubuntu nodes with passwordless SSH:

  * **Master** (example IP: `10.65.4.2`)
  * **Worker** (example IP: `10.65.4.3`)
* Python 3.8+ with virtualenv
* NVIDIA GPUs with CUDA 11.8

---

## Directory Structure

```
/                    # project root
├── data/            # raw & processed data shards
├── scripts/         # Python pipeline scripts
├── finalizedata.sh  # orchestrates full crawl + processing pipeline
├── commandsequence.sh # runs data→train→infer end-to-end
├── train/           # training & inference scripts
├── inference/       # CLI for model querying
├── tokenizer/       # custom tokenizer tools (optional)
├── requirements.txt
├── setup.txt        # detailed distributed setup instructions
└── README.md
```

---

## Setup (Multi‑Node SSH & Environment)

1. **On both nodes:**

   ```bash
   sudo apt update && sudo apt install -y pdsh sshpass
   ssh-keygen -t ed25519
   # Append public key to ~/.ssh/authorized_keys
   ```
2. **Configure SSH & network:**

   ```bash
   # On master:
   ssh-copy-id root@10.65.4.3
   # On both nodes:
   ssh-keyscan -H 10.65.4.2 >> ~/.ssh/known_hosts
   ssh-keyscan -H 10.65.4.3 >> ~/.ssh/known_hosts
   ```
3. **Activate Python venv & install deps:**

   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Set NCCL & DeepSpeed ENV vars:**

   ```bash
   export MASTER_ADDR=10.65.4.2
   export MASTER_PORT=29500
   export NCCL_SOCKET_IFNAME=eth1
   export NCCL_IB_DISABLE=1
   ```

---

## Data Pipeline

All preprocessing is driven by `finalizedata.sh`:

```bash
#Download Llama-3-8B tokenizer.model from huggingface.
huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/tokenizer.model" --local-dir .

./finalizedata.sh \
  --seeds seeds.txt \
  --domains domains.txt \
  --output-dir data \
  --threads 8 \
  --max-pages 100000 \
  --delay 1.0
```

This will:

1. **Crawl** seeds/domains → per-thread `raw_html_threadN.txt`
2. **Extract** text → `extracted_threadN.txt`
3. **Filter** language/keywords → `filtered_threadN.txt`
4. **Deduplicate** by SHA256 → `deduped_threadN.txt`
5. **PII scrub** → `scrubbed_threadN.txt`
6. **Tokenize & pack** with LLaMA‐3 tokenizer → `train_threadN.seq` & `test_threadN.seq`
7. **Count tokens** for balanced shards

All intermediate files live in `data/`. You can inspect per‐thread stats in the final summary table.

Download Llama-3-8B tokenizer.model from huggingface.
huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/tokenizer.model" --local-dir .

---

## Fine‑Tuning

Under `train/` you have:

* `train_llama3_full_ft.py` – DeepSpeed multi‑GPU finetuning script
* `inference_llama3_ft.py` – conversion + CLI inference setup
* `deepspeed_config.json` – your Zero3 config

**Example:**

```bash
deepseed --hostfile extras/hostfile train/train_llama3_full_ft.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --train_sequences data/train_thread*.seq \
  --output_dir output/ll3-8b-ft \
  --deepspeed train/deepspeed_config.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_train_tokens 3e11 \
  --logging_steps 100
```

---

## Inference

Once saved, convert checkpoint to HF format:

```bash
python train/modelcreate.py \
  --checkpoint output/ll3-8b-ft/checkpoint-xxx \
  --out_dir inference/llama3-8b-ft
```

Run the CLI:

```bash
python inference/inference_cli.py \
  --model_dir inference/llama3-8b-ft \
  --prompt "Hello, world"
```

---

## Command Sequence

`commandsequence.sh` ties everything together:

```bash
#!/usr/bin/env bash
./finalizedata.sh ...
# then train
# then inference
```

---
