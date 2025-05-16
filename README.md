# Crawler and Dataset Creator for Deep Learning

## Overview

This project implements a scalable, parallelized pipeline for crawling web data, extracting and preprocessing text, deduplicating content, scrubbing personally identifiable information, tokenizing, and packing data into fixed-length sequences suitable for training large language models (e.g., LLaMA-3-8B). It also includes orchestration scripts for fine-tuning the model with DeepSpeed on multi-node/multi-GPU environments and for running inference.

## Repository Structure

```
/
├── data/                      # Storage for intermediate and final data shards
├── scripts1/                  # Optimized, parallel Python scripts for each pipeline stage
│   ├── crawl_threaded.py      # Multi-threaded web crawler
│   ├── extract_text.py        # HTML-to-text extraction
│   ├── filter_data.py         # Language detection & keyword filtering
│   ├── dedupe.py              # SHA256-based deduplication
│   ├── scrub_pii.py           # PII masking/removal
│   ├── tokenize_and_pack.py   # Tokenization & packing into train/test splits
│   └── count_tokens.py        # Token counting per shard
├── scripts/                   # Original reference scripts
├── finalizedata.sh            # Orchestrates the entire data preparation pipeline
├── commandsequence.sh         # Runs data pipeline, training, and inference in order
├── trainingcommand.sh         # DeepSpeed multi-node/multi-GPU training launcher
├── requirements.txt           # Python dependencies
├── tokenizer/                 # Custom tokenizer training or configuration (optional)
└── train/                     # Training & inference scripts for fine-tuning LLaMA
```

## Scripts

### finalizedata.sh

This Bash script runs the data preprocessing pipeline in **NUM\_SHARDS** parallel shards:

```bash
#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Configuration
NUM_SHARDS=8             # Number of parallel shards\NWORKERS=4              # Multiprocessing workers
SEEDS=seeds.txt
DOMAINS=domains.txt
OUTDIR=data
KEYWORDS=cyber_keywords.txt
SPM_MODEL=tokenizer.model # SentencePiece model file
SEQ_LEN=4096             # Sequence length for training
TRAIN_FRAC=0.9           # Fraction of data as training set

# Helper: print line & byte counts per shard
print_counts(){ ... }

# 1) Crawl
# 2) Extract Text
# 3) Filter Data
# 4) Deduplicate
# 5) Scrub PII
# 6) Tokenize & Pack
# 7) Count Tokens
```

#### Pipeline Stages

1. **Crawl**: Fetch pages using `crawl_threaded.py` → `raw_html_thread<N>.txt`
2. **Extract Text**: `extract_text.py` parses HTML → `extracted_thread<N>.txt`
3. **Filter Data**: `filter_data.py` applies language detection & keyword matching → `filtered_thread<N>.txt`
4. **Deduplicate**: `dedupe.py` removes duplicate lines via SHA256 → `deduped_thread<N>.txt`
5. **Scrub PII**: `scrub_pii.py` masks/removes emails, IPs, etc. → `scrubbed_thread<N>.txt`
6. **Tokenize & Pack**: `tokenize_and_pack.py` tokenizes with SentencePiece, splits train/test → `train_thread<N>.seq`, `test_thread<N>.seq`
7. **Count Tokens**: `count_tokens.py` reports token counts per shard and total.

**Run Data Preparation**

```bash
bash finalizedata.sh
```

---

### commandsequence.sh

High-level orchestration of data preparation, model training, and inference:

```bash
#!/usr/bin/env bash
set -e

# 1) Data finalization\nbash finalizedata.sh
# 2) Training\nbash trainingcommand.sh
# 3) Inference example
python train/inference_cli.py \
  --model_dir output/ll3-8b-ft \
  --prompt "Explain the MITRE ATT&CK framework." \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 0.95
```

* **trainingcommand.sh**: Configures environment and launches DeepSpeed with `train_llama3_full_ft.py` using generated `.seq` files.
* **inference\_cli.py**: Loads the fine-tuned model (`output/ll3-8b-ft`) and generates text from CLI.

---

## Python Scripts (in `scripts1/`)

* **crawl\_threaded.py**: Splits seed URLs across threads, fetches HTML concurrently, writes per-thread raw HTML shards.
* **extract\_text.py**: Reads HTML shards, uses BeautifulSoup + lxml to extract visible text, outputs plain-text shards.
* **filter\_data.py**: Applies language detection (fastText or langdetect) and keyword filtering against `cyber_keywords.txt`, enforces minimum length.
* **dedupe.py**: Hashes each line (SHA256) to drop duplicates within each shard.
* **scrub\_pii.py**: Masks/removes PII (emails, IPs, phone numbers) via regex and optional NLP.
* **tokenize\_and\_pack.py**: Loads SentencePiece model (`--model`), tokenizes text, packs into fixed-length sequences (`--seq_len`), splits into train/test by `--train_frac`.
* **count\_tokens.py**: Counts tokens in `.seq` files per shard; can aggregate for token-budget-based training.

---

## Training & Inference

* **trainingcommand.sh**: Sets NCCL and master node env vars, then calls:

  ```bash
  ```

de\$\$pspeed --hostfile hostfile train\_llama3\_full\_ft.py&#x20;
\--model\_name\_or\_path meta-llama/Llama-3-8B&#x20;
\--train\_sequences data/train\_thread\*.seq&#x20;
\--deepspeed deepspeed\_config.json&#x20;
\--per\_device\_train\_batch\_size 1&#x20;
\--gradient\_accumulation\_steps 16&#x20;
\--max\_train\_tokens 3e11

````
- **train_llama3_full_ft.py**: Uses Transformers `Trainer` with DeepSpeed stage 3, gradient checkpointing.
- **inference_cli.py**: Loads `inference-llama3-8b/` directory and generates tokens for a given prompt.

**Usage**
```bash
# Fine-tune model
bash trainingcommand.sh

# Run inference
python train/inference_cli.py --model_dir output/ll3-8b-ft --prompt "Hello world" --max_new_tokens 50
````

---

## Customization

* Adjust `NUM_SHARDS`, `WORKERS`, and thread counts for your hardware.
* Provide your own `seeds.txt`, `domains.txt`, and `cyber_keywords.txt`.
* Ensure `tokenizer.model` matches your target LLaMA base model tokenizer.

*README generated based on project scripts.*

