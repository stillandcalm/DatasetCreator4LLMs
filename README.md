# CrawlerAndDatasetCreator4DL

A fully parallelized web-crawler and data‑preparation pipeline for generating large-scale language-model training data, plus end‑to‑end scripts to fine‑tune and infer with LLaMA‑3 8B models using DeepSpeed.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Directory Structure](#directory-structure)
* [Data Preparation Pipeline](#data-preparation-pipeline)
* [Training](#training)
* [Inference](#inference)
* [Tokenizer](#tokenizer)
* [Extras](#extras)

## Prerequisites

* Ubuntu 20.04+ or equivalent
* Python 3.8+ with `venv`
* NVIDIA GPU(s) with CUDA 11.8
* `git` command line

## Installation

```bash
# Clone and enter repo
git clone git@github.com:stillandcalm/CrawlerandDatasetCreator4DL.git
cd CrawlerandDatasetCreator4DL

# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Directory Structure

```text
.
├── .gitignore               # Files/directories ignored by Git
├── commandsequence.sh       # Orchestrates full pipeline + training + inference
├── finalizedata.sh          # Runs just the data‑prep pipeline
├── requirements.txt         # Python libraries
├── cyber_keywords.txt       # Keywords for filtering extracted text
├── domains.txt              # Allowed crawl domains
├── seeds.txt                # Initial seed URLs for the crawler
├── data/                    # Intermediate and final data shards
├── extras/                  # Helper configs & scripts for distributed training
│   ├── hostfile             # DeepSpeed hostfile template
│   ├── NCCL.sh              # NCCL environment setup
│   └── run_deepspeed.sh     # Wrapper for multi-node DeepSpeed launch
├── inference/               # CLI script for model inference
│   └── inference_cli.py
├── tokenizer/               # Tokenizer training and configuration
│   ├── bpetokenizer.py      # Train a custom BPE tokenizer
│   ├── special_tokens_map.json
│   ├── tokencount.py        # Count tokens in raw text
│   ├── tokenize_and_pack.py # Tokenize & pack into .seq files
│   ├── tokenizer_config.json
│   └── tokenizer.json       # SentencePiece model config
├── train/                   # Fine‑tuning scripts & DeepSpeed settings
│   ├── deepspeed_config.json
│   ├── train_llama3_full_ft.py
│   └── inference_llama3_ft.py
└── scripts/                 # Parallelized data‑processing stages
    ├── crawl_threaded.py    # Multi‑threaded web crawler
    ├── extract_text.py      # HTML→plain‑text extraction
    ├── filter_data.py       # Language detection & keyword filtering
    ├── dedupe.py            # Fast deduplication (SHA‑256 hashing)
    ├── scrub_pii.py         # PII detection & masking
    ├── tokenize_and_pack.py # Tokenization + train/test packing by shard
    └── count_tokens.py      # Token count verification per shard
```

## Data Preparation Pipeline

To run the entire data pipeline (crawl → extract → filter → dedupe → scrub → tokenize):

```bash
./finalizedata.sh
```

`finalizedata.sh` invokes, in sequence:

1. **crawl\_threaded.py** — splits `seeds.txt` across threads; writes `data/raw_html_thread*.txt`
2. **extract\_text.py** — strips HTML tags; outputs `data/extracted_thread*.txt`
3. **filter\_data.py** — retains lines in target language containing `cyber_keywords.txt`; writes `data/filtered_thread*.txt`
4. **dedupe.py** — SHA‑256 dedupe; writes `data/deduped_thread*.txt`
5. **scrub\_pii.py** — masks/removes PII; writes `data/scrubbed_thread*.txt`
6. **tokenize\_and\_pack.py** — applies SentencePiece tokenizer; packs into `data/train_thread*.seq` and `data/test_thread*.seq`
7. **count\_tokens.py** — reports token counts to verify even sharding

Intermediate and final shards all live under `data/`.

## Training

Example DeepSpeed multi‑node launch using the helper script:

```bash
bash extras/run_deepspeed.sh \
  --model meta-llama/Llama-3-8B \
  --data-dir data/ \
  --output-dir output/llama3-8b-ft
```

Alternatively, you can call the training script directly:

```bash
deepspeed train/train_llama3_full_ft.py \
  --model_name_or_path meta-llama/Llama-3-8B \
  --train_file data/train.jsonl \
  --deepspeed train/deepspeed_config.json
```

After training you can package the model for inference via `train/inference_llama3_ft.py`.

## Inference

Once a fine‑tuned model is saved to `output/llama3-8b-ft`, run:

```bash
python inference/inference_cli.py \
  --model_dir output/llama3-8b-ft \
  --prompt "Your question here"
```

## Tokenizer

* **Train a custom tokenizer** from your corpus:

  ```bash
  python tokenizer/bpetokenizer.py \
    --input data/scrubbed.txt \
    --model_out tokenizer/tokenizer.model
  ```
* **Reuse HF LLaMA‑3 tokenizer**:

  ```bash
  python scripts/get_llama3-8b_tokenizer.py
  ```

## Extras

* `extras/hostfile` — configure multi‑node hosts
* `extras/NCCL.sh` — set NCCL environment variables
* `extras/run_deepspeed.sh` — example DeepSpeed launcher wrapper
