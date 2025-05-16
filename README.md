# Crawler and Dataset Creator for LLM training (README)

## Overview

This repository provides a scalable, parallelized data collection and preprocessing pipeline, along with training and inference orchestration for cybersecurity-focused language models. The core components are two Bash scripts that automate the workflow:

1. **`finalizedata.sh`**: Cleans, filters, deduplicates, and tokenizes raw HTML data into training and test sequences. citeturn5file2
2. **`commandsequence.sh`**: Runs the data finalization, model training, and inference steps in sequence. citeturn5file0

## Prerequisites

* Unix-like shell (`bash`)
* Python 3 with required packages:

  ```bash
  pip install -r requirements.txt
  ```
* SentencePiece model file (`tokenizer.model`) built or downloaded for your desired tokenizer.
* Seeds (`seeds.txt`) and domain list (`domains.txt`) files.

## Scripts

### 1. `finalizedata.sh`

This script orchestrates the data preprocessing pipeline over **$NUM_SHARDS$** parallel shards:

```bash
#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Configuration variables
NUM_SHARDS=8             # number of parallel pieces
WORKERS=4                # multiprocessing workers
SEEDS=seeds.txt
DOMAINS=domains.txt
OUTDIR=data
KEYWORDS=cyber_keywords.txt
SPM_MODEL=tokenizer.model # LLaMA-3-8B tokenizer.model
SEQ_LEN=4096             # sequence length
TRAIN_FRAC=0.9           # train/test split fraction

# Helper to print counts for file globs
print_counts(){
  local stage=$1
  local glob=$2
  echo
  echo "=== ${stage} counts ==="
  printf  " shard |   lines |    bytes\n"
  printf  "-------:|--------:|--------:\n"
  for path in ${OUTDIR}/${glob}; do
    [[ -e $path ]] || continue
    shard=$(basename $path | grep -o '[0-9]\+')
    l=$(wc -l <"$path" 2>/dev/null || echo 0)
    b=$(wc -c <"$path" 2>/dev/null || echo 0)
    printf "   %2s   | %7s | %7s\n" "$shard" "$l" "$b"
  done
}
```

**Pipeline stages:**

1. **Crawl**: Fetch raw HTML pages in parallel threads. Writes `raw_html_threadN.txt` citeturn5file5.
2. **Extract Text**: Use `scripts/extract_text.py` to extract clean text from HTML shards. Writes `extracted_threadN.txt` citeturn5file4
3. **Filter Data**: `scripts/filter_data.py` applies language detection and keyword filtering. Writes `filtered_threadN.txt` citeturn5file4.
4. **Deduplicate**: Remove duplicate content via SHA256 hashing. Writes `deduped_threadN.txt` citeturn5file4.
5. **Scrub PII**: `scripts/scrub_pii.py` masks or removes personally identifiable information. Writes `scrubbed_threadN.txt` citeturn5file2.
6. **Tokenize and Pack**: `scripts/tokenize_and_pack.py` uses the SentencePiece model to tokenize, split into train/test, and produce `train_threadN.seq` and `test_threadN.seq` citeturn5file2.
7. **Count Tokens**: `scripts/count_tokens.py` reports token counts per shard and grand total. Optionally reshards to equalize token budgets.

To run data finalization:

```bash
sh finalizedata.sh
```

### 2. `commandsequence.sh`

This script ties together data finalization, training, and inference:

```bash
# Run data finalization pipeline
sh finalizedata.sh

# Training
sh trainingcommand.sh

# Inference examples
python scripts/inference_cli.py \
  --model_dir output/ll3-8b-ft \
  --prompt "Explain the MITRE ATT&CK framework." \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 0.95
```

* **Training (`trainingcommand.sh`)**: invokes `train.py` or `deepspeed train.py` with appropriate flags.
* **Inference (`inference_cli.py`)**: Provides a command-line interface to load the fine-tuned model and generate text.
