# Crawler and Dataset Creator for Distributed LLM Training

This repository automates the end-to-end pipeline for building a high-quality training dataset and fine-tuning a LLaMA-based model (e.g., LLaMA-3-8B) in a multi-node, multi-GPU environment.

---

## 1. Setup & Prework

NOTE: Using two ip addresses as an example: 10.65.4.2 10.65.4.3 
Before running the data pipeline or training, prepare your distributed cluster and environment as follows:

```bash
# On all nodes (master and slaves):
# 1) Create a Python virtual environment
python3 -m venv llama3
source llama3/bin/activate

# 2) Install system packages
sudo apt update -y
sudo apt install -y pdsh vim iputils-ping

# 3) Enable SSH root login & restart
sudo sed -i 's|^#\?\s*PermitRootLogin\s\+.*|PermitRootLogin yes|' /etc/ssh/sshd_config
sudo sed -i 's|^#\?\s*PasswordAuthentication\s\+.*|PasswordAuthentication yes|' /etc/ssh/sshd_config
sudo sed -i 's|^#\?\s*PermitEmptyPasswords\s\+.*|PermitEmptyPasswords no|' /etc/ssh/sshd_config
sudo service ssh restart

# 4) On master: generate and distribute SSH keys
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
data=(10.65.4.2 10.65.4.3)
for ip in "${data[@]}"; do
  ssh-copy-id -i ~/.ssh/id_ed25519.pub root@${ip}
  ssh-keyscan -H ${ip} >> ~/.ssh/known_hosts
done
chmod 700 ~/.ssh && chmod 600 ~/.ssh/*

# 5) Install Python dependencies
pip install --upgrade pip
pip install transformers accelerate datasets bitsandbytes sentencepiece protobuf huggingface-hub
deps=(torch torchvision torchaudio deepspeed)
pip uninstall -y ${deps[*]}
pip install --upgrade --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
deepspeed

# 6) Login to Hugging Face
huggingface-cli login

# 7) Configure NCCL for inter-node communication
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1   # Adjust to your network interface
export NCCL_IB_DISABLE=1          # Disable InfiniBand if not used
export NCCL_P2P_LEVEL=NVL
export MASTER_ADDR=10.65.4.2      # Master node IP
export MASTER_PORT=29500
```

> **Source:** Detailed setup instructions from `setup.txt` citeturn6file0

---

## 2. Repository Layout

```text
lion1b/                   # Project root
├── data/                 # Intermediate & final data shards
│   ├── raw_html_thread*.txt
│   ├── extracted_thread*.txt
│   ├── filtered_thread*.txt
│   ├── deduped_thread*.txt
│   ├── scrubbed_thread*.txt
│   ├── train_thread*.seq
│   ├── test_thread*.seq
│   └── ...
├── scripts/              # Parallelized data preprocessing scripts
│   ├── crawl_threaded.py
│   ├── extract_text.py
│   ├── filter_data.py
│   ├── dedupe.py
│   ├── scrub_pii.py
│   ├── tokenize_and_pack.py
│   └── count_tokens.py
├── finalizedata.sh        # Orchestrate full data prep pipeline
├── parall.sh             # Alternate pipeline orchestration script
├── commandsequence.sh    # Runs data, training, and inference end-to-end
├── requirements.txt      # Python dependencies
├── tokenizer/            # Optional: custom tokenizer tools
└── train/                # Training & inference scripts
    ├── deepspeed_config.json
    ├── train_llama3_full_ft.py
    └── inference_llama3_ft.py
```

---

## 3. Data Preparation Pipeline

### 3.1 Crawl the Web

Run a multi-threaded crawler to partition your seed URLs across threads:

```bash
# Using finalize pipeline
./finalizedata.sh            # Executes crawling + all downstream stages

# Or manually:
./parall.sh                  # Launches threaded crawl & parallel scripts
```

### 3.2 Extract Text

Strips HTML into plaintext per shard:

```bash
bash >> scripts/extract_text.py --input data/raw_html_thread$T.txt \
    --output data/extracted_thread$T.txt --threads 4 --part-id $T --num-parts 8
```

### 3.3 Filter & Clean

* **filter\_data.py:** language detection & keyword filter
* **dedupe.py:** fast SHA256 deduplication
* **scrub\_pii.py:** remove/mask PII

Each runs in parallel across shards.

### 3.4 Tokenize & Pack

Uses SentencePiece model to split into fixed-length sequences:

```bash
python scripts/tokenize_and_pack.py \
  --input data/scrubbed_thread$T.txt \
  --model tokenizer.model \
  --seq_len 4096 \
  --train_out data/train_thread$T.seq \
  --test_out data/test_thread$T.seq \
  --part-id $T --num-parts 8
```

Verify token counts with `count_tokens.py`:

```bash
python scripts/count_tokens.py --input data/train_thread$T.seq \
  --part-id $T --num-parts 8
```

---

## 4. Fine-tuning

Launch distributed training via DeepSpeed:

```bash
# On master node:
deepseed --hostfile extras/hostfile \
  train/train.py \
  --model_name_or_path meta-llama/Llama-3-8B \
  --train_file data/train_thread*.seq \
  --output_dir output/ll3-8b-ft \
  --deepspeed train/deepspeed_config.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --max_seq_length 4096
```

> **Tip:** Replace `--train_file` glob with a JSONL manifest if needed by `datasets` loader.

---

## 5. Inference

After fine-tuning, run inference:

```bash
python inference/inference_cli.py \
  --model output/ll3-8b-ft \
  --prompt "Hello, world!" \
  --max_length 200
```

---
