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
```
node0(master) node1(slave)
passwd 
cd /root
python3 -m venv llama3
source llama3/bin/activate

apt update -y
apt install -y pdsh
apt-get install vim
apt-get install -y iputils-ping

sed -i 's|^#\?\s*PermitRootLogin\s\+.*|PermitRootLogin yes|' /etc/ssh/sshd_config
sed -i 's|^#\?\s*PasswordAuthentication\s\+.*|PasswordAuthentication yes|' /etc/ssh/sshd_config
sed -i 's|^#\?\s*PermitEmptyPasswords\s\+.*|PermitEmptyPasswords no|' /etc/ssh/sshd_config
service ssh restart

#node0
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@10.65.0.3On 

#node0 node1
ssh-keyscan -H 10.65.4.2 >> ~/.ssh/known_hosts
ssh-keyscan -H 10.65.4.3 >> ~/.ssh/known_hosts

#on node0
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chmod 600 ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub

#Test 
ssh root@ip-address-of-node0  # local (master)
ssh root@ip-address-of-node1  # remote (slave)

pip install transformers accelerate datasets bitsandbytes sentencepiece protobuf huggingface-hub
python3 -c "import sentencepiece; print(sentencepiece.__version__)"
python3 -c "import google.protobuf; print(google.protobuf.__version__)"

which python    # should be something like /root/llama3/bin/python
which pip       # should point under /root/llama3/bin/pip

huggingface-cli login

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1   # <— your real inter-node interface
export NCCL_IB_DISABLE=1   # disable InfiniBand
export NCCL_P2P_LEVEL=NVL  # optionally force NVL-level P2P (can help on multi-GPU NICs)
export MASTER_ADDR=10.65.4.2  # The master node’s IP address
export MASTER_PORT=29500
#export NCCL_DEBUG=WARN

pip uninstall -y torch torchvision torchaudio deepspeed
pip install --upgrade --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install deepspeed

```

---

## Data Pipeline

All preprocessing is driven by `finalizedata.sh`:

```bash
#Download the DatasetCreator4LLMs repo
git clone https://github.com/stillandcalm/DatasetCreator4LLMs.git

#Download Llama-3-8B tokenizer.model from huggingface.
huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/tokenizer.model" --local-dir .
cd DatasetCreator4LLMs/
pip install -r requirements.txt
cp ../original/tokenizer.model .
mkdir data

./finalizedata.sh \
  --seeds seeds.txt \
  --domains domains.txt \
  --output-dir data \
  --threads 1 \
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

deepspeed_config.json (for single A100 env). For multi GPU based training use the config file defined in my multi-node project

{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "fp16": {
    "enabled": true
  },
  "train_micro_batch_size_per_gpu": 1,
  "train_batch_size": 16,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  }
}

deepspeed --hostfile hostfile train/train_llama3_full_ft.py   \
  --model_name_or_path meta-llama/Meta-Llama-3-8B  \
  --train_sequences data/train_thread*.seq  \
  --output_dir output/ll3-8b-ft  \
  --deepspeed deepspeed_config.json  \
  --per_device_train_batch_size 1  \
  --gradient_accumulation_steps 16 \
  --max_train_tokens 12800 \
  --logging_steps 100

Kept max_train_tokens to small number to demo fast.

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
