#!/usr/bin/env bash
set -e

# 1) Activate your venv
source /root/llama3/bin/activate

# 2) SSH setup is assumed already done (authorized_keys, known_hosts, etc.)

# 3) Load NCCL/MASTER env
source extras/.env.sh

# 4) Launch DeepSpeed
deepspeed \
  --hostfile extras/hostfile \
  --num_nodes 2 \
  --num_gpus 4 \
  train_llama3_full_ft.py \
    --model_name_or_path meta-llama/Llama-3-8B \
    --train_sequences data/train.sequences \
    --output_dir output/ll3-8b-ft \
    --deepspeed_config deepspeed_config.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_train_tokens 300000000000 \
    --logging_steps 100 \
    --num_workers 4

