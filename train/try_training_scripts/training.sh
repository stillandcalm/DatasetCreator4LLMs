export NCCL_DEBUG=INFO
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29500

deepspeed \
  --num_nodes 2 \
  --num_gpus 4 \
  train_llama3_seq_ft.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --train_sequences data/train_thread*.seq \
    --output_dir output/ll3-8b-finetuned \
    --deepspeed deepspeed_config.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --max_seq_length 4096 \
    --logging_steps 100

