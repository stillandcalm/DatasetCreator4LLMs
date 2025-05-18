#!/usr/bin/env python3
import logging
import os
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

def main():
    parser = argparse.ArgumentParser(description="Finetune LLaMA-3 on a single H100 node using pre-tokenized input")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Running on {torch.cuda.device_count()} GPU(s)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    logger.info(f"Loading pre-tokenized token ID data from: {args.train_file}")
    samples = []
    with open(args.train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                token_ids = list(map(int, line.split()))
                samples.append({
                    "input_ids": token_ids,
                    "labels": token_ids.copy()
                })
            except ValueError:
                logger.warning(f"Skipping malformed line: {line}")

    train_ds = Dataset.from_list(samples)

    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        save_total_limit=2,
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    main()
