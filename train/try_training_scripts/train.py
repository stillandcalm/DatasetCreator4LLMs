#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


class SeqDataset(Dataset):
    """
    A simple Dataset that reads space-separated token IDs from .seq files
    and returns them as input_ids + labels.
    """
    def __init__(self, seq_dir, prefix="train_thread"):
        self.lines = []
        pattern = os.path.join(seq_dir, f"{prefix}*.seq")
        for path in sorted(glob.glob(pattern)):
            with open(path, 'r', encoding='utf-8') as f:
                for l in f:
                    toks = [int(tok) for tok in l.strip().split() if tok]
                    if toks:
                        self.lines.append(toks)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        ids = torch.tensor(self.lines[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids}


def main():
    parser = argparse.ArgumentParser(
        description="Finetune LLaMA-3 on token-sequence data with DeepSpeed"
    )
    parser.add_argument(
        "--model_name_or_path", required=True,
        help="Hugging Face model ID or local path"
    )
    parser.add_argument(
        "--train_seq_dir", required=True,
        help="Directory containing train_thread*.seq files"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Where to write checkpoints and final model"
    )
    parser.add_argument(
        "--deepspeed", dest="deepspeed_config", required=True,
        help="Path to DeepSpeed config JSON"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Training on {torch.cuda.device_count()} GPUs")

    # 1) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # 2) Model
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    # 3) Dataset
    train_ds = SeqDataset(args.train_seq_dir)

    # 4) Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        deepspeed=args.deepspeed_config,
        remove_unused_columns=True,
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7) Train
    trainer.train()

    # 8) Save final model and tokenizer
    logger.info("Saving final model and tokenizer to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

