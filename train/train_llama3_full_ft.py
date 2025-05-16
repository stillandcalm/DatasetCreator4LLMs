# train_llama3_full_ft.py
#!/usr/bin/env python3
import argparse
import logging
import os
import torch
import re
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, LlamaForCausalLM

# DeepSpeed import guard
try:
    import deepspeed
    _has_deepspeed = True
except ImportError:
    _has_deepspeed = False

class SequenceStream(IterableDataset):
    """
    Streams prepacked token-ID sequences (one sequence per line) forever.
    """
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        def generator():
            while True:
                with open(self.path, 'r') as f:
                    for line in f:
                        ids = list(map(int, line.strip().split()))
                        yield torch.tensor(ids, dtype=torch.long)
        return generator()

PAD_ID = 0

def collate_fn(batch):
    """Pads a batch of variable-length ID sequences."""
    seqs = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    return {"input_ids": seqs, "labels": seqs}


def main():
    parser = argparse.ArgumentParser(description="Full fine-tune LLaMA-3-8B with DeepSpeed + infinite data")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_sequences", type=str, required=True,
                        help="Path to packed seq file (4096-token lines)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_train_tokens", type=int, default=300_000_000_000,
                        help="Total tokens to train on (e.g. 300B)")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    # DeepSpeed launcher will inject local_rank
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Using DeepSpeed: {_has_deepspeed}")

    # 1) Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    # 2) Data loader (infinite stream)
    ds = SequenceStream(args.train_sequences)
    loader = DataLoader(
        ds,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # 3) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 4) DeepSpeed initialization
    if _has_deepspeed:
        model, optimizer, _, _ = deepspeed.initialize(
            config=args.deepspeed_config,
            model=model,
            model_parameters=model.parameters(),
        )

    # 5) Training loop by token count
    model.train()
    tokens_processed = 0
    for step, batch in enumerate(loader, start=1):
        outputs = model(**batch)
        loss = outputs.loss
        if _has_deepspeed:
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # count tokens = batch_size * seq_len
        B, T = batch["input_ids"].size()
        tokens_processed += B * T

        if step % args.logging_steps == 0:
            logger.info(f"Step {step} | Tokens {tokens_processed:,}/{args.max_train_tokens:,} | Loss {loss.item():.4f}")

        if tokens_processed >= args.max_train_tokens:
            logger.info(f"Reached target tokens. Exiting training loop.")
            break

    # 6) Save checkpoint
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        if _has_deepspeed:
            model.save_checkpoint(args.output_dir)
        else:
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved model to {args.output_dir}")

if __name__ == "__main__":
    main()
