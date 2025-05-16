#!/usr/bin/env python3
import argparse
import math
from transformers import AutoTokenizer

def pack_sequences(docs, tokenizer, seq_len, out_path):
    """Tokenize a list of docs into non-overlapping fixed-length sequences."""
    buffer = []
    with open(out_path, "w", encoding="utf-8") as fo:
        for doc in docs:
            # encode without truncation; we'll chop in the buffer
            ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
            buffer.extend(ids)
            # emit every full seq_len
            while len(buffer) >= seq_len:
                seq, buffer = buffer[:seq_len], buffer[seq_len:]
                fo.write(" ".join(map(str, seq)) + "\n")

def main():
    p = argparse.ArgumentParser(
        description="Partitioned tokenize-and-pack using HF fast tokenizer"
    )
    p.add_argument("--input",      required=True,  help="One document per line")
    p.add_argument("--model",      required=True,  help="HF tokenizer dir (tokenizer.json etc.)")
    p.add_argument("--seq_len",    type=int, required=True,  help="Sequence length (e.g. 4096)")
    p.add_argument("--train_out",  required=True,  help="Where to write train.sequences")
    p.add_argument("--test_out",   required=True,  help="Where to write test.sequences")
    p.add_argument("--part-id",    type=int, required=True, help="This worker’s partition ID")
    p.add_argument("--num-parts",  type=int, required=True, help="Total number of partitions")
    p.add_argument("--train-frac", type=float, default=0.9,
                   help="Fraction of docs→train, rest→test (default 0.9)")
    args = p.parse_args()

    # 1) Load all docs, assign this worker its slice
    with open(args.input, encoding="utf-8") as fi:
        lines = [l.strip() for l in fi if l.strip()]
    docs = lines[args.part_id :: args.num_parts]

    # 2) Split into train/test by fraction
    split_idx = math.floor(len(docs) * args.train_frac)
    train_docs, test_docs = docs[:split_idx], docs[split_idx:]

    # 3) Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # ensure pad token is defined (not strictly needed for causal LM packing)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4) Pack and write
    pack_sequences(train_docs, tokenizer, args.seq_len, args.train_out)
    pack_sequences(test_docs,  tokenizer, args.seq_len, args.test_out)

if __name__ == "__main__":
    main()

