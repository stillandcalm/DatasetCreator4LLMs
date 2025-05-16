#!/usr/bin/env python3
import glob, argparse

def main():
    p = argparse.ArgumentParser(
        description="Re-shard a set of .seq files into N new shards balanced by token count"
    )
    p.add_argument(
        "--input-glob", type=str, default="data/train_thread*.seq",
        help="glob matching your per-thread train .seq files"
    )
    p.add_argument(
        "--output-prefix", type=str, default="data/train_balanced_",
        help="prefix for new shards (will append 0..N-1 + .seq)"
    )
    p.add_argument(
        "--num-shards", type=int, required=True,
        help="how many balanced output shards to create"
    )
    args = p.parse_args()

    # 1) load all sequences into memory
    seqs = []
    for fn in sorted(glob.glob(args.input_glob)):
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens: continue
                seqs.append((line.rstrip("\n"), len(tokens)))

    if not seqs:
        print("No sequences found with:", args.input_glob)
        return

    total_tokens = sum(length for _, length in seqs)
    target_per_shard = total_tokens / args.num_shards
    print(f"Total tokens: {total_tokens:,}; target ≃ {target_per_shard:,.0f} per shard")

    # 2) greedy assign
    shards = [[] for _ in range(args.num_shards)]
    shard_token_counts = [0] * args.num_shards
    shard_idx = 0
    for seq, length in seqs:
        # if this shard has already hit its target, move on
        if shard_idx < args.num_shards - 1 and shard_token_counts[shard_idx] >= target_per_shard:
            shard_idx += 1
        shards[shard_idx].append(seq)
        shard_token_counts[shard_idx] += length

    # 3) write out
    for i, (lines, tok_count) in enumerate(zip(shards, shard_token_counts)):
        out_fn = f"{args.output_prefix}{i}.seq"
        with open(out_fn, "w", encoding="utf-8") as fo:
            fo.write("\n".join(lines) + "\n")
        print(f"Shard {i}: {len(lines):6d} lines, {tok_count:8d} tokens → {out_fn}")

if __name__ == "__main__":
    main()

