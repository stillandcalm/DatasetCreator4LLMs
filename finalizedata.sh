#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─── CONFIG ────────────────────────────────────────────────────────────────────
NUM_SHARDS=8             # number of parallel pieces
WORKERS=4                # for CPU‐bound stages
MAX_PAGES=100000         # crawler limit
DELAY=1.0                # politeness
CONCURRENCY=20           # asyncio fetch concurrency

SEEDS=seeds.txt
DOMAINS=domains.txt
OUTDIR=data
KEYWORDS=cyber_keywords.txt

SPM_MODEL=tokenizer.model    # your SentencePiece model
SEQ_LEN=4096
TRAIN_FRAC=0.9               # per-shard train/test split

# ─── helper to print counts for a file‐glob ────────────────────────────────────
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
  echo
}

# ─── 1) crawl_threaded.py ─────────────────────────────────────────────────────
echo "=== 1) Threaded crawl (per-thread raw_html) ==="
python3 scripts/crawl_threaded.py \
  --seeds       "$SEEDS" \
  --domains     "$DOMAINS" \
  --output-dir  "$OUTDIR" \
  --threads     "$NUM_SHARDS" \
  --max-pages   "$MAX_PAGES" \
  --delay       "$DELAY" \
  --concurrency "$CONCURRENCY"
print_counts "raw_html" "raw_html_thread*.txt"

# ─── 2) extract_text.py ───────────────────────────────────────────────────────
echo "=== 2) Parallel extract_text ==="
for ((i=0;i<NUM_SHARDS;i++)); do
  python3 scripts/extract_text.py \
    --input     "${OUTDIR}/raw_html_thread${i}.txt" \
    --output    "${OUTDIR}/extracted_thread${i}.txt" \
    --workers   "$WORKERS" \
    --part-id   "$i" \
    --num-parts "$NUM_SHARDS" &
done
wait
print_counts "extracted" "extracted_thread*.txt"

# ─── 3) filter_data.py ────────────────────────────────────────────────────────
echo "=== 3) Parallel filter_data ==="
for ((i=0;i<NUM_SHARDS;i++)); do
  python3 scripts/filter_data.py \
    --input     "${OUTDIR}/extracted_thread${i}.txt" \
    --output    "${OUTDIR}/filtered_thread${i}.txt" \
    --keywords  "$KEYWORDS" \
    --min-len   100 \
    --workers   "$WORKERS" \
    --part-id   "$i" \
    --num-parts "$NUM_SHARDS" &
done
wait
print_counts "filtered" "filtered_thread*.txt"

# ─── 4) dedupe.py (sha256) ────────────────────────────────────────────────────
echo "=== 4) Parallel dedupe ==="
for ((i=0;i<NUM_SHARDS;i++)); do
  python3 scripts/dedupe.py \
    --input     "${OUTDIR}/filtered_thread${i}.txt" \
    --output    "${OUTDIR}/deduped_thread${i}.txt" \
    --part-id   "$i" \
    --num-parts "$NUM_SHARDS" &
done
wait
print_counts "deduped" "deduped_thread*.txt"

# ─── 5) scrub_pii.py ──────────────────────────────────────────────────────────
echo "=== 5) Parallel scrub_pii ==="
for ((i=0;i<NUM_SHARDS;i++)); do
  python3 scripts/scrub_pii.py \
    --input     "${OUTDIR}/deduped_thread${i}.txt" \
    --output    "${OUTDIR}/scrubbed_thread${i}.txt" \
    --workers   "$WORKERS" \
    --part-id   "$i" \
    --num-parts "$NUM_SHARDS" &
done
wait
print_counts "scrubbed" "scrubbed_thread*.txt"

# ─── 6) tokenize_and_pack.py ─────────────────────────────────────────────────
echo "=== 6) Parallel tokenize_and_pack ==="
for ((i=0;i<NUM_SHARDS;i++)); do
  python3 scripts/tokenize_and_pack.py \
    --input      "${OUTDIR}/scrubbed_thread${i}.txt" \
    --model      "$SPM_MODEL" \
    --seq_len    "$SEQ_LEN" \
    --train-out  "${OUTDIR}/train_thread${i}.seq" \
    --test-out   "${OUTDIR}/test_thread${i}.seq" \
    --part-id    "$i" \
    --num-parts  "$NUM_SHARDS" \
    --train-frac "$TRAIN_FRAC" &
done
wait
print_counts "train_sequences" "train_thread*.seq"
print_counts "test_sequences " "test_thread*.seq"

# ─── 7) count_tokens.py ───────────────────────────────────────────────────────
echo "=== 7) Count tokens per shard and total ==="
grand=0
for ((i=0;i<NUM_SHARDS;i++)); do
  cnt=$(python3 scripts/count_tokens.py \
    --input     "${OUTDIR}/train_thread${i}.seq" \
    --part-id   "$i" \
    --num-parts "$NUM_SHARDS")
  printf " Shard %2d: %6d tokens\n" "$i" "$cnt"
  (( grand += cnt ))
done
echo "---------------------------------"
echo " Grand total tokens: $grand"

echo
echo "=== 8) Re-shard into balanced token shards ==="
python3 scripts/reshard_by_tokens.py \
  --input-glob   "data/train_thread*.seq" \
  --output-prefix "data/train_balanced_" \
  --num-shards   "$NUM_SHARDS"

