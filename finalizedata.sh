#!/usr/bin/env bash
set -euo pipefail

# —————————————————————————————————————
# CONFIGURATION
THREADS=${THREADS:-1}
MAX_PAGES=${MAX_PAGES:-100000}
DELAY=${DELAY:-1.0}
SEEDS="seeds.txt"
DOMAINS="domains.txt"
DATA_DIR="data"
KEYWORDS="cyber_keywords.txt"
TOKENIZER_DIR="tokenizer"    # <— your local folder with tokenizer.json etc.
SEQ_LEN=64
# —————————————————————————————————————

mkdir -p "${DATA_DIR}"

echo "=== Preflight: install HTML-clean + extraction deps ==="
pip install --quiet lxml-html-clean trafilatura justext

print_counts() {
  local prefix=$1 label=$2
  echo
  echo "=== ${label} counts ==="
  printf -- " shard |   lines |    bytes\n"
  printf -- "------:|--------:|--------:\n"
  for SHARD in $(seq 0 $((THREADS-1))); do
    local f="${DATA_DIR}/${prefix}_thread${SHARD}.txt"
    if [[ -f $f ]]; then
      LINES=$(wc -l < "$f" | xargs)
      BYTES=$(wc -c < "$f" | xargs)
    else
      LINES=0; BYTES=0
    fi
    printf -- "%6d | %7d | %8d\n" "$SHARD" "$LINES" "$BYTES"
  done
}

echo
echo "=== 1) Threaded crawl (per-thread raw_html) ==="
python3 scripts/crawl_threaded.py \
  --seeds      "${SEEDS}" \
  --domains    "${DOMAINS}" \
  --output-dir "${DATA_DIR}" \
  --threads    "${THREADS}" \
  --max-pages  "${MAX_PAGES}" \
  --delay      "${DELAY}"

print_counts raw_html "raw_html"

echo
echo "=== 2) Parallel extract_text ==="
for SHARD in $(seq 0 $((THREADS-1))); do
  python3 scripts/extract_text.py \
    --input     "${DATA_DIR}/raw_html_thread${SHARD}.txt" \
    --output    "${DATA_DIR}/extracted_thread${SHARD}.txt" \
    --workers   "${THREADS}" \
    --part-id   "${SHARD}" \
    --num-parts "${THREADS}" &
done
wait

print_counts extracted "extracted"

echo
echo "=== 3) Parallel filter_data ==="
for SHARD in $(seq 0 $((THREADS-1))); do
# no keywords, no length filter, skip language → output ≡ input
  python3 scripts/filter_data.py \
    --input     data/extracted_thread${SHARD}.txt \
    --output    data/filtered_thread${SHARD}.txt \
    --skip-lang \
    --part-id   "${SHARD}" \
    --num-parts "${THREADS}" &
done
wait

print_counts filtered "filtered"

echo
echo "=== 4) Parallel dedupe (fast SHA256) ==="
for SHARD in $(seq 0 $((THREADS-1))); do
  python3 scripts/dedupe.py \
    --input     "${DATA_DIR}/filtered_thread${SHARD}.txt" \
    --output    "${DATA_DIR}/deduped_thread${SHARD}.txt" \
    --part-id   "${SHARD}" \
    --num-parts "${THREADS}" &
done
wait

print_counts deduped "deduped"

echo
echo "=== 5) Parallel scrub_pii ==="
for SHARD in $(seq 0 $((THREADS-1))); do
  python3 scripts/scrub_pii.py \
    --input     "${DATA_DIR}/deduped_thread${SHARD}.txt" \
    --output    "${DATA_DIR}/scrubbed_thread${SHARD}.txt" \
    --workers   "${THREADS}" \
    --part-id   "${SHARD}" \
    --num-parts "${THREADS}" &
done
wait

print_counts scrubbed "scrubbed"

echo
echo "=== 6) Parallel tokenize_and_pack ==="
for SHARD in $(seq 0 $((THREADS-1))); do
  python3 scripts/tokenize_and_pack.py \
    --input      "${DATA_DIR}/scrubbed_thread${SHARD}.txt" \
    --model      "${TOKENIZER_DIR}" \
    --seq_len    "${SEQ_LEN}" \
    --train_out  "${DATA_DIR}/train_thread${SHARD}.seq" \
    --test_out   "${DATA_DIR}/test_thread${SHARD}.seq" \
    --part-id    "${SHARD}" \
    --num-parts  "${THREADS}" &
done
wait

echo
echo "=== train/test sequence file sizes ==="
printf -- " type  | shard | lines | bytes\n"
printf -- "------:|------:|------:|------:\n"
for SHARD in $(seq 0 $((THREADS-1))); do
  for TYPE in train test; do
    F="${DATA_DIR}/${TYPE}_thread${SHARD}.seq"
    if [[ -f $F ]]; then
      LINES=$(wc -l < "$F" | xargs)
      BYTES=$(wc -c < "$F" | xargs)
    else
      LINES=0; BYTES=0
    fi
    printf -- "%6s | %5d | %5d | %5d\n" "$TYPE" "$SHARD" "$LINES" "$BYTES"
  done
done

echo
echo "=== 7) Count tokens per thread and total ==="
TOTAL=0
for SHARD in $(seq 0 $((THREADS-1))); do
  TOK=$(python3 scripts/count_tokens.py \
    --input     "${DATA_DIR}/train_thread${SHARD}.seq" \
    --part-id   "${SHARD}" \
    --num-parts "${THREADS}")
  echo " Thread ${SHARD}: ${TOK} tokens"
  TOTAL=$((TOTAL + TOK))
done
echo "-----------------------------------"
echo " Total tokens across all threads: ${TOTAL}"
