#!/usr/bin/env bash
set -e

NUM_PARTS=8
THREADS=4

# 1) Crawl into raw_html_part*.txt
#for PART in $(seq 0 $((NUM_PARTS-1))); do
#  python3 scripts/crawl.py \
#    --seeds      seeds.txt \
#    --domains    domains.txt \
#    --output     data/raw_html_part${PART}.txt \
#    --threads    $THREADS \
#    --delay      1.0 \
#    --max-pages  100000 \
#    --part-id    ${PART} \
#    --num-parts  ${NUM_PARTS} &
#done
#wait

# 2) Extract text into extracted_part*.txt
for PART in $(seq 0 $((NUM_PARTS-1))); do
  python3 scripts/extract_text.py \
    --input      data/raw_html_part${PART}.txt \
    --output     data/extracted_part${PART}.txt \
    --workers    $THREADS \
    --part-id    ${PART} \
    --num-parts  ${NUM_PARTS} &
done
wait

# 3) Filter into filtered_part*.txt
for PART in $(seq 0 $((NUM_PARTS-1))); do
  python3 scripts/filter_data.py \
    --input      data/extracted_part${PART}.txt \
    --output     data/filtered_part${PART}.txt \
    --keywords   cyber_keywords.txt \
    --min-len    100 \
    --workers    $THREADS \
    --part-id    ${PART} \
    --num-parts  ${NUM_PARTS} &
done
wait

# 4) Deduplicate into deduped_part*.txt
for PART in $(seq 0 $((NUM_PARTS-1))); do
  python3 scripts/dedupe.py \
    --input      data/filtered_part${PART}.txt \
    --output     data/deduped_part${PART}.txt \
    --threshold  0.8 \
    --workers    $THREADS \
    --part-id    ${PART} \
    --num-parts  ${NUM_PARTS} &
done
wait

# 5) Scrub PII into scrubbed_part*.txt
for PART in $(seq 0 $((NUM_PARTS-1))); do
  python3 scripts/scrub_pii.py \
    --input      data/deduped_part${PART}.txt \
    --output     data/scrubbed_part${PART}.txt \
    --workers    $THREADS \
    --part-id    ${PART} \
    --num-parts  ${NUM_PARTS} &
done
wait

# 6) Tokenize & pack into train_part*.seq / test_part*.seq
for PART in $(seq 0 $((NUM_PARTS-1))); do
  python3 scripts/tokenize_and_pack.py \
    --input      data/scrubbed_part${PART}.txt \
    --model      llama8b_tokenizer.model \
    --seq_len    4096 \
    --train_out  data/train_part${PART}.seq \
    --test_out   data/test_part${PART}.seq \
    --workers    $THREADS \
    --part-id    ${PART} \
    --num-parts  ${NUM_PARTS} &
done
wait

# 7) Count tokens per partition
echo "=== Token counts per partition ==="
for PART in $(seq 0 $((NUM_PARTS-1))); do
  CNT=$(python3 scripts/count_tokens.py \
    --input      data/train_part${PART}.seq \
    --workers    1 \
    --part-id    ${PART} \
    --num-parts  ${NUM_PARTS})
  echo "Part ${PART}: ${CNT}"
done

# (Optional) Sum them up:
TOTAL=0
for PART in $(seq 0 $((NUM_PARTS-1))); do
  CNT=$(python3 scripts/count_tokens.py \
    --input      data/train_part${PART}.seq \
    --workers    1 \
    --part-id    ${PART} \
    --num-parts  ${NUM_PARTS})
  TOTAL=$((TOTAL + CNT))
done
echo "Total tokens across all partitions: $TOTAL"

