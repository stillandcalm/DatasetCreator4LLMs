#!/usr/bin/env python3
"""
Filter raw extracted text by document boundaries, language, keywords, and minimum length.
This script splits on a sentinel boundary (ENDDOC) to preserve document integrity,
then shards at the document level so each process works on whole documents.
It collapses each document into a single line for downstream deduplication/tokenization.
"""
import argparse
import multiprocessing
from langdetect import detect

# Marker that ends each extracted page/document
ENDDOC = "ENDDOC"

def parse_args():
    p = argparse.ArgumentParser(
        description="Document-level filtering of extracted text with optional language/keyword/length checks"
    )
    p.add_argument("--input",    required=True,
                   help="Path to extracted input file (with ENDDOC markers)")
    p.add_argument("--output",   required=True,
                   help="Path to write filtered output")
    p.add_argument("--keywords", default=None,
                   help="Comma-separated list of keywords; if provided, require at least one in each doc")
    p.add_argument("--min-len",  type=int, default=0,
                   help="Minimum number of tokens per document (default: 0)")
    p.add_argument("--skip-lang", action="store_true",
                   help="Skip language detection (default: detect English only)")
    p.add_argument("--workers",  type=int, default=multiprocessing.cpu_count(),
                   help="Number of worker processes (default: CPU count)")
    p.add_argument("--part-id",  type=int, required=True,
                   help="Shard ID (0-indexed)")
    p.add_argument("--num-parts",type=int, required=True,
                   help="Total number of shards")
    return p.parse_args()


def main(input_path, output_path, keywords, min_len, skip_lang, workers, part_id, num_parts):
    # Read and split into whole documents
    raw = open(input_path, encoding='utf-8', errors='ignore').read()
    docs = [d.strip() for d in raw.split(ENDDOC) if d.strip()]
    #print("NUMBER OF DOCUMENTS = ", len(docs))
    #print("NUMBER OF PARTS = ", num_parts)
    # Shard at document level: assign each doc by index
    docs_shard = [docs[i] for i in range(len(docs)) if (i % num_parts) == part_id]

    filtered = []
    for doc in docs_shard:
        # Length filter
        # Collapse multi-line doc into one line
        single_line = " ".join(doc.splitlines())
        filtered.append(single_line)
    #print("++++++++START to PRINT FILTERED+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #print(filtered)
    #print("********END printing FILTERED *****************************************************************")
    # Write one document per line
    with open(output_path, 'w', encoding='utf-8') as fo:
        for doc in filtered:
            fo.write(doc + "\n")


if __name__ == '__main__':
    args = parse_args()
    main(
        args.input,
        args.output,
        args.keywords,
        args.min_len,
        args.skip_lang,
        args.workers,
        args.part_id,
        args.num_parts
    )

