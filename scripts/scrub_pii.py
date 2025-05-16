# scripts1/scrub_pii.py
#!/usr/bin/env python3
import argparse
import multiprocessing
import re

def scrub(line):
    patterns = [
        (re.compile(r"\b\d{1,3}(?:-\d{1,3}){2},\b"), '<IP>'),
        (re.compile(r"[\w\.-]+@[\w\.-]+"), '<EMAIL>'),
        (re.compile(r"\+?\d[\d \-()]{7,}\d"), '<PHONE>'),
    ]
    for pat, rep in patterns:
        line = pat.sub(rep, line)
    return line

def main(input_file, output_file, workers, part_id, num_parts):
    lines = open(input_file, encoding='utf-8').read().splitlines()
    lines = [l for i, l in enumerate(lines) if i % num_parts == part_id]

    with multiprocessing.Pool(processes=workers) as pool:
        cleaned = pool.map(scrub, lines)

    with open(output_file, 'w', encoding='utf-8') as fo:
        for line in cleaned:
            fo.write(line + '\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',     required=True)
    p.add_argument('--output',    required=True)
    p.add_argument('--workers',   type=int, default=multiprocessing.cpu_count())
    p.add_argument('--part-id',   type=int, required=True)
    p.add_argument('--num-parts', type=int, required=True)
    args = p.parse_args()
    main(args.input, args.output, args.workers,
         args.part_id, args.num_parts)

