# scripts1/count_tokens.py
#!/usr/bin/env python3
import argparse
import multiprocessing

def count_line(line):
    return len(line.split())

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',     required=True)
    p.add_argument('--workers',   type=int, default=1)
    p.add_argument('--part-id',   type=int, required=True)
    p.add_argument('--num-parts', type=int, required=True)
    args = p.parse_args()

    lines = open(args.input, encoding='utf-8').read().splitlines()
    lines = [l for i, l in enumerate(lines) if i % args.num_parts == args.part_id]

    with multiprocessing.Pool(processes=args.workers) as pool:
        counts = pool.map(count_line, lines)

    print(sum(counts))

