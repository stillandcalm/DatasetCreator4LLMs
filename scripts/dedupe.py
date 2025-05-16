#!/usr/bin/env python3
import argparse
import hashlib

def main(input_file, output_file, part_id, num_parts):
    """
    Fast exact duplicate removal by SHA-256 hashing per line.
    """
    seen = set()
    with open(input_file, encoding='utf-8') as fi, \
         open(output_file, 'w', encoding='utf-8') as fo:
        for idx, line in enumerate(fi):
            # partition by line index
            if idx % num_parts != part_id:
                continue
            doc = line.strip()
            if not doc:
                continue
            # compute hash
            h = hashlib.sha256(doc.encode('utf-8')).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            fo.write(doc + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fast dedupe via hash")
    parser.add_argument('--input',     required=True, help='Filtered input text file')
    parser.add_argument('--output',    required=True, help='Output deduped text file')
    parser.add_argument('--part-id',   type=int, required=True, help='Shard ID')
    parser.add_argument('--num-parts', type=int, required=True, help='Total shards')
    args = parser.parse_args()
    main(args.input, args.output, args.part_id, args.num_parts)

