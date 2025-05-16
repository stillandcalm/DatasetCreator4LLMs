# scripts1/filter_data.py
#!/usr/bin/env python3
import argparse
import multiprocessing
from langdetect import detect

def filter_doc(args):
    doc, min_len, keywords = args
    if len(doc.split()) < min_len:
        return None
    try:
        if detect(doc) != 'en':
            return None
    except:
        return None
    low = doc.lower()
    if any(k in low for k in keywords):
        return doc
    return None

def main(input_file, output_file, keywords_file, min_len, workers, part_id, num_parts):
    keywords = [w.strip().lower() for w in open(keywords_file) if w.strip()]

    # Read extracted docs
    docs = []
    buf = []
    with open(input_file, encoding='utf-8') as fi:
        for line in fi:
            if line.strip() == '---ENDDOC---':
                docs.append('\n'.join(buf).strip())
                buf = []
            else:
                buf.append(line)
    if buf:
        docs.append('\n'.join(buf).strip())

    # Partition
    docs = [doc for i, doc in enumerate(docs) if i % num_parts == part_id]

    # Parallel filter
    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(filter_doc, [(doc, min_len, keywords) for doc in docs])

    # Write out
    with open(output_file, 'w', encoding='utf-8') as fo:
        for doc in results:
            if doc:
                fo.write(doc + '\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',    required=True)
    p.add_argument('--output',   required=True)
    p.add_argument('--keywords', required=True)
    p.add_argument('--min-len',  type=int, default=100)
    p.add_argument('--workers',  type=int, default=multiprocessing.cpu_count())
    p.add_argument('--part-id',  type=int, required=True)
    p.add_argument('--num-parts',type=int, required=True)
    args = p.parse_args()
    main(args.input, args.output, args.keywords, args.min_len,
         args.workers, args.part_id, args.num_parts)

