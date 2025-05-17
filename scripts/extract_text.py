# scripts1/extract_text.py
#!/usr/bin/env python3
import argparse
import multiprocessing
import trafilatura

def extract_html(doc):
    return trafilatura.extract(doc) or ''

def main(input_file, output_file, workers, part_id, num_parts):
    # Read raw HTML docs
    docs = []
    buf = []
    with open(input_file, encoding='utf-8') as fi:
        for line in fi:
            if line.startswith('URL:'):
                if buf:
                    docs.append(''.join(buf))
                buf = []
            else:
                buf.append(line)
    if buf:
        docs.append(''.join(buf))

    # Partition the docs by index
    docs = [doc for i, doc in enumerate(docs) if i % num_parts == part_id]
#    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PRINTING DOCS ", docs)
    # Parallel extract
    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(extract_html, docs)
    # Write out
    with open(output_file, 'w', encoding='utf-8') as fo:
        for text in results:
#            print(">>>>>>>>>>>>>>>>>>>>GOING TO WRITE text.strip ", text.strip())
            fo.write(text.strip() + '\n---ENDDOC---\n')
#            print(">>>>>>>>>>>>>>>>>>>>GOING TO WRITE done")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',    required=True)
    p.add_argument('--output',   required=True)
    p.add_argument('--workers',  type=int, default=multiprocessing.cpu_count())
    p.add_argument('--part-id',  type=int, required=True)
    p.add_argument('--num-parts',type=int, required=True)
    args = p.parse_args()
    main(args.input, args.output, args.workers, args.part_id, args.num_parts)

