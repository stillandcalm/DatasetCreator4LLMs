# scripts/crawl.py
#!/usr/bin/env python3
import argparse
import threading
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from queue import Queue

def worker(queue, visited, lock, domains, out, delay):
    while True:
        url = queue.get()
        if url is None:
            queue.task_done()
            break
        with lock:
            if url in visited:
                queue.task_done()
                continue
            visited.add(url)
        try:
            resp = requests.get(url, timeout=10)
            html = resp.text
            with lock:
                out.write(f"URL: {url}\n<html>\n{html}\n</html>\n---ENDDOC---\n")
            soup = BeautifulSoup(html, 'html.parser')
            for a in soup.find_all('a', href=True):
                new_url = urljoin(url, a['href'])
                if new_url.startswith(tuple(domains)):
                    with lock:
                        if new_url not in visited:
                            queue.put(new_url)
        except Exception as e:
            print(f"[WARN] {url}: {e}")
        time.sleep(delay)
        queue.task_done()

def crawl(seeds, domains_file, max_pages, output, delay, threads, part_id, num_parts):
    all_seeds = [u.strip() for u in open(seeds) if u.strip()]
    seeds = [u for i, u in enumerate(all_seeds) if i % num_parts == part_id]
    domains = [d.strip() for d in open(domains_file) if d.strip()]
    q = Queue()
    for u in seeds:
        q.put(u)
    visited = set()
    lock = threading.Lock()
    out = open(output, 'w', encoding='utf-8')
    workers = []
    for _ in range(threads):
        t = threading.Thread(target=worker, args=(q, visited, lock, domains, out, delay))
        t.daemon = True
        t.start()
        workers.append(t)
    q.join()
    for _ in workers:
        q.put(None)
    for t in workers:
        t.join()
    out.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', required=True)
    p.add_argument('--domains', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-pages', type=int, default=100000)
    p.add_argument('--delay', type=float, default=1.0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--part-id', type=int, required=True)
    p.add_argument('--num-parts', type=int, required=True)
    args = p.parse_args()
    crawl(args.seeds, args.domains, args.max_pages, args.output,
          args.delay, args.threads, args.part_id, args.num_parts)


