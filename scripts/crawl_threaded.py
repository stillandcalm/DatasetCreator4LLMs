# scripts1/crawl_threaded.py
#!/usr/bin/env python3
import argparse
import threading
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from queue import Queue

def worker(thread_id, queue, visited, visit_lock, domains, output_dir, delay, max_pages):
    out_path = f"{output_dir}/raw_html_thread{thread_id}.txt"
    with open(out_path, 'w', encoding='utf-8') as out:
        while True:
            url = queue.get()
            if url is None:
                queue.task_done()
                break

            # Check global visited set
            with visit_lock:
                if url in visited or len(visited) >= max_pages:
                    queue.task_done()
                    continue
                visited.add(url)

            try:
                resp = requests.get(url, timeout=10)
                html = resp.text
                # write to this thread's file
                out.write(f"URL: {url}\n<html>\n{html}\n</html>\n---ENDDOC---\n")

                # parse and enqueue new links
                soup = BeautifulSoup(html, 'lxml')
                for a in soup.find_all('a', href=True):
                    nxt = urljoin(url, a['href'])
                    if any(nxt.startswith(d) for d in domains):
                        with visit_lock:
                            if nxt not in visited:
                                queue.put(nxt)

            except Exception as e:
                print(f"[WARN] Thread {thread_id} failed to fetch {url}: {e}")

            time.sleep(delay)
            queue.task_done()


def main(seeds_file, domains_file, output_dir, threads, max_pages, delay):
    seeds = [u.strip() for u in open(seeds_file) if u.strip()]
    domains = [d.strip() for d in open(domains_file) if d.strip()]

    q = Queue()
    for u in seeds:
        q.put(u)

    visited = set()
    visit_lock = threading.Lock()

    # start worker threads
    workers = []
    for tid in range(threads):
        t = threading.Thread(target=worker,
                             args=(tid, q, visited, visit_lock,
                                   domains, output_dir, delay, max_pages))
        t.daemon = True
        t.start()
        workers.append(t)

    # wait for queue to drain or max_pages reached
    q.join()

    # stop workers
    for _ in workers:
        q.put(None)
    for t in workers:
        t.join()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Threaded web crawler with per-thread outputs")
    p.add_argument('--seeds',      required=True, help='File with seed URLs')
    p.add_argument('--domains',    required=True, help='File with allowed domain prefixes')
    p.add_argument('--output-dir', required=True, help='Directory to write raw_html_thread*.txt')
    p.add_argument('--threads',    type=int, default=8, help='Number of crawler threads')
    p.add_argument('--max-pages',  type=int,   default=100000, help='Maximum pages to crawl total')
    p.add_argument('--delay',      type=float, default=1.0, help='Delay between requests (sec)')
    args = p.parse_args()

    main(args.seeds, args.domains, args.output_dir,
         args.threads, args.max_pages, args.delay)

