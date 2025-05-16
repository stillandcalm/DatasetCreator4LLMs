#!/usr/bin/env python3
# scripts1/crawl_async.py

import argparse
import asyncio
import aiohttp
import aiofiles
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from urllib.parse import urljoin
from zlib import crc32

# Silence the XML-as-HTML warning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def should_handle(url: str, part_id: int, num_parts: int) -> bool:
    """CRC32-based sharding for *discovered* URLs only."""
    h = crc32(url.encode('utf-8')) & 0xFFFFFFFF
    return (h % num_parts) == part_id

async def fetch(session: aiohttp.ClientSession, url: str, timeout: int):
    """Fetch page text or return None on error."""
    try:
        async with session.get(url, timeout=timeout) as resp:
            return await resp.text()
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return None

async def worker(queue: asyncio.Queue,
                 visited: set,
                 writer,
                 session: aiohttp.ClientSession,
                 domains: list,
                 args,
                 seed_set: set):
    """Async worker: fetch, write, enqueue new links."""
    while True:
        url = await queue.get()
        if url is None:
            queue.task_done()
            break

        # skip if already seen
        if url in visited:
            queue.task_done()
            continue

        # only enforce partitioning for *non-seed* URLs
        if url not in seed_set and not should_handle(url, args.part_id, args.num_parts):
            queue.task_done()
            continue

        visited.add(url)
        html = await fetch(session, url, args.timeout)
        if html:
            # write out like crawl.py: URL + <html> + ---ENDDOC---
            await writer.write(f"URL: {url}\n<html>\n{html}\n</html>\n---ENDDOC---\n")
            # parse and enqueue links
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                nxt = urljoin(url, a["href"])
                if any(nxt.startswith(d) for d in domains) and nxt not in visited:
                    await queue.put(nxt)

        await asyncio.sleep(args.delay)
        queue.task_done()

async def crawl(args):
    # load seeds & domains
    seeds = [u.strip() for u in open(args.seeds) if u.strip()]
    seed_set = set(seeds)
    domains = [d.strip() for d in open(args.domains) if d.strip()]

    queue = asyncio.Queue()
    for u in seeds:
        await queue.put(u)

    visited = set()

    async with aiofiles.open(args.output, "w", encoding="utf-8") as writer, \
               aiohttp.ClientSession() as session:

        # start workers
        tasks = [
            asyncio.create_task(
                worker(queue, visited, writer, session, domains, args, seed_set)
            )
            for _ in range(args.concurrency)
        ]

        # run until max_pages or queue empty
        while len(visited) < args.max_pages and not queue.empty():
            await asyncio.sleep(1)

        # shutdown
        for _ in tasks:
            await queue.put(None)
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Partitioned async crawler")
    p.add_argument("--seeds",       required=True, help="Seed URLs file")
    p.add_argument("--domains",     required=True, help="Allowed domains file")
    p.add_argument("--output",      required=True, help="Shard output file")
    p.add_argument("--max-pages",   type=int,   default=100_000, help="Max pages per shard")
    p.add_argument("--delay",       type=float, default=1.0,    help="Delay between fetches")
    p.add_argument("--concurrency", type=int,   default=10,     help="Async tasks per shard")
    p.add_argument("--timeout",     type=int,   default=15,     help="Request timeout")
    p.add_argument("--part-id",     type=int,   required=True,  help="Shard ID (0-based)")
    p.add_argument("--num-parts",   type=int,   required=True,  help="Total number of shards")
    args = p.parse_args()

    asyncio.run(crawl(args))

