"""Benchmark FlashMLX server with concurrent requests."""
import asyncio
import json
import time
import argparse

# Use urllib for simplicity (no aiohttp dependency needed for sequential)
import urllib.request

def send_request(url, prompt, max_tokens):
    data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    t1 = time.perf_counter()
    tokens = result.get("usage", {}).get("completion_tokens", 0)
    return {"tokens": tokens, "time_s": t1 - t0}

def bench_sequential(url, num_requests, max_tokens, prompt):
    """Sequential requests — measures single-request throughput."""
    # Warmup
    send_request(url, "Hi", 5)

    results = []
    t0 = time.perf_counter()
    for i in range(num_requests):
        r = send_request(url, prompt, max_tokens)
        results.append(r)
    t1 = time.perf_counter()

    total_tokens = sum(r["tokens"] for r in results)
    return {
        "mode": "sequential",
        "requests": num_requests,
        "total_tokens": total_tokens,
        "wall_time_s": t1 - t0,
        "total_tok_s": total_tokens / (t1 - t0),
        "avg_request_s": sum(r["time_s"] for r in results) / len(results),
    }

def bench_concurrent(url, concurrency, max_tokens, prompt):
    """Concurrent requests using threads — measures batched throughput."""
    import concurrent.futures

    # Warmup
    send_request(url, "Hi", 5)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_request, url, prompt, max_tokens) for _ in range(concurrency)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    t1 = time.perf_counter()

    total_tokens = sum(r["tokens"] for r in results)
    return {
        "mode": f"concurrent-{concurrency}",
        "requests": concurrency,
        "total_tokens": total_tokens,
        "wall_time_s": t1 - t0,
        "total_tok_s": total_tokens / (t1 - t0),
        "avg_request_s": sum(r["time_s"] for r in results) / len(results),
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashMLX server")
    parser.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain how transformer attention works.")
    args = parser.parse_args()

    print("=" * 60)
    print("FlashMLX Server Benchmark")
    print(f"URL: {args.url}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 60)

    # Sequential
    r = bench_sequential(args.url, 3, args.max_tokens, args.prompt)
    print(f"\nSequential (3 requests):")
    print(f"  Total: {r['total_tok_s']:.1f} tok/s ({r['total_tokens']} tokens in {r['wall_time_s']:.1f}s)")

    # Concurrent
    for c in [2, 4, 8]:
        try:
            r = bench_concurrent(args.url, c, args.max_tokens, args.prompt)
            print(f"\nConcurrent-{c}:")
            print(f"  Total: {r['total_tok_s']:.1f} tok/s ({r['total_tokens']} tokens in {r['wall_time_s']:.1f}s)")
            print(f"  Per-request avg: {r['avg_request_s']:.2f}s")
        except Exception as e:
            print(f"\nConcurrent-{c}: FAILED ({e})")

if __name__ == "__main__":
    main()
