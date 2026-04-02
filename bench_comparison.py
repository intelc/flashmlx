"""Benchmark FlashMLX vs vllm-mlx (or any OpenAI-compatible server)."""
import json
import time
import argparse
import concurrent.futures
import urllib.request


def send_request(url, prompt, max_tokens, model=None):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if model:
        body["model"] = model
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    t1 = time.perf_counter()
    tokens = result.get("usage", {}).get("completion_tokens", 0)
    return {"tokens": tokens, "time_s": t1 - t0}


def bench_sequential(url, num_requests, max_tokens, prompt, model=None):
    # Warmup
    send_request(url, "Hi", 5, model)

    results = []
    t0 = time.perf_counter()
    for _ in range(num_requests):
        r = send_request(url, prompt, max_tokens, model)
        results.append(r)
    t1 = time.perf_counter()

    total_tokens = sum(r["tokens"] for r in results)
    return {
        "total_tokens": total_tokens,
        "wall_time_s": t1 - t0,
        "total_tok_s": total_tokens / (t1 - t0) if t1 > t0 else 0,
    }


def bench_concurrent(url, concurrency, max_tokens, prompt, model=None):
    # Warmup
    send_request(url, "Hi", 5, model)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_request, url, prompt, max_tokens, model) for _ in range(concurrency)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    t1 = time.perf_counter()

    total_tokens = sum(r["tokens"] for r in results)
    return {
        "total_tokens": total_tokens,
        "wall_time_s": t1 - t0,
        "total_tok_s": total_tokens / (t1 - t0) if t1 > t0 else 0,
        "avg_request_s": sum(r["time_s"] for r in results) / len(results),
    }


def run_bench(name, url, max_tokens, prompt, model=None):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  URL: {url}")
    print(f"{'=' * 60}")

    # Sequential (C=1)
    r = bench_sequential(url, 3, max_tokens, prompt, model)
    print(f"  Sequential (3 req):  {r['total_tok_s']:6.1f} tok/s  ({r['total_tokens']} tokens in {r['wall_time_s']:.1f}s)")

    # Concurrent
    for c in [2, 4, 8]:
        try:
            r = bench_concurrent(url, c, max_tokens, prompt, model)
            print(f"  Concurrent C={c}:      {r['total_tok_s']:6.1f} tok/s  ({r['total_tokens']} tokens in {r['wall_time_s']:.1f}s)")
        except Exception as e:
            print(f"  Concurrent C={c}:      FAILED ({e})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain how transformer attention works in 3 sentences.")
    parser.add_argument("--flashmlx-url", default="http://localhost:8080/v1/chat/completions")
    parser.add_argument("--flashmlx-only", action="store_true")
    parser.add_argument("--vllm-mlx-url", default="http://localhost:8090/v1/chat/completions")
    parser.add_argument("--vllm-mlx-only", action="store_true")
    parser.add_argument("--vllm-mlx-model", default="mlx-community/Qwen3-0.6B-4bit")
    args = parser.parse_args()

    print(f"Max tokens: {args.max_tokens}")
    print(f"Prompt: {args.prompt[:60]}...")

    if not args.vllm_mlx_only:
        try:
            run_bench("FlashMLX (C++ server)", args.flashmlx_url, args.max_tokens, args.prompt)
        except Exception as e:
            print(f"\nFlashMLX: UNAVAILABLE ({e})")

    if not args.flashmlx_only:
        try:
            run_bench("vllm-mlx (continuous batching)", args.vllm_mlx_url, args.max_tokens, args.prompt, model=args.vllm_mlx_model)
        except Exception as e:
            print(f"\nvllm-mlx: UNAVAILABLE ({e})")

    print()


if __name__ == "__main__":
    main()
