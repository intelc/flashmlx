"""Quick benchmark: measure total tok/s at concurrency=4 against the server."""
import concurrent.futures
import json
import time
import urllib.request

URL = "http://localhost:8080/v1/chat/completions"
CONCURRENCY = 4
MAX_TOKENS = 64
PROMPT = "Explain how transformer attention works."


def send_request(i):
    data = json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    t1 = time.perf_counter()
    return result.get("usage", {}).get("completion_tokens", 0), t1 - t0


# Warmup
send_request(-1)

# Concurrent benchmark
t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    futures = [pool.submit(send_request, i) for i in range(CONCURRENCY)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
t1 = time.perf_counter()

total_tokens = sum(r[0] for r in results)
tok_s = total_tokens / (t1 - t0)
print(f"{tok_s:.2f}")
