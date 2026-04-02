#!/bin/bash
# Autoresearch: C=8, Qwen3-0.6B-4bit, 1024 tokens, 2-run median
set -e
cd /Users/yihengchen/codestuff/aiexperiments/flashmlx
cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -j$(sysctl -n hw.ncpu) _flashmlx_engine > /dev/null 2>&1
cp _flashmlx_engine*.so ../python/
cd /Users/yihengchen/codestuff/aiexperiments/flashmlx
MODEL_PATH=~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/$(ls ~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/)
.venv/bin/python -m server.run "$MODEL_PATH" --port 8082 > /dev/null 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT
for i in $(seq 1 30); do curl -s http://localhost:8082/health 2>/dev/null | grep -q ok && break; sleep 1; done
python3.12 -c "
import json, time, urllib.request, concurrent.futures, statistics
URL = 'http://localhost:8082/v1/chat/completions'
PROMPT = 'Write a detailed tutorial on building a REST API with Python Flask. Cover routing, error handling, authentication, database integration, and deployment.'
def send(p, mt):
    data = json.dumps({'messages': [{'role': 'user', 'content': p}], 'max_tokens': mt, 'temperature': 0, 'stream': False}).encode()
    req = urllib.request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=600)
    result = json.loads(resp.read())
    return result.get('usage',{}).get('completion_tokens',0), time.perf_counter()-t0
send(PROMPT, 5)
runs = []
for _ in range(2):
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(8) as p:
        results = list(p.map(lambda _: send(PROMPT, 1024), range(8)))
    toks = sum(r[0] for r in results)
    runs.append(toks/(time.perf_counter()-t0))
print(f'{statistics.median(runs):.1f}')
" 2>/dev/null
