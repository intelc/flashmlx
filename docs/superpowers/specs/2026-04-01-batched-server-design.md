# FlashMLX Batched Inference Server — Design Spec

## Goal

Build an OpenAI-compatible inference server that batches concurrent requests into a single GPU forward pass, multiplying total throughput on Apple Silicon. Python HTTP layer + C++ extension for the hot path.

## Context

FlashMLX currently achieves 63.1 tok/s per-token decode for Meta-Llama-3-8B-4bit on M1 Max — matching LM Studio. But this is single-request only. The M1 Max reads 4.5GB of model weights per token regardless of batch size. With batching, one weight read serves N sequences simultaneously, approaching N× total throughput.

**Target**: 8 concurrent requests, ~4× total throughput over single-request (250+ total tok/s for 8B).

## Architecture

```
Clients ──HTTP──▶ Python (FastAPI/uvicorn)
                       │
                  submit_request(prompt, params)
                       │
                       ▼
              ┌─────────────────┐
              │  C++ Extension   │  ← pybind11 module: libflashmlx
              │  (libflashmlx)   │
              │                  │
              │  ┌────────────┐  │
              │  │ Scheduler  │  │  ← request queue, batch formation
              │  └─────┬──────┘  │
              │        │         │
              │  ┌─────▼──────┐  │
              │  │ BatchLoop  │  │  ← C++ thread, tight decode loop
              │  └─────┬──────┘  │
              │        │         │
              │  ┌─────▼──────┐  │
              │  │ KVCachePool│  │  ← pre-allocated slots, assign/free
              │  └─────┬──────┘  │
              │        │         │
              │        ▼         │
              │   MLX C++ API    │  ← model forward, Metal GPU
              └─────────────────┘
```

### Layer 1: Python HTTP (FastAPI)

Handles HTTP concerns only — not performance-critical.

- `POST /v1/chat/completions` — OpenAI-compatible chat endpoint
- `GET /health` — model status, active requests, memory usage
- Accepts streaming (`"stream": true`) and non-streaming requests
- Applies chat template (via `transformers.AutoTokenizer`) before submitting to C++ engine
- Each request gets a unique ID and a `asyncio.Queue` for receiving tokens back

**Request flow:**
1. FastAPI handler tokenizes the prompt
2. Calls `engine.submit_request(request_id, token_ids, max_tokens, temperature)`
3. For streaming: yields SSE chunks as tokens arrive on the queue
4. For non-streaming: collects all tokens, returns complete response

### Layer 2: C++ Extension (libflashmlx)

The performance-critical core. A pybind11 module exposing:

```cpp
class Engine {
public:
    Engine(const std::string& model_path, int max_batch_size, int max_context_len);

    // Submit a new request — thread-safe, called from Python
    void submit_request(
        const std::string& request_id,
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        float temperature
    );

    // Poll for completed tokens — returns {request_id: [token_ids]}
    // Called from Python's async loop
    std::map<std::string, std::vector<int>> poll_tokens();

    // Cancel a request
    void cancel_request(const std::string& request_id);

    // Stats
    EngineStats get_stats();
};
```

**Internal components:**

#### BatchScheduler

Manages the lifecycle of requests:

```
States: QUEUED → PREFILLING → DECODING → DONE

Each tick:
1. Take new requests from submit queue
2. Run prefill for new requests (can batch multiple prefills if same length)
3. Collect all DECODING requests into a batch
4. Run one batched forward pass: model([B, 1]) → [B, vocab_size]
5. Sample tokens per-request (respecting each request's temperature)
6. Push tokens to output queues
7. Remove DONE requests, free their KV slots
```

The scheduler runs on a dedicated C++ thread. No Python GIL involvement during the decode loop.

#### KVCachePool

Pre-allocates a fixed pool of KV cache memory at startup:

```
Pool size = max_batch_size × max_context_len × n_layers × 2 × n_kv_heads × head_dim × sizeof(float16)

For 8B model, batch=8, context=2048:
= 8 × 2048 × 32 × 2 × 8 × 128 × 2 bytes
= 4.29 GB
```

Each request is assigned a slot (index 0–7). The slot's KV buffers are used for that request's cache. When the request completes, the slot is returned to the free list.

This eliminates all allocation during generation — the cache buffers are pre-allocated contiguous memory regions that the model writes into via slice assignment.

#### Batched Forward Pass

The key optimization. Instead of calling the model N times for N requests:

```cpp
// Collect next tokens from all active requests
// input_ids shape: [B, 1] where B = number of active requests
auto input_ids = mx::concatenate(batch_tokens, /*axis=*/0);

// Single forward pass — reads model weights once, computes B sequences
auto logits = model(input_ids, batch_caches);  // [B, 1, vocab_size]

// Distribute logits back to each request
for (int i = 0; i < B; i++) {
    auto token = sample(logits[i], requests[i].temperature);
    requests[i].emit_token(token);
}
```

The model weights (4.5GB) are read once from unified memory regardless of B. The compute scales linearly with B but is small relative to the memory read. Net result: near-linear throughput scaling with batch size until compute-bound.

### Model Loading

The C++ extension loads the model using MLX's C++ API:

1. Read `config.json` to determine architecture and quantization
2. Build the model graph (same Llama architecture as current Python code)
3. Load safetensors weights via `mx::load()`
4. Apply quantization structure matching the weight format
5. Keep model in GPU memory for the lifetime of the server

For V1, the model architecture (Llama family) is hardcoded in C++. This avoids the complexity of a generic model registry.

### Tokenization

Stays in Python. The C++ engine works with token IDs only. The Python layer:
1. Loads `AutoTokenizer` at startup
2. Applies chat template per request
3. Encodes prompt → token IDs → submit to C++ engine
4. Decodes token IDs → text → SSE response

### Sampling

Each request has its own sampling parameters (temperature, top_p). Sampling happens in C++ after the batched forward pass:

- `temperature == 0`: `mx::argmax` (greedy)
- `temperature > 0`: `mx::random::categorical(logits / temperature)`

Top-p/top-k filtering can be added later.

## Build System

```
server/
├── CMakeLists.txt          # C++ build
├── src/
│   ├── engine.cpp          # Engine class, pybind11 bindings
│   ├── scheduler.cpp       # BatchScheduler
│   ├── kv_pool.cpp         # KVCachePool
│   ├── model.cpp           # Model loading + forward pass (Llama arch)
│   └── sampling.cpp        # Token sampling
├── include/
│   └── flashmlx/
│       ├── engine.h
│       ├── scheduler.h
│       ├── kv_pool.h
│       ├── model.h
│       └── sampling.h
├── python/
│   ├── server.py           # FastAPI app
│   └── __init__.py         # Python wrapper around C++ module
└── tests/
    ├── test_engine.cpp     # C++ unit tests
    └── test_server.py      # Python integration tests
```

**Dependencies:**
- MLX C++ library (already installed via pip — headers at `.venv/lib/python3.14/site-packages/mlx/include/`)
- pybind11 (for Python bindings)
- nlohmann/json (for config parsing, header-only)
- FastAPI + uvicorn (Python HTTP server)

**Build:** CMake builds `libflashmlx.so` (or `.dylib` on macOS). The Python module imports it via pybind11.

## API Specification

### POST /v1/chat/completions

**Request:**
```json
{
  "model": "mlx-community/Meta-Llama-3-8B-4bit",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "stream": true
}
```

**Streaming response** (SSE):
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: [DONE]
```

**Non-streaming response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}
}
```

### GET /health

```json
{
  "status": "ok",
  "model": "mlx-community/Meta-Llama-3-8B-4bit",
  "active_requests": 3,
  "total_requests": 1247,
  "avg_tok_s": 58.2,
  "batch_utilization": 0.375,
  "memory_mb": 5120
}
```

## Performance Targets

On Apple M1 Max (64GB), Meta-Llama-3-8B-4bit:

| Metric | Single-request (current) | Batched server (target) |
|--------|------------------------:|------------------------:|
| Decode tok/s per request | 63 | ~55 (slight overhead) |
| Total tok/s (batch=1) | 63 | ~60 |
| Total tok/s (batch=4) | — | ~200 |
| Total tok/s (batch=8) | — | ~350 |
| TTFT (single) | 250ms | <200ms |
| Max concurrent | 1 | 8 (design for 16) |

## What's NOT in V1

- Multi-model support (one model loaded at startup)
- Prompt caching / prefix sharing across requests
- Quantized KV cache (add in V2 — doubles max batch size)
- Speculative decoding
- Top-p / top-k sampling (greedy + temperature only)
- Authentication / rate limiting
- Request cancellation mid-generation
- Graceful shutdown with request draining
- Metrics endpoint (Prometheus)

## Testing Strategy

1. **C++ unit tests**: KVCachePool allocation/free, BatchScheduler state machine, sampling correctness
2. **Python integration tests**: Submit request → receive tokens, concurrent requests, streaming SSE
3. **Benchmark**: `bench_compare.py` updated to test the server endpoint alongside ollama/LM Studio/bodega
4. **Stress test**: 16 concurrent requests, verify no OOM, no deadlock, correct output per request

## Success Criteria

1. Server starts, loads model, responds to `/v1/chat/completions`
2. Handles 8 concurrent requests without OOM or deadlock
3. Total throughput at batch=4 exceeds 200 tok/s (4× single-request)
4. Per-request latency stays within 2× of single-request baseline
5. Output correctness: greedy decode matches single-request output for same prompt
