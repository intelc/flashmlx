# FlashMLX

Auto-research driven MLX inference engine for Apple Silicon.

FlashMLX is a modular inference engine that loads HuggingFace models in MLX format and runs them on Apple Silicon's Metal GPU. It includes an autobench harness that iteratively discovers faster configurations.

## Performance

Benchmarked on Apple M1 Max (64GB), generating 128 tokens. All engines use 4-bit quantized models.

### Qwen3-0.6B (small model)

| Engine | TTFT | Decode tok/s | E2E tok/s |
|--------|-----:|------------:|----------:|
| **FlashMLX** | 27.8ms | **249** | **248** |
| ollama 0.19 | **5.8ms** | 172.9 | 171.5 |
| LM Studio | 166.9ms | 192.9 | 153.1 |
| bodega | 136.5ms | 141.0 | 123.5 |

At small model sizes, FlashMLX dominates — **+45% e2e over ollama**, +62% over LM Studio.

### Meta-Llama-3-8B (medium model)

| Engine | TTFT | Decode tok/s | E2E tok/s |
|--------|-----:|------------:|----------:|
| **FlashMLX** | ~250ms | **63.1** | **58.5** |
| LM Studio | 171.6ms | 68.2 | 62.9 |
| bodega | 128.4ms | 67.0 | 63.3 |
| ollama 0.19 | **18.2ms** | 50.1 | 49.8 |

At 8B scale, FlashMLX's per-token decode (63.1 tok/s) matches LM Studio. E2E is slightly lower due to prefill overhead. ollama has the best TTFT. **Next: C++ batched inference server to push total throughput beyond single-request limits.**

### Key Optimizations

- **Correct float16 weight loading**: Fixed critical bug where HF weight keys were silently skipped, causing float32 fallback (+13% on 8B, +30% on 0.6B)
- **`mx.compile` on decode step**: Compiles full model forward pass with PreAllocKVCache for Metal kernel fusion (+7% on 8B)
- **N-step graph batching**: Builds up to 64 sequential forward passes before `mx.async_eval`, amortizing dispatch overhead (+24% on 0.6B)
- **Compiled fused ops**: `fused_residual_rms_norm` and `fused_rms_norm` reduce kernel dispatches
- **Async evaluation**: `mx.async_eval` overlaps Metal GPU compute with Python execution (+6%)
- **Concat-based KV cache**: Simpler `mx.concatenate` instead of pre-allocated slice assignment — better MLX graph optimization
- **Single-pass prefill**: Small prompts processed in one forward pass instead of chunked
- **Last-token-only lm_head**: Only computes logits for the final position during generation

## Quick Start

```bash
# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Generate text
python -c "
import mlx.core as mx
from engine import load_model, generate

model, tokenizer = load_model('mlx-community/Qwen3-0.6B-4bit')
prompt = tokenizer.encode('The meaning of life is')
for token in generate(model, mx.array(prompt), max_tokens=100):
    print(tokenizer.decode([token]), end='', flush=True)
print()
"
```

## Inference Server

FlashMLX includes a C++ inference server with an OpenAI-compatible API:

```bash
# Build the C++ extension
cd server && mkdir -p build && cd build
cmake .. && make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
cd ../..

# Start the server
python -m server.run mlx-community/Qwen3-0.6B-4bit --port 8080

# Query it
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":50,"stream":true}'

# Benchmark
python server/bench_server.py --url http://localhost:8080/v1/chat/completions
```

The server uses a C++ engine (pybind11) with MLX's C++ API for model loading and inference. The generation loop runs on a dedicated C++ thread with no Python GIL involvement.

## Architecture

```
engine/          # Core inference engine
  kv_cache.py    # KV cache with concat-based updates
  attention.py   # Scaled dot-product attention with GQA + causal mask
  quantize.py    # Quantization config parsing
  model_config.py # Model architecture (Llama family), loading, forward pass
  generate.py    # Generation loop with N-step graph batching
harness/         # Benchmark harness
  models.yaml    # Model registry (small/medium tiers)
  benchmark.py   # tok/s, TTFT, memory measurement
  validate.py    # Perplexity correctness gate
  prepare.py     # One-time model download and baseline computation
autobench/       # Autoresearch optimization loop
  program.md     # Research directives
  loop.py        # Propose -> commit -> benchmark -> keep/discard cycle
  analyze.py     # Post-hoc Pareto analysis
baselines/       # Comparison against other engines
  compare.py     # FlashMLX vs mlx-lm comparison
  benchmark-*.json # Saved benchmark results
```

## Supported Models

Any model in the Llama architecture family (GQA + SwiGLU + RMSNorm):

- Llama 2/3
- Mistral
- Qwen 2/3
- Gemma 1/2
- Phi-3
- Cohere Command

Models must be in MLX safetensors format (e.g. from `mlx-community/` on HuggingFace).

## Tuning

The `generate()` function exposes `eval_batch_size` (default 16) which controls how many tokens are computed in the MLX graph before evaluating. Higher values increase throughput but add latency. Optimal value depends on model size and hardware:

```python
# Higher throughput, more latency per chunk
tokens = list(generate(model, prompt, eval_batch_size=32))

# Lower latency per token, slightly less throughput  
tokens = list(generate(model, prompt, eval_batch_size=4))
```

## Running Benchmarks

```bash
# FlashMLX only
python bench.py

# Compare against ollama, LM Studio, bodega
python bench_compare.py

# Run the autobench research loop
python -m harness.prepare        # download models, compute baselines
python -m autobench.loop         # start optimization loop
```

## Tests

```bash
python -m pytest tests/ -v
```

45 tests covering KV cache, attention, quantization, model config, generation, benchmarking, validation, and end-to-end integration.

## Tech Stack

Python 3.12+, MLX, huggingface_hub, transformers (tokenizer only)

## How It Was Built

FlashMLX was built using an autoresearch methodology: an autonomous iteration loop that proposes code changes, benchmarks them, and keeps improvements while reverting regressions. Over 25 iterations, the engine went from 190 tok/s (naive implementation) to 247 tok/s (+30%), with the N-step graph batching optimization providing the single largest improvement.
