# FlashMLX

Auto-research driven MLX inference engine for Apple Silicon.

FlashMLX is a modular inference engine that loads HuggingFace models in MLX format and runs them on Apple Silicon's Metal GPU. It includes an autobench harness that iteratively discovers faster configurations.

## Performance

Benchmarked on Apple M1 Max (64GB), 4-bit quantized models. FlashMLX includes both a Python engine and a C++ inference server.

### Qwen3-0.6B — Server Throughput (tok/s)

| Engine | C=1 | C=4 | C=8 |
|--------|----:|----:|----:|
| **FlashMLX C++ Server** | **232** | 218 | **233** |
| Python FlashMLX (seq) | — | 259 | — |
| LM Studio | 227 | **326** | — |
| ollama 0.19 | 180 | 199 | — |
| bodega | 124 | 105 | — |

FlashMLX C++ server beats ollama by 29% and matches LM Studio at C=1. LM Studio leads at C=4 due to continuous batching with heterogeneous offsets.

### Qwen3-8B — Server Throughput (tok/s)

| Engine | C=1 | C=4 |
|--------|----:|----:|
| **FlashMLX C++ Server** | **52** | **53** |
| Python FlashMLX (seq) | 52 | — |
| ollama | 41 | — |

FlashMLX matches its Python engine and beats ollama by 28% on Qwen3-8B.

### Meta-Llama-3-8B — Server Throughput (tok/s)

| Engine | C=1 | C=4 |
|--------|----:|----:|
| LM Studio | 59 | **83** |
| Python FlashMLX (seq) | — | 60 |
| ollama | **58** | 63 |
| FlashMLX C++ Server | 43 | 59 |

At 8B with non-quantized embeddings, the C++ server has higher graph overhead. Scales to match Python at C=4.

### Qwen1.5-MoE-A2.7B — MoE Server Throughput (tok/s)

| Engine | C=1 | C=4 |
|--------|----:|----:|
| **FlashMLX C++ Server** | **103** | **135** |
| mlx-lm (sequential) | 106 | — |

First MoE model supported! 60 experts, top-4 routing, 2.7B active parameters. FlashMLX uses `mx::gather_qmm` for efficient indexed expert execution.

### Key Optimizations

**Python Engine:**
- **Correct float16 weight loading**: Fixed critical bug where HF weight keys were silently skipped (+13% on 8B, +30% on 0.6B)
- **`mx.compile` on decode step**: Metal kernel fusion with PreAllocKVCache (+7%)
- **N-step graph batching**: Up to 64 sequential forward passes before `mx.async_eval` (+24% on 0.6B)
- **Compiled fused ops** / **async evaluation** / **concat KV cache**

**C++ Server (from 60 → 232 tok/s = +287%):**
- **bfloat16 KV cache**: dtype mismatch fix eliminated conversion kernels (+56%)
- **Pre-dequantized embedding**: Was dequantizing 155M elements per token (+70%)
- **Quantized lm_head**: `quantized_matmul` for tied embeddings (4× less bandwidth)
- **Batched forward pass**: Same-offset requests processed in single GPU call
- **N=32 step graph batching** + **int-offset forward** (no mx::eval sync)
- **Weight reference caching**: Pre-resolved all array refs at construction

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
