# MLX Optimization Notes

Lessons learned from 25 autoresearch iterations optimizing FlashMLX on Apple M1 Max.

## What Worked

### N-step Graph Batching (+30% total, single biggest win)
Build N sequential model forward passes in the MLX computation graph before calling `mx.async_eval`. This lets Metal fuse kernels and pipeline execution across multiple decode steps. The MLX lazy evaluation model means constructing 16 sequential `model(token) -> sample -> model(token) -> ...` calls builds a single large graph that Metal can optimize holistically.

- N=2: +12% over baseline
- N=4: +16%
- N=8: +21%
- N=16: +30% (optimal)
- N=32: no further improvement (graph compilation overhead dominates)

**Why it works:** Each `mx.async_eval` call has fixed overhead (graph submission to Metal, synchronization). By batching 16 forward passes into one graph, we reduce this overhead by 16x while Metal optimizes the fused kernel pipeline.

### mx.async_eval (+6%)
Replace `mx.eval(y)` with `mx.async_eval(y)` in the decode loop. This submits the computation to Metal without blocking Python. The actual sync happens at `y.item()` when we need the value.

**Caveat:** Only works for the decode loop. Prefill *must* use synchronous `mx.eval` to materialize KV cache states before the next chunk uses them (iteration 7 confirmed: async prefill = 2x regression).

### Concat-based KV Cache (simplicity win)
`mx.concatenate([cached, new], axis=2)` outperforms pre-allocated buffers with slice assignment (`buffer[:, :, offset:offset+T, :] = new`) for the model sizes tested. The concat approach produces a simpler computation graph that MLX optimizes better.

### Single-pass Prefill
For prompts that fit within `prefill_chunk_size`, process the entire prompt in one model forward pass instead of splitting into chunks. Eliminates unnecessary intermediate eval/sync.

### Metal Cache Limit
`mx.set_cache_limit(4GB)` reduces Metal memory allocation overhead by allowing MLX to cache and reuse GPU memory buffers.

## What Didn't Work

### mx.compile on Decode Step (iteration 4: -64%)
Wrapping the decode step in `@mx.compile` caused severe regression. The KV cache mutates on every call (growing), which forces `mx.compile` to retrace the graph every invocation — worse than not compiling at all.

### Removing Cache Eval During Prefill (iteration 1: -53%)
The `mx.eval([c.state for c in cache])` after each prefill chunk is essential. Without it, the lazy evaluation graph explodes in size as subsequent chunks reference unevaluated cache states.

### Batching RoPE Application (iteration 10: -69%)
Concatenating Q and K along the head dimension, applying RoPE once, then splitting back was dramatically slower. The concat/split overhead far exceeds the cost of two separate RoPE calls.

### Pre-allocated KV Buffers (iteration 9: -6%)
Slice assignment into pre-allocated buffers was slower than simple concatenation. MLX's graph optimizer handles concat better than scatter/slice-write operations.

### Inlining the SDPA Wrapper (iteration 11: -74%)
Calling `mx.fast.scaled_dot_product_attention` directly instead of through our thin wrapper function caused an unexpected severe regression. Likely related to how MLX traces the computation graph differently when the call site changes.

### Python-level Micro-optimizations
Reshape avoidance (iteration 3), squeeze vs slice (iteration 20), inline sampling (iteration 22) — all within measurement noise. Python overhead is negligible compared to Metal compute for these model sizes.

## Measurement Notes

- Benchmark variance is ~5-10% run-to-run on M1 Max, even with nothing else running
- 5 runs with median gives stable results; 3 runs was too noisy
- 64 warmup tokens (vs 20) significantly reduces cold-start variance
- System load (other GPU/CPU workloads) can cause 2-3x measurement swings
- Always do a confirmation run before keeping/discarding marginal changes

## Comparison with Other Engines

On Qwen3-0.6B-4bit, 128 generated tokens (2026-04-01):

| Engine | TTFT | Decode tok/s | E2E tok/s |
|--------|-----:|------------:|----------:|
| FlashMLX | 27.8ms | 225.1 | 215.4 |
| ollama 0.19 | 5.8ms | 172.9 | 171.5 |
| LM Studio | 166.9ms | 192.9 | 153.1 |
| bodega | 136.5ms | 141.0 | 123.5 |

ollama's TTFT advantage comes from llama.cpp's highly optimized GGUF prompt processing. FlashMLX's throughput advantage comes primarily from N-step graph batching which no other engine currently implements.
