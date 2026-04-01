# Model-Specific Optimization Configs

Different model sizes have different bottlenecks and optimal configurations.

## Small Models (≤1B) — e.g., Qwen3-0.6B

**Bottleneck:** Python/C++ overhead dominates (model compute is tiny)
**Best config:** Maximum graph batching + concat KV + pre-dequantized embedding

| Parameter | Value | Why |
|-----------|-------|-----|
| KV cache | concat | Simpler graph → better MLX optimization (+14% over slice_update) |
| eval_batch_size | 16 (Python) / 32 (C++) | Amortize eval overhead, 16→32 gives diminishing returns |
| Embedding | Pre-dequantize at load | Avoids 155M element dequant per token (+70%) |
| lm_head (tied) | quantized_matmul | 4× less bandwidth than dequantized matmul |
| KV pool dtype | bfloat16 | Must match model weight dtype (was float16 → +56%) |
| mx.compile | Yes (Python) | +8% from graph fusion |
| Prefill | Single pass | No chunking needed at 128-2048 tokens |
| N-step graph batch | N=32 | Build 32 forward passes before eval |

**C++ server results:** 269 tok/s C=1, 360 tok/s C=4

## Medium Models (7-8B) — e.g., Llama-3-8B, Qwen3-8B

**Bottleneck:** Memory bandwidth for weight reads (4.5GB per token at 400GB/s)
**Best config:** TBD — currently investigating

| Parameter | Current | Notes |
|-----------|---------|-------|
| KV cache | concat (trimmed after prefill) | Marginal gain over slice_update at 8B |
| eval_batch_size | 32 (C++) | Graph too large at 64+ |
| Embedding | Pre-dequant if quantized (Qwen3) | Non-quantized embeds (Llama) unaffected |
| lm_head | quantized_matmul | Already optimal |
| KV pool dtype | bfloat16 | Critical — same as 0.6B |
| mx.compile | No effect at 8B | GPU-bound, compile doesn't help |
| N-step graph batch | N=32 | |

**Qwen3-8B C++ server:** 54 tok/s C=1, 59 tok/s C=8 (beats ollama 40-42)
**Llama-3-8B C++ server:** 45 tok/s C=1, 69 tok/s C=4 (beats ollama 63, behind LMS 83)

### Known 8B Issues
- C++ forward 22ms/tok vs Python 16ms/tok (Llama-3-8B) — 6ms GPU graph overhead
- Weight caching confirmed this is GPU-side, not CPU string-lookup overhead
- Concat KV helps marginally at 8B (21.3ms vs 22.1ms)
- The 6ms gap may require mx::compile in C++ or architectural changes

### Why Qwen3-8B is faster than Llama-3-8B on FlashMLX
- Qwen3's quantized embedding → pre-dequant optimization applies (+70% on embed)
- Llama-3's float16 embedding → no pre-dequant benefit
- Qwen3's tied lm_head uses quantized_matmul → less bandwidth
- Llama-3's separate lm_head is already quantized but model has different graph structure
