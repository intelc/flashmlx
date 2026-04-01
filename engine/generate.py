# engine/generate.py
from typing import Iterator

import mlx.core as mx

from engine.kv_cache import KVCache


def generate(
    model,
    prompt: mx.array,
    max_tokens: int = 256,
    temperature: float = 0.0,
    prefill_chunk_size: int = 2048,
    eval_batch_size: int = 16,
    prealloc_cache: bool = False,
) -> Iterator[int]:
    """Generate tokens autoregressively.

    Args:
        model: Model with __call__(input_ids, cache) -> logits
        prompt: 1D array of token IDs
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        prefill_chunk_size: Process prompt in chunks of this size
        eval_batch_size: Number of tokens to compute before evaluating (higher = faster but more latency)
        prealloc_cache: Use pre-allocated KV cache (better for large models 8B+)

    Yields:
        Token IDs one at a time
    """
    if max_tokens == 0:
        return

    # Set generous Metal limits to reduce allocation overhead
    mx.set_cache_limit(8 * 1024 * 1024 * 1024)  # 8GB cache
    mx.set_memory_limit(32 * 1024 * 1024 * 1024)  # 32GB total

    cache = model.make_cache(prealloc=prealloc_cache)

    # Prefill: process prompt in chunks
    prompt = prompt.reshape(1, -1)  # [1, seq_len]
    prompt_len = prompt.shape[1]

    if prompt_len <= prefill_chunk_size:
        # Small prompt — single forward pass
        logits = model(prompt, cache=cache)
        logits = logits[:, -1, :]  # [1, vocab_size]
    else:
        # Large prompt — chunked prefill
        processed = 0
        while prompt_len - processed > prefill_chunk_size:
            chunk = prompt[:, processed : processed + prefill_chunk_size]
            model(chunk, cache=cache)
            mx.eval([c.state for c in cache])
            processed += prefill_chunk_size
        # Process remaining tokens
        logits = model(prompt[:, processed:], cache=cache)
        logits = logits[:, -1, :]

    y = _sample(logits, temperature)
    mx.async_eval(y)

    if eval_batch_size <= 1:
        # Simple per-token loop — optimal for large models
        for _ in range(max_tokens):
            yield y.item()
            logits = model(y.reshape(1, 1), cache=cache)
            logits = logits[:, -1, :]
            y = _sample(logits, temperature)
            mx.async_eval(y)
    else:
        # N-step graph batching — optimal for small models
        BATCH_STEPS = eval_batch_size
        generated = 0
        while generated < max_tokens:
            yield y.item()
            generated += 1
            if generated >= max_tokens:
                break

            remaining = min(BATCH_STEPS, max_tokens - generated)
            tokens = []
            prev = y
            for _ in range(remaining):
                logits = model(prev.reshape(1, 1), cache=cache)
                logits = logits[:, -1, :]
                prev = _sample(logits, temperature)
                tokens.append(prev)

            mx.async_eval(*tokens)

            for t in tokens[:-1]:
                yield t.item()
                generated += 1

            y = tokens[-1]


def _sample(logits: mx.array, temperature: float) -> mx.array:
    """Sample a token from logits.

    Args:
        logits: [1, vocab_size]
        temperature: 0.0 for greedy, > 0 for sampling

    Returns:
        [1] array with sampled token ID
    """
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits * (1.0 / temperature))
