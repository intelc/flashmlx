# baselines/compare.py
"""Compare FlashMLX engine performance against mlx-lm baseline."""

import time
from dataclasses import dataclass

import mlx.core as mx

from engine import load_model, generate


@dataclass
class ComparisonResult:
    engine: str
    tok_s: float
    ttft_ms: float
    memory_mb: float


def benchmark_flashmlx(
    model_path: str,
    prompt_tokens: list[int],
    gen_tokens: int = 256,
) -> ComparisonResult:
    """Benchmark FlashMLX engine."""
    model, _ = load_model(model_path)
    prompt = mx.array(prompt_tokens)

    # Warmup
    for _ in generate(model, prompt, max_tokens=10, temperature=0.0):
        pass

    # Benchmark
    gen_iter = generate(model, prompt, max_tokens=gen_tokens, temperature=0.0)
    t_start = time.perf_counter()
    first = next(gen_iter)
    t_first = time.perf_counter()
    tokens = [first]
    for t in gen_iter:
        tokens.append(t)
    t_end = time.perf_counter()

    memory_mb = mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx, "metal") else 0

    return ComparisonResult(
        engine="flashmlx",
        tok_s=len(tokens) / (t_end - t_start),
        ttft_ms=(t_first - t_start) * 1000,
        memory_mb=memory_mb,
    )


def benchmark_mlx_lm(
    model_path: str,
    prompt_tokens: list[int],
    gen_tokens: int = 256,
) -> ComparisonResult | None:
    """Benchmark mlx-lm for comparison. Returns None if mlx-lm is not installed."""
    try:
        import mlx_lm
    except ImportError:
        return None

    model, tokenizer = mlx_lm.load(model_path)

    # Warmup
    prompt_text = tokenizer.decode(prompt_tokens)
    mlx_lm.generate(model, tokenizer, prompt=prompt_text, max_tokens=10, verbose=False)

    # Benchmark
    t_start = time.perf_counter()
    output = mlx_lm.generate(model, tokenizer, prompt=prompt_text, max_tokens=gen_tokens, verbose=False)
    t_end = time.perf_counter()

    memory_mb = mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx, "metal") else 0
    num_tokens = len(tokenizer.encode(output)) - len(prompt_tokens)

    return ComparisonResult(
        engine="mlx-lm",
        tok_s=num_tokens / (t_end - t_start),
        ttft_ms=0,  # mlx-lm doesn't expose TTFT easily in non-streaming mode
        memory_mb=memory_mb,
    )


def compare(model_path: str, prompt_tokens: list[int], gen_tokens: int = 256):
    """Run comparison and print results."""
    print(f"Benchmarking {model_path}...")
    print(f"  Prompt: {len(prompt_tokens)} tokens, Generate: {gen_tokens} tokens\n")

    flash_result = benchmark_flashmlx(model_path, prompt_tokens, gen_tokens)
    print(f"  FlashMLX:  {flash_result.tok_s:.1f} tok/s, TTFT: {flash_result.ttft_ms:.1f}ms, "
          f"Memory: {flash_result.memory_mb:.0f}MB")

    mlx_result = benchmark_mlx_lm(model_path, prompt_tokens, gen_tokens)
    if mlx_result:
        print(f"  mlx-lm:    {mlx_result.tok_s:.1f} tok/s, Memory: {mlx_result.memory_mb:.0f}MB")
        speedup = flash_result.tok_s / mlx_result.tok_s if mlx_result.tok_s > 0 else 0
        print(f"\n  Speedup: {speedup:.2f}x")
    else:
        print("  mlx-lm: not installed (pip install mlx-lm to compare)")


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-0.6B-4bit"
    compare(model, list(range(1, 129)))
