# harness/benchmark.py
import time
import statistics
from dataclasses import dataclass

import mlx.core as mx

from engine import load_model, generate


@dataclass
class BenchmarkResult:
    tok_s: float         # Tokens per second (generation)
    ttft_ms: float       # Time to first token in milliseconds
    memory_mb: float     # Peak Metal memory in MB
    tokens_generated: int


def run_benchmark(
    model_path: str,
    prompt_tokens: list[int],
    gen_tokens: int = 256,
    num_runs: int = 3,
    warmup_tokens: int = 20,
    temperature: float = 0.0,
) -> BenchmarkResult:
    """Run the benchmark protocol on a model.

    Args:
        model_path: Path to model directory or HF repo
        prompt_tokens: List of prompt token IDs
        gen_tokens: Number of tokens to generate per run
        num_runs: Number of runs (takes median)
        warmup_tokens: Tokens to generate for warmup (discarded)
        temperature: Sampling temperature

    Returns:
        BenchmarkResult with median metrics across runs
    """
    model, _ = load_model(model_path)
    prompt = mx.array(prompt_tokens)

    # Warmup: prime Metal shader caches
    if warmup_tokens > 0:
        for _ in generate(model, prompt, max_tokens=warmup_tokens, temperature=temperature):
            pass

    tok_s_runs = []
    ttft_runs = []

    for _ in range(num_runs):
        gen_iter = generate(model, prompt, max_tokens=gen_tokens, temperature=temperature)
        tokens_out = []

        # Measure TTFT
        t_start = time.perf_counter()
        first_token = next(gen_iter)
        t_first = time.perf_counter()
        tokens_out.append(first_token)

        # Measure generation throughput
        for token in gen_iter:
            tokens_out.append(token)
        t_end = time.perf_counter()

        ttft_ms = (t_first - t_start) * 1000
        total_time = t_end - t_start
        tok_s = len(tokens_out) / total_time if total_time > 0 else 0

        tok_s_runs.append(tok_s)
        ttft_runs.append(ttft_ms)

    # Peak memory
    memory_mb = mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx, "metal") else 0

    return BenchmarkResult(
        tok_s=statistics.median(tok_s_runs),
        ttft_ms=statistics.median(ttft_runs),
        memory_mb=memory_mb,
        tokens_generated=gen_tokens,
    )
