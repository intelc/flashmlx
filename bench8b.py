"""Benchmark script for 8B model autoresearch."""
import time
import mlx.core as mx
from engine import load_model, generate

MODEL_REPO = "mlx-community/Meta-Llama-3-8B-4bit"
PROMPT_TOKENS = list(range(1, 129))  # 128 tokens
GEN_TOKENS = 128
NUM_RUNS = 5
WARMUP_TOKENS = 20


def bench():
    model, _ = load_model(MODEL_REPO)
    prompt = mx.array(PROMPT_TOKENS)

    # Warmup
    for _ in generate(model, prompt, max_tokens=WARMUP_TOKENS, temperature=0.0, eval_batch_size=256, prealloc_cache=True):
        pass

    tok_s_runs = []
    for _ in range(NUM_RUNS):
        gen_iter = generate(model, prompt, max_tokens=GEN_TOKENS, temperature=0.0, eval_batch_size=256, prealloc_cache=True)
        t_start = time.perf_counter()
        tokens = [t for t in gen_iter]
        t_end = time.perf_counter()
        tok_s = len(tokens) / (t_end - t_start)
        tok_s_runs.append(tok_s)

    tok_s_runs.sort()
    median = tok_s_runs[len(tok_s_runs) // 2]
    print(f"{median:.2f}")


if __name__ == "__main__":
    bench()
