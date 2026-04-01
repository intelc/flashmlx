"""Comprehensive benchmark: TTFT, decode-only, and end-to-end tok/s across engines."""
import json
import time
import urllib.request
import statistics
import mlx.core as mx
from engine import load_model, generate

PROMPT_TEXT = (
    "Explain how transformer attention works in detail with formulas. "
    "Cover query key value matrices and multi-head attention. "
    "Explain the computational complexity."
)
NUM_PREDICT = 128
NUM_RUNS = 5
WARMUP_RUNS = 1

# ─── FlashMLX ───────────────────────────────────────────────────────────────

def bench_flashmlx():
    model, tokenizer = load_model("mlx-community/Qwen3-0.6B-4bit")
    # Use the same token IDs as our standard bench
    prompt = mx.array(list(range(1, 129)))  # 128 tokens

    # Warmup
    for _ in generate(model, prompt, max_tokens=20, temperature=0.0):
        pass

    results = []
    for _ in range(NUM_RUNS):
        gen = generate(model, prompt, max_tokens=NUM_PREDICT, temperature=0.0)

        t_start = time.perf_counter()
        first_tok = next(gen)
        t_first = time.perf_counter()

        rest = [first_tok]
        for tok in gen:
            rest.append(tok)
        t_end = time.perf_counter()

        ttft_ms = (t_first - t_start) * 1000
        total_tokens = len(rest)
        decode_tokens = total_tokens - 1  # exclude first token
        decode_time = t_end - t_first
        e2e_time = t_end - t_start

        results.append({
            "ttft_ms": ttft_ms,
            "decode_tok_s": decode_tokens / decode_time if decode_time > 0 else 0,
            "e2e_tok_s": total_tokens / e2e_time if e2e_time > 0 else 0,
            "total_tokens": total_tokens,
        })
    return results


# ─── Ollama ─────────────────────────────────────────────────────────────────

def bench_ollama():
    # Warmup
    _ollama_call()

    results = []
    for _ in range(NUM_RUNS):
        data = _ollama_call()
        if not data:
            continue

        # ollama exposes granular timing
        prompt_eval_ns = data.get("prompt_eval_duration", 0)
        eval_ns = data.get("eval_duration", 0)
        eval_count = data.get("eval_count", 0)
        prompt_count = data.get("prompt_eval_count", 0)

        ttft_ms = prompt_eval_ns / 1e6  # prompt processing ≈ TTFT
        decode_tok_s = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0
        total_time_s = (prompt_eval_ns + eval_ns) / 1e9
        e2e_tok_s = eval_count / total_time_s if total_time_s > 0 else 0

        results.append({
            "ttft_ms": ttft_ms,
            "decode_tok_s": decode_tok_s,
            "e2e_tok_s": e2e_tok_s,
            "total_tokens": eval_count,
        })
    return results


def _ollama_call():
    data = json.dumps({
        "model": "qwen3:0.6b",
        "prompt": PROMPT_TEXT,
        "options": {"num_predict": NUM_PREDICT, "temperature": 0},
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data, headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return json.loads(resp.read())
    except Exception as e:
        print(f"  ollama error: {e}")
        return None


# ─── OpenAI-compatible (LM Studio / Bodega) ─────────────────────────────────

def bench_openai_compat(name, port, model_id):
    """Benchmark via streaming to measure TTFT, then compute decode and e2e."""
    # Warmup
    _openai_stream(port, model_id)

    results = []
    for _ in range(NUM_RUNS):
        r = _openai_stream(port, model_id)
        if r:
            results.append(r)
    return results


def _openai_stream(port, model_id):
    """Use streaming to measure TTFT accurately."""
    data = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT_TEXT}],
        "max_tokens": NUM_PREDICT,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"},
    )
    try:
        t_start = time.perf_counter()
        resp = urllib.request.urlopen(req, timeout=120)

        t_first = None
        token_count = 0

        for line in resp:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "") or delta.get("reasoning_content", "")
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    token_count += 1  # approximate: 1 chunk ≈ 1 token
            except json.JSONDecodeError:
                continue

        t_end = time.perf_counter()

        if t_first is None or token_count == 0:
            return None

        ttft_ms = (t_first - t_start) * 1000
        decode_time = t_end - t_first
        e2e_time = t_end - t_start

        return {
            "ttft_ms": ttft_ms,
            "decode_tok_s": (token_count - 1) / decode_time if decode_time > 0 and token_count > 1 else 0,
            "e2e_tok_s": token_count / e2e_time if e2e_time > 0 else 0,
            "total_tokens": token_count,
        }
    except Exception as e:
        print(f"  stream error: {e}")
        return None


# ─── Report ─────────────────────────────────────────────────────────────────

def median(values):
    s = sorted(values)
    return s[len(s) // 2]


def report(name, results):
    if not results:
        print(f"| {name:<20} | {'FAILED':>10} | {'—':>10} | {'—':>10} | {'—':>6} |")
        return

    ttfts = [r["ttft_ms"] for r in results]
    decodes = [r["decode_tok_s"] for r in results]
    e2es = [r["e2e_tok_s"] for r in results]
    tokens = [r["total_tokens"] for r in results]

    m_ttft = median(ttfts)
    m_decode = median(decodes)
    m_e2e = median(e2es)
    m_tokens = median(tokens)

    print(f"| {name:<20} | {m_ttft:>9.1f}ms | {m_decode:>9.1f} | {m_e2e:>9.1f} | {m_tokens:>6} |")


if __name__ == "__main__":
    print("=" * 78)
    print("  Inference Engine Comparison — Qwen3-0.6B, 128 generated tokens")
    print(f"  {NUM_RUNS} runs per engine, median reported")
    print("=" * 78)
    print()
    print(f"| {'Engine':<20} | {'TTFT':>10} | {'Decode':>9} | {'E2E':>9} | {'Tokens':>6} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*11}|{'-'*11}|{'-'*8}|")

    # FlashMLX
    print("  Benchmarking FlashMLX...", end="", flush=True)
    r = bench_flashmlx()
    print("\r", end="")
    report("FlashMLX (ours)", r)

    # Ollama
    print("  Benchmarking ollama...", end="", flush=True)
    r = bench_ollama()
    print("\r", end="")
    report("ollama", r)

    # LM Studio
    print("  Benchmarking LM Studio...", end="", flush=True)
    r = bench_openai_compat("lm-studio", 1234, "qwen/qwen3-0.6b")
    print("\r", end="")
    report("LM Studio", r)

    # Bodega
    print("  Benchmarking bodega...", end="", flush=True)
    r = bench_openai_compat("bodega", 44468, "qwen/qwen3-0.6b")
    print("\r", end="")
    report("bodega", r)

    print()
    print("TTFT = Time to First Token (lower is better)")
    print("Decode = Decode-only tok/s (higher is better)")
    print("E2E = End-to-end tok/s after model loaded (higher is better)")
