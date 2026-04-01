"""One-time setup: download models, compute baselines, cache eval data."""

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml

from engine import load_model, generate
from harness.benchmark import run_benchmark
from harness.validate import compute_perplexity

HARNESS_DIR = Path(__file__).parent
MODELS_YAML = HARNESS_DIR / "models.yaml"
BASELINES_JSON = HARNESS_DIR / "baselines.json"

# Fixed eval prompt (first 128 tokens repeated — will be replaced by WikiText-2 in production)
EVAL_PROMPT_SHORT = list(range(1, 129))   # 128 tokens
EVAL_PROMPT_LONG = list(range(1, 1025))   # 1024 tokens


def get_hardware_info() -> dict:
    """Detect Apple Silicon hardware via system_profiler."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        chip = result.stdout.strip()
    except Exception:
        chip = "unknown"

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes / (1024 ** 3)
    except Exception:
        ram_gb = 0

    return {"chip": chip, "ram_gb": round(ram_gb, 1)}


def load_models_yaml() -> dict:
    """Load the model registry."""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)


def prepare(tiers: list[str] | None = None):
    """Download models and compute baselines.

    Args:
        tiers: Which tiers to prepare. Default: ["small_tier"].
    """
    if tiers is None:
        tiers = ["small_tier"]

    registry = load_models_yaml()
    hardware = get_hardware_info()
    baselines = {"hardware": hardware, "models": {}}

    for tier in tiers:
        models = registry.get(tier, [])
        for model_info in models:
            name = model_info["name"]
            repo = model_info["repo"]
            print(f"Preparing {name} from {repo}...")

            try:
                result = run_benchmark(
                    model_path=repo,
                    prompt_tokens=EVAL_PROMPT_SHORT,
                    gen_tokens=64,
                    num_runs=3,
                    warmup_tokens=20,
                )
                baselines["models"][name] = {
                    "repo": repo,
                    "tier": tier,
                    "tok_s": result.tok_s,
                    "ttft_ms": result.ttft_ms,
                    "memory_mb": result.memory_mb,
                }
                print(f"  tok/s: {result.tok_s:.1f}, TTFT: {result.ttft_ms:.1f}ms, Memory: {result.memory_mb:.0f}MB")
            except Exception as e:
                print(f"  FAILED: {e}")
                baselines["models"][name] = {"repo": repo, "tier": tier, "error": str(e)}

    with open(BASELINES_JSON, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\nBaselines saved to {BASELINES_JSON}")


if __name__ == "__main__":
    tiers = sys.argv[1:] if len(sys.argv) > 1 else None
    prepare(tiers)
