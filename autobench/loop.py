# autobench/loop.py
"""Core autoresearch loop: propose -> commit -> benchmark -> keep/discard."""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

AUTOBENCH_DIR = Path(__file__).parent
RESULTS_TSV = AUTOBENCH_DIR / "results.tsv"
PROGRAM_MD = AUTOBENCH_DIR / "program.md"
PROJECT_ROOT = AUTOBENCH_DIR.parent

TSV_HEADER = "commit\tmodule\ttok_s\tttft_ms\tmemory_mb\tperplexity\tppl_delta_pct\tcomposite_score\tstatus\tdescription"


@dataclass
class ExperimentResult:
    commit: str
    module: str
    tok_s: float
    ttft_ms: float
    memory_mb: float
    perplexity: float
    ppl_delta_pct: float
    composite_score: float
    status: str  # "keep", "discard", "crash", "rejected"
    description: str


def parse_results_tsv(path: str) -> list[ExperimentResult]:
    """Parse results.tsv into a list of ExperimentResult."""
    results = []
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return results

    lines = p.read_text().strip().split("\n")
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        results.append(ExperimentResult(
            commit=parts[0],
            module=parts[1],
            tok_s=float(parts[2]),
            ttft_ms=float(parts[3]),
            memory_mb=float(parts[4]),
            perplexity=float(parts[5]),
            ppl_delta_pct=float(parts[6]),
            composite_score=float(parts[7]),
            status=parts[8],
            description=parts[9],
        ))
    return results


def append_result(path: str, result: ExperimentResult):
    """Append an experiment result to the TSV file."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        p.write_text(TSV_HEADER + "\n")

    line = "\t".join([
        result.commit, result.module,
        f"{result.tok_s:.1f}", f"{result.ttft_ms:.1f}",
        f"{result.memory_mb:.0f}", f"{result.perplexity:.4f}",
        f"{result.ppl_delta_pct:.2f}", f"{result.composite_score:.4f}",
        result.status, result.description,
    ])
    with open(p, "a") as f:
        f.write(line + "\n")


def compute_composite_score(
    tok_s: float, ttft_ms: float,
    baseline_tok_s: float, baseline_ttft_ms: float,
) -> float:
    """Compute composite score: 0.7 * normalized_tok_s + 0.3 * normalized_ttft.

    Both components are ratios where > 1.0 means improvement over baseline.
    """
    norm_tok_s = tok_s / baseline_tok_s if baseline_tok_s > 0 else 0
    norm_ttft = baseline_ttft_ms / ttft_ms if ttft_ms > 0 else 0
    return 0.7 * norm_tok_s + 0.3 * norm_ttft


def get_git_head() -> str:
    """Get current HEAD commit hash (short)."""
    result = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    return result.stdout.strip().split()[0] if result.stdout.strip() else "unknown"


def git_commit(message: str, files: list[str]):
    """Stage files and commit."""
    for f in files:
        subprocess.run(["git", "add", f], cwd=str(PROJECT_ROOT))
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(PROJECT_ROOT), capture_output=True,
    )


def git_revert_last():
    """Revert the last commit (discard experiment)."""
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        cwd=str(PROJECT_ROOT), capture_output=True,
    )


def load_baselines() -> dict:
    """Load baseline metrics from harness/baselines.json."""
    baselines_path = PROJECT_ROOT / "harness" / "baselines.json"
    with open(baselines_path) as f:
        return json.load(f)


def get_best_score(results: list[ExperimentResult]) -> float:
    """Get the best composite score from kept experiments."""
    kept = [r for r in results if r.status == "keep"]
    if not kept:
        return 1.0  # baseline score
    return max(r.composite_score for r in kept)


# Cycling strategy: which module to target based on experiment number
CYCLE_MODULES = [
    ("attention", 20),
    ("kv_cache", 20),
    ("quantize", 20),
    ("generate", 20),
    ("cross-module", 20),
]


def get_target_module(experiment_num: int) -> str:
    """Determine which module to optimize based on experiment number."""
    cycle_len = sum(count for _, count in CYCLE_MODULES)
    pos = experiment_num % cycle_len
    cumulative = 0
    for module, count in CYCLE_MODULES:
        cumulative += count
        if pos < cumulative:
            return module
    return "cross-module"


def run_loop(
    model_name: str = "qwen3-0.6b",
    max_experiments: int | None = None,
    validation_interval: int = 10,
):
    """Run the autoresearch loop.

    Args:
        model_name: Which model from models.yaml to benchmark against
        max_experiments: Stop after N experiments (None = run forever)
        validation_interval: Run medium-model validation every N experiments
    """
    baselines = load_baselines()
    model_baseline = baselines["models"].get(model_name, {})
    baseline_tok_s = model_baseline.get("tok_s", 100.0)
    baseline_ttft_ms = model_baseline.get("ttft_ms", 50.0)
    model_repo = model_baseline.get("repo", "")

    results = parse_results_tsv(str(RESULTS_TSV))
    best_score = get_best_score(results)
    experiment_num = len(results)

    print(f"Starting autobench loop. Baseline: {baseline_tok_s:.1f} tok/s, {baseline_ttft_ms:.1f}ms TTFT")
    print(f"Best score so far: {best_score:.4f} ({experiment_num} prior experiments)")

    while max_experiments is None or experiment_num < max_experiments:
        target_module = get_target_module(experiment_num)
        print(f"\n--- Experiment {experiment_num + 1}: targeting {target_module} ---")

        # The agent interaction would happen here:
        # 1. Read program.md, current module source, results.tsv tail, git log
        # 2. Send to LLM agent
        # 3. Agent returns a diff
        # 4. Apply diff, commit, benchmark, keep/discard
        #
        # For now, this is a placeholder that will be connected to an LLM API.
        print(f"  [Agent interaction needed — target: engine/{target_module}.py]")
        print(f"  Connect an LLM agent to propose modifications to engine/{target_module}.py")
        break  # Exit until agent is connected

        experiment_num += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FlashMLX AutoBench Loop")
    parser.add_argument("--model", default="qwen3-0.6b", help="Model name from models.yaml")
    parser.add_argument("--max-experiments", type=int, default=None, help="Max experiments (default: unlimited)")
    parser.add_argument("--validation-interval", type=int, default=10, help="Medium-model validation interval")
    args = parser.parse_args()
    run_loop(args.model, args.max_experiments, args.validation_interval)
