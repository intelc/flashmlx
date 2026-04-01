# autobench/analyze.py
"""Post-hoc analysis of autobench experiments."""

import json
from collections import defaultdict
from pathlib import Path

from autobench.loop import parse_results_tsv, ExperimentResult, RESULTS_TSV

AUTOBENCH_DIR = Path(__file__).parent
PROJECT_ROOT = AUTOBENCH_DIR.parent


def analyze(results_path: str | None = None) -> dict:
    """Analyze experiment results and return summary statistics.

    Returns dict with:
        - total_experiments: int
        - kept: int
        - discarded: int
        - crashed: int
        - best_tok_s: float
        - best_ttft_ms: float
        - best_composite: float
        - improvement_over_baseline: float (ratio)
        - by_module: dict mapping module -> {kept, discarded, best_score}
        - timeline: list of {experiment_num, composite_score, status}
    """
    path = results_path or str(RESULTS_TSV)
    results = parse_results_tsv(path)

    if not results:
        return {"total_experiments": 0, "message": "No experiments found"}

    kept = [r for r in results if r.status == "keep"]
    discarded = [r for r in results if r.status == "discard"]
    crashed = [r for r in results if r.status == "crash"]

    best_tok_s = max((r.tok_s for r in kept), default=0)
    best_ttft = min((r.ttft_ms for r in kept), default=0) if kept else 0
    best_composite = max((r.composite_score for r in kept), default=0)

    # Per-module breakdown
    by_module = defaultdict(lambda: {"kept": 0, "discarded": 0, "crashed": 0, "best_score": 0})
    for r in results:
        mod = by_module[r.module]
        if r.status == "keep":
            mod["kept"] += 1
            mod["best_score"] = max(mod["best_score"], r.composite_score)
        elif r.status == "discard":
            mod["discarded"] += 1
        elif r.status == "crash":
            mod["crashed"] += 1

    # Timeline
    timeline = [
        {"experiment_num": i, "composite_score": r.composite_score, "status": r.status}
        for i, r in enumerate(results)
    ]

    return {
        "total_experiments": len(results),
        "kept": len(kept),
        "discarded": len(discarded),
        "crashed": len(crashed),
        "best_tok_s": best_tok_s,
        "best_ttft_ms": best_ttft,
        "best_composite_score": best_composite,
        "by_module": dict(by_module),
        "timeline": timeline,
    }


def print_report(results_path: str | None = None):
    """Print a human-readable analysis report."""
    stats = analyze(results_path)

    if stats.get("total_experiments", 0) == 0:
        print("No experiments found.")
        return

    print("=" * 60)
    print("FlashMLX AutoBench Analysis Report")
    print("=" * 60)
    print(f"Total experiments: {stats['total_experiments']}")
    print(f"  Kept:      {stats['kept']}")
    print(f"  Discarded: {stats['discarded']}")
    print(f"  Crashed:   {stats['crashed']}")
    print(f"\nBest results:")
    print(f"  tok/s:           {stats['best_tok_s']:.1f}")
    print(f"  TTFT:            {stats['best_ttft_ms']:.1f}ms")
    print(f"  Composite score: {stats['best_composite_score']:.4f}")

    if stats["by_module"]:
        print(f"\nPer-module breakdown:")
        for module, data in sorted(stats["by_module"].items()):
            print(f"  {module}: {data['kept']} kept, {data['discarded']} discarded, "
                  f"{data['crashed']} crashed, best={data['best_score']:.4f}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    print_report(path)
