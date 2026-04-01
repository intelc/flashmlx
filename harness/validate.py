# harness/validate.py
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class ValidationResult:
    perplexity: float
    ppl_delta_pct: float
    passed: bool


def compute_perplexity(logits: mx.array, targets: mx.array) -> float:
    """Compute perplexity from model logits and target token IDs.

    Args:
        logits: [B, seq_len, vocab_size] — raw logits from model
        targets: [B, seq_len] — ground truth token IDs

    Returns:
        Perplexity (lower is better)
    """
    B, T, V = logits.shape
    # Log-softmax
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # Gather log-probs of target tokens
    targets_flat = targets.reshape(-1)
    log_probs_flat = log_probs.reshape(-1, V)
    # Index into log_probs for each target
    target_log_probs = mx.take_along_axis(
        log_probs_flat, targets_flat.reshape(-1, 1), axis=1
    ).squeeze(-1)
    # Mean negative log-likelihood
    nll = -mx.mean(target_log_probs)
    ppl = float(mx.exp(nll).item())
    return ppl


def check_correctness(
    current_ppl: float,
    baseline_ppl: float,
    threshold_pct: float = 1.0,
) -> ValidationResult:
    """Check if current perplexity is within threshold of baseline.

    Args:
        current_ppl: Perplexity from current engine configuration
        baseline_ppl: Perplexity from FP16 baseline
        threshold_pct: Maximum allowed degradation in percent

    Returns:
        ValidationResult with pass/fail and delta
    """
    if baseline_ppl == 0:
        ppl_delta_pct = 0.0 if current_ppl == 0 else float("inf")
    else:
        ppl_delta_pct = (current_ppl - baseline_ppl) / baseline_ppl * 100.0
    passed = ppl_delta_pct < threshold_pct
    return ValidationResult(
        perplexity=current_ppl,
        ppl_delta_pct=ppl_delta_pct,
        passed=passed,
    )
