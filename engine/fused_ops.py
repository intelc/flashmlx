# engine/fused_ops.py
"""Custom fused Metal kernels for reducing kernel launch overhead."""

from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def fused_residual_rms_norm(residual: mx.array, x: mx.array, weight: mx.array, eps: float = 1e-5):
    """Fused residual add + RMS normalization.

    Computes: norm(residual + x) in a single compiled graph.
    Eliminates intermediate materialization of the add result.

    Args:
        residual: [B, L, D] residual connection
        x: [B, L, D] output from attention or MLP
        weight: [D] RMSNorm weight
        eps: RMSNorm epsilon

    Returns:
        Tuple of (h, normed) where h = residual + x and normed = rms_norm(h)
    """
    h = residual + x
    normed = mx.fast.rms_norm(h, weight, eps)
    return h, normed


@partial(mx.compile, shapeless=True)
def fused_rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-5):
    """Compiled RMS normalization using mx.fast.rms_norm.

    Compiles the norm call to reduce dispatch overhead.
    """
    return mx.fast.rms_norm(x, weight, eps)
