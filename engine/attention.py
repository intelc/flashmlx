# engine/attention.py
import mlx.core as mx


def create_causal_mask(seq_len: int, offset: int = 0) -> mx.array:
    """Create additive causal attention mask.

    Returns a [seq_len, offset + seq_len] mask where masked positions are -inf
    and allowed positions are 0.
    """
    total_len = offset + seq_len
    # Row indices: the new tokens at positions [offset, offset+seq_len)
    row_idx = mx.arange(offset, offset + seq_len).reshape(-1, 1)
    # Col indices: all positions [0, total_len)
    col_idx = mx.arange(total_len).reshape(1, -1)
    mask = mx.where(col_idx <= row_idx, 0.0, float("-inf"))
    return mask


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask: mx.array | None = None,
) -> mx.array:
    """Scaled dot-product attention with GQA support.

    Uses mx.fast.scaled_dot_product_attention which natively handles
    n_q_heads != n_kv_heads by broadcasting KV heads.

    Args:
        q: [B, n_q_heads, L_q, head_dim]
        k: [B, n_kv_heads, L_kv, head_dim]
        v: [B, n_kv_heads, L_kv, head_dim]
        scale: typically head_dim ** -0.5
        mask: [L_q, L_kv] additive mask (0 = attend, -inf = mask)

    Returns:
        [B, n_q_heads, L_q, head_dim]
    """
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
