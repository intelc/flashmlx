# tests/test_attention.py
import mlx.core as mx
import mlx.nn as nn
from engine.attention import scaled_dot_product_attention, create_causal_mask
from engine.kv_cache import KVCache


class TestCausalMask:
    def test_causal_mask_shape(self):
        mask = create_causal_mask(seq_len=4, offset=0)
        assert mask.shape == (4, 4)

    def test_causal_mask_is_lower_triangular(self):
        mask = create_causal_mask(seq_len=3, offset=0)
        # Upper triangle should be -inf (masked), lower triangle + diagonal should be 0
        assert mask[0, 1].item() == float("-inf")
        assert mask[0, 2].item() == float("-inf")
        assert mask[1, 2].item() == float("-inf")
        assert mask[0, 0].item() == 0.0
        assert mask[1, 1].item() == 0.0
        assert mask[2, 2].item() == 0.0

    def test_causal_mask_with_offset(self):
        # With offset=5, generating 1 new token that can attend to 5 cached + 1 new = 6
        mask = create_causal_mask(seq_len=1, offset=5)
        assert mask.shape == (1, 6)
        # Single new token can attend to everything
        assert mx.all(mask == 0.0).item()


class TestScaledDotProductAttention:
    def test_output_shape_standard(self):
        B, n_heads, L, D = 1, 4, 8, 16
        q = mx.random.normal((B, n_heads, L, D))
        k = mx.random.normal((B, n_heads, L, D))
        v = mx.random.normal((B, n_heads, L, D))
        mask = create_causal_mask(L, offset=0)
        out = scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)
        assert out.shape == (B, n_heads, L, D)

    def test_output_shape_gqa(self):
        """GQA: 8 query heads, 2 KV heads — should broadcast correctly."""
        B, L, D = 1, 4, 16
        n_q_heads, n_kv_heads = 8, 2
        q = mx.random.normal((B, n_q_heads, L, D))
        k = mx.random.normal((B, n_kv_heads, L, D))
        v = mx.random.normal((B, n_kv_heads, L, D))
        mask = create_causal_mask(L, offset=0)
        out = scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)
        assert out.shape == (B, n_q_heads, L, D)

    def test_attention_respects_mask(self):
        """First token should only attend to itself (causal mask)."""
        B, n_heads, L, D = 1, 1, 4, 8
        q = mx.random.normal((B, n_heads, L, D))
        k = mx.random.normal((B, n_heads, L, D))
        # Set value of token 0 to all 1s, rest to all 0s
        v = mx.zeros((B, n_heads, L, D))
        v[:, :, 0:1, :] = mx.ones((B, n_heads, 1, D))
        mask = create_causal_mask(L, offset=0)
        out = scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)
        # First token output should be exactly [1,1,...,1] since it only attends to token 0
        assert mx.allclose(out[:, :, 0, :], mx.ones((B, n_heads, D)), atol=1e-5).item()
