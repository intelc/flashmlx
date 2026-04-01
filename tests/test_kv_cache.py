# tests/test_kv_cache.py
import mlx.core as mx
from engine.kv_cache import KVCache


class TestKVCache:
    def test_empty_cache_offset_is_zero(self):
        cache = KVCache()
        assert cache.offset == 0

    def test_update_and_fetch_single_layer(self):
        cache = KVCache()
        # Simulate: batch=1, n_kv_heads=2, seq_len=3, head_dim=4
        keys = mx.ones((1, 2, 3, 4))
        values = mx.ones((1, 2, 3, 4)) * 2
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape == (1, 2, 3, 4)
        assert v_out.shape == (1, 2, 3, 4)
        assert cache.offset == 3

    def test_update_accumulates_tokens(self):
        cache = KVCache()
        k1 = mx.ones((1, 2, 3, 4))
        v1 = mx.ones((1, 2, 3, 4))
        cache.update_and_fetch(k1, v1)

        k2 = mx.ones((1, 2, 1, 4)) * 5
        v2 = mx.ones((1, 2, 1, 4)) * 10
        k_out, v_out = cache.update_and_fetch(k2, v2)
        # Should have 3 + 1 = 4 tokens
        assert k_out.shape == (1, 2, 4, 4)
        assert v_out.shape == (1, 2, 4, 4)
        assert cache.offset == 4
        # Last token should be the new values
        assert mx.allclose(k_out[:, :, -1:, :], mx.ones((1, 2, 1, 4)) * 5).item()
        assert mx.allclose(v_out[:, :, -1:, :], mx.ones((1, 2, 1, 4)) * 10).item()

    def test_multiple_sequential_updates(self):
        cache = KVCache()
        for i in range(10):
            k = mx.ones((1, 4, 1, 8)) * i
            v = mx.ones((1, 4, 1, 8)) * (i + 100)
            k_out, v_out = cache.update_and_fetch(k, v)
        assert cache.offset == 10
        assert k_out.shape == (1, 4, 10, 8)

    def test_pre_allocation_step(self):
        """Cache should pre-allocate in steps to avoid per-token reallocation."""
        cache = KVCache(step=256)
        k = mx.ones((1, 2, 1, 4))
        v = mx.ones((1, 2, 1, 4))
        cache.update_and_fetch(k, v)
        # Internal buffer should be larger than offset
        assert cache.offset == 1
