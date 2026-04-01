# engine/kv_cache.py
import mlx.core as mx


class KVCache:
    """KV cache for transformer attention layers using concatenation.

    Stores keys and values as [B, n_kv_heads, seq_len, head_dim] tensors.
    Uses concatenation instead of pre-allocated buffers for simpler
    memory management and better MLX graph optimization.
    """

    def __init__(self, step: int = 256):
        self.step = step
        self.offset = 0
        self._keys = None
        self._values = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append new keys/values and return the full cached history.

        Args:
            keys: [B, n_kv_heads, new_tokens, head_dim]
            values: [B, n_kv_heads, new_tokens, head_dim]

        Returns:
            Tuple of (all_keys, all_values) each [B, n_kv_heads, total_tokens, head_dim]
        """
        T = keys.shape[2]

        if self._keys is None:
            self._keys = keys
            self._values = values
        else:
            self._keys = mx.concatenate([self._keys, keys], axis=2)
            self._values = mx.concatenate([self._values, values], axis=2)

        self.offset += T
        return self._keys, self._values

    @property
    def state(self):
        """Return current cache state for mx.eval materialization."""
        if self._keys is None:
            return []
        return [self._keys, self._values]
