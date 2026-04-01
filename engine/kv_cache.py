# engine/kv_cache.py
import mlx.core as mx


class KVCache:
    """Pre-allocating KV cache for transformer attention layers.

    Stores keys and values as [B, n_kv_heads, seq_len, head_dim] tensors.
    Pre-allocates in chunks of `step` tokens to avoid per-token reallocation.
    """

    def __init__(self, step: int = 256):
        self.step = step
        self.offset = 0
        self._keys = None
        self._values = None
        self._capacity = 0

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
        B, H, T, D = keys.shape
        new_offset = self.offset + T

        if self._keys is None:
            # First call — allocate buffer
            self._capacity = max(self.step, new_offset)
            self._keys = mx.zeros((B, H, self._capacity, D))
            self._values = mx.zeros((B, H, self._capacity, D))

        if new_offset > self._capacity:
            # Grow buffer
            new_capacity = self._capacity
            while new_capacity < new_offset:
                new_capacity += self.step
            k_new = mx.zeros((B, H, new_capacity, D))
            v_new = mx.zeros((B, H, new_capacity, D))
            k_new[:, :, : self.offset, :] = self._keys[:, :, : self.offset, :]
            v_new[:, :, : self.offset, :] = self._values[:, :, : self.offset, :]
            self._keys = k_new
            self._values = v_new
            self._capacity = new_capacity

        # Write new tokens into pre-allocated buffer
        self._keys[:, :, self.offset : new_offset, :] = keys
        self._values[:, :, self.offset : new_offset, :] = values
        self.offset = new_offset

        return self._keys[:, :, : self.offset, :], self._values[:, :, : self.offset, :]

    @property
    def state(self):
        """Return current cache state for mx.eval materialization."""
        if self._keys is None:
            return []
        return [self._keys, self._values]
