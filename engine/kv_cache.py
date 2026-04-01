# engine/kv_cache.py
import mlx.core as mx


class KVCache:
    """Pre-allocating KV cache matching mlx-lm's pattern.

    Allocates in steps of 256 tokens. Uses slice assignment for O(1)
    per-token updates. Grows by concatenating new zero blocks when needed.
    """

    step = 256

    def __init__(self, step: int = 256):
        self.step = step
        self.offset = 0
        self._keys = None
        self._values = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        prev = self.offset
        if self._keys is None or (prev + keys.shape[2]) > self._keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self._keys is not None:
                if prev % self.step != 0:
                    self._keys = self._keys[..., :prev, :]
                    self._values = self._values[..., :prev, :]
                self._keys = mx.concatenate([self._keys, new_k], axis=2)
                self._values = mx.concatenate([self._values, new_v], axis=2)
            else:
                self._keys, self._values = new_k, new_v

        self.offset += keys.shape[2]
        self._keys[..., prev : self.offset, :] = keys
        self._values[..., prev : self.offset, :] = values
        return self._keys[..., : self.offset, :], self._values[..., : self.offset, :]

    @property
    def state(self):
        """Return current cache state for mx.eval materialization."""
        if self._keys is None:
            return []
        if self.offset == self._keys.shape[2]:
            return [self._keys, self._values]
        return [self._keys[..., : self.offset, :], self._values[..., : self.offset, :]]
