# engine/kv_cache.py
import mlx.core as mx


class KVCache:
    """KV cache for transformer attention layers using concatenation.

    Uses concatenation for updates. Simple and fast for small-to-medium models
    where MLX's graph optimizer handles concat efficiently.
    """

    def __init__(self, step: int = 256):
        self.step = step
        self.offset = 0
        self._keys = None
        self._values = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
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
        if self._keys is None:
            return []
        return [self._keys, self._values]
