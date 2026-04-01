# engine/kv_cache.py
import mlx.core as mx


class KVCache:
    """Pre-allocating KV cache with quantized step growth.

    Uses pre-allocated buffers with slice assignment for O(1) per-token updates
    instead of O(n) concatenation. Grows in steps when capacity is exceeded.
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
        B, H, T, D = keys.shape
        new_offset = self.offset + T

        if self._keys is None:
            # First call — allocate buffer rounded up to step
            self._capacity = ((new_offset + self.step - 1) // self.step) * self.step
            self._keys = mx.zeros((B, H, self._capacity, D))
            self._values = mx.zeros((B, H, self._capacity, D))

        if new_offset > self._capacity:
            new_capacity = ((new_offset + self.step - 1) // self.step) * self.step
            k_new = mx.zeros((B, H, new_capacity, D))
            v_new = mx.zeros((B, H, new_capacity, D))
            k_new[:, :, : self.offset, :] = self._keys[:, :, : self.offset, :]
            v_new[:, :, : self.offset, :] = self._values[:, :, : self.offset, :]
            self._keys = k_new
            self._values = v_new
            self._capacity = new_capacity

        self._keys[:, :, self.offset : new_offset, :] = keys
        self._values[:, :, self.offset : new_offset, :] = values
        self.offset = new_offset

        return self._keys[:, :, : self.offset, :], self._values[:, :, : self.offset, :]

    @property
    def state(self):
        if self._keys is None:
            return []
        return [self._keys, self._values]
