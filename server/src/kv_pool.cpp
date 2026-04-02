#include "flashmlx/kv_pool.h"
#include <stdexcept>

namespace flashmlx {

KVCachePool::KVCachePool(int max_slots, int max_context_len, int num_layers,
                         int n_kv_heads, int head_dim, mx::Dtype dtype)
    : max_slots_(max_slots),
      max_context_len_(max_context_len),
      num_layers_(num_layers),
      slot_free_(max_slots, true) {

    // Pre-allocate all KV cache buffers
    cache_keys_.reserve(max_slots);
    cache_values_.reserve(max_slots);

    std::vector<mx::array> all_arrays;
    all_arrays.reserve(max_slots * num_layers * 2);

    for (int s = 0; s < max_slots; ++s) {
        cache_keys_.emplace_back();
        cache_values_.emplace_back();
        cache_keys_[s].reserve(num_layers);
        cache_values_[s].reserve(num_layers);

        for (int l = 0; l < num_layers; ++l) {
            cache_keys_[s].emplace_back(
                mx::zeros({1, n_kv_heads, max_context_len, head_dim}, dtype));
            cache_values_[s].emplace_back(
                mx::zeros({1, n_kv_heads, max_context_len, head_dim}, dtype));

            all_arrays.push_back(cache_keys_[s][l]);
            all_arrays.push_back(cache_values_[s][l]);
        }
    }

    // Materialize all buffers
    mx::eval(all_arrays);
}

int KVCachePool::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < max_slots_; ++i) {
        if (slot_free_[i]) {
            slot_free_[i] = false;
            return i;
        }
    }
    return -1;  // Pool is full
}

void KVCachePool::free(int slot_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot_idx < 0 || slot_idx >= max_slots_) {
        throw std::out_of_range("KVCachePool::free: invalid slot index");
    }
    slot_free_[slot_idx] = true;
}

mx::array& KVCachePool::keys(int slot_idx, int layer) {
    if (slot_idx < 0 || slot_idx >= max_slots_) {
        throw std::out_of_range("KVCachePool::keys: invalid slot index");
    }
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("KVCachePool::keys: invalid layer index");
    }
    return cache_keys_[slot_idx][layer];
}

mx::array& KVCachePool::values(int slot_idx, int layer) {
    if (slot_idx < 0 || slot_idx >= max_slots_) {
        throw std::out_of_range("KVCachePool::values: invalid slot index");
    }
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("KVCachePool::values: invalid layer index");
    }
    return cache_values_[slot_idx][layer];
}

void KVCachePool::write_slot(int slot_idx,
                              const std::vector<mx::array>& keys,
                              const std::vector<mx::array>& values) {
    for (int l = 0; l < num_layers_; ++l) {
        cache_keys_[slot_idx][l] = keys[l];
        cache_values_[slot_idx][l] = values[l];
    }
}

int KVCachePool::num_free() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (bool f : slot_free_) {
        if (f) ++count;
    }
    return count;
}

} // namespace flashmlx
