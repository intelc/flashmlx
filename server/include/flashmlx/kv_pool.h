#pragma once
#include <vector>
#include <mutex>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace flashmlx {

class KVCachePool {
public:
    KVCachePool(int max_slots, int max_context_len, int num_layers,
                int n_kv_heads, int head_dim, mx::Dtype dtype = mx::float16);

    int allocate();                              // Returns slot index or -1 if full
    void free(int slot_idx);                     // Return slot to pool
    mx::array& keys(int slot_idx, int layer);    // Get key cache for slot+layer
    mx::array& values(int slot_idx, int layer);  // Get value cache for slot+layer
    int num_free() const;
    int max_slots() const { return max_slots_; }

private:
    int max_slots_;
    int max_context_len_;
    int num_layers_;
    // cache_keys_[slot][layer] = [1, n_kv_heads, max_context, head_dim] in float16
    std::vector<std::vector<mx::array>> cache_keys_;
    std::vector<std::vector<mx::array>> cache_values_;
    std::vector<bool> slot_free_;
    mutable std::mutex mutex_;
};

} // namespace flashmlx
