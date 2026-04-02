# Left-Padded Batch KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-step KV cache concat/split with a persistent left-padded batched cache (à la mlx-lm BatchKVCache), eliminating the main bottleneck for multi-request throughput.

**Architecture:** Maintain a single `[B, n_kv, write_pos, hd]` cache per layer that all active requests share. Shorter prompts are left-padded so all sequences align at the right edge. New tokens are written at position `write_pos` via slice assignment (one op per layer, not per-sequence). A pre-built mask excludes left-padded positions from attention. The cache only restructures when batch composition changes (request joins/completes), not every decode step.

**Tech Stack:** C++17, MLX C++ API (`mx::slice_update`, `mx::fast::scaled_dot_product_attention` with explicit mask, `mx::fast::rope` with per-sequence offsets)

---

## Design Overview

### How left-padding works

```
Request A (prompt=10 tokens):  [pad pad pad pad pad pad pad pad pad pad A0 A1 A2 A3 A4 A5 A6 A7 A8 A9]
Request B (prompt=20 tokens):  [B0  B1  B2  B3  B4  B5  B6  B7  B8  B9  B10 B11 B12 B13 B14 B15 B16 B17 B18 B19]
                                                                        ↑ write_pos (= max prompt length = 20)
```

After 3 decode steps:
```
Request A: [pad ... pad A0..A9 dec0 dec1 dec2]
Request B: [B0..B19 dec0 dec1 dec2]
                                   ↑ write_pos = 23
```

- **RoPE offset** for A: `write_pos - left_padding_A = 23 - 10 = 13` (A has 10+3=13 real tokens)
- **RoPE offset** for B: `write_pos - left_padding_B = 23 - 0 = 23` (B has 20+3=23 real tokens)
- **Mask**: position `p` is valid for batch element `b` iff `p >= left_padding[b]`
- **KV write**: ALL sequences write at `write_pos` (single slice assignment for entire batch)

### What changes vs. current approach

| Per-step operation | Current (scatter + split) | New (left-padded) |
|---|---|---|
| KV cache update | `put_along_axis` per-batch + `slice_update` per-layer writeback | Single `slice_update` at `write_pos` per layer |
| Mask | Built every step from offsets | Pre-built, sliced to `write_pos` |
| RoPE | Array offsets from `cache_offset` | Array offsets from `write_pos - left_padding` |
| Batch concat | 28 layers × B-way concat on cache miss | Only on batch composition change |
| Batch split | 28 layers × B-way slice + slice_update | Only on batch composition change |
| Array ops per step | ~336 (28×(B slices + concat + B slice_updates)) | ~84 (28×(1 slice_update keys + 1 slice_update values + 1 slice mask)) |

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `server/include/flashmlx/batch_kv_cache.h` | `BatchKVCache` class — left-padded batched KV cache |
| Create | `server/src/batch_kv_cache.cpp` | Implementation: create, update, merge, filter |
| Modify | `server/include/flashmlx/model.h` | Add `forward_batched()` using BatchKVCache |
| Modify | `server/src/model.cpp` | Implement `attention_batched()` + `forward_batched()` |
| Modify | `server/src/scheduler.cpp` | Replace `decode_batch_heterogeneous()` with BatchKVCache-based decode |
| Modify | `server/include/flashmlx/scheduler.h` | Store `BatchKVCache` instance + new method |
| Modify | `server/CMakeLists.txt` | Add `batch_kv_cache.cpp` to build |
| Create | `server/tests/test_batch_kv_cache.cpp` | Correctness test |

---

### Task 1: Create BatchKVCache data structure

**Files:**
- Create: `server/include/flashmlx/batch_kv_cache.h`
- Create: `server/src/batch_kv_cache.cpp`
- Modify: `server/CMakeLists.txt`

- [ ] **Step 1: Write batch_kv_cache.h**

```cpp
#pragma once
#include <vector>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace flashmlx {

/// Left-padded batched KV cache for efficient multi-request decode.
/// All sequences share a single [B, n_kv_heads, buf_len, head_dim] buffer per layer.
/// Shorter sequences are left-padded so they align at the right edge.
class BatchKVCache {
public:
    BatchKVCache() = default;

    /// Build from individually-prefilled per-slot caches.
    /// slot_keys[slot][layer] = [1, n_kv, seq_len, hd] (variable seq_len per slot)
    /// Merges into left-padded [B, n_kv, max_seq_len, hd] buffers.
    void build(const std::vector<std::vector<mx::array>>& slot_keys,
               const std::vector<std::vector<mx::array>>& slot_values,
               const std::vector<int>& seq_lengths,
               int num_layers, int n_kv_heads, int head_dim, mx::Dtype dtype);

    /// Update: write new K/V at write_pos for all B sequences.
    /// k, v: [B, n_kv_heads, 1, head_dim]
    void update(const mx::array& k, const mx::array& v, int layer);

    /// Get cache arrays for layer (for SDPA)
    /// Returns [B, n_kv_heads, write_pos, head_dim]
    mx::array get_keys(int layer) const;
    mx::array get_values(int layer) const;

    /// Get the pre-built mask [B, 1, 1, write_pos]
    /// Valid where position >= left_padding[b]
    mx::array get_mask() const;

    /// Get per-sequence RoPE offsets [B] = write_pos - left_padding[b]
    mx::array get_rope_offsets() const;

    /// Advance write position by 1 (call after update for all layers)
    void advance();

    /// Remove sequences that completed. keep_indices: which batch elements to retain.
    void filter(const std::vector<int>& keep_indices);

    /// Flush current state back to per-slot pool caches (for when batch changes)
    void flush_to_slots(std::vector<std::vector<mx::array>>& slot_keys,
                        std::vector<std::vector<mx::array>>& slot_values) const;

    int batch_size() const { return batch_size_; }
    int write_pos() const { return write_pos_; }
    bool valid() const { return valid_; }
    void invalidate() { valid_ = false; }

private:
    std::vector<mx::array> keys_;    // [B, n_kv, buf_len, hd] per layer
    std::vector<mx::array> values_;
    std::vector<int> left_padding_;  // per-sequence left padding
    mx::array mask_;                 // [B, 1, 1, buf_len] — pre-built, sliced at get_mask()
    int write_pos_ = 0;             // current write position (same for all sequences)
    int buf_len_ = 0;               // allocated buffer length
    int batch_size_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int num_layers_ = 0;
    mx::Dtype dtype_ = mx::bfloat16;
    bool valid_ = false;
};

} // namespace flashmlx
```

- [ ] **Step 2: Write batch_kv_cache.cpp**

```cpp
#include "flashmlx/batch_kv_cache.h"
#include <algorithm>

namespace flashmlx {

void BatchKVCache::build(
    const std::vector<std::vector<mx::array>>& slot_keys,
    const std::vector<std::vector<mx::array>>& slot_values,
    const std::vector<int>& seq_lengths,
    int num_layers, int n_kv_heads, int head_dim, mx::Dtype dtype) {

    batch_size_ = static_cast<int>(seq_lengths.size());
    num_layers_ = num_layers;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    dtype_ = dtype;

    // Find max sequence length — this becomes the right-aligned write position
    int max_len = *std::max_element(seq_lengths.begin(), seq_lengths.end());
    write_pos_ = max_len;

    // Allocate buffer with 256-token step size for growth headroom
    buf_len_ = ((max_len + 255) / 256) * 256 + 256;

    // Compute left padding per sequence
    left_padding_.resize(batch_size_);
    for (int b = 0; b < batch_size_; b++) {
        left_padding_[b] = max_len - seq_lengths[b];
    }

    // Build left-padded caches
    keys_.clear();
    values_.clear();
    keys_.reserve(num_layers);
    values_.reserve(num_layers);

    for (int l = 0; l < num_layers; l++) {
        auto buf_k = mx::zeros({batch_size_, n_kv_heads, buf_len_, head_dim}, dtype);
        auto buf_v = mx::zeros({batch_size_, n_kv_heads, buf_len_, head_dim}, dtype);

        // Copy each sequence's KV data right-justified
        for (int b = 0; b < batch_size_; b++) {
            int pad = left_padding_[b];
            int len = seq_lengths[b];
            // slot_keys[b][l] shape: [1, n_kv, len, hd]
            // Write to buf[b, :, pad:pad+len, :]
            buf_k = mx::slice_update(buf_k, slot_keys[b][l],
                {b, 0, pad, 0}, {b + 1, n_kv_heads, pad + len, head_dim});
            buf_v = mx::slice_update(buf_v, slot_values[b][l],
                {b, 0, pad, 0}, {b + 1, n_kv_heads, pad + len, head_dim});
        }

        keys_.push_back(buf_k);
        values_.push_back(buf_v);
    }

    // Build full mask [B, 1, 1, buf_len]
    // Valid where position >= left_padding[b]
    auto positions = mx::arange(0, buf_len_, mx::int32);  // [buf_len]
    std::vector<int> pad_vec(left_padding_.begin(), left_padding_.end());
    auto pads = mx::array(pad_vec.data(), {batch_size_, 1, 1, 1}, mx::int32);
    mask_ = mx::where(
        mx::greater_equal(positions, pads),
        mx::array(0.0f, dtype),
        mx::array(-1e9f, dtype));  // [B, 1, 1, buf_len]

    valid_ = true;
}

void BatchKVCache::update(const mx::array& k, const mx::array& v, int layer) {
    // k, v: [B, n_kv, 1, hd]
    // Write at write_pos_ for all sequences
    keys_[layer] = mx::slice_update(keys_[layer], k,
        {0, 0, write_pos_, 0},
        {batch_size_, n_kv_heads_, write_pos_ + 1, head_dim_});
    values_[layer] = mx::slice_update(values_[layer], v,
        {0, 0, write_pos_, 0},
        {batch_size_, n_kv_heads_, write_pos_ + 1, head_dim_});
}

mx::array BatchKVCache::get_keys(int layer) const {
    return mx::slice(keys_[layer], {0, 0, 0, 0},
        {batch_size_, n_kv_heads_, write_pos_, head_dim_});
}

mx::array BatchKVCache::get_values(int layer) const {
    return mx::slice(values_[layer], {0, 0, 0, 0},
        {batch_size_, n_kv_heads_, write_pos_, head_dim_});
}

mx::array BatchKVCache::get_mask() const {
    return mx::slice(mask_, {0, 0, 0, 0},
        {batch_size_, 1, 1, write_pos_});
}

mx::array BatchKVCache::get_rope_offsets() const {
    // RoPE offset = write_pos - left_padding (actual token count per sequence)
    std::vector<int> offsets(batch_size_);
    for (int b = 0; b < batch_size_; b++) {
        offsets[b] = write_pos_ - left_padding_[b];
    }
    return mx::array(offsets.data(), {batch_size_}, mx::int32);
}

void BatchKVCache::advance() {
    write_pos_++;
    // Grow buffers if we hit the allocated limit
    if (write_pos_ >= buf_len_) {
        int new_len = buf_len_ + 256;
        for (int l = 0; l < num_layers_; l++) {
            auto pad_k = mx::zeros({batch_size_, n_kv_heads_, 256, head_dim_}, dtype_);
            auto pad_v = mx::zeros({batch_size_, n_kv_heads_, 256, head_dim_}, dtype_);
            keys_[l] = mx::concatenate({keys_[l], pad_k}, 2);
            values_[l] = mx::concatenate({values_[l], pad_v}, 2);
        }
        // Extend mask
        auto extra_positions = mx::arange(buf_len_, new_len, mx::int32);
        std::vector<int> pad_vec(left_padding_.begin(), left_padding_.end());
        auto pads = mx::array(pad_vec.data(), {batch_size_, 1, 1, 1}, mx::int32);
        auto extra_mask = mx::where(
            mx::greater_equal(extra_positions, pads),
            mx::array(0.0f, dtype_),
            mx::array(-1e9f, dtype_));
        mask_ = mx::concatenate({mask_, extra_mask}, 3);
        buf_len_ = new_len;
    }
}

void BatchKVCache::filter(const std::vector<int>& keep_indices) {
    auto idx = mx::array(keep_indices.data(),
        {static_cast<int>(keep_indices.size())}, mx::int32);
    for (int l = 0; l < num_layers_; l++) {
        keys_[l] = mx::take(keys_[l], idx, 0);
        values_[l] = mx::take(values_[l], idx, 0);
    }
    mask_ = mx::take(mask_, idx, 0);

    std::vector<int> new_padding;
    for (int i : keep_indices) {
        new_padding.push_back(left_padding_[i]);
    }
    left_padding_ = new_padding;
    batch_size_ = static_cast<int>(keep_indices.size());

    if (batch_size_ == 0) {
        valid_ = false;
    }
}

void BatchKVCache::flush_to_slots(
    std::vector<std::vector<mx::array>>& slot_keys,
    std::vector<std::vector<mx::array>>& slot_values) const {
    // Extract per-sequence caches (right portion, excluding left padding)
    for (int b = 0; b < batch_size_; b++) {
        for (int l = 0; l < num_layers_; l++) {
            int pad = left_padding_[b];
            slot_keys[b][l] = mx::slice(keys_[l],
                {b, 0, pad, 0}, {b + 1, n_kv_heads_, write_pos_, head_dim_});
            slot_values[b][l] = mx::slice(values_[l],
                {b, 0, pad, 0}, {b + 1, n_kv_heads_, write_pos_, head_dim_});
        }
    }
}

} // namespace flashmlx
```

- [ ] **Step 3: Add to CMakeLists.txt**

Add `src/batch_kv_cache.cpp` to the source lists for both `_flashmlx_engine` and `test_heterogeneous_attention`.

- [ ] **Step 4: Build to verify**

Run: `cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)`

- [ ] **Step 5: Commit**

```bash
git add server/include/flashmlx/batch_kv_cache.h server/src/batch_kv_cache.cpp server/CMakeLists.txt
git commit -m "feat: add BatchKVCache — left-padded batched KV cache"
```

---

### Task 2: Add `attention_batched()` and `forward_batched()` to LlamaModel

**Files:**
- Modify: `server/include/flashmlx/model.h`
- Modify: `server/src/model.cpp`

A new attention path that takes pre-sliced K/V caches and a pre-built mask from BatchKVCache. Unlike `attention_heterogeneous` which does scatter writes, this path does NO cache writes (the scheduler handles that via `BatchKVCache::update()`). It returns new K/V tensors alongside logits so the scheduler can write them.

- [ ] **Step 1: Add declarations to model.h**

Add to LlamaModel private section:

```cpp
    /// Batched attention using pre-sliced caches from BatchKVCache.
    /// Does NOT modify caches — returns new_k, new_v for caller to write.
    /// Returns {attn_output, new_k, new_v}
    std::tuple<mx::array, mx::array, mx::array> attention_batched(
        const mx::array& x, int layer,
        const mx::array& cache_k, const mx::array& cache_v,
        const mx::array& rope_offsets,
        const mx::array& mask);
```

Add to LlamaModel public section:

```cpp
    /// Batched forward using external cache management.
    /// Returns {logits, vector of new_k per layer, vector of new_v per layer}
    struct BatchedForwardResult {
        mx::array logits;
        std::vector<mx::array> new_keys;   // [B, n_kv, 1, hd] per layer
        std::vector<mx::array> new_values;
    };
    BatchedForwardResult forward_batched(
        const mx::array& input_ids,
        const std::vector<mx::array>& cache_keys,   // from BatchKVCache::get_keys()
        const std::vector<mx::array>& cache_values,
        const mx::array& rope_offsets,
        const mx::array& mask);
```

Add corresponding pure virtual to ModelBase and stub to NemotronHModel.

- [ ] **Step 2: Implement attention_batched**

```cpp
std::tuple<mx::array, mx::array, mx::array> LlamaModel::attention_batched(
    const mx::array& x, int layer,
    const mx::array& cache_k, const mx::array& cache_v,
    const mx::array& rope_offsets,
    const mx::array& mask) {

    const auto& lw = layer_weights_[layer];
    int B = x.shape(0);
    int L = x.shape(1);
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int hd = head_dim_;

    auto q = linear_fast(x, lw.q_w, lw.q_s, lw.q_b);
    auto k = linear_fast(x, lw.k_w, lw.k_s, lw.k_b);
    auto v = linear_fast(x, lw.v_w, lw.v_s, lw.v_b);

    if (lw.q_bias) q = mx::add(q, *lw.q_bias);
    if (lw.k_bias) k = mx::add(k, *lw.k_bias);
    if (lw.v_bias) v = mx::add(v, *lw.v_bias);

    q = mx::transpose(mx::reshape(q, {B, L, n_heads, hd}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B, L, n_kv_heads, hd}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B, L, n_kv_heads, hd}), {0, 2, 1, 3});

    if (lw.has_q_norm) {
        q = rms_norm(q, lw.q_norm_w);
        k = rms_norm(k, lw.k_norm_w);
    }

    // RoPE with per-sequence offsets (actual token positions, not buffer positions)
    q = mx::fast::rope(q, hd, false, config_.rope_theta, 1.0f, rope_offsets);
    k = mx::fast::rope(k, hd, false, config_.rope_theta, 1.0f, rope_offsets);

    // Concatenate new K/V to cached K/V for this step's attention
    auto full_k = mx::concatenate({cache_k, k}, 2);  // [B, n_kv, write_pos+1, hd]
    auto full_v = mx::concatenate({cache_v, v}, 2);

    // SDPA with mask (mask has write_pos entries, but full_k has write_pos+1)
    // Extend mask by 1 column (the new token is always valid)
    auto valid_col = mx::zeros({B, 1, 1, 1}, mask.dtype());
    auto full_mask = mx::concatenate({mask, valid_col}, 3);

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto attn_out = mx::fast::scaled_dot_product_attention(
        q, full_k, full_v, scale, "", full_mask);

    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, n_heads * hd});
    auto output = linear_fast(attn_out, lw.o_w, lw.o_s, lw.o_b);

    return {output, k, v};  // Return new K/V for caller to write to cache
}
```

- [ ] **Step 3: Implement forward_batched**

```cpp
LlamaModel::BatchedForwardResult LlamaModel::forward_batched(
    const mx::array& input_ids,
    const std::vector<mx::array>& cache_keys,
    const std::vector<mx::array>& cache_values,
    const mx::array& rope_offsets,
    const mx::array& mask) {

    auto h = embed(input_ids);

    std::vector<mx::array> all_new_keys, all_new_values;
    all_new_keys.reserve(config_.num_hidden_layers);
    all_new_values.reserve(config_.num_hidden_layers);

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        const auto& lw = layer_weights_[i];
        auto normed = rms_norm(h, lw.input_norm_w);

        auto [attn_out, new_k, new_v] = attention_batched(
            normed, i, cache_keys[i], cache_values[i], rope_offsets, mask);
        all_new_keys.push_back(new_k);
        all_new_values.push_back(new_v);

        h = mx::add(h, attn_out);
        auto normed2 = rms_norm(h, lw.post_norm_w);
        auto mlp_out = [&]() -> mx::array {
            if (!layer_is_moe_.empty() && i < (int)layer_is_moe_.size() && layer_is_moe_[i]) {
                return moe_block(normed2, i);
            } else {
                return mlp_fast(normed2, i);
            }
        }();
        h = mx::add(h, mlp_out);
    }

    h = rms_norm(h, *norm_w_);
    auto logits = lm_head(h);

    return {logits, std::move(all_new_keys), std::move(all_new_values)};
}
```

- [ ] **Step 4: Build to verify**

Run: `cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)`

- [ ] **Step 5: Commit**

```bash
git add server/include/flashmlx/model.h server/src/model.cpp server/src/nemotron_h.cpp
git commit -m "feat: add forward_batched — returns new K/V for external cache management"
```

---

### Task 3: Integrate BatchKVCache into scheduler

**Files:**
- Modify: `server/include/flashmlx/scheduler.h`
- Modify: `server/src/scheduler.cpp`

Replace `decode_batch_heterogeneous()` with a new `decode_batched()` that uses `BatchKVCache`.

- [ ] **Step 1: Add BatchKVCache to scheduler.h**

Replace the persistent cache members:
```cpp
    // Replace these:
    // std::vector<mx::array> batch_cache_k_, batch_cache_v_;
    // std::vector<std::string> batch_ids_;
    // bool batch_cache_valid_ = false;

    // With:
    #include "flashmlx/batch_kv_cache.h"
    BatchKVCache batch_kv_cache_;
    std::vector<std::string> batch_ids_;

    void decode_batched(const std::vector<std::string>& ids,
                        std::unordered_map<std::string, std::vector<int>>& new_tokens,
                        std::vector<std::string>& done_ids);
```

- [ ] **Step 2: Implement decode_batched**

```cpp
void BatchScheduler::decode_batched(
    const std::vector<std::string>& ids,
    std::unordered_map<std::string, std::vector<int>>& new_tokens,
    std::vector<std::string>& done_ids) {

    int B = static_cast<int>(ids.size());
    int num_layers = pool_.num_layers();
    int n_kv = model_.config().num_key_value_heads;
    int hd = model_.config().head_dim > 0 ? model_.config().head_dim
             : model_.config().hidden_size / model_.config().num_attention_heads;

    // 1. Build input tokens
    std::vector<int> tok_ids;
    tok_ids.reserve(B);
    for (auto& id : ids) {
        tok_ids.push_back(active_[id].next_token.item<int>());
    }
    auto input_ids = mx::array(tok_ids.data(), {B, 1}, mx::int32);

    // 2. Build or reuse BatchKVCache
    if (!batch_kv_cache_.valid() || ids != batch_ids_) {
        // Build from pool slots
        std::vector<std::vector<mx::array>> slot_keys(B), slot_values(B);
        std::vector<int> seq_lengths(B);
        for (int b = 0; b < B; b++) {
            auto& req = active_[ids[b]];
            seq_lengths[b] = req.cache_offset;
            slot_keys[b].reserve(num_layers);
            slot_values[b].reserve(num_layers);
            for (int l = 0; l < num_layers; l++) {
                // Slice pool cache to actual sequence length
                slot_keys[b].push_back(mx::slice(pool_.keys(req.kv_slot, l),
                    {0, 0, 0, 0}, {1, n_kv, req.cache_offset, hd}));
                slot_values[b].push_back(mx::slice(pool_.values(req.kv_slot, l),
                    {0, 0, 0, 0}, {1, n_kv, req.cache_offset, hd}));
            }
        }
        batch_kv_cache_.build(slot_keys, slot_values, seq_lengths,
                              num_layers, n_kv, hd, model_.config().activation_dtype);
        batch_ids_ = ids;
    }

    // 3. Get cache state for forward pass
    std::vector<mx::array> cache_k, cache_v;
    cache_k.reserve(num_layers);
    cache_v.reserve(num_layers);
    for (int l = 0; l < num_layers; l++) {
        cache_k.push_back(batch_kv_cache_.get_keys(l));
        cache_v.push_back(batch_kv_cache_.get_values(l));
    }
    auto rope_offsets = batch_kv_cache_.get_rope_offsets();
    auto mask = batch_kv_cache_.get_mask();

    // 4. Forward pass — returns logits + new K/V per layer
    auto result = model_.forward_batched(input_ids, cache_k, cache_v, rope_offsets, mask);

    // 5. Write new K/V to cache and advance
    for (int l = 0; l < num_layers; l++) {
        batch_kv_cache_.update(result.new_keys[l], result.new_values[l], l);
    }
    batch_kv_cache_.advance();

    // 6. Sample all tokens at once
    auto all_logits = mx::reshape(result.logits, {B, result.logits.shape(2)});
    auto all_tokens = mx::argmax(all_logits, -1);
    mx::eval({all_tokens});

    bool any_done = false;
    for (int b = 0; b < B; b++) {
        auto& req = active_[ids[b]];
        int tok_id = mx::slice(all_tokens, {b}, {b + 1}).item<int>();

        req.next_token = mx::array({tok_id}, mx::int32);
        req.output_tokens.push_back(tok_id);
        req.generated_count++;
        req.cache_offset++;

        new_tokens[ids[b]].push_back(tok_id);

        if (req.generated_count >= req.max_tokens) {
            req.state = RequestState::DONE;
            done_ids.push_back(ids[b]);
            any_done = true;
        }
    }

    // 7. If any request completed, flush caches back to pool and invalidate
    if (any_done) {
        // Flush back to pool slots before invalidation
        std::vector<std::vector<mx::array>> flush_keys(B), flush_values(B);
        for (int b = 0; b < B; b++) {
            flush_keys[b].resize(num_layers);
            flush_values[b].resize(num_layers);
        }
        batch_kv_cache_.flush_to_slots(flush_keys, flush_values);
        for (int b = 0; b < B; b++) {
            int slot = active_[ids[b]].kv_slot;
            int max_ctx = pool_.max_context_len();
            for (int l = 0; l < num_layers; l++) {
                auto full = mx::zeros({1, n_kv, max_ctx, hd}, model_.config().activation_dtype);
                int len = flush_keys[b][l].shape(2);
                pool_.keys(slot, l) = mx::slice_update(full, flush_keys[b][l],
                    {0, 0, 0, 0}, {1, n_kv, len, hd});
                full = mx::zeros({1, n_kv, max_ctx, hd}, model_.config().activation_dtype);
                pool_.values(slot, l) = mx::slice_update(full, flush_values[b][l],
                    {0, 0, 0, 0}, {1, n_kv, len, hd});
            }
        }
        batch_kv_cache_.invalidate();
        batch_ids_.clear();
    }
}
```

- [ ] **Step 3: Update step() to call decode_batched instead of decode_batch_heterogeneous**

Replace the dispatch:
```cpp
    } else if (decoding_ids.size() > 1) {
        decode_batched(decoding_ids, new_tokens, done_ids);
    }
```

- [ ] **Step 4: Build and run correctness test**

```bash
cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)
./test_heterogeneous_attention $MODEL_PATH
```

- [ ] **Step 5: Server smoke test**

```bash
cp _flashmlx_engine*.so ../python/
python -m server.run $MODEL_PATH --port 8081 &
sleep 3
curl -s http://localhost:8081/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10,"temperature":0,"stream":false}'
# Two concurrent
curl -s ... -d '{"messages":[{"role":"user","content":"Hello"}],...}' &
curl -s ... -d '{"messages":[{"role":"user","content":"What?"}],...}' &
wait
pkill -f server.run
```

- [ ] **Step 6: Commit**

```bash
git add server/include/flashmlx/scheduler.h server/src/scheduler.cpp
git commit -m "feat: use BatchKVCache for multi-request decode — eliminates per-step concat/split"
```

---

### Task 4: Benchmark

- [ ] **Step 1: Run benchmark**

```bash
bash autoresearch_bench.sh
```

Compare against baseline (321 tok/s) and current best (353 tok/s).

**Target**: >400 tok/s at C=4 (eliminating ~250 array ops per step should give 15-20% improvement).

- [ ] **Step 2: Run full suite** (C=1 through C=8, 64/256/512 tokens)

Verify C=1 performance is preserved (N-step path unchanged).

---

## Known Risks

1. **Buffer growth via concat (every 256 tokens)** — amortized cost, but each growth event is expensive. For 512-token generation, only 1-2 growth events. For 2048 tokens, ~8 events.

2. **Flush-to-pool overhead on completion** — when a request finishes, we flush all B sequences' caches back to the pool. This is O(B × layers) slice operations. But it only happens once per request completion, not per step.

3. **RoPE offset correctness** — must verify that `write_pos - left_padding` gives the correct absolute token position for RoPE. The first token of a sequence at position `left_padding` should get RoPE offset 0, the second should get 1, etc.

4. **Mask correctness** — the mask extends by 1 column each step (for the new token). `attention_batched` handles this by concatenating a valid column. This means the mask from `BatchKVCache::get_mask()` always has `write_pos` columns, and the actual SDPA mask has `write_pos + 1` columns.
