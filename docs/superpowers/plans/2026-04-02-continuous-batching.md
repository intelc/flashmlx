# Continuous Batching with Heterogeneous Offsets

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable FlashMLX to batch decode requests at different cache offsets together, closing the throughput gap with vllm-mlx at high concurrency (currently 237 vs 440 tok/s at C=8, 256 tokens).

**Architecture:** Replace the scheduler's offset-grouping strategy with a unified batch that handles heterogeneous offsets. The model's int-offset attention path gets a new sibling that accepts per-sequence offsets as an array and builds an explicit attention mask. The KV pool already pre-allocates `[1, n_kv_heads, max_context_len, head_dim]` per slot, so caches are already "padded" — we just need a mask to ignore the padding. Keep the N-step single-request fast path (it's still optimal at C=1).

**Tech Stack:** C++17, MLX C++ API (`mx::fast::scaled_dot_product_attention` with `mask` parameter, `mx::fast::rope` with array offsets), pybind11

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `server/src/scheduler.cpp` | Remove offset-grouping, add `decode_batch_heterogeneous()` |
| Modify | `server/include/flashmlx/scheduler.h` | Declare new `decode_batch_heterogeneous()` method |
| Modify | `server/src/model.cpp` | Add `attention_heterogeneous()` for per-sequence offset + mask decode |
| Modify | `server/include/flashmlx/model.h` | Declare `attention_heterogeneous()` and `forward_heterogeneous()` |
| Modify | `server/include/flashmlx/kv_pool.h` | Add `max_context_len()` accessor |
| Create | `server/tests/test_heterogeneous_attention.cpp` | Unit test: multi-offset batched decode correctness |
| Create | `server/tests/test_continuous_batching.cpp` | Integration test: scheduler batches mixed-offset requests |

---

### Task 1: Add `max_context_len()` accessor to KVCachePool

**Files:**
- Modify: `server/include/flashmlx/kv_pool.h`

The heterogeneous attention path needs to know the max context length to build masks over the full pre-allocated KV dimension. Currently `max_context_len_` is private with no accessor.

- [ ] **Step 1: Add the accessor**

In `server/include/flashmlx/kv_pool.h`, add after the existing `num_layers()` accessor:

```cpp
int max_context_len() const { return max_context_len_; }
```

- [ ] **Step 2: Build to verify**

Run: `cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Compiles successfully

- [ ] **Step 3: Commit**

```bash
git add server/include/flashmlx/kv_pool.h
git commit -m "feat: add max_context_len() accessor to KVCachePool"
```

---

### Task 2: Add heterogeneous attention to LlamaModel

**Files:**
- Modify: `server/include/flashmlx/model.h:92-142` (LlamaModel class)
- Modify: `server/src/model.cpp` (after line 887)

This task adds a new attention method that takes per-sequence offsets as an `mx::array` and builds an explicit boolean mask, avoiding the `mx::eval()` sync that the existing array-offset path uses. The key insight: since KV caches are pre-allocated to `max_context_len`, we concatenate the new K/V token at each sequence's offset position using `scatter`, then mask out invalid positions in SDPA.

However, scatter-based writes are complex and slow in MLX. A simpler approach: use the existing concat-based int-offset attention path's structure, but operate on the *full pre-allocated buffer*. Each slot's KV cache already contains valid data at positions `[0, offset)` from prior decode steps. We just need to:
1. Append the new K/V at each sequence's position (different positions per batch element)
2. Pass the full buffer to SDPA with a mask that marks valid positions per sequence

For the append step, we use `mx::scatter` to write one token per batch element at different positions. For the mask, we build a `[B, 1, 1, max_ctx+1]` boolean mask.

- [ ] **Step 1: Declare the new methods in model.h**

In `server/include/flashmlx/model.h`, add to the LlamaModel private section (after `attention` overloads, around line 142):

```cpp
    // Heterogeneous-offset attention: per-sequence offsets, explicit mask
    // offsets: [B] int32 array with each sequence's cache position
    // Uses full pre-allocated KV buffers + boolean mask for SDPA
    mx::array attention_heterogeneous(
        const mx::array& x, int layer,
        mx::array& cache_k, mx::array& cache_v,
        const mx::array& offsets, int max_kv_len);
    mx::array forward_heterogeneous(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        const mx::array& offsets, int max_kv_len);
```

Also add the same to `ModelBase` (around line 86):

```cpp
    virtual mx::array forward_heterogeneous(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        const mx::array& offsets, int max_kv_len) = 0;
```

And to `NemotronHModel` as a stub that falls back to the array-offset path (around line 201):

```cpp
    mx::array forward_heterogeneous(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        const mx::array& offsets, int max_kv_len) override {
        // Nemotron-H doesn't support heterogeneous batching yet; fall back
        return forward(input_ids, cache_keys, cache_values, offsets);
    }
```

- [ ] **Step 2: Implement `attention_heterogeneous` in model.cpp**

Add after the existing int-offset `attention()` (after line 887 in `model.cpp`):

```cpp
mx::array LlamaModel::attention_heterogeneous(
    const mx::array& x, int layer,
    mx::array& cache_k, mx::array& cache_v,
    const mx::array& offsets, int max_kv_len) {

    const auto& lw = layer_weights_[layer];
    int B = x.shape(0);
    int L = x.shape(1);  // Always 1 for decode
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int hd = head_dim_;

    // Q, K, V projections
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

    // RoPE with per-sequence offsets
    q = mx::fast::rope(q, hd, false, config_.rope_theta, 1.0f, offsets);
    k = mx::fast::rope(k, hd, false, config_.rope_theta, 1.0f, offsets);

    // KV cache update: write new k,v at each sequence's offset position
    // cache_k shape: [B, n_kv_heads, max_kv_len, hd] (pre-allocated full buffer)
    // k shape: [B, n_kv_heads, 1, hd]
    //
    // We need to write k[b] at position offsets[b] for each batch element.
    // Use slice_update with the max offset, which writes to a contiguous block.
    // But offsets differ per sequence — so we use a loop-free scatter approach:
    //
    // Strategy: build index array and use scatter to place each token.
    // For each batch element b, write at cache_k[b, :, offsets[b], :] = k[b, :, 0, :]
    //
    // Simpler approach for decode (L=1): iterate batch elements.
    // B is small (<=8), so a small loop is fine.
    // We avoid mx::eval() — just build the graph.

    // Build updated caches by scattering new tokens at per-sequence positions
    // For B<=8 this loop is negligible vs. the attention compute
    std::vector<mx::array> k_updates, v_updates;
    for (int b = 0; b < B; b++) {
        // Extract this sequence's cache and new token
        auto cache_k_b = mx::slice(cache_k, {b, 0, 0, 0}, {b+1, n_kv_heads, max_kv_len, hd});
        auto cache_v_b = mx::slice(cache_v, {b, 0, 0, 0}, {b+1, n_kv_heads, max_kv_len, hd});
        auto k_b = mx::slice(k, {b, 0, 0, 0}, {b+1, n_kv_heads, 1, hd});
        auto v_b = mx::slice(v, {b, 0, 0, 0}, {b+1, n_kv_heads, 1, hd});

        // Get this sequence's offset
        auto off_b = mx::slice(offsets, {b}, {b+1});

        // Concatenate: append new token after valid portion, producing [1, n_kv, offset+1, hd]
        // But we're working with fixed-size buffers. Use slice_update instead.
        // Write k_b at position [0, 0, offset_b, 0] in cache_k_b
        // Since offset_b is a graph node (not evaluated), we need a graph-compatible write.
        // MLX scatter: cache_k_b[:, :, offset_b, :] = k_b[:, :, 0, :]
        // Use mx::put_along_axis or manual index construction.

        // Build scatter indices: [1, 1, 1, 1] with value = offset_b, axis=2
        auto idx = mx::reshape(off_b, {1, 1, 1, 1});
        idx = mx::broadcast_to(idx, {1, n_kv_heads, 1, hd});
        cache_k_b = mx::put_along_axis(cache_k_b, idx, k_b, 2);
        cache_v_b = mx::put_along_axis(cache_v_b, idx, v_b, 2);

        k_updates.push_back(cache_k_b);
        v_updates.push_back(cache_v_b);
    }
    cache_k = mx::concatenate(k_updates, 0);
    cache_v = mx::concatenate(v_updates, 0);

    // Build attention mask: [B, 1, 1, max_kv_len]
    // For each batch element b, positions [0, offsets[b]] are valid (offsets[b]+1 total)
    // Create: mask[b, 0, 0, p] = (p <= offsets[b]) ? 0.0 : -inf
    auto positions = mx::arange(max_kv_len);  // [max_kv_len]
    positions = mx::reshape(positions, {1, max_kv_len});  // [1, max_kv_len]
    auto off_col = mx::reshape(offsets, {B, 1});  // [B, 1]
    // valid where position <= offset (offset is the position we just wrote to)
    auto valid = mx::less_equal(positions, off_col);  // [B, max_kv_len]
    // Convert to float mask: 0.0 for valid, -inf for invalid
    auto mask = mx::where(valid,
        mx::zeros({1}, config_.activation_dtype),
        mx::full({1}, -std::numeric_limits<float>::infinity(), config_.activation_dtype));
    mask = mx::reshape(mask, {B, 1, 1, max_kv_len});  // [B, 1, 1, max_kv_len]

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto attn_out = mx::fast::scaled_dot_product_attention(
        q, cache_k, cache_v, scale, /*mask=*/mask);

    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, n_heads * hd});
    return linear_fast(attn_out, lw.o_w, lw.o_s, lw.o_b);
}
```

- [ ] **Step 3: Implement `forward_heterogeneous` in model.cpp**

Add after `attention_heterogeneous`:

```cpp
mx::array LlamaModel::forward_heterogeneous(
    const mx::array& input_ids,
    std::vector<mx::array>& cache_keys,
    std::vector<mx::array>& cache_values,
    const mx::array& offsets, int max_kv_len) {

    auto h = embed(input_ids);
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        const auto& lw = layer_weights_[i];
        auto normed = rms_norm(h, lw.input_norm_w);
        auto attn_out = attention_heterogeneous(normed, i, cache_keys[i], cache_values[i], offsets, max_kv_len);
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
    return lm_head(h);
}
```

- [ ] **Step 4: Build to verify**

Run: `cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -10`
Expected: Compiles successfully

- [ ] **Step 5: Commit**

```bash
git add server/include/flashmlx/model.h server/src/model.cpp
git commit -m "feat: add heterogeneous-offset attention for continuous batching"
```

---

### Task 3: Write correctness test for heterogeneous attention

**Files:**
- Create: `server/tests/test_heterogeneous_attention.cpp`

Before changing the scheduler, verify that `forward_heterogeneous` produces the same logits as the int-offset `forward` path for individual sequences. Run two single-sequence forward passes at different offsets, then one batched heterogeneous forward, and assert logits match.

- [ ] **Step 1: Write the test**

```cpp
// server/tests/test_heterogeneous_attention.cpp
//
// Verifies that forward_heterogeneous(batch, [off_a, off_b]) produces the same
// logits as running forward(seq_a, off_a) and forward(seq_b, off_b) separately.

#include <iostream>
#include <cmath>
#include <cassert>
#include <string>

#include <mlx/mlx.h>
#include "flashmlx/model.h"
#include "flashmlx/kv_pool.h"

namespace mx = mlx::core;
using namespace flashmlx;

bool arrays_close(const mx::array& a, const mx::array& b, float atol = 1e-2f) {
    mx::eval({a, b});
    auto diff = mx::abs(mx::subtract(mx::astype(a, mx::float32), mx::astype(b, mx::float32)));
    mx::eval({diff});
    auto max_diff = mx::max(diff);
    mx::eval({max_diff});
    float md = max_diff.item<float>();
    if (md > atol) {
        std::cerr << "  max_diff = " << md << " > atol = " << atol << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];

    std::cout << "Loading model from " << model_path << std::endl;
    LlamaModel model(model_path);
    auto& cfg = model.config();
    int n_kv = cfg.num_key_value_heads;
    int hd = cfg.head_dim > 0 ? cfg.head_dim : cfg.hidden_size / cfg.num_attention_heads;
    int n_layers = cfg.num_hidden_layers;
    int max_ctx = 128;  // Small for testing

    // Create two KV cache pools (one per sequence for reference)
    // and one batched pool for heterogeneous test
    auto make_cache = [&]() {
        std::vector<mx::array> keys, vals;
        for (int l = 0; l < n_layers; l++) {
            keys.push_back(mx::zeros({1, n_kv, 0, hd}, cfg.activation_dtype));
            vals.push_back(mx::zeros({1, n_kv, 0, hd}, cfg.activation_dtype));
        }
        return std::make_pair(keys, vals);
    };

    // Prefill sequence A with 5 tokens, sequence B with 10 tokens
    std::vector<int> prompt_a = {1, 2, 3, 4, 5};
    std::vector<int> prompt_b = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    auto [keys_a, vals_a] = make_cache();
    auto [keys_b, vals_b] = make_cache();

    // Prefill A
    auto ids_a = mx::array(prompt_a.data(), {1, 5}, mx::int32);
    auto off_a = mx::array({0}, mx::int32);
    model.forward(ids_a, keys_a, vals_a, off_a);
    // Trim caches to prompt length
    for (int l = 0; l < n_layers; l++) {
        keys_a[l] = mx::slice(keys_a[l], {0,0,0,0}, {1, n_kv, 5, hd});
        vals_a[l] = mx::slice(vals_a[l], {0,0,0,0}, {1, n_kv, 5, hd});
    }

    // Prefill B
    auto ids_b = mx::array(prompt_b.data(), {1, 10}, mx::int32);
    auto off_b = mx::array({0}, mx::int32);
    model.forward(ids_b, keys_b, vals_b, off_b);
    for (int l = 0; l < n_layers; l++) {
        keys_b[l] = mx::slice(keys_b[l], {0,0,0,0}, {1, n_kv, 10, hd});
        vals_b[l] = mx::slice(vals_b[l], {0,0,0,0}, {1, n_kv, 10, hd});
    }

    // Decode one token each (separately) to get reference logits
    int tok_a = 42, tok_b = 99;
    auto decode_a = mx::array({tok_a}, {1, 1}, mx::int32);
    auto decode_b = mx::array({tok_b}, {1, 1}, mx::int32);

    // Clone caches for reference (forward modifies them via concat)
    auto clone_cache = [](const std::vector<mx::array>& k, const std::vector<mx::array>& v) {
        std::vector<mx::array> ck, cv;
        for (size_t l = 0; l < k.size(); l++) {
            ck.push_back(mx::copy(k[l]));
            cv.push_back(mx::copy(v[l]));
        }
        mx::eval(ck); mx::eval(cv);
        return std::make_pair(ck, cv);
    };
    auto [ref_keys_a, ref_vals_a] = clone_cache(keys_a, vals_a);
    auto [ref_keys_b, ref_vals_b] = clone_cache(keys_b, vals_b);

    auto logits_a = model.forward(decode_a, ref_keys_a, ref_vals_a, 5);  // offset = prompt_len
    auto logits_b = model.forward(decode_b, ref_keys_b, ref_vals_b, 10);
    mx::eval({logits_a, logits_b});

    std::cout << "Reference logits_a shape: [" << logits_a.shape(0) << ", " << logits_a.shape(1) << ", " << logits_a.shape(2) << "]" << std::endl;
    std::cout << "Reference logits_b shape: [" << logits_b.shape(0) << ", " << logits_b.shape(1) << ", " << logits_b.shape(2) << "]" << std::endl;

    // Now test heterogeneous batched forward
    // Build batched KV caches: pad shorter cache to match longer, using pre-allocated buffers
    int max_off = 10;  // max of the two offsets
    std::vector<mx::array> batch_keys, batch_vals;
    for (int l = 0; l < n_layers; l++) {
        // Pad A's cache from [1, n_kv, 5, hd] to [1, n_kv, max_ctx, hd]
        auto padded_a = mx::zeros({1, n_kv, max_ctx, hd}, cfg.activation_dtype);
        padded_a = mx::slice_update(padded_a, keys_a[l], {0,0,0,0}, {1, n_kv, 5, hd});
        auto padded_b = mx::zeros({1, n_kv, max_ctx, hd}, cfg.activation_dtype);
        padded_b = mx::slice_update(padded_b, keys_b[l], {0,0,0,0}, {1, n_kv, 10, hd});
        batch_keys.push_back(mx::concatenate({padded_a, padded_b}, 0));  // [2, n_kv, max_ctx, hd]

        auto padded_va = mx::zeros({1, n_kv, max_ctx, hd}, cfg.activation_dtype);
        padded_va = mx::slice_update(padded_va, vals_a[l], {0,0,0,0}, {1, n_kv, 5, hd});
        auto padded_vb = mx::zeros({1, n_kv, max_ctx, hd}, cfg.activation_dtype);
        padded_vb = mx::slice_update(padded_vb, vals_b[l], {0,0,0,0}, {1, n_kv, 10, hd});
        batch_vals.push_back(mx::concatenate({padded_va, padded_vb}, 0));
    }
    mx::eval(batch_keys); mx::eval(batch_vals);

    auto batch_ids = mx::array({tok_a, tok_b}, {2, 1}, mx::int32);
    auto batch_offsets = mx::array({5, 10}, mx::int32);

    auto batch_logits = model.forward_heterogeneous(batch_ids, batch_keys, batch_vals, batch_offsets, max_ctx);
    mx::eval({batch_logits});

    std::cout << "Batch logits shape: [" << batch_logits.shape(0) << ", " << batch_logits.shape(1) << ", " << batch_logits.shape(2) << "]" << std::endl;

    // Compare: batch_logits[0] should match logits_a, batch_logits[1] should match logits_b
    auto het_logits_a = mx::slice(batch_logits, {0, 0, 0}, {1, 1, batch_logits.shape(2)});
    auto het_logits_b = mx::slice(batch_logits, {1, 0, 0}, {2, 1, batch_logits.shape(2)});

    std::cout << "Comparing sequence A logits... ";
    if (arrays_close(het_logits_a, logits_a, 0.05f)) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        return 1;
    }

    std::cout << "Comparing sequence B logits... ";
    if (arrays_close(het_logits_b, logits_b, 0.05f)) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        return 1;
    }

    std::cout << "\nAll heterogeneous attention tests PASSED!" << std::endl;
    return 0;
}
```

- [ ] **Step 2: Add test to CMakeLists.txt**

Find the test section in `server/CMakeLists.txt` and add:

```cmake
add_executable(test_heterogeneous_attention tests/test_heterogeneous_attention.cpp ${ENGINE_SOURCES})
target_include_directories(test_heterogeneous_attention PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_heterogeneous_attention PRIVATE mlx)
```

- [ ] **Step 3: Build and run the test**

Run:
```bash
cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu) test_heterogeneous_attention 2>&1 | tail -5
MODEL_PATH=~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/$(ls ~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/)
./test_heterogeneous_attention "$MODEL_PATH"
```
Expected: Both sequence comparisons print PASS

- [ ] **Step 4: Commit**

```bash
git add server/tests/test_heterogeneous_attention.cpp server/CMakeLists.txt
git commit -m "test: verify heterogeneous attention matches per-sequence forward"
```

---

### Task 4: Update scheduler for continuous batching

**Files:**
- Modify: `server/include/flashmlx/scheduler.h:56-61`
- Modify: `server/src/scheduler.cpp:36-73` (the `step()` decode section)

Replace the offset-grouping logic with a strategy that:
1. If only 1 request is decoding → use N-step single-request path (unchanged, fastest for C=1)
2. If 2+ requests are decoding → batch ALL of them via `forward_heterogeneous`, regardless of offset

This eliminates the per-offset-group limitation. Each step decodes one token per request (no N-step batching for multi-request case, since requests finish at different times and we want to admit new requests promptly).

- [ ] **Step 1: Add `decode_batch_heterogeneous` declaration to scheduler.h**

In `server/include/flashmlx/scheduler.h`, add after the `decode_batch` declaration (line 61):

```cpp
    void decode_batch_heterogeneous(const std::vector<std::string>& ids,
                                    std::unordered_map<std::string, std::vector<int>>& new_tokens,
                                    std::vector<std::string>& done_ids);
```

- [ ] **Step 2: Replace the decode section of `step()` in scheduler.cpp**

Replace lines 36-73 of `scheduler.cpp` (the entire "Batched decode" section after the admission phase) with:

```cpp
    // 2. Decode: batch all decoding requests together
    std::vector<std::string> done_ids;

    std::vector<std::string> decoding_ids;
    for (auto& [id, req] : active_) {
        if (req.state == RequestState::DECODING) {
            decoding_ids.push_back(id);
        }
    }

    if (decoding_ids.size() == 1) {
        // Single request — N-step graph batching (fastest for C=1)
        auto& req = active_[decoding_ids[0]];
        int prev_count = req.generated_count;
        decode_request(req);
        for (int i = prev_count; i < req.generated_count; i++) {
            new_tokens[decoding_ids[0]].push_back(req.output_tokens[i]);
        }
        if (req.generated_count >= req.max_tokens) {
            req.state = RequestState::DONE;
            done_ids.push_back(decoding_ids[0]);
        }
    } else if (decoding_ids.size() > 1) {
        // Multiple requests — heterogeneous batched decode
        decode_batch_heterogeneous(decoding_ids, new_tokens, done_ids);
    }
```

- [ ] **Step 3: Implement `decode_batch_heterogeneous` in scheduler.cpp**

Add after the existing `decode_batch` method (after line 339):

```cpp
void BatchScheduler::decode_batch_heterogeneous(
    const std::vector<std::string>& ids,
    std::unordered_map<std::string, std::vector<int>>& new_tokens,
    std::vector<std::string>& done_ids) {

    int B = static_cast<int>(ids.size());
    int num_layers = pool_.num_layers();
    int max_kv_len = pool_.max_context_len();

    // 1. Build batched input_ids [B, 1]
    std::vector<int> tok_ids;
    tok_ids.reserve(B);
    for (auto& id : ids) {
        tok_ids.push_back(active_[id].next_token.item<int>());
    }
    auto input_ids = mx::array(tok_ids.data(), {B, 1}, mx::int32);

    // 2. Build per-sequence offsets [B]
    std::vector<int> offset_vals;
    offset_vals.reserve(B);
    for (auto& id : ids) {
        offset_vals.push_back(active_[id].cache_offset);
    }
    auto offsets = mx::array(offset_vals.data(), {B}, mx::int32);

    // 3. Build batched KV caches from pool slots [B, n_kv, max_kv_len, hd]
    //    Pool caches are already [1, n_kv, max_kv_len, hd] — just concatenate
    std::vector<mx::array> batch_cache_k, batch_cache_v;
    batch_cache_k.reserve(num_layers);
    batch_cache_v.reserve(num_layers);

    for (int l = 0; l < num_layers; l++) {
        std::vector<mx::array> k_parts, v_parts;
        for (auto& id : ids) {
            int slot = active_[id].kv_slot;
            k_parts.push_back(pool_.keys(slot, l));
            v_parts.push_back(pool_.values(slot, l));
        }
        batch_cache_k.push_back(mx::concatenate(k_parts, 0));
        batch_cache_v.push_back(mx::concatenate(v_parts, 0));
    }

    // 4. Heterogeneous forward pass
    mx::array logits = model_.forward_heterogeneous(
        input_ids, batch_cache_k, batch_cache_v, offsets, max_kv_len);

    // 5. Split updated KV caches back to pool slots
    for (int l = 0; l < num_layers; l++) {
        for (int b = 0; b < B; b++) {
            int slot = active_[ids[b]].kv_slot;
            pool_.keys(slot, l) = mx::slice(batch_cache_k[l], {b, 0, 0, 0},
                {b + 1, batch_cache_k[l].shape(1), batch_cache_k[l].shape(2), batch_cache_k[l].shape(3)});
            pool_.values(slot, l) = mx::slice(batch_cache_v[l], {b, 0, 0, 0},
                {b + 1, batch_cache_v[l].shape(1), batch_cache_v[l].shape(2), batch_cache_v[l].shape(3)});
        }
    }

    // 6. Sample tokens per-request
    for (int b = 0; b < B; b++) {
        auto& req = active_[ids[b]];
        auto req_logits = mx::slice(logits, {b, 0, 0}, {b + 1, 1, logits.shape(2)});
        req_logits = mx::reshape(req_logits, {1, -1});

        mx::array token = sample_token(req_logits, req.temperature);
        mx::eval({token});
        int tok_id = token.item<int>();

        req.next_token = token;
        req.output_tokens.push_back(tok_id);
        req.generated_count++;
        req.cache_offset++;

        new_tokens[ids[b]].push_back(tok_id);

        if (req.generated_count >= req.max_tokens) {
            req.state = RequestState::DONE;
            done_ids.push_back(ids[b]);
        }
    }
}
```

- [ ] **Step 4: Build to verify**

Run: `cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -10`
Expected: Compiles successfully

- [ ] **Step 5: Commit**

```bash
git add server/include/flashmlx/scheduler.h server/src/scheduler.cpp
git commit -m "feat: continuous batching — batch all decoding requests regardless of offset"
```

---

### Task 5: Update KV cache handling for pre-allocated buffers

**Files:**
- Modify: `server/src/scheduler.cpp` (`prefill_request` method, lines 141-148)

The current prefill path trims KV caches to `[1, n_kv, prompt_len, hd]` after prefill (line 146-147). This was needed for the concat-based int-offset decode path, which appends tokens by concatenation. But the heterogeneous path expects full `[1, n_kv, max_context_len, hd]` buffers — it writes at arbitrary positions via scatter.

Change prefill to write into the pre-allocated buffer using `slice_update` instead of trimming, preserving the full buffer size.

- [ ] **Step 1: Modify `prefill_request` to use slice_update instead of trimming**

Replace lines 139-148 of `scheduler.cpp` (from the forward call through the cache write-back) with:

```cpp
    // Forward pass — use the existing array-offset path for prefill
    // But we need the caches to stay at full max_context_len size for heterogeneous decode.
    // Strategy: forward produces caches with data at [0:prompt_len].
    // We write this data back into the pool's pre-allocated buffers using slice_update.

    // Create temporary caches for prefill (start empty, will be filled by forward)
    int n_kv = model_.config().num_key_value_heads;
    int hd = model_.config().head_dim > 0 ? model_.config().head_dim
             : model_.config().hidden_size / model_.config().num_attention_heads;
    int max_ctx = pool_.max_context_len();

    // Use fresh zero caches for prefill (the array-offset path uses slice_update internally)
    std::vector<mx::array> prefill_keys, prefill_vals;
    prefill_keys.reserve(num_layers);
    prefill_vals.reserve(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        prefill_keys.push_back(mx::zeros({1, n_kv, max_ctx, hd}, model_.config().activation_dtype));
        prefill_vals.push_back(mx::zeros({1, n_kv, max_ctx, hd}, model_.config().activation_dtype));
    }

    mx::array logits = model_.forward(input_ids, prefill_keys, prefill_vals, cache_offsets);

    // Write prefilled caches back to pool (they're already max_ctx sized)
    for (int l = 0; l < num_layers; ++l) {
        pool_.keys(slot, l) = prefill_keys[l];
        pool_.values(slot, l) = prefill_vals[l];
    }
```

Also remove the now-unnecessary `n_kv` and `hd` computation that was at lines 142-144, since we moved it earlier.

- [ ] **Step 2: Update `decode_request` (N-step single path) to work with full buffers**

The N-step path at lines 232-243 gathers caches from pool and passes them to `model_.forward(... int cache_offset)`. The int-offset attention path uses concat: `cache_k = mx::concatenate({cache_k, k}, 2)`. This would grow the buffer beyond `max_context_len`.

We need to change the N-step path to also use the full pre-allocated buffers. Replace the cache gathering and writing in the N-step loop (lines 234-244) to use `slice_update` to insert the new token at the right position, and pass the full buffer to attention.

Actually, the simplest approach: keep the int-offset path with concat for single-request N-step batching. After N steps of concat, the caches grow from `[1, n_kv, offset, hd]` to `[1, n_kv, offset+N, hd]`. At the end, we write this back into the pool's full buffer using `slice_update`.

Modify `decode_request` for dense models. After the existing N-step loop (after line 258), replace the implicit cache write-back with an explicit `slice_update` into the pool's full buffer:

```cpp
        // After N-step loop: caches grew via concat to [1, n_kv, offset+N, hd]
        // Write back into pool's full pre-allocated buffer
        int new_len = req.cache_offset;  // cache_offset was incremented N times in the loop above
        int max_ctx = pool_.max_context_len();
        for (int l = 0; l < num_layers; ++l) {
            // pool_.keys(slot, l) is [1, n_kv, max_ctx, hd]
            // step produced [1, n_kv, new_len, hd] — write into the first new_len positions
            pool_.keys(slot, l) = mx::slice_update(
                pool_.keys(slot, l),
                mx::slice(cache_keys_final[l], {0,0,0,0}, {1, cache_keys_final[l].shape(1), new_len, cache_keys_final[l].shape(3)}),
                {0, 0, 0, 0}, {1, cache_keys_final[l].shape(1), new_len, cache_keys_final[l].shape(3)});
            pool_.values(slot, l) = mx::slice_update(
                pool_.values(slot, l),
                mx::slice(cache_values_final[l], {0,0,0,0}, {1, cache_values_final[l].shape(1), new_len, cache_values_final[l].shape(3)}),
                {0, 0, 0, 0}, {1, cache_values_final[l].shape(1), new_len, cache_values_final[l].shape(3)});
        }
```

Wait — this is getting complex. A simpler approach for the N-step path: before the loop, slice the pool cache down to `[1, n_kv, offset, hd]` (the valid portion). Run the loop with concat as before. After the loop, write the grown cache back into the full buffer.

Restructure `decode_request` for dense models (replace lines 224-259):

```cpp
    } else {
        // Dense models: N-step graph batching
        int batch_n = 32;
        int N = std::min(batch_n, req.max_tokens - req.generated_count);
        int start_offset = req.cache_offset;

        // Slice pool caches to valid portion for concat-based decode
        std::vector<mx::array> cache_keys_local, cache_values_local;
        cache_keys_local.reserve(num_layers); cache_values_local.reserve(num_layers);
        int n_kv = model_.config().num_key_value_heads;
        int hd_val = model_.config().head_dim > 0 ? model_.config().head_dim
                 : model_.config().hidden_size / model_.config().num_attention_heads;
        for (int l = 0; l < num_layers; ++l) {
            cache_keys_local.push_back(mx::slice(pool_.keys(slot, l),
                {0, 0, 0, 0}, {1, n_kv, start_offset, hd_val}));
            cache_values_local.push_back(mx::slice(pool_.values(slot, l),
                {0, 0, 0, 0}, {1, n_kv, start_offset, hd_val}));
        }

        std::vector<mx::array> step_tokens;
        mx::array prev_token = req.next_token;

        for (int s = 0; s < N; s++) {
            auto input_ids = mx::reshape(prev_token, {1, 1});
            // Pass local caches — forward with int offset will concat
            mx::array logits = model_.forward(input_ids, cache_keys_local, cache_values_local,
                                              start_offset + s);
            mx::array last_logits = mx::reshape(logits, {1, -1});
            mx::array token = sample_token(last_logits, req.temperature);
            step_tokens.push_back(token);
            prev_token = token;
        }

        mx::async_eval(step_tokens);
        for (auto& tok : step_tokens) {
            int tok_id = tok.item<int>();
            req.output_tokens.push_back(tok_id);
            req.generated_count++;
            req.cache_offset++;
        }
        req.next_token = step_tokens.back();

        // Write grown caches back into pool's pre-allocated buffers
        int new_len = req.cache_offset;
        int max_ctx = pool_.max_context_len();
        for (int l = 0; l < num_layers; ++l) {
            pool_.keys(slot, l) = mx::slice_update(
                mx::zeros({1, n_kv, max_ctx, hd_val}, model_.config().activation_dtype),
                cache_keys_local[l],
                {0, 0, 0, 0}, {1, n_kv, new_len, hd_val});
            pool_.values(slot, l) = mx::slice_update(
                mx::zeros({1, n_kv, max_ctx, hd_val}, model_.config().activation_dtype),
                cache_values_local[l],
                {0, 0, 0, 0}, {1, n_kv, new_len, hd_val});
        }
    }
```

- [ ] **Step 3: Build and run tests**

Run:
```bash
cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -5
MODEL_PATH=~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/$(ls ~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/)
./test_heterogeneous_attention "$MODEL_PATH"
```
Expected: PASS

- [ ] **Step 4: End-to-end server test**

Start the server and run a quick smoke test:
```bash
python -m server.run "$MODEL_PATH" --port 8080 &
sleep 3
# Sequential request
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10,"temperature":0,"stream":false}'
# Two concurrent requests
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10,"temperature":0,"stream":false}' &
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10,"temperature":0,"stream":false}' &
wait
pkill -f server.run
```
Expected: All three requests return valid JSON with completion tokens

- [ ] **Step 5: Commit**

```bash
git add server/src/scheduler.cpp
git commit -m "feat: use pre-allocated KV buffers for both prefill and decode paths"
```

---

### Task 6: Benchmark and validate

**Files:**
- None created or modified — benchmark only

Run the comparison benchmark to measure the impact of continuous batching.

- [ ] **Step 1: Build the final binary**

```bash
cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
```

- [ ] **Step 2: Start FlashMLX server**

```bash
MODEL_PATH=~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/$(ls ~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/)
source .venv/bin/activate && python -m server.run "$MODEL_PATH" --port 8080 &
sleep 3
```

- [ ] **Step 3: Run benchmark at 64 tokens**

```bash
python3.12 -c "
import json, time, urllib.request, concurrent.futures
URL = 'http://localhost:8080/v1/chat/completions'
PROMPT = 'Explain how transformer attention works in 3 sentences.'
def send(p, mt):
    data = json.dumps({'messages': [{'role': 'user', 'content': p}], 'max_tokens': mt, 'temperature': 0, 'stream': False}).encode()
    req = urllib.request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    return result.get('usage',{}).get('completion_tokens',0), time.perf_counter()-t0

for mt in [64, 256]:
    print(f'=== {mt} tokens ===')
    # C=1
    t0 = time.perf_counter()
    toks = sum(send(PROMPT, mt)[0] for _ in range(3))
    print(f'  C=1: {toks/(time.perf_counter()-t0):.1f} tok/s')
    for c in [2, 4, 8]:
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(c) as p:
            results = list(p.map(lambda _: send(PROMPT, mt), range(c)))
        toks = sum(r[0] for r in results)
        print(f'  C={c}: {toks/(time.perf_counter()-t0):.1f} tok/s')
"
```

- [ ] **Step 4: Compare results**

Record the numbers and compare against the baseline:

| | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| **Before (64 tok)** | 250 | 298 | 305 | 209 |
| **After (64 tok)** | ? | ? | ? | ? |
| **Before (256 tok)** | 277 | 282 | 338 | 237 |
| **After (256 tok)** | ? | ? | ? | ? |
| **vllm-mlx (64 tok)** | 150 | 183 | 210 | 210 |
| **vllm-mlx (256 tok)** | 242 | 338 | 409 | 440 |

**Success criteria:**
- C=1 should be within 5% of baseline (N-step path unchanged)
- C=4 and C=8 should improve significantly (no offset-grouping bottleneck)
- C=8/256tok target: >350 tok/s (up from 237)

- [ ] **Step 5: Kill server and commit benchmark results to README if satisfied**

```bash
pkill -f server.run
```

If results are good, update the benchmark table in README.md.

---

## Known Risks

1. **`mx::put_along_axis` performance** — Scatter writes may be slower than the concat approach for single-request decode. This is why we keep the N-step concat path for C=1. If scatter is too slow for the batched path, an alternative is to pad all caches to `max_context_len` and use `slice_update` with evaluated offsets (one `mx::eval` per step, but shared across all sequences).

2. **SDPA over full `max_context_len`** — The heterogeneous path runs attention over the full pre-allocated buffer (2048 tokens) even if most positions are padding. This wastes compute proportional to the padding ratio. For short sequences on large `max_context_len`, this could be slower than the old approach. Mitigation: use `max_kv_len = max(offsets) + 1` instead of `pool_.max_context_len()` and slice caches to that length before SDPA.

3. **Correctness with bfloat16 masks** — The mask uses `config_.activation_dtype` which may be bfloat16. Verify that `-inf` in bfloat16 works correctly with SDPA. If not, use float32 for the mask.
