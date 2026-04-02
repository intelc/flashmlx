# Prefix Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add prompt prefix caching so that requests sharing the same prompt tokens skip prefill and reuse pre-computed KV caches, closing the remaining throughput gap with vllm-mlx at C=8.

**Architecture:** The C++ scheduler maintains an LRU cache mapping prompt token hashes to pre-computed KV arrays. After each `prefill_request()`, it stores a copy of the KV data. Before each prefill, it checks the cache — on a hit, it writes cached KV to the pool slot, runs a single-token forward for the last prompt token to get logits for sampling, and skips the full prefill. Entirely transparent to the Python layer (no API changes). Also add `mx::clear_cache()` on request completion to free Metal command buffers.

**Tech Stack:** C++17, MLX C++ API, no Python changes needed

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `server/include/flashmlx/scheduler.h` | Add `CachedPrefill` struct, `prefix_cache_`, `hash_tokens()` |
| Modify | `server/src/scheduler.cpp` | Cache check/store in `prefill_request()` |
| Modify | `server/include/flashmlx/kv_pool.h` | Add `write_slot()` for bulk KV write |
| Modify | `server/src/kv_pool.cpp` | Implement `write_slot()` |
| Modify | `server/src/engine.cpp` | Add `mx::clear_cache()` on completion |

---

### Task 1: Add `write_slot` to KVCachePool

**Files:**
- Modify: `server/include/flashmlx/kv_pool.h`
- Modify: `server/src/kv_pool.cpp`

- [ ] **Step 1: Add declaration to kv_pool.h**

Add after `max_context_len()`:

```cpp
    /// Bulk-write all layers' KV caches for a slot from external arrays
    void write_slot(int slot_idx,
                    const std::vector<mx::array>& keys,
                    const std::vector<mx::array>& values);
```

- [ ] **Step 2: Implement in kv_pool.cpp**

Add after `free()`:

```cpp
void KVCachePool::write_slot(int slot_idx,
                              const std::vector<mx::array>& keys,
                              const std::vector<mx::array>& values) {
    for (int l = 0; l < num_layers_; ++l) {
        cache_keys_[slot_idx][l] = keys[l];
        cache_values_[slot_idx][l] = values[l];
    }
}
```

- [ ] **Step 3: Build to verify**

Run: `cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -3`

- [ ] **Step 4: Commit**

```bash
git add server/include/flashmlx/kv_pool.h server/src/kv_pool.cpp
git commit -m "feat: add write_slot to KVCachePool for bulk KV injection"
```

---

### Task 2: Add prefix cache to scheduler

**Files:**
- Modify: `server/include/flashmlx/scheduler.h`
- Modify: `server/src/scheduler.cpp`

This is the core task. The scheduler caches KV data after prefill and reuses it on subsequent requests with identical prompt tokens.

- [ ] **Step 1: Add cache data structures to scheduler.h**

Add to the private section (after `completed_`):

```cpp
    // Prefix cache: hash of prompt tokens -> pre-computed KV data
    struct CachedPrefill {
        std::vector<mx::array> keys;    // [1, n_kv, max_ctx, hd] per layer
        std::vector<mx::array> values;
        int offset;                      // prompt length
    };
    std::unordered_map<size_t, CachedPrefill> prefix_cache_;
    static constexpr int kPrefixCacheMaxEntries = 32;

    static size_t hash_tokens(const std::vector<int>& tokens);
```

- [ ] **Step 2: Implement hash function in scheduler.cpp**

Add after the constructor:

```cpp
size_t BatchScheduler::hash_tokens(const std::vector<int>& tokens) {
    size_t seed = tokens.size();
    for (auto& t : tokens) {
        seed ^= std::hash<int>{}(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
```

- [ ] **Step 3: Add cache check at the start of `prefill_request()`**

In `prefill_request()`, after `req.state = RequestState::PREFILLING;` (line ~116), add the cache check:

```cpp
    // Check prefix cache
    size_t token_hash = hash_tokens(req.prompt_tokens);
    auto cache_it = prefix_cache_.find(token_hash);
    if (cache_it != prefix_cache_.end()) {
        auto& cached = cache_it->second;
        // Write cached KV data to pool slot
        pool_.write_slot(slot, cached.keys, cached.values);

        // Run a single-token forward on the last prompt token to get logits for sampling.
        // Use the int-offset path with sliced caches (concat-based).
        int n_kv = model_.config().num_key_value_heads;
        int hd = model_.config().head_dim > 0 ? model_.config().head_dim
                 : model_.config().hidden_size / model_.config().num_attention_heads;
        int max_ctx = pool_.max_context_len();

        auto last_token = mx::array({req.prompt_tokens.back()}, {1, 1}, mx::int32);
        std::vector<mx::array> ck, cv;
        ck.reserve(num_layers); cv.reserve(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            ck.push_back(mx::slice(pool_.keys(slot, l),
                {0, 0, 0, 0}, {1, n_kv, cached.offset - 1, hd}));
            cv.push_back(mx::slice(pool_.values(slot, l),
                {0, 0, 0, 0}, {1, n_kv, cached.offset - 1, hd}));
        }
        mx::array logits = model_.forward(last_token, ck, cv, cached.offset - 1);

        // Write grown caches back to pool (concat added 1 token)
        int new_len = cached.offset;
        for (int l = 0; l < num_layers; ++l) {
            auto full_buf = mx::zeros({1, n_kv, max_ctx, hd}, model_.config().activation_dtype);
            pool_.keys(slot, l) = mx::slice_update(full_buf, ck[l],
                {0, 0, 0, 0}, {1, n_kv, new_len, hd});
            full_buf = mx::zeros({1, n_kv, max_ctx, hd}, model_.config().activation_dtype);
            pool_.values(slot, l) = mx::slice_update(full_buf, cv[l],
                {0, 0, 0, 0}, {1, n_kv, new_len, hd});
        }

        // Sample first token
        int logits_seq_len = logits.shape(1);
        mx::array last_logits = mx::slice(logits, {0, logits_seq_len - 1, 0},
                                           {1, logits_seq_len, logits.shape(2)});
        last_logits = mx::reshape(last_logits, {1, -1});
        mx::array token = sample_token(last_logits, req.temperature);
        mx::eval({token});
        int tok_id = token.item<int>();

        req.next_token = token;
        req.output_tokens.push_back(tok_id);
        req.generated_count = 1;
        req.cache_offset = cached.offset;
        req.state = RequestState::DECODING;
        return;
    }
```

- [ ] **Step 4: Add cache storage at end of `prefill_request()`**

At the very end of `prefill_request()`, after `req.state = RequestState::DECODING;`, add:

```cpp
    // Store in prefix cache for future reuse
    {
        std::vector<mx::array> cached_k, cached_v;
        cached_k.reserve(num_layers);
        cached_v.reserve(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            cached_k.push_back(mx::copy(pool_.keys(slot, l)));
            cached_v.push_back(mx::copy(pool_.values(slot, l)));
        }
        mx::eval(cached_k);
        mx::eval(cached_v);
        prefix_cache_[token_hash] = CachedPrefill{
            std::move(cached_k), std::move(cached_v), req.cache_offset
        };
        // Evict oldest if over capacity
        if ((int)prefix_cache_.size() > kPrefixCacheMaxEntries) {
            prefix_cache_.erase(prefix_cache_.begin());
        }
    }
```

Note: `token_hash` is computed at the top of the function but after the cache check. Move the hash computation above the cache check (it's already there from step 3).

- [ ] **Step 5: Build and test correctness**

```bash
cd server/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)
./test_heterogeneous_attention $MODEL_PATH
```
Expected: PASS (prefix cache doesn't affect heterogeneous attention test)

- [ ] **Step 6: Server smoke test with cache**

```bash
cp server/build/_flashmlx_engine*.so server/python/
MODEL_PATH=~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/$(ls ~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/)
source .venv/bin/activate && python -m server.run "$MODEL_PATH" --port 8081 &
sleep 3

# First request: cache miss, full prefill
curl -s http://localhost:8081/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10,"temperature":0,"stream":false}'
echo

# Second request, same prompt: cache hit, skip prefill
curl -s http://localhost:8081/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10,"temperature":0,"stream":false}'
echo

# Third request, different prompt: cache miss
curl -s http://localhost:8081/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10,"temperature":0,"stream":false}'

pkill -f server.run
```
Expected: All return valid JSON. First two return identical content (deterministic at temp=0).

- [ ] **Step 7: Commit**

```bash
git add server/include/flashmlx/scheduler.h server/src/scheduler.cpp
git commit -m "feat: C++-side prefix cache — skip prefill on repeated prompts"
```

---

### Task 3: Add mx::clear_cache() on request completion

**Files:**
- Modify: `server/src/engine.cpp`

- [ ] **Step 1: Add cache clearing after completed requests**

In `engine.cpp`, in the `loop()` method, after the completed requests block (after `output_queue_.push(TokenOutput{req_id, {}, true});`), add:

```cpp
        // Free Metal command buffer cache after completing requests
        if (!completed.empty()) {
            mx::clear_cache();
        }
```

The `mx::clear_cache()` function is in `<mlx/mlx.h>` which is already transitively included.

- [ ] **Step 2: Build to verify**

Run: `cd server/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -3`

- [ ] **Step 3: Commit**

```bash
git add server/src/engine.cpp
git commit -m "feat: clear Metal cache on request completion"
```

---

### Task 4: Benchmark with prefix caching

**Files:**
- None modified

- [ ] **Step 1: Build and start server**

```bash
cd server/build && make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
MODEL_PATH=~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/$(ls ~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/)
source .venv/bin/activate && python -m server.run "$MODEL_PATH" --port 8081 &
sleep 3
```

- [ ] **Step 2: Run benchmark (cache primed first)**

```bash
python3.12 -c "
import json, time, urllib.request, concurrent.futures
URL = 'http://localhost:8081/v1/chat/completions'
PROMPT = 'Explain how transformer attention works in 3 sentences.'
def send(p, mt):
    data = json.dumps({'messages': [{'role': 'user', 'content': p}], 'max_tokens': mt, 'temperature': 0, 'stream': False}).encode()
    req = urllib.request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    return result.get('usage',{}).get('completion_tokens',0), time.perf_counter()-t0

# Prime cache
send(PROMPT, 5)
print('Cache primed')

for mt in [64, 256]:
    print(f'\n=== {mt} tokens ===')
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
pkill -f server.run
```

- [ ] **Step 3: Record and compare**

Target numbers (Qwen3-0.6B-4bit, M1 Max):

| | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| **FlashMLX baseline** | 250 | 298 | 305 | 209 |
| **+ continuous batching** | 272 | 272 | 363 | 393 |
| **+ prefix cache** | ? | ? | ? | ? |
| **vllm-mlx (64 tok)** | 150 | 183 | 210 | 210 |
| **vllm-mlx (256 tok)** | 242 | 338 | 409 | 440 |

**Success criteria:** C=8/256tok should exceed 440 tok/s (match or beat vllm-mlx) because we now skip prefill for 7 of 8 requests.

---

## Known Risks

1. **Hash collisions** — `hash_tokens` uses boost-style hash combining. For a 32-entry cache the collision probability is negligible. If needed, store the full token vector for verification.

2. **Memory usage** — Each cache entry stores full KV arrays. For Qwen3-0.6B (28 layers, 8 KV heads, 128 head dim, bfloat16, max_ctx=2048): `28 × 2 × 1 × 8 × 2048 × 128 × 2 bytes ≈ 230MB` per entry. 32 entries = ~7.4GB. Adjust `kPrefixCacheMaxEntries` for memory-constrained systems.

3. **Cache stores post-prefill KV (includes all prompt tokens)** — The cached KV is from after the array-offset prefill path. On cache hit, we re-run the last prompt token through the int-offset concat path to get sampling logits. This adds one forward pass cost but avoids the full prompt prefill.

4. **Thread safety** — `prefix_cache_` is only accessed from `prefill_request()` within `step()`, which runs exclusively on the engine background thread. No synchronization needed.
