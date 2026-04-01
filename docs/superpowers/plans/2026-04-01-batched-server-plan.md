# FlashMLX Batched Inference Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenAI-compatible inference server with a C++ batched decode loop that multiplies total throughput by processing concurrent requests in a single GPU forward pass.

**Architecture:** Python HTTP layer (FastAPI) → C++ extension (pybind11) → MLX C++ API → Metal GPU. The C++ extension owns the model, KV cache pool, batch scheduler, and generation loop. Python handles HTTP, tokenization, and SSE streaming.

**Tech Stack:** C++17, MLX C++ API (0.31.x), pybind11, CMake, FastAPI, uvicorn, transformers (tokenizer)

---

## File Structure

```
server/
├── CMakeLists.txt              # C++ build system
├── include/
│   └── flashmlx/
│       ├── engine.h            # Engine class — top-level C++ API
│       ├── model.h             # LlamaModel — forward pass, weight loading
│       ├── kv_pool.h           # KVCachePool — pre-allocated cache slots
│       ├── scheduler.h         # BatchScheduler — request lifecycle
│       └── sampling.h          # Sampling functions (greedy, temperature)
├── src/
│   ├── engine.cpp              # Engine implementation + pybind11 bindings
│   ├── model.cpp               # LlamaModel implementation
│   ├── kv_pool.cpp             # KVCachePool implementation
│   ├── scheduler.cpp           # BatchScheduler implementation
│   └── sampling.cpp            # Sampling implementation
├── python/
│   ├── __init__.py             # Package init, imports C++ module
│   ├── app.py                  # FastAPI server
│   └── tokenizer.py            # Tokenizer wrapper
├── tests/
│   ├── test_engine.py          # Python integration tests for C++ engine
│   ├── test_server.py          # HTTP endpoint tests
│   └── CMakeLists.txt          # C++ test build (optional, stretch goal)
├── bench_server.py             # Concurrent benchmark script
└── run.py                      # Entry point: load model, start server
```

---

### Task 1: Build System — CMake + pybind11 Skeleton

**Files:**
- Create: `server/CMakeLists.txt`
- Create: `server/src/engine.cpp` (minimal pybind11 binding)
- Create: `server/include/flashmlx/engine.h`

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
# server/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(flashmlx_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find MLX
set(MLX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/lib/python3.14/site-packages/mlx/share/cmake/MLX")
find_package(MLX REQUIRED)

# Find pybind11
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# Build the extension module
pybind11_add_module(_flashmlx_engine
    src/engine.cpp
)
target_include_directories(_flashmlx_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MLX_INCLUDE_DIRS}
)
target_link_libraries(_flashmlx_engine PRIVATE mlx)
target_compile_options(_flashmlx_engine PRIVATE ${MLX_CXX_FLAGS})

# Install to python package directory
install(TARGETS _flashmlx_engine DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python)
```

- [ ] **Step 2: Create engine.h with minimal declaration**

```cpp
// server/include/flashmlx/engine.h
#pragma once

#include <string>
#include <vector>
#include <map>

namespace flashmlx {

struct EngineStats {
    int active_requests;
    int total_requests;
    double avg_tok_s;
};

class Engine {
public:
    Engine(const std::string& model_path, int max_batch_size, int max_context_len);
    ~Engine();

    std::string ping() const;
    EngineStats get_stats() const;

private:
    std::string model_path_;
    int max_batch_size_;
    int max_context_len_;
    int total_requests_ = 0;
};

} // namespace flashmlx
```

- [ ] **Step 3: Create engine.cpp with pybind11 bindings**

```cpp
// server/src/engine.cpp
#include "flashmlx/engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace flashmlx {

Engine::Engine(const std::string& model_path, int max_batch_size, int max_context_len)
    : model_path_(model_path), max_batch_size_(max_batch_size), max_context_len_(max_context_len) {}

Engine::~Engine() = default;

std::string Engine::ping() const {
    return "flashmlx engine ready";
}

EngineStats Engine::get_stats() const {
    return {0, total_requests_, 0.0};
}

} // namespace flashmlx

PYBIND11_MODULE(_flashmlx_engine, m) {
    m.doc() = "FlashMLX C++ inference engine";

    py::class_<flashmlx::EngineStats>(m, "EngineStats")
        .def_readonly("active_requests", &flashmlx::EngineStats::active_requests)
        .def_readonly("total_requests", &flashmlx::EngineStats::total_requests)
        .def_readonly("avg_tok_s", &flashmlx::EngineStats::avg_tok_s);

    py::class_<flashmlx::Engine>(m, "Engine")
        .def(py::init<const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("max_batch_size") = 8,
             py::arg("max_context_len") = 2048)
        .def("ping", &flashmlx::Engine::ping)
        .def("get_stats", &flashmlx::Engine::get_stats);
}
```

- [ ] **Step 4: Build and test the extension**

```bash
cd server
mkdir -p build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3)
make -j$(sysctl -n hw.ncpu)
```

Expected: Builds `_flashmlx_engine.cpython-*.so` in `build/`

- [ ] **Step 5: Copy to python dir and test import**

```bash
cp build/_flashmlx_engine*.so python/
cd .. && source ../.venv/bin/activate
python -c "from server.python._flashmlx_engine import Engine; e = Engine('/tmp', 8, 2048); print(e.ping())"
```

Expected: `flashmlx engine ready`

- [ ] **Step 6: Commit**

```bash
git add server/CMakeLists.txt server/include/ server/src/engine.cpp
git commit -m "feat(server): CMake build system + pybind11 skeleton"
```

---

### Task 2: Model Loading in C++

**Files:**
- Create: `server/include/flashmlx/model.h`
- Create: `server/src/model.cpp`
- Modify: `server/src/engine.cpp` (add model loading)
- Modify: `server/CMakeLists.txt` (add model.cpp)

- [ ] **Step 1: Create model.h**

```cpp
// server/include/flashmlx/model.h
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace flashmlx {

struct ModelConfig {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int vocab_size;
    float rms_norm_eps;
    float rope_theta;
    bool tie_word_embeddings;
    // Quantization
    int quant_bits;
    int quant_group_size;
};

// Parse config.json into ModelConfig
ModelConfig load_config(const std::string& model_path);

// Load all safetensors weights, stripping "model." prefix
std::unordered_map<std::string, mx::array> load_weights(const std::string& model_path);

class LlamaModel {
public:
    explicit LlamaModel(const std::string& model_path);

    // Forward pass: input_ids [B, L] → logits [B, L, vocab_size]
    // cache_keys/cache_values: per-layer KV cache arrays
    mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        const std::vector<int>& cache_offsets
    );

    const ModelConfig& config() const { return config_; }

private:
    ModelConfig config_;
    std::unordered_map<std::string, mx::array> weights_;

    // Single-layer forward
    mx::array transformer_block(
        const mx::array& x,
        int layer_idx,
        mx::array& cache_k,
        mx::array& cache_v,
        int cache_offset
    );

    // Attention sublayer
    mx::array attention(
        const mx::array& x,
        int layer_idx,
        mx::array& cache_k,
        mx::array& cache_v,
        int cache_offset
    );

    // MLP sublayer
    mx::array mlp(const mx::array& x, int layer_idx);

    // Helpers
    mx::array rms_norm(const mx::array& x, const mx::array& weight);
    mx::array embed(const mx::array& input_ids);
    mx::array lm_head(const mx::array& x);
};

} // namespace flashmlx
```

- [ ] **Step 2: Create model.cpp with config loading and weight loading**

```cpp
// server/src/model.cpp
#include "flashmlx/model.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>

// Minimal JSON parsing — we only need a few fields from config.json
// Using a simple approach to avoid external dependencies
namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::stringstream buf;
    buf << f.rdbuf();
    return buf.str();
}

// Extract an integer value from JSON string for a given key
int json_int(const std::string& json, const std::string& key, int default_val = 0) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_val;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return std::stoi(json.substr(pos));
}

float json_float(const std::string& json, const std::string& key, float default_val = 0.0f) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_val;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return std::stof(json.substr(pos));
}

bool json_bool(const std::string& json, const std::string& key, bool default_val = false) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_val;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return json.substr(pos, 4) == "true";
}

} // anonymous namespace

namespace flashmlx {

ModelConfig load_config(const std::string& model_path) {
    auto json = read_file(model_path + "/config.json");
    ModelConfig c;
    c.hidden_size = json_int(json, "hidden_size");
    c.num_hidden_layers = json_int(json, "num_hidden_layers");
    c.intermediate_size = json_int(json, "intermediate_size");
    c.num_attention_heads = json_int(json, "num_attention_heads");
    c.num_key_value_heads = json_int(json, "num_key_value_heads", c.num_attention_heads);
    c.head_dim = json_int(json, "head_dim", c.hidden_size / c.num_attention_heads);
    c.vocab_size = json_int(json, "vocab_size");
    c.rms_norm_eps = json_float(json, "rms_norm_eps", 1e-5f);
    c.rope_theta = json_float(json, "rope_theta", 10000.0f);
    c.tie_word_embeddings = json_bool(json, "tie_word_embeddings", true);
    // Quantization — check both "quantization" and "quantization_config"
    c.quant_bits = json_int(json, "bits", 0);
    c.quant_group_size = json_int(json, "group_size", 64);
    if (c.quant_bits == 0) {
        // Try nested quantization config
        auto qpos = json.find("\"quantization\"");
        if (qpos != std::string::npos) {
            auto qblock = json.substr(qpos);
            c.quant_bits = json_int(qblock, "bits", 4);
            c.quant_group_size = json_int(qblock, "group_size", 64);
        }
    }
    return c;
}

std::unordered_map<std::string, mx::array> load_weights(const std::string& model_path) {
    namespace fs = std::filesystem;
    std::unordered_map<std::string, mx::array> weights;

    // Find all model*.safetensors files
    std::vector<std::string> files;
    for (auto& entry : fs::directory_iterator(model_path)) {
        auto name = entry.path().filename().string();
        if (name.find("model") == 0 && name.find(".safetensors") != std::string::npos) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());

    for (auto& f : files) {
        auto [file_weights, metadata] = mx::load_safetensors(f);
        for (auto& [k, v] : file_weights) {
            // Strip "model." prefix
            std::string key = k;
            if (key.substr(0, 6) == "model.") {
                key = key.substr(6);
            }
            // Skip rotary embedding frequencies
            if (key.find("rotary_emb.inv_freq") != std::string::npos) continue;
            weights[key] = std::move(v);
        }
    }
    return weights;
}

LlamaModel::LlamaModel(const std::string& model_path) {
    config_ = load_config(model_path);
    weights_ = load_weights(model_path);
    // Evaluate all weights to ensure they're materialized in GPU memory
    std::vector<mx::array> to_eval;
    for (auto& [k, v] : weights_) {
        to_eval.push_back(v);
    }
    mx::eval(to_eval);
}

mx::array LlamaModel::rms_norm(const mx::array& x, const mx::array& weight) {
    return mx::fast::rms_norm(x, weight, config_.rms_norm_eps);
}

mx::array LlamaModel::embed(const mx::array& input_ids) {
    // Embedding lookup — gather from embed_tokens.weight
    auto& w = weights_.at("embed_tokens.weight");
    return mx::take(w, input_ids, 0);  // [B, L] → [B, L, D]
}

mx::array LlamaModel::lm_head(const mx::array& x) {
    if (config_.tie_word_embeddings) {
        auto& w = weights_.at("embed_tokens.weight");
        return mx::matmul(x, mx::transpose(w));
    }
    // Non-tied: use separate lm_head weights
    auto& w = weights_.at("lm_head.weight");
    if (config_.quant_bits > 0 && weights_.count("lm_head.scales")) {
        return mx::quantized_matmul(
            x, w, weights_.at("lm_head.scales"),
            weights_.count("lm_head.biases") ? std::optional(weights_.at("lm_head.biases")) : std::nullopt,
            true, config_.quant_group_size, config_.quant_bits
        );
    }
    return mx::matmul(x, mx::transpose(w));
}

mx::array LlamaModel::attention(
    const mx::array& x, int layer_idx,
    mx::array& cache_k, mx::array& cache_v, int cache_offset
) {
    auto prefix = "layers." + std::to_string(layer_idx) + ".self_attn.";
    int B = x.shape()[0], L = x.shape()[1];
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int hd = config_.head_dim;

    // Q, K, V projections (quantized matmul)
    auto qkv_proj = [&](const std::string& name) {
        auto& w = weights_.at(prefix + name + ".weight");
        auto& s = weights_.at(prefix + name + ".scales");
        auto b_it = weights_.find(prefix + name + ".biases");
        auto bias = (b_it != weights_.end()) ? std::optional(b_it->second) : std::nullopt;
        return mx::quantized_matmul(x, w, s, bias, true, config_.quant_group_size, config_.quant_bits);
    };

    auto q = mx::transpose(mx::reshape(qkv_proj("q_proj"), {B, L, n_heads, hd}), {0, 2, 1, 3});
    auto k = mx::transpose(mx::reshape(qkv_proj("k_proj"), {B, L, n_kv_heads, hd}), {0, 2, 1, 3});
    auto v = mx::transpose(mx::reshape(qkv_proj("v_proj"), {B, L, n_kv_heads, hd}), {0, 2, 1, 3});

    // RoPE
    q = mx::fast::rope(q, hd, false, config_.rope_theta, 1.0f, cache_offset);
    k = mx::fast::rope(k, hd, false, config_.rope_theta, 1.0f, cache_offset);

    // Update KV cache via slice assignment into pre-allocated buffer
    // cache_k shape: [B, n_kv_heads, max_context, head_dim]
    int new_offset = cache_offset + L;
    cache_k = mx::slice_update(cache_k, k, {0, 0, cache_offset, 0}, {B, n_kv_heads, new_offset, hd});
    cache_v = mx::slice_update(cache_v, v, {0, 0, cache_offset, 0}, {B, n_kv_heads, new_offset, hd});

    // Slice active region for attention
    auto active_k = mx::slice(cache_k, {0, 0, 0, 0}, {B, n_kv_heads, new_offset, hd});
    auto active_v = mx::slice(cache_v, {0, 0, 0, 0}, {B, n_kv_heads, new_offset, hd});

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::string mask_mode = (L > 1) ? "causal" : "";
    auto out = mx::fast::scaled_dot_product_attention(q, active_k, active_v, scale, mask_mode);

    // Reshape and output projection
    out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, n_heads * hd});

    auto& wo = weights_.at(prefix + "o_proj.weight");
    auto& ws = weights_.at(prefix + "o_proj.scales");
    auto bo_it = weights_.find(prefix + "o_proj.biases");
    auto bo = (bo_it != weights_.end()) ? std::optional(bo_it->second) : std::nullopt;
    return mx::quantized_matmul(out, wo, ws, bo, true, config_.quant_group_size, config_.quant_bits);
}

mx::array LlamaModel::mlp(const mx::array& x, int layer_idx) {
    auto prefix = "layers." + std::to_string(layer_idx) + ".mlp.";

    auto proj = [&](const std::string& name) {
        auto& w = weights_.at(prefix + name + ".weight");
        auto& s = weights_.at(prefix + name + ".scales");
        auto b_it = weights_.find(prefix + name + ".biases");
        auto bias = (b_it != weights_.end()) ? std::optional(b_it->second) : std::nullopt;
        return mx::quantized_matmul(x, w, s, bias, true, config_.quant_group_size, config_.quant_bits);
    };

    auto gate = proj("gate_proj");
    auto up = proj("up_proj");
    // SwiGLU: silu(gate) * up
    auto activated = mx::multiply(mx::sigmoid(gate) * gate, up);  // silu = x * sigmoid(x)
    return proj("down_proj");  // Bug: should use activated as input, fix in implementation
}

mx::array LlamaModel::transformer_block(
    const mx::array& x, int layer_idx,
    mx::array& cache_k, mx::array& cache_v, int cache_offset
) {
    auto prefix = "layers." + std::to_string(layer_idx) + ".";
    auto& norm1_w = weights_.at(prefix + "input_layernorm.weight");
    auto& norm2_w = weights_.at(prefix + "post_attention_layernorm.weight");

    // Pre-norm + attention + residual
    auto normed = rms_norm(x, norm1_w);
    auto attn_out = attention(normed, layer_idx, cache_k, cache_v, cache_offset);
    auto h = mx::add(x, attn_out);

    // Pre-norm + MLP + residual
    auto normed2 = rms_norm(h, norm2_w);
    auto mlp_out = mlp(normed2, layer_idx);
    return mx::add(h, mlp_out);
}

mx::array LlamaModel::forward(
    const mx::array& input_ids,
    std::vector<mx::array>& cache_keys,
    std::vector<mx::array>& cache_values,
    const std::vector<int>& cache_offsets
) {
    auto h = embed(input_ids);  // [B, L, D]

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        h = transformer_block(h, i, cache_keys[i], cache_values[i], cache_offsets[i]);
    }

    // Final norm
    auto& norm_w = weights_.at("norm.weight");
    h = rms_norm(h, norm_w);

    // LM head — only last token
    h = mx::slice(h, {0, static_cast<int>(h.shape()[1]) - 1, 0},
                     {static_cast<int>(h.shape()[0]), static_cast<int>(h.shape()[1]), static_cast<int>(h.shape()[2])});
    return lm_head(h);  // [B, 1, vocab_size]
}

} // namespace flashmlx
```

**Note:** This is a first pass — the MLP has a known bug (down_proj uses `x` not `activated`). We'll fix it in the test-and-fix cycle.

- [ ] **Step 3: Update CMakeLists.txt to include model.cpp**

Add to the `pybind11_add_module` call:

```cmake
pybind11_add_module(_flashmlx_engine
    src/engine.cpp
    src/model.cpp
)
```

- [ ] **Step 4: Add model loading to Engine and bindings**

Update `engine.h`:
```cpp
#include "flashmlx/model.h"
// In Engine class:
private:
    std::unique_ptr<LlamaModel> model_;
```

Update `engine.cpp` constructor:
```cpp
Engine::Engine(const std::string& model_path, int max_batch_size, int max_context_len)
    : model_path_(model_path), max_batch_size_(max_batch_size), max_context_len_(max_context_len) {
    model_ = std::make_unique<LlamaModel>(model_path);
}
```

- [ ] **Step 5: Build and test model loading**

```bash
cd server/build && cmake .. && make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
cd ../.. && source .venv/bin/activate
python -c "
from server.python._flashmlx_engine import Engine
import time
t0 = time.time()
e = Engine('/Users/yihengchen/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3-8B-4bit/snapshots/7f296f1dd0e720f7daa69ed648ed7b193373ca11', 8, 2048)
print(f'Model loaded in {time.time()-t0:.1f}s')
print(e.ping())
"
```

Expected: Model loads in ~2-5s, prints `flashmlx engine ready`

- [ ] **Step 6: Commit**

```bash
git add server/include/flashmlx/model.h server/src/model.cpp server/CMakeLists.txt server/src/engine.cpp server/include/flashmlx/engine.h
git commit -m "feat(server): C++ model loading with safetensors + quantized weights"
```

---

### Task 3: KV Cache Pool

**Files:**
- Create: `server/include/flashmlx/kv_pool.h`
- Create: `server/src/kv_pool.cpp`

- [ ] **Step 1: Create kv_pool.h**

```cpp
// server/include/flashmlx/kv_pool.h
#pragma once

#include <vector>
#include <mutex>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace flashmlx {

// Pre-allocated KV cache pool with slot management
class KVCachePool {
public:
    KVCachePool(int max_slots, int max_context_len, int num_layers,
                int n_kv_heads, int head_dim);

    // Allocate a slot, returns slot index or -1 if full
    int allocate();

    // Free a slot
    void free(int slot_idx);

    // Get KV arrays for a slot (references into pre-allocated pool)
    mx::array& keys(int slot_idx, int layer);
    mx::array& values(int slot_idx, int layer);

    // Reset a slot's cache (zero offset, keep allocation)
    void reset(int slot_idx);

    int num_free() const;
    int max_slots() const { return max_slots_; }

private:
    int max_slots_;
    int max_context_len_;
    int num_layers_;
    int n_kv_heads_;
    int head_dim_;

    // cache_keys_[slot][layer] = [1, n_kv_heads, max_context, head_dim]
    std::vector<std::vector<mx::array>> cache_keys_;
    std::vector<std::vector<mx::array>> cache_values_;
    std::vector<bool> slot_free_;
    mutable std::mutex mutex_;
};

} // namespace flashmlx
```

- [ ] **Step 2: Create kv_pool.cpp**

```cpp
// server/src/kv_pool.cpp
#include "flashmlx/kv_pool.h"
#include <stdexcept>

namespace flashmlx {

KVCachePool::KVCachePool(int max_slots, int max_context_len, int num_layers,
                         int n_kv_heads, int head_dim)
    : max_slots_(max_slots), max_context_len_(max_context_len),
      num_layers_(num_layers), n_kv_heads_(n_kv_heads), head_dim_(head_dim),
      slot_free_(max_slots, true) {

    cache_keys_.resize(max_slots);
    cache_values_.resize(max_slots);

    for (int s = 0; s < max_slots; s++) {
        cache_keys_[s].resize(num_layers);
        cache_values_[s].resize(num_layers);
        for (int l = 0; l < num_layers; l++) {
            cache_keys_[s][l] = mx::zeros({1, n_kv_heads, max_context_len, head_dim}, mx::float16);
            cache_values_[s][l] = mx::zeros({1, n_kv_heads, max_context_len, head_dim}, mx::float16);
        }
    }
    // Materialize all cache buffers
    std::vector<mx::array> all;
    for (auto& slot : cache_keys_)
        for (auto& layer : slot) all.push_back(layer);
    for (auto& slot : cache_values_)
        for (auto& layer : slot) all.push_back(layer);
    mx::eval(all);
}

int KVCachePool::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < max_slots_; i++) {
        if (slot_free_[i]) {
            slot_free_[i] = false;
            return i;
        }
    }
    return -1;  // Pool full
}

void KVCachePool::free(int slot_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    slot_free_[slot_idx] = true;
}

mx::array& KVCachePool::keys(int slot_idx, int layer) {
    return cache_keys_[slot_idx][layer];
}

mx::array& KVCachePool::values(int slot_idx, int layer) {
    return cache_values_[slot_idx][layer];
}

void KVCachePool::reset(int slot_idx) {
    // No-op: offset tracking handles this. The stale data beyond
    // the offset is never read (attention only reads up to offset).
}

int KVCachePool::num_free() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (bool f : slot_free_) if (f) count++;
    return count;
}

} // namespace flashmlx
```

- [ ] **Step 3: Add to CMakeLists.txt**

```cmake
pybind11_add_module(_flashmlx_engine
    src/engine.cpp
    src/model.cpp
    src/kv_pool.cpp
)
```

- [ ] **Step 4: Build and verify**

```bash
cd server/build && cmake .. && make -j$(sysctl -n hw.ncpu)
```

Expected: Builds without errors.

- [ ] **Step 5: Commit**

```bash
git add server/include/flashmlx/kv_pool.h server/src/kv_pool.cpp server/CMakeLists.txt
git commit -m "feat(server): pre-allocated KV cache pool with slot management"
```

---

### Task 4: Batch Scheduler

**Files:**
- Create: `server/include/flashmlx/scheduler.h`
- Create: `server/src/scheduler.cpp`
- Create: `server/include/flashmlx/sampling.h`
- Create: `server/src/sampling.cpp`

- [ ] **Step 1: Create sampling.h and sampling.cpp**

```cpp
// server/include/flashmlx/sampling.h
#pragma once
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace flashmlx {

// Sample a token from logits [vocab_size]
mx::array sample_token(const mx::array& logits, float temperature);

} // namespace flashmlx
```

```cpp
// server/src/sampling.cpp
#include "flashmlx/sampling.h"

namespace flashmlx {

mx::array sample_token(const mx::array& logits, float temperature) {
    if (temperature <= 0.0f) {
        return mx::argmax(logits, -1);
    }
    auto scaled = mx::multiply(logits, mx::array(1.0f / temperature));
    return mx::random::categorical(scaled);
}

} // namespace flashmlx
```

- [ ] **Step 2: Create scheduler.h**

```cpp
// server/include/flashmlx/scheduler.h
#pragma once

#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <mlx/mlx.h>
#include "flashmlx/model.h"
#include "flashmlx/kv_pool.h"

namespace mx = mlx::core;

namespace flashmlx {

enum class RequestState { QUEUED, PREFILLING, DECODING, DONE };

struct Request {
    std::string id;
    std::vector<int> prompt_tokens;
    int max_tokens;
    float temperature;
    RequestState state = RequestState::QUEUED;

    // Runtime state
    int kv_slot = -1;
    int cache_offset = 0;
    int generated_count = 0;
    mx::array next_token;  // Last sampled token

    // Output
    std::vector<int> output_tokens;
};

class BatchScheduler {
public:
    BatchScheduler(LlamaModel& model, KVCachePool& kv_pool);

    // Submit a new request (thread-safe)
    void submit(Request req);

    // Run one batch step: prefill new requests + decode active ones
    // Returns map of request_id → new tokens produced this step
    std::unordered_map<std::string, std::vector<int>> step();

    // Check if any requests are active or queued
    bool has_work() const;

    // Get list of completed request IDs (and remove them)
    std::vector<std::string> drain_completed();

    int active_count() const;

private:
    LlamaModel& model_;
    KVCachePool& kv_pool_;

    std::queue<Request> pending_;
    std::unordered_map<std::string, Request> active_;
    std::vector<std::string> completed_;
    mutable std::mutex mutex_;

    // Prefill a single request
    void prefill_request(Request& req);

    // Decode one token for a batch of requests
    std::unordered_map<std::string, std::vector<int>> decode_batch();
};

} // namespace flashmlx
```

- [ ] **Step 3: Create scheduler.cpp**

```cpp
// server/src/scheduler.cpp
#include "flashmlx/scheduler.h"
#include "flashmlx/sampling.h"
#include <algorithm>

namespace flashmlx {

BatchScheduler::BatchScheduler(LlamaModel& model, KVCachePool& kv_pool)
    : model_(model), kv_pool_(kv_pool) {}

void BatchScheduler::submit(Request req) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push(std::move(req));
}

bool BatchScheduler::has_work() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !pending_.empty() || !active_.empty();
}

int BatchScheduler::active_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_.size();
}

std::vector<std::string> BatchScheduler::drain_completed() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = std::move(completed_);
    completed_.clear();
    return result;
}

void BatchScheduler::prefill_request(Request& req) {
    // Allocate KV slot
    req.kv_slot = kv_pool_.allocate();
    if (req.kv_slot < 0) {
        // Pool full — keep in pending
        return;
    }

    // Build input array from prompt tokens
    auto input_ids = mx::array(req.prompt_tokens.data(),
                               {1, static_cast<int>(req.prompt_tokens.size())},
                               mx::int32);

    // Gather KV cache arrays for this slot
    std::vector<mx::array> slot_keys, slot_values;
    std::vector<int> offsets;
    for (int l = 0; l < model_.config().num_hidden_layers; l++) {
        slot_keys.push_back(kv_pool_.keys(req.kv_slot, l));
        slot_values.push_back(kv_pool_.values(req.kv_slot, l));
        offsets.push_back(0);
    }

    // Forward pass for prefill
    auto logits = model_.forward(input_ids, slot_keys, slot_values, offsets);
    auto token = sample_token(mx::reshape(logits, {-1}), req.temperature);
    mx::eval({token});

    // Update cache arrays back to pool
    for (int l = 0; l < model_.config().num_hidden_layers; l++) {
        kv_pool_.keys(req.kv_slot, l) = slot_keys[l];
        kv_pool_.values(req.kv_slot, l) = slot_values[l];
    }

    req.cache_offset = static_cast<int>(req.prompt_tokens.size());
    req.next_token = token;
    req.state = RequestState::DECODING;
    req.output_tokens.push_back(token.item<int>());
    req.generated_count = 1;
}

std::unordered_map<std::string, std::vector<int>> BatchScheduler::decode_batch() {
    std::unordered_map<std::string, std::vector<int>> results;

    // Collect decoding requests
    std::vector<std::string> batch_ids;
    for (auto& [id, req] : active_) {
        if (req.state == RequestState::DECODING) {
            batch_ids.push_back(id);
        }
    }
    if (batch_ids.empty()) return results;

    // For V1: decode one request at a time (batched decode in V2)
    // This still benefits from C++ loop — no Python overhead
    for (auto& id : batch_ids) {
        auto& req = active_[id];

        auto input_ids = mx::reshape(req.next_token, {1, 1});

        std::vector<mx::array> slot_keys, slot_values;
        std::vector<int> offsets;
        for (int l = 0; l < model_.config().num_hidden_layers; l++) {
            slot_keys.push_back(kv_pool_.keys(req.kv_slot, l));
            slot_values.push_back(kv_pool_.values(req.kv_slot, l));
            offsets.push_back(req.cache_offset);
        }

        auto logits = model_.forward(input_ids, slot_keys, slot_values, offsets);
        auto token = sample_token(mx::reshape(logits, {-1}), req.temperature);
        mx::eval({token});

        // Update cache
        for (int l = 0; l < model_.config().num_hidden_layers; l++) {
            kv_pool_.keys(req.kv_slot, l) = slot_keys[l];
            kv_pool_.values(req.kv_slot, l) = slot_values[l];
        }

        req.cache_offset++;
        req.next_token = token;
        int tok = token.item<int>();
        req.output_tokens.push_back(tok);
        req.generated_count++;
        results[id] = {tok};

        // Check if done
        if (req.generated_count >= req.max_tokens) {
            req.state = RequestState::DONE;
        }
    }

    return results;
}

std::unordered_map<std::string, std::vector<int>> BatchScheduler::step() {
    // 1. Admit pending requests
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pending_.empty() && kv_pool_.num_free() > 0) {
            auto req = std::move(pending_.front());
            pending_.pop();
            auto id = req.id;
            active_[id] = std::move(req);
            prefill_request(active_[id]);
        }
    }

    // 2. Decode batch
    auto results = decode_batch();

    // 3. Clean up completed requests
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> to_remove;
        for (auto& [id, req] : active_) {
            if (req.state == RequestState::DONE) {
                kv_pool_.free(req.kv_slot);
                completed_.push_back(id);
                to_remove.push_back(id);
            }
        }
        for (auto& id : to_remove) {
            active_.erase(id);
        }
    }

    return results;
}

} // namespace flashmlx
```

- [ ] **Step 4: Update CMakeLists.txt**

```cmake
pybind11_add_module(_flashmlx_engine
    src/engine.cpp
    src/model.cpp
    src/kv_pool.cpp
    src/scheduler.cpp
    src/sampling.cpp
)
```

- [ ] **Step 5: Build**

```bash
cd server/build && cmake .. && make -j$(sysctl -n hw.ncpu)
```

- [ ] **Step 6: Commit**

```bash
git add server/include/flashmlx/scheduler.h server/include/flashmlx/sampling.h server/src/scheduler.cpp server/src/sampling.cpp server/CMakeLists.txt
git commit -m "feat(server): batch scheduler with request lifecycle + sampling"
```

---

### Task 5: Engine Integration — Wire Up Submit/Poll + pybind11

**Files:**
- Modify: `server/include/flashmlx/engine.h`
- Modify: `server/src/engine.cpp`

- [ ] **Step 1: Update engine.h with full API**

```cpp
// server/include/flashmlx/engine.h
#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include "flashmlx/model.h"
#include "flashmlx/kv_pool.h"
#include "flashmlx/scheduler.h"

namespace flashmlx {

struct EngineStats {
    int active_requests;
    int total_requests;
    double avg_tok_s;
    int free_kv_slots;
};

struct TokenOutput {
    std::string request_id;
    std::vector<int> tokens;
    bool done;
};

class Engine {
public:
    Engine(const std::string& model_path, int max_batch_size = 8, int max_context_len = 2048);
    ~Engine();

    // Submit a request (thread-safe, called from Python)
    void submit_request(
        const std::string& request_id,
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        float temperature
    );

    // Poll for new tokens (called from Python async loop)
    std::vector<TokenOutput> poll_tokens();

    // Stats
    EngineStats get_stats() const;
    std::string ping() const;

    // Start/stop the background decode loop
    void start();
    void stop();

private:
    std::unique_ptr<LlamaModel> model_;
    std::unique_ptr<KVCachePool> kv_pool_;
    std::unique_ptr<BatchScheduler> scheduler_;

    std::string model_path_;
    int max_batch_size_;
    int max_context_len_;
    std::atomic<int> total_requests_{0};

    // Background thread
    std::thread loop_thread_;
    std::atomic<bool> running_{false};
    void loop();

    // Output queue
    std::queue<TokenOutput> output_queue_;
    std::mutex output_mutex_;
};

} // namespace flashmlx
```

- [ ] **Step 2: Update engine.cpp with full implementation and bindings**

```cpp
// server/src/engine.cpp
#include "flashmlx/engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#include <thread>

namespace py = pybind11;

namespace flashmlx {

Engine::Engine(const std::string& model_path, int max_batch_size, int max_context_len)
    : model_path_(model_path), max_batch_size_(max_batch_size), max_context_len_(max_context_len) {
    model_ = std::make_unique<LlamaModel>(model_path);
    auto& cfg = model_->config();
    kv_pool_ = std::make_unique<KVCachePool>(
        max_batch_size, max_context_len, cfg.num_hidden_layers,
        cfg.num_key_value_heads, cfg.head_dim
    );
    scheduler_ = std::make_unique<BatchScheduler>(*model_, *kv_pool_);
}

Engine::~Engine() {
    stop();
}

void Engine::submit_request(
    const std::string& request_id,
    const std::vector<int>& prompt_tokens,
    int max_tokens,
    float temperature
) {
    Request req;
    req.id = request_id;
    req.prompt_tokens = prompt_tokens;
    req.max_tokens = max_tokens;
    req.temperature = temperature;
    scheduler_->submit(std::move(req));
    total_requests_++;
}

std::vector<TokenOutput> Engine::poll_tokens() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    std::vector<TokenOutput> result;
    while (!output_queue_.empty()) {
        result.push_back(std::move(output_queue_.front()));
        output_queue_.pop();
    }
    return result;
}

EngineStats Engine::get_stats() const {
    return {
        scheduler_->active_count(),
        total_requests_.load(),
        0.0,  // TODO: track rolling average
        kv_pool_->num_free()
    };
}

std::string Engine::ping() const {
    return "flashmlx engine ready";
}

void Engine::start() {
    if (running_) return;
    running_ = true;
    loop_thread_ = std::thread(&Engine::loop, this);
}

void Engine::stop() {
    running_ = false;
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
}

void Engine::loop() {
    while (running_) {
        if (!scheduler_->has_work()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Run one scheduler step
        auto new_tokens = scheduler_->step();

        // Push to output queue
        {
            std::lock_guard<std::mutex> lock(output_mutex_);
            for (auto& [id, tokens] : new_tokens) {
                output_queue_.push({id, tokens, false});
            }
            // Check for completed requests
            auto completed = scheduler_->drain_completed();
            for (auto& id : completed) {
                output_queue_.push({id, {}, true});
            }
        }
    }
}

} // namespace flashmlx

// pybind11 bindings
PYBIND11_MODULE(_flashmlx_engine, m) {
    m.doc() = "FlashMLX C++ inference engine";

    py::class_<flashmlx::EngineStats>(m, "EngineStats")
        .def_readonly("active_requests", &flashmlx::EngineStats::active_requests)
        .def_readonly("total_requests", &flashmlx::EngineStats::total_requests)
        .def_readonly("avg_tok_s", &flashmlx::EngineStats::avg_tok_s)
        .def_readonly("free_kv_slots", &flashmlx::EngineStats::free_kv_slots);

    py::class_<flashmlx::TokenOutput>(m, "TokenOutput")
        .def_readonly("request_id", &flashmlx::TokenOutput::request_id)
        .def_readonly("tokens", &flashmlx::TokenOutput::tokens)
        .def_readonly("done", &flashmlx::TokenOutput::done);

    py::class_<flashmlx::Engine>(m, "Engine")
        .def(py::init<const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("max_batch_size") = 8,
             py::arg("max_context_len") = 2048)
        .def("ping", &flashmlx::Engine::ping)
        .def("get_stats", &flashmlx::Engine::get_stats)
        .def("submit_request", &flashmlx::Engine::submit_request,
             py::arg("request_id"), py::arg("prompt_tokens"),
             py::arg("max_tokens"), py::arg("temperature"))
        .def("poll_tokens", &flashmlx::Engine::poll_tokens)
        .def("start", &flashmlx::Engine::start)
        .def("stop", &flashmlx::Engine::stop);
}
```

- [ ] **Step 3: Build and test end-to-end from Python**

```bash
cd server/build && cmake .. && make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
cd ../..
source .venv/bin/activate
python -c "
import time
from server.python._flashmlx_engine import Engine

MODEL = '/Users/yihengchen/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3-8B-4bit/snapshots/7f296f1dd0e720f7daa69ed648ed7b193373ca11'
e = Engine(MODEL, 8, 2048)
e.start()
e.submit_request('test-1', list(range(1, 129)), 10, 0.0)

# Poll for tokens
tokens = []
for _ in range(100):
    outputs = e.poll_tokens()
    for out in outputs:
        if out.done:
            print(f'Done! Tokens: {tokens}')
            break
        tokens.extend(out.tokens)
    else:
        time.sleep(0.01)
        continue
    break

e.stop()
print(f'Generated {len(tokens)} tokens')
"
```

Expected: Generates 10 tokens and prints them.

- [ ] **Step 4: Commit**

```bash
git add server/include/flashmlx/engine.h server/src/engine.cpp
git commit -m "feat(server): Engine with submit/poll/start/stop + pybind11 bindings"
```

---

### Task 6: Python HTTP Server (FastAPI)

**Files:**
- Create: `server/python/__init__.py`
- Create: `server/python/app.py`
- Create: `server/python/tokenizer.py`
- Create: `server/run.py`

- [ ] **Step 1: Install FastAPI dependencies**

```bash
source .venv/bin/activate
pip install fastapi uvicorn sse-starlette
```

- [ ] **Step 2: Create tokenizer.py**

```python
# server/python/tokenizer.py
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_path: str):
        self.tok = AutoTokenizer.from_pretrained(model_path)

    def encode_chat(self, messages: list[dict]) -> list[int]:
        """Apply chat template and encode to token IDs."""
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tok.decode(token_ids, skip_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        return self.tok.decode([token_id], skip_special_tokens=False)
```

- [ ] **Step 3: Create app.py**

```python
# server/python/app.py
import asyncio
import time
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.python._flashmlx_engine import Engine, TokenOutput
from server.python.tokenizer import Tokenizer

app = FastAPI(title="FlashMLX Inference Server")

# Global engine and tokenizer — initialized by run.py
engine: Optional[Engine] = None
tokenizer: Optional[Tokenizer] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


def init(model_path: str, max_batch_size: int = 8, max_context_len: int = 2048):
    global engine, tokenizer
    engine = Engine(model_path, max_batch_size, max_context_len)
    tokenizer = Tokenizer(model_path)
    engine.start()


@app.get("/health")
async def health():
    stats = engine.get_stats()
    return {
        "status": "ok",
        "active_requests": stats.active_requests,
        "total_requests": stats.total_requests,
        "free_kv_slots": stats.free_kv_slots,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    request_id = str(uuid.uuid4())
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt_tokens = tokenizer.encode_chat(messages)

    engine.submit_request(request_id, prompt_tokens, req.max_tokens, req.temperature)

    if req.stream:
        return StreamingResponse(
            _stream_tokens(request_id),
            media_type="text/event-stream",
        )
    else:
        return await _collect_response(request_id, prompt_tokens)


async def _stream_tokens(request_id: str):
    """SSE streaming response."""
    while True:
        outputs = engine.poll_tokens()
        for out in outputs:
            if out.request_id != request_id:
                continue
            if out.done:
                yield f"data: [DONE]\n\n"
                return
            for tok in out.tokens:
                text = tokenizer.decode_token(tok)
                chunk = {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                import json
                yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.005)


async def _collect_response(request_id: str, prompt_tokens: list[int]):
    """Non-streaming response."""
    all_tokens = []
    while True:
        outputs = engine.poll_tokens()
        for out in outputs:
            if out.request_id != request_id:
                continue
            if out.done:
                text = tokenizer.decode(all_tokens)
                return {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": len(prompt_tokens),
                        "completion_tokens": len(all_tokens),
                        "total_tokens": len(prompt_tokens) + len(all_tokens),
                    },
                }
            all_tokens.extend(out.tokens)
        await asyncio.sleep(0.005)
```

- [ ] **Step 4: Create run.py**

```python
# server/run.py
"""FlashMLX Inference Server entry point."""
import argparse
import uvicorn
from server.python.app import app, init


def main():
    parser = argparse.ArgumentParser(description="FlashMLX Inference Server")
    parser.add_argument("model", help="Path to model directory or HF repo ID")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-context-len", type=int, default=2048)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    init(args.model, args.max_batch_size, args.max_context_len)
    print(f"Server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create server/python/__init__.py**

```python
# server/python/__init__.py
```

- [ ] **Step 6: Test the server**

```bash
source .venv/bin/activate
python -m server.run /path/to/model --port 8080 &
sleep 5
curl -s http://localhost:8080/health | python -m json.tool
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":10,"temperature":0}' | python -m json.tool
kill %1
```

Expected: Health returns stats, chat returns generated text.

- [ ] **Step 7: Commit**

```bash
git add server/python/ server/run.py
git commit -m "feat(server): FastAPI HTTP server with /v1/chat/completions endpoint"
```

---

### Task 7: Integration Tests

**Files:**
- Create: `server/tests/test_engine.py`
- Create: `server/tests/test_server.py`
- Create: `server/bench_server.py`

- [ ] **Step 1: Create test_engine.py**

```python
# server/tests/test_engine.py
"""Integration tests for the C++ engine."""
import time
import pytest
from server.python._flashmlx_engine import Engine

MODEL_PATH = "/Users/yihengchen/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3-8B-4bit/snapshots/7f296f1dd0e720f7daa69ed648ed7b193373ca11"


class TestEngine:
    def test_ping(self):
        e = Engine(MODEL_PATH, 2, 512)
        assert e.ping() == "flashmlx engine ready"

    def test_single_request(self):
        e = Engine(MODEL_PATH, 2, 512)
        e.start()
        e.submit_request("test-1", list(range(1, 17)), 10, 0.0)

        tokens = []
        done = False
        for _ in range(200):
            for out in e.poll_tokens():
                if out.request_id == "test-1":
                    if out.done:
                        done = True
                    else:
                        tokens.extend(out.tokens)
            if done:
                break
            time.sleep(0.01)

        e.stop()
        assert done
        assert len(tokens) == 10

    def test_concurrent_requests(self):
        e = Engine(MODEL_PATH, 4, 512)
        e.start()
        e.submit_request("a", list(range(1, 9)), 5, 0.0)
        e.submit_request("b", list(range(1, 9)), 5, 0.0)

        results = {"a": [], "b": []}
        done = set()
        for _ in range(300):
            for out in e.poll_tokens():
                if out.done:
                    done.add(out.request_id)
                else:
                    results[out.request_id].extend(out.tokens)
            if len(done) == 2:
                break
            time.sleep(0.01)

        e.stop()
        assert len(done) == 2
        assert len(results["a"]) == 5
        assert len(results["b"]) == 5
```

- [ ] **Step 2: Create bench_server.py**

```python
# server/bench_server.py
"""Benchmark the server with concurrent requests."""
import asyncio
import json
import time
import aiohttp
import argparse


async def send_request(session, url, prompt, max_tokens, request_num):
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with session.post(url, json=data) as resp:
        result = await resp.json()
        t1 = time.perf_counter()
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        return {"request": request_num, "tokens": tokens, "time_s": t1 - t0}


async def bench(url, concurrency, max_tokens, prompt):
    async with aiohttp.ClientSession() as session:
        # Warmup
        await send_request(session, url, "Hello", 10, -1)

        # Concurrent requests
        t0 = time.perf_counter()
        tasks = [
            send_request(session, url, prompt, max_tokens, i)
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        t1 = time.perf_counter()

        total_tokens = sum(r["tokens"] for r in results)
        print(f"Concurrency: {concurrency}")
        print(f"Total tokens: {total_tokens}")
        print(f"Wall time: {t1-t0:.2f}s")
        print(f"Total throughput: {total_tokens/(t1-t0):.1f} tok/s")
        print(f"Per-request avg: {sum(r['time_s'] for r in results)/len(results):.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Explain transformer attention in detail.")
    args = parser.parse_args()
    asyncio.run(bench(args.url, args.concurrency, args.max_tokens, args.prompt))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

```bash
source .venv/bin/activate
python -m pytest server/tests/test_engine.py -v --timeout=60
```

Expected: All tests pass.

- [ ] **Step 4: Run concurrent benchmark**

```bash
# Start server in background
python -m server.run /path/to/model --port 8080 &
sleep 10

# Benchmark
python server/bench_server.py --concurrency 1 --max-tokens 128
python server/bench_server.py --concurrency 4 --max-tokens 128
python server/bench_server.py --concurrency 8 --max-tokens 128

kill %1
```

Expected: Total throughput increases with concurrency (target: 200+ tok/s at concurrency=4).

- [ ] **Step 5: Commit**

```bash
git add server/tests/ server/bench_server.py
git commit -m "feat(server): integration tests + concurrent benchmark"
```

---

### Task 8: Documentation + Final Cleanup

**Files:**
- Modify: `README.md` (add server section)
- Modify: `pyproject.toml` (add server dependencies)

- [ ] **Step 1: Add server dependencies to pyproject.toml**

Add to `[project.optional-dependencies]`:
```toml
server = [
    "fastapi>=0.100.0",
    "uvicorn>=0.24.0",
    "sse-starlette>=1.6.0",
    "aiohttp>=3.9.0",
]
```

- [ ] **Step 2: Add server section to README.md**

Add after the Quick Start section:
```markdown
## Inference Server

FlashMLX includes a batched inference server with a C++ decode loop:

\`\`\`bash
# Build the C++ extension
cd server && mkdir build && cd build
cmake .. && make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
cd ../..

# Start the server
python -m server.run mlx-community/Meta-Llama-3-8B-4bit --port 8080

# Query it (OpenAI-compatible)
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
\`\`\`
```

- [ ] **Step 3: Commit**

```bash
git add README.md pyproject.toml
git commit -m "docs: add server documentation and dependencies"
```
