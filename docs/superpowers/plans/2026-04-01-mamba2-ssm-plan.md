# Mamba-2 SSM + Hybrid Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Mamba-2 SSM layer support to the C++ server, enabling Nemotron-3-Nano-4B (hybrid Mamba-2 + Attention).

**Architecture:** Add a new `NemotronHModel` class alongside the existing `LlamaModel`. The engine detects `model_type: nemotron_h` and instantiates the right model. NemotronH has a flat list of blocks (not nested [attn+mlp] layers) — each block is one of: Mamba-2 mixer (`M`), Attention (`*`), MLP (`-`), or MoE (`E`). The Mamba-2 SSM uses a custom Metal kernel for the selective scan and a conv1d sliding window for input processing.

**Tech Stack:** C++17, MLX C++ API (`mx::fast::metal_kernel` for SSM, `mx::conv_general` for conv1d), pybind11

**CRITICAL ARCHITECTURE DIFFERENCE:** Nemotron-H blocks are NOT [attention + MLP] pairs like Llama. Each block is a SINGLE operation: `x → RMSNorm → mixer → residual`. The pattern `M-M-M-MM-M-M*-...` maps one character per block. There are 42 blocks total. The MLP uses relu² (squared ReLU), not SwiGLU. Attention has NO RoPE.

---

## File Structure

```
server/
├── include/flashmlx/
│   ├── model.h          # Add: ModelBase interface, NemotronHModel class
│   └── kv_pool.h        # Add: MambaState, hybrid state management
├── src/
│   ├── model.cpp         # Existing LlamaModel (unchanged)
│   ├── nemotron_h.cpp    # NEW: NemotronHModel implementation
│   ├── kv_pool.cpp       # Extend: allocate Mamba states
│   └── engine.cpp        # Modify: detect model_type, instantiate correct model
└── CMakeLists.txt        # Add nemotron_h.cpp
```

Key decision: **new file `nemotron_h.cpp`** for the NemotronH model, keeping `model.cpp` untouched for Llama. Both models share the same `Engine` interface via a base class or common forward signature.

---

### Task 1: Model Base Interface + Config Detection

**Files:**
- Modify: `server/include/flashmlx/model.h`
- Modify: `server/src/engine.cpp`

- [ ] **Step 1: Add ModelBase interface to model.h**

Add before the LlamaModel class:

```cpp
// Abstract base for all model types
class ModelBase {
public:
    virtual ~ModelBase() = default;
    
    // Forward pass: input_ids [B, L] → logits [B, L, vocab_size]
    virtual mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        int cache_offset) = 0;
    
    // Forward pass with array offsets (for prefill/heterogeneous batching)
    virtual mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        const mx::array& cache_offsets) = 0;
    
    virtual const ModelConfig& config() const = 0;
    virtual std::vector<int> debug_forward(const std::vector<int>& token_ids) = 0;
    virtual std::vector<float> debug_embed(const std::vector<int>& token_ids) = 0;
};
```

Make LlamaModel inherit from ModelBase:
```cpp
class LlamaModel : public ModelBase {
    // ... existing code, add 'override' to all virtual methods ...
};
```

- [ ] **Step 2: Add nemotron_h model type detection to engine.cpp**

In the Engine constructor, before creating the model:

```cpp
// Detect model type from config.json
std::string model_type = "llama";  // default
{
    auto config_path = model_path + "/config.json";
    // ... read and parse model_type field ...
}

if (model_type == "nemotron_h") {
    model_ = std::make_unique<NemotronHModel>(model_path);
} else {
    model_ = std::make_unique<LlamaModel>(model_path);
}
```

Change `model_` type from `std::unique_ptr<LlamaModel>` to `std::unique_ptr<ModelBase>`.

- [ ] **Step 3: Build and verify existing models still work**

```bash
cd server/build && PATH=/opt/homebrew/bin:$PATH cmake .. && make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
cd ../..
.venv/bin/python -m pytest server/tests/test_engine.py -x -q --timeout=60
```

Expected: All 4 tests pass (Qwen3-0.6B still works through ModelBase interface).

- [ ] **Step 4: Commit**

```bash
git add server/
git commit -m "refactor(server): add ModelBase interface for multi-architecture support"
```

---

### Task 2: NemotronH Config Parsing + Skeleton

**Files:**
- Modify: `server/include/flashmlx/model.h` (add NemotronH config fields)
- Create: `server/src/nemotron_h.cpp`
- Modify: `server/CMakeLists.txt`

- [ ] **Step 1: Add Nemotron-H config fields to ModelConfig**

```cpp
    // Nemotron-H / Mamba-2 parameters
    std::string hybrid_pattern;  // e.g., "M-M-M-MM-M-M*-..."
    int mamba_num_heads = 0;
    int mamba_head_dim = 0;
    int ssm_state_size = 0;
    int conv_kernel = 4;
    int n_groups = 1;
    float time_step_min = 0.001f;
    float time_step_max = 0.1f;
    float layer_norm_epsilon = 1e-5f;
    bool use_conv_bias = true;
    bool mamba_proj_bias = false;
    bool mlp_bias = false;
```

- [ ] **Step 2: Create nemotron_h.cpp skeleton**

```cpp
// server/src/nemotron_h.cpp
#include "flashmlx/model.h"
#include <mlx/compile.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

namespace flashmlx {

class NemotronHModel : public ModelBase {
public:
    explicit NemotronHModel(const std::string& model_path);
    
    mx::array forward(const mx::array& input_ids,
                      std::vector<mx::array>& cache_keys,
                      std::vector<mx::array>& cache_values,
                      int cache_offset) override;
    
    mx::array forward(const mx::array& input_ids,
                      std::vector<mx::array>& cache_keys,
                      std::vector<mx::array>& cache_values,
                      const mx::array& cache_offsets) override;
    
    const ModelConfig& config() const override { return config_; }
    std::vector<int> debug_forward(const std::vector<int>& token_ids) override;
    std::vector<float> debug_embed(const std::vector<int>& token_ids) override;

private:
    ModelConfig config_;
    std::unordered_map<std::string, mx::array> weights_;
    
    enum class BlockType { MAMBA, ATTENTION, MLP, MOE };
    std::vector<BlockType> block_types_;
    
    void load_config(const std::string& path);
    void load_weights(const std::string& path);
    void parse_hybrid_pattern();
    
    mx::array get_weight(const std::string& name) const;
    bool has_weight(const std::string& name) const;
    
    // Building blocks
    mx::array embed(const mx::array& input_ids);
    mx::array lm_head(const mx::array& x);
    mx::array rms_norm(const mx::array& x, const mx::array& weight);
    mx::array linear(const mx::array& x, const std::string& prefix);
    
    // Block types
    mx::array mamba_block(const mx::array& x, int block_idx, /* state */);
    mx::array attention_block(const mx::array& x, int block_idx, /* KV cache */);
    mx::array mlp_block(const mx::array& x, int block_idx);
};

} // namespace flashmlx
```

- [ ] **Step 3: Implement config loading and pattern parsing**

Parse the `hybrid_override_pattern` from config.json:
```cpp
void NemotronHModel::parse_hybrid_pattern() {
    block_types_.clear();
    for (char c : config_.hybrid_pattern) {
        switch (c) {
            case 'M': block_types_.push_back(BlockType::MAMBA); break;
            case '*': block_types_.push_back(BlockType::ATTENTION); break;
            case '-': block_types_.push_back(BlockType::MLP); break;
            case 'E': block_types_.push_back(BlockType::MOE); break;
            // Other characters (separators, etc.) are skipped
        }
    }
    // Verify block count matches num_hidden_layers
    if ((int)block_types_.size() != config_.num_hidden_layers) {
        std::cerr << "[nemotron_h] WARNING: pattern gives " << block_types_.size()
                  << " blocks but config has " << config_.num_hidden_layers << " layers" << std::endl;
    }
}
```

- [ ] **Step 4: Add to CMakeLists.txt**

```cmake
pybind11_add_module(_flashmlx_engine
    src/engine.cpp
    src/model.cpp
    src/nemotron_h.cpp
    src/kv_pool.cpp
    src/scheduler.cpp
    src/sampling.cpp
)
```

- [ ] **Step 5: Build and test — NemotronH detected but not yet functional**

```bash
cd server/build && PATH=/opt/homebrew/bin:$PATH cmake .. && make -j$(sysctl -n hw.ncpu)
```

Test that config parsing works:
```python
import sys; sys.path.insert(0, '.')
from server.python._flashmlx_engine import Engine
# This should detect nemotron_h and try to create NemotronHModel
# May crash on missing implementation — just verify detection
e = Engine('<NEMOTRON_PATH>', 2, 512)
```

- [ ] **Step 6: Commit**

```bash
git add server/
git commit -m "feat(server): NemotronH model skeleton with hybrid pattern parsing"
```

---

### Task 3: SSM Metal Kernel + Conv1D

**Files:**
- Modify: `server/src/nemotron_h.cpp`

This is the core Mamba-2 compute. The SSM kernel is a custom Metal shader and the conv1d uses MLX's built-in convolution.

- [ ] **Step 1: Implement the SSM Metal kernel**

Port the Metal shader from mlx-lm's `ssm.py`:

```cpp
// At file scope in nemotron_h.cpp:

static const char* SSM_KERNEL_SOURCE = R"(
    auto n = thread_position_in_grid.z;
    auto h_idx = n % H;
    auto g_idx = n / G;
    constexpr int n_per_t = Ds / 32;

    auto x = X + n * Dh;
    out += n * Dh;
    auto i_state = state_in + n * Dh * Ds;
    auto o_state = state_out + n * Dh * Ds;

    auto C_ = C + g_idx * Ds;
    auto B_ = B + g_idx * Ds;

    auto ds_idx = thread_position_in_threadgroup.x;
    auto d_idx = thread_position_in_grid.y;

    auto dt_ = static_cast<float>(dt[n]);
    auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
    auto dA = fast::exp(A * dt_);

    float acc = 0.0;
    auto x_ = static_cast<float>(x[d_idx]);

    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * ds_idx + i;
        auto idx = d_idx * Ds + s_idx;
        auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
        auto state = dA * i_state[idx] + dB_by_x;
        o_state[idx] = static_cast<T>(state);
        acc += state * C_[s_idx];
    }
    acc = simd_sum(acc);
    if (thread_index_in_simdgroup == 0) {
        out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
    }
)";

static auto ssm_kernel_fn = mx::fast::metal_kernel(
    "ssm_kernel",
    {"X", "A_log", "B", "C", "D", "dt", "state_in"},
    {"out", "state_out"},
    SSM_KERNEL_SOURCE
);
```

- [ ] **Step 2: Implement compute_dt helper**

```cpp
// Compiled dt computation: softplus + clip
static auto compiled_compute_dt = mx::compile(
    [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto dt = inputs[0];
        auto dt_bias = inputs[1];
        auto dt_sum = mx::add(dt, dt_bias);
        // softplus: log(1 + exp(x))
        auto sp = mx::log(mx::add(mx::array(1.0f), mx::exp(dt_sum)));
        // clip to time_step_limit
        auto clipped = mx::clip(sp, mx::array(0.001f), mx::array(0.1f));
        return {clipped};
    },
    /*shapeless=*/true
);
```

- [ ] **Step 3: Implement ssm_update wrapper**

```cpp
// Wraps the Metal kernel call with proper shapes and template args
std::pair<mx::array, mx::array> NemotronHModel::ssm_update(
    const mx::array& hidden_states,  // [B, 1, n_heads, head_dim]
    const mx::array& A_log,          // [n_heads]
    const mx::array& B,              // [B, 1, n_groups, state_size]
    const mx::array& C,              // [B, 1, n_groups, state_size]
    const mx::array& D,              // [n_heads]
    const mx::array& dt,             // [B, 1, n_heads]
    const mx::array& dt_bias,        // [n_heads]
    const mx::array& state           // [B, n_groups, n_heads, head_dim, state_size]
) {
    int n = hidden_states.shape(0);  // batch
    int h = config_.mamba_num_heads;
    int d = config_.mamba_head_dim;
    int ds = config_.ssm_state_size;
    int hb = config_.n_groups;
    
    // Compute dt
    auto dt_computed = compiled_compute_dt({dt, dt_bias})[0];
    
    auto input_type = hidden_states.dtype();
    
    return ssm_kernel_fn(
        /*inputs=*/{hidden_states, A_log, B, C, D, dt_computed, state},
        /*template=*/{{"T", input_type}, {"Dh", d}, {"Ds", ds}, {"H", h}, {"G", h / hb}},
        /*grid=*/{32, d, h * n},
        /*threadgroup=*/{32, 8, 1},
        /*output_shapes=*/{{n, 1, h, d}, state.shape()},
        /*output_dtypes=*/{input_type, input_type}
    );
}
```

NOTE: The exact C++ API for `metal_kernel` call syntax needs to be verified against the MLX C++ headers. The Python API uses keyword arguments; the C++ API may differ. Check `mx::fast::metal_kernel` return type and call signature in `.venv/lib/python3.14/site-packages/mlx/include/mlx/fast.h`.

- [ ] **Step 4: Implement conv1d with sliding window state**

```cpp
mx::array NemotronHModel::conv1d_with_state(
    const mx::array& x,           // [B, 1, conv_dim]
    const mx::array& conv_weight,  // [conv_dim, 1, conv_kernel]
    mx::array& conv_state          // [B, conv_kernel-1, conv_dim] — mutated
) {
    // Append new input to sliding window
    auto padded = mx::concatenate({conv_state, x}, 1);  // [B, conv_kernel, conv_dim]
    
    // Update state: keep last (conv_kernel-1) entries
    int keep = config_.conv_kernel - 1;
    conv_state = mx::slice(padded, {0, 1, 0}, 
                           {(int)padded.shape(0), (int)padded.shape(1), (int)padded.shape(2)});
    
    // Depthwise conv1d
    // MLX conv1d expects [B, L, C_in] input and [C_out, 1, K] weight for groups=C_in
    auto conv_out = mx::conv_general(padded, conv_weight,
                                     /*stride=*/{1}, /*padding_lo=*/{0}, /*padding_hi=*/{0},
                                     /*kernel_dilation=*/{1}, /*input_dilation=*/{1},
                                     /*groups=*/(int)x.shape(2));
    
    return mx::multiply(conv_out, mx::sigmoid(conv_out));  // silu
}
```

- [ ] **Step 5: Build to verify compilation**

```bash
cd server/build && PATH=/opt/homebrew/bin:$PATH cmake .. && make -j$(sysctl -n hw.ncpu)
```

- [ ] **Step 6: Commit**

```bash
git add server/
git commit -m "feat(server): SSM Metal kernel + conv1d for Mamba-2"
```

---

### Task 4: Mamba Mixer + Block Assembly

**Files:**
- Modify: `server/src/nemotron_h.cpp`

- [ ] **Step 1: Implement MambaRMSNormGated**

```cpp
mx::array NemotronHModel::mamba_rms_norm_gated(
    const mx::array& x, const mx::array& gate, const mx::array& weight, float eps) {
    // SwiGLU gate then group RMSNorm
    auto gated = swiglu(gate, x);  // reuse compiled swiglu from LlamaModel
    int group_size = config_.mamba_num_heads * config_.mamba_head_dim / config_.n_groups;
    // Reshape to groups, norm per group, reshape back
    auto grouped = mx::unflatten(gated, -1, {-1, group_size});
    auto normed = mx::fast::rms_norm(grouped, std::nullopt, eps);
    return mx::multiply(weight, mx::flatten(normed, -2));
}
```

- [ ] **Step 2: Implement the full mamba_mixer**

```cpp
mx::array NemotronHModel::mamba_mixer(
    const mx::array& x, int block_idx,
    mx::array& conv_state, mx::array& ssm_state) {
    
    auto prefix = "layers." + std::to_string(block_idx) + ".mixer.";
    int d_inner = config_.mamba_num_heads * config_.mamba_head_dim;
    int n_groups = config_.n_groups;
    int state_size = config_.ssm_state_size;
    int conv_dim = d_inner + 2 * n_groups * state_size;
    
    // 1. Input projection
    auto proj = linear(x, prefix + "in_proj");
    
    // Split: [gate, conv_input, dt]
    auto gate = mx::slice(proj, {0, 0, 0}, {(int)x.shape(0), 1, d_inner});
    auto conv_input = mx::slice(proj, {0, 0, d_inner}, {(int)x.shape(0), 1, d_inner + conv_dim});
    auto dt = mx::slice(proj, {0, 0, d_inner + conv_dim}, {(int)x.shape(0), 1, (int)proj.shape(2)});
    
    // 2. Conv1D with sliding window
    auto conv_weight = get_weight(prefix + "conv1d.weight");
    auto conv_out = conv1d_with_state(conv_input, conv_weight, conv_state);
    
    // 3. Split conv output: [hidden_states, B, C]
    auto hidden = mx::slice(conv_out, {0, 0, 0}, {(int)x.shape(0), 1, d_inner});
    auto B = mx::slice(conv_out, {0, 0, d_inner}, {(int)x.shape(0), 1, d_inner + n_groups * state_size});
    auto C = mx::slice(conv_out, {0, 0, d_inner + n_groups * state_size}, {(int)x.shape(0), 1, (int)conv_out.shape(2)});
    
    // Reshape for SSM
    hidden = mx::reshape(hidden, {(int)x.shape(0), 1, config_.mamba_num_heads, config_.mamba_head_dim});
    B = mx::reshape(B, {(int)x.shape(0), 1, n_groups, state_size});
    C = mx::reshape(C, {(int)x.shape(0), 1, n_groups, state_size});
    
    // 4. SSM update
    auto A_log = get_weight(prefix + "A_log");
    auto D_param = get_weight(prefix + "D");
    auto dt_bias = get_weight(prefix + "dt_bias");
    
    auto [y, new_state] = ssm_update(hidden, A_log, B, C, D_param, dt, dt_bias, ssm_state);
    ssm_state = new_state;
    
    // 5. Reshape output and apply gated norm
    y = mx::reshape(y, {(int)x.shape(0), 1, d_inner});
    auto norm_weight = get_weight(prefix + "norm.weight");
    y = mamba_rms_norm_gated(y, gate, norm_weight, config_.layer_norm_epsilon);
    
    // 6. Output projection
    return linear(y, prefix + "out_proj");
}
```

- [ ] **Step 3: Implement attention_block (no RoPE)**

```cpp
mx::array NemotronHModel::attention_block(
    const mx::array& x, int block_idx,
    mx::array& cache_k, mx::array& cache_v) {
    
    auto prefix = "layers." + std::to_string(block_idx) + ".mixer.";
    int B = x.shape(0), L = x.shape(1);
    int n_heads = config_.num_attention_heads;
    int n_kv = config_.num_key_value_heads;
    int hd = config_.head_dim > 0 ? config_.head_dim : config_.hidden_size / n_heads;
    
    auto q = linear(x, prefix + "q_proj");
    auto k = linear(x, prefix + "k_proj");
    auto v = linear(x, prefix + "v_proj");
    
    q = mx::transpose(mx::reshape(q, {B, L, n_heads, hd}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B, L, n_kv, hd}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B, L, n_kv, hd}), {0, 2, 1, 3});
    
    // NO RoPE for Nemotron-H attention
    
    // KV cache update (concat-based)
    cache_k = mx::concatenate({cache_k, k}, 2);
    cache_v = mx::concatenate({cache_v, v}, 2);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::string mask_mode = (L > 1) ? "causal" : "";
    auto out = mx::fast::scaled_dot_product_attention(q, cache_k, cache_v, scale, mask_mode);
    out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, n_heads * hd});
    return linear(out, prefix + "o_proj");
}
```

- [ ] **Step 4: Implement mlp_block (relu² not SwiGLU)**

```cpp
mx::array NemotronHModel::mlp_block(const mx::array& x, int block_idx) {
    auto prefix = "layers." + std::to_string(block_idx) + ".mixer.";
    auto up = linear(x, prefix + "up_proj");
    // relu² = relu(x)²
    auto activated = mx::square(mx::maximum(up, mx::array(0.0f)));
    return linear(activated, prefix + "down_proj");
}
```

- [ ] **Step 5: Build**

```bash
cd server/build && PATH=/opt/homebrew/bin:$PATH cmake .. && make -j$(sysctl -n hw.ncpu)
```

- [ ] **Step 6: Commit**

```bash
git add server/
git commit -m "feat(server): Mamba mixer + attention + MLP blocks for NemotronH"
```

---

### Task 5: Hybrid Forward Pass + State Management

**Files:**
- Modify: `server/src/nemotron_h.cpp`
- Modify: `server/include/flashmlx/kv_pool.h`
- Modify: `server/src/kv_pool.cpp`
- Modify: `server/src/scheduler.cpp`

- [ ] **Step 1: Implement the NemotronH forward pass**

The forward iterates through blocks, dispatching based on type. State is managed differently for each block type:
- Mamba blocks: conv_state + ssm_state (fixed size, updated in place)
- Attention blocks: KV cache (grows via concat)
- MLP blocks: no state

```cpp
mx::array NemotronHModel::forward(
    const mx::array& input_ids,
    std::vector<mx::array>& cache_keys,    // for attention blocks
    std::vector<mx::array>& cache_values,
    int cache_offset) {
    
    auto h = embed(input_ids);
    
    int attn_idx = 0;  // index into cache_keys/values
    int mamba_idx = 0;  // index into mamba states
    
    for (int i = 0; i < (int)block_types_.size(); i++) {
        auto normed = rms_norm(h, get_weight("layers." + std::to_string(i) + ".norm.weight"));
        
        mx::array block_out;
        switch (block_types_[i]) {
            case BlockType::MAMBA:
                block_out = mamba_mixer(normed, i, 
                    mamba_conv_states_[mamba_idx], mamba_ssm_states_[mamba_idx]);
                mamba_idx++;
                break;
            case BlockType::ATTENTION:
                block_out = attention_block(normed, i,
                    cache_keys[attn_idx], cache_values[attn_idx]);
                attn_idx++;
                break;
            case BlockType::MLP:
                block_out = mlp_block(normed, i);
                break;
            default:
                block_out = normed;  // passthrough for unsupported
        }
        h = mx::add(h, block_out);
    }
    
    h = rms_norm(h, get_weight("norm_f.weight"));
    return lm_head(h);
}
```

NOTE: The Mamba states (`mamba_conv_states_`, `mamba_ssm_states_`) need to be managed per-request, similar to KV cache slots. The simplest approach for V1: store them as member variables on the model, initialized per request. For batched serving, they'll need to be in the pool.

- [ ] **Step 2: Extend KV pool for hybrid state**

Add Mamba state storage to the pool. For Nemotron-H, each request needs:
- KV caches for attention blocks (4 attention blocks × [k, v])
- Mamba states for Mamba blocks (~38 blocks × [conv_state, ssm_state])

For V1 (single-request at a time through Mamba blocks), store Mamba state on the model. For batched serving, extend the pool.

- [ ] **Step 3: Wire up in engine.cpp**

The scheduler calls `model_->forward()` which dispatches to either LlamaModel or NemotronHModel. The forward signature is the same (via ModelBase), but NemotronH uses the cache arrays differently (only for attention blocks).

- [ ] **Step 4: Download test model and verify end-to-end**

```bash
# Download if not already present
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit',
    allow_patterns=['*.json', 'model*.safetensors', 'tokenizer*', '*.txt'])
"
```

Test:
```python
import sys, time; sys.path.insert(0, '.')
from server.python._flashmlx_engine import Engine

e = Engine('<NEMOTRON_PATH>', 2, 512)
e.start()
e.submit_request('mamba-test', list(range(1, 17)), 10, 0.0)
# ... poll for tokens ...
```

- [ ] **Step 5: Verify output against mlx-lm**

Compare top-5 tokens with mlx-lm for correctness.

- [ ] **Step 6: Commit**

```bash
git add server/
git commit -m "feat(server): NemotronH hybrid forward pass with Mamba-2 + Attention"
```

---

### Task 6: Test + Benchmark + Documentation

**Files:**
- Modify: `server/tests/test_engine.py`
- Modify: `docs/model-configs.md`
- Modify: `README.md`

- [ ] **Step 1: Add NemotronH test**

```python
NEMOTRON_PATH = "<path>"

class TestNemotronHEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        e = Engine(NEMOTRON_PATH, 2, 512)
        e.start()
        yield e
        e.stop()

    def test_generates_tokens(self, engine):
        engine.submit_request("nemotron-test", list(range(1, 17)), 10, 0.0)
        tokens = collect_tokens(engine, "nemotron-test")
        assert len(tokens) == 10
```

- [ ] **Step 2: Run all tests (regression)**

```bash
.venv/bin/python -m pytest server/tests/test_engine.py -v --timeout=120
.venv/bin/python -m pytest tests/ -x -q
```

- [ ] **Step 3: Benchmark and compare with LM Studio**

LM Studio gets 63 tok/s C=1 on the 30B variant. The 4B should be much faster. Benchmark FlashMLX and compare.

- [ ] **Step 4: Update documentation**

Add Nemotron-H to model-configs.md and README.md supported models.

- [ ] **Step 5: Commit**

```bash
git add server/ docs/ README.md
git commit -m "feat(server): NemotronH tests, benchmarks, documentation"
```
