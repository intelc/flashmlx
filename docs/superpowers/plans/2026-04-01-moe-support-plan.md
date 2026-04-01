# MoE Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Mixture-of-Experts layer support to the C++ inference server so it can run Qwen1.5-MoE-A2.7B and other MoE models.

**Architecture:** Extend the existing LlamaModel with per-layer MoE detection. MoE layers replace the MLP sublayer with a router (top-k expert selection via softmax + argpartition) + expert execution (via `mx::gather_qmm` for quantized indexed matmul) + optional shared expert. Attention and KV cache are unchanged.

**Tech Stack:** C++17, MLX C++ API (`mx::gather_qmm`, `mx::softmax`, `mx::argpartition`, `mx::take_along_axis`), pybind11

---

## File Structure

```
server/
├── include/flashmlx/
│   └── model.h          # Add: MoE config fields, MoEWeights struct, moe_block(), per-layer is_moe flag
├── src/
│   └── model.cpp         # Add: MoE config parsing, MoE weight loading + stacking, moe_block(), switch_mlp()
└── tests/
    └── test_engine.py    # Add: MoE model test (load + generate)
```

No new files — all MoE logic lives in model.h/model.cpp alongside existing Llama code.

---

### Task 1: Download Test Model + Extend Config Parsing

**Files:**
- Modify: `server/include/flashmlx/model.h`
- Modify: `server/src/model.cpp`

- [ ] **Step 1: Download Qwen1.5-MoE-A2.7B-Chat-4bit**

```bash
source .venv/bin/activate
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit',
    allow_patterns=['*.json', 'model*.safetensors', 'tokenizer*', '*.txt'])
print(path)
"
```

Expected: Downloads ~2GB, prints cache path.

- [ ] **Step 2: Add MoE config fields to ModelConfig**

In `server/include/flashmlx/model.h`, add after the quantization fields:

```cpp
    // MoE parameters (num_experts == 0 means not MoE)
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    bool norm_topk_prob = false;
```

- [ ] **Step 3: Parse MoE config fields from JSON**

In `server/src/model.cpp`, in `load_config()`, add after the quantization parsing:

```cpp
    // MoE config
    config_.num_experts = json_int(json, "num_experts", 0);
    config_.num_experts_per_tok = json_int(json, "num_experts_per_tok", 0);
    config_.moe_intermediate_size = json_int(json, "moe_intermediate_size", 0);
    config_.shared_expert_intermediate_size = json_int(json, "shared_expert_intermediate_size", 0);
    config_.norm_topk_prob = json_bool(json, "norm_topk_prob", false);
```

Also update the config print line to show MoE info:

```cpp
    if (config_.num_experts > 0) {
        std::cout << "[flashmlx] MoE: " << config_.num_experts << " experts, top-"
                  << config_.num_experts_per_tok << ", intermediate=" << config_.moe_intermediate_size
                  << std::endl;
    }
```

- [ ] **Step 4: Build and verify config parsing**

```bash
cd server/build && PATH=/opt/homebrew/bin:$PATH cmake .. && make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
cd ../..
source .venv/bin/activate
python -c "
import sys; sys.path.insert(0, '.')
from server.python._flashmlx_engine import Engine
e = Engine('<QWEN_MOE_PATH>', 2, 512)
print(e.ping())
"
```

Expected: Prints MoE config and `flashmlx engine ready`. May fail on weight loading (experts not supported yet) — that's OK for this step, just verify config parsing works.

- [ ] **Step 5: Commit**

```bash
git add server/include/flashmlx/model.h server/src/model.cpp
git commit -m "feat(server): parse MoE config fields (num_experts, top_k, moe_intermediate_size)"
```

---

### Task 2: MoE Weight Loading + Stacking

**Files:**
- Modify: `server/include/flashmlx/model.h`
- Modify: `server/src/model.cpp`

- [ ] **Step 1: Add MoEWeights struct and per-layer storage**

In `server/include/flashmlx/model.h`, add the MoEWeights struct:

```cpp
struct MoEWeights {
    mx::array router_w;  // [num_experts, hidden_size] — NOT quantized (float16/bf16)
    // Stacked expert weights: [num_experts, dim1, dim2] — quantized
    mx::array gate_w, gate_s, up_w, up_s, down_w, down_s;
    std::optional<mx::array> gate_b, up_b, down_b;
    // Shared expert (standard SwiGLU MLP weights)
    bool has_shared = false;
    mx::array shared_gate_w, shared_gate_s, shared_up_w, shared_up_s, shared_down_w, shared_down_s;
    std::optional<mx::array> shared_gate_b, shared_up_b, shared_down_b;
    mx::array shared_expert_gate_w;  // [hidden_size, 1] — sigmoid gating
    std::optional<mx::array> shared_expert_gate_s;
    std::optional<mx::array> shared_expert_gate_b;
};
```

Add to the LlamaModel class:

```cpp
    std::vector<bool> layer_is_moe_;
    std::vector<MoEWeights> moe_weights_;
    void build_moe_weight_cache();
```

- [ ] **Step 2: Implement MoE weight stacking in build_moe_weight_cache()**

In `server/src/model.cpp`, add after `build_weight_cache()`:

```cpp
void LlamaModel::build_moe_weight_cache() {
    if (config_.num_experts == 0) return;

    int n_layers = config_.num_hidden_layers;
    int n_experts = config_.num_experts;
    layer_is_moe_.resize(n_layers, false);
    moe_weights_.reserve(n_layers);

    for (int i = 0; i < n_layers; i++) {
        std::string expert0_key = "layers." + std::to_string(i) + ".mlp.experts.0.gate_proj.weight";
        if (!has_weight(expert0_key)) {
            // Dense layer — no MoE weights needed
            moe_weights_.emplace_back();  // placeholder
            continue;
        }

        layer_is_moe_[i] = true;
        MoEWeights mw;

        // Router weight (not quantized — linear without scales)
        std::string router_key = "layers." + std::to_string(i) + ".mlp.gate.weight";
        mw.router_w = get_weight(router_key);

        // Stack expert weights: iterate over experts, stack into [num_experts, ...]
        std::string prefix = "layers." + std::to_string(i) + ".mlp.experts.";
        auto stack_expert_weights = [&](const std::string& proj_name, const std::string& component) {
            std::vector<mx::array> parts;
            for (int e = 0; e < n_experts; e++) {
                std::string key = prefix + std::to_string(e) + "." + proj_name + "." + component;
                parts.push_back(get_weight(key));
            }
            return mx::stack(parts, 0);
        };

        mw.gate_w = stack_expert_weights("gate_proj", "weight");
        mw.gate_s = stack_expert_weights("gate_proj", "scales");
        mw.up_w = stack_expert_weights("up_proj", "weight");
        mw.up_s = stack_expert_weights("up_proj", "scales");
        mw.down_w = stack_expert_weights("down_proj", "weight");
        mw.down_s = stack_expert_weights("down_proj", "scales");

        // Optional biases
        std::string bias_key = prefix + "0.gate_proj.biases";
        if (has_weight(bias_key)) {
            mw.gate_b = stack_expert_weights("gate_proj", "biases");
            mw.up_b = stack_expert_weights("up_proj", "biases");
            mw.down_b = stack_expert_weights("down_proj", "biases");
        }

        // Shared expert
        std::string shared_prefix = "layers." + std::to_string(i) + ".mlp.shared_expert.";
        if (has_weight(shared_prefix + "gate_proj.weight")) {
            mw.has_shared = true;
            mw.shared_gate_w = get_weight(shared_prefix + "gate_proj.weight");
            mw.shared_gate_s = get_weight(shared_prefix + "gate_proj.scales");
            mw.shared_up_w = get_weight(shared_prefix + "up_proj.weight");
            mw.shared_up_s = get_weight(shared_prefix + "up_proj.scales");
            mw.shared_down_w = get_weight(shared_prefix + "down_proj.weight");
            mw.shared_down_s = get_weight(shared_prefix + "down_proj.scales");
            if (has_weight(shared_prefix + "gate_proj.biases")) {
                mw.shared_gate_b = get_weight(shared_prefix + "gate_proj.biases");
                mw.shared_up_b = get_weight(shared_prefix + "up_proj.biases");
                mw.shared_down_b = get_weight(shared_prefix + "down_proj.biases");
            }

            // Shared expert gate (sigmoid)
            std::string sg_prefix = "layers." + std::to_string(i) + ".mlp.shared_expert_gate.";
            mw.shared_expert_gate_w = get_weight(sg_prefix + "weight");
            if (has_weight(sg_prefix + "scales")) {
                mw.shared_expert_gate_s = get_weight(sg_prefix + "scales");
            }
            if (has_weight(sg_prefix + "biases")) {
                mw.shared_expert_gate_b = get_weight(sg_prefix + "biases");
            }
        }

        moe_weights_.push_back(std::move(mw));
    }

    // Eval stacked weights
    std::vector<mx::array> to_eval;
    for (auto& mw : moe_weights_) {
        if (!layer_is_moe_.empty()) {
            to_eval.push_back(mw.gate_w);
            to_eval.push_back(mw.up_w);
            to_eval.push_back(mw.down_w);
        }
    }
    if (!to_eval.empty()) mx::eval(to_eval);

    int moe_count = 0;
    for (bool b : layer_is_moe_) if (b) moe_count++;
    std::cout << "[flashmlx] Built MoE weight cache: " << moe_count << " MoE layers, "
              << n_experts << " experts stacked" << std::endl;
}
```

- [ ] **Step 3: Call build_moe_weight_cache() from constructor**

In the constructor, after `build_weight_cache()`:

```cpp
    build_moe_weight_cache();
```

- [ ] **Step 4: Build and verify weight loading**

```bash
cd server/build && PATH=/opt/homebrew/bin:$PATH cmake .. && make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
cd ../..
python -c "
import sys; sys.path.insert(0, '.')
from server.python._flashmlx_engine import Engine
e = Engine('<QWEN_MOE_PATH>', 2, 512)
print(e.ping())
"
```

Expected: Prints MoE config, stacked weight info, and `flashmlx engine ready`.

- [ ] **Step 5: Commit**

```bash
git add server/
git commit -m "feat(server): MoE weight loading with expert stacking via mx::stack"
```

---

### Task 3: MoE Block + Switch MLP Implementation

**Files:**
- Modify: `server/include/flashmlx/model.h`
- Modify: `server/src/model.cpp`

- [ ] **Step 1: Add method declarations to model.h**

In the LlamaModel class:

```cpp
    mx::array moe_block(const mx::array& x, int layer);
    mx::array switch_mlp(const mx::array& x, const mx::array& indices, int layer);
    mx::array shared_expert_mlp(const mx::array& x, int layer);
```

- [ ] **Step 2: Implement moe_block()**

In `server/src/model.cpp`:

```cpp
mx::array LlamaModel::moe_block(const mx::array& x, int layer) {
    auto& mw = moe_weights_[layer];
    int k = config_.num_experts_per_tok;

    // 1. Router: project to expert logits, softmax, top-k
    // router_w shape: [num_experts, hidden_size] — use matmul(x, router_w.T)
    auto gate_logits = mx::matmul(x, mx::transpose(mw.router_w, {1, 0}));  // [B, L, num_experts]
    auto scores = mx::softmax(gate_logits, -1);

    // Top-k selection
    auto neg_scores = mx::negative(scores);
    auto top_k_inds = mx::argpartition(neg_scores, k - 1, -1);
    // Slice to keep only top-k
    int num_experts = config_.num_experts;
    auto inds = mx::slice(top_k_inds, {0, 0, 0},
                          {(int)x.shape(0), (int)x.shape(1), k});  // [B, L, k]
    auto top_k_scores = mx::take_along_axis(scores, inds, -1);     // [B, L, k]

    if (config_.norm_topk_prob) {
        auto score_sum = mx::sum(top_k_scores, -1, true);
        top_k_scores = mx::divide(top_k_scores, score_sum);
    }

    // 2. Expert execution via gather_qmm
    auto y = switch_mlp(x, inds, layer);  // [B, L, k, hidden_size]
    // Weight by routing scores and sum over experts
    y = mx::multiply(y, mx::expand_dims(top_k_scores, -1));  // [B, L, k, hidden_size]
    y = mx::sum(y, -2);  // [B, L, hidden_size]

    // 3. Shared expert (optional)
    if (mw.has_shared) {
        auto shared_out = shared_expert_mlp(x, layer);
        // Sigmoid gate
        mx::array gate_val;
        if (mw.shared_expert_gate_s.has_value()) {
            gate_val = mx::quantized_matmul(x, mw.shared_expert_gate_w,
                                            *mw.shared_expert_gate_s, mw.shared_expert_gate_b,
                                            true, config_.quant_group_size, config_.quant_bits);
        } else {
            gate_val = mx::matmul(x, mx::transpose(mw.shared_expert_gate_w, {1, 0}));
        }
        gate_val = mx::sigmoid(gate_val);
        y = mx::add(y, mx::multiply(gate_val, shared_out));
    }

    return y;
}
```

- [ ] **Step 3: Implement switch_mlp()**

```cpp
mx::array LlamaModel::switch_mlp(const mx::array& x, const mx::array& indices, int layer) {
    auto& mw = moe_weights_[layer];

    // x shape: [B, L, hidden_size]
    // indices shape: [B, L, k]
    // Expert weights: [num_experts, intermediate, hidden] (stacked)

    auto x_expanded = mx::expand_dims(x, {-2, -3});  // [B, L, 1, 1, hidden_size]

    // Optionally sort indices for better gather_qmm cache behavior
    bool do_sort = indices.size() >= 64;
    auto idx = indices;
    mx::array inv_order;

    if (do_sort) {
        int M = indices.shape(-1);  // k
        auto flat_idx = mx::reshape(indices, {-1});
        auto order = mx::argsort(flat_idx);
        inv_order = mx::argsort(order);
        auto flat_x = mx::reshape(x_expanded, {-1, 1, 1, (int)x.shape(-1)});
        x_expanded = mx::take(flat_x, mx::divide(order, mx::array(M)), 0);
        idx = mx::take(flat_idx, order, 0);
    }

    // gather_qmm: indexed quantized matmul
    auto x_gate = mx::gather_qmm(x_expanded, mw.gate_w, mw.gate_s, mw.gate_b,
                                  std::nullopt, idx, true,
                                  config_.quant_group_size, config_.quant_bits,
                                  "affine", do_sort);
    auto x_up = mx::gather_qmm(x_expanded, mw.up_w, mw.up_s, mw.up_b,
                                std::nullopt, idx, true,
                                config_.quant_group_size, config_.quant_bits,
                                "affine", do_sort);

    // SwiGLU activation
    auto activated = swiglu(x_gate, x_up);

    auto x_down = mx::gather_qmm(activated, mw.down_w, mw.down_s, mw.down_b,
                                  std::nullopt, idx, true,
                                  config_.quant_group_size, config_.quant_bits,
                                  "affine", do_sort);

    // Unsort if we sorted
    if (do_sort) {
        auto flat = mx::reshape(x_down, {-1, (int)x_down.shape(-1)});
        flat = mx::take(flat, inv_order, 0);
        x_down = mx::reshape(flat, {(int)x.shape(0), (int)x.shape(1), config_.num_experts_per_tok, -1});
    }

    return mx::squeeze(x_down, -2);
}
```

- [ ] **Step 4: Implement shared_expert_mlp()**

```cpp
mx::array LlamaModel::shared_expert_mlp(const mx::array& x, int layer) {
    auto& mw = moe_weights_[layer];
    auto gate = mx::quantized_matmul(x, mw.shared_gate_w, mw.shared_gate_s,
                                     mw.shared_gate_b, true,
                                     config_.quant_group_size, config_.quant_bits);
    auto up = mx::quantized_matmul(x, mw.shared_up_w, mw.shared_up_s,
                                   mw.shared_up_b, true,
                                   config_.quant_group_size, config_.quant_bits);
    auto activated = swiglu(gate, up);
    return mx::quantized_matmul(activated, mw.shared_down_w, mw.shared_down_s,
                                mw.shared_down_b, true,
                                config_.quant_group_size, config_.quant_bits);
}
```

- [ ] **Step 5: Build and verify compilation**

```bash
cd server/build && PATH=/opt/homebrew/bin:$PATH cmake .. && make -j$(sysctl -n hw.ncpu)
```

Expected: Builds without errors.

- [ ] **Step 6: Commit**

```bash
git add server/
git commit -m "feat(server): MoE block with gather_qmm expert routing + shared expert"
```

---

### Task 4: Per-Layer MoE Dispatch in Forward Pass

**Files:**
- Modify: `server/src/model.cpp`

- [ ] **Step 1: Update the int-offset transformer_block to dispatch MoE**

In the int-offset `transformer_block()`, replace the MLP call with conditional dispatch:

```cpp
mx::array LlamaModel::transformer_block(
    const mx::array& x, int layer,
    mx::array& cache_k, mx::array& cache_v,
    int cache_offset) {

    const auto& lw = layer_weights_[layer];

    auto normed = rms_norm(x, lw.input_norm_w);
    auto attn_out = attention(x_normed_to_attn...);  // existing attention code
    auto h = mx::add(x, attn_out);

    auto normed2 = rms_norm(h, lw.post_norm_w);

    mx::array mlp_out;
    if (!layer_is_moe_.empty() && layer_is_moe_[layer]) {
        mlp_out = moe_block(normed2, layer);
    } else {
        mlp_out = mlp_fast(normed2, layer);
    }

    return mx::add(h, mlp_out);
}
```

Do the same for the array-offset transformer_block used during prefill.

- [ ] **Step 2: Handle attention bias (Qwen2-MoE has q/k/v bias)**

Qwen2-MoE attention has `self_attn.q_proj.bias` (linear bias, separate from quantization biases). Check if `q_proj.bias` exists and add it after the quantized_matmul:

In the attention function, after Q/K/V projections:
```cpp
    // Add attention bias if present (Qwen2-MoE)
    std::string q_bias_key = prefix + ".q_proj.bias";
    if (has_weight(q_bias_key)) {
        q = mx::add(q, get_weight(q_bias_key));
        k = mx::add(k, get_weight(prefix + ".k_proj.bias"));
        v = mx::add(v, get_weight(prefix + ".v_proj.bias"));
    }
```

- [ ] **Step 3: Build, copy, and test end-to-end**

```bash
cd server/build && PATH=/opt/homebrew/bin:$PATH make -j$(sysctl -n hw.ncpu)
cp _flashmlx_engine*.so ../python/
cd ../..
source .venv/bin/activate
python -c "
import sys; sys.path.insert(0, '.')
from server.python._flashmlx_engine import Engine
import time

MODEL = '<QWEN_MOE_PATH>'
e = Engine(MODEL, 2, 512)
e.start()
e.submit_request('test-moe', list(range(1, 17)), 10, 0.0)

tokens = []
for _ in range(500):
    for out in e.poll_tokens():
        if out.request_id == 'test-moe':
            if out.done:
                break
            tokens.extend(out.tokens)
    else:
        time.sleep(0.01)
        continue
    break

e.stop()
print(f'Generated {len(tokens)} tokens: {tokens}')
"
```

Expected: Generates 10 tokens. Output may be gibberish initially — debug if it crashes.

- [ ] **Step 4: Verify output quality against mlx-lm**

```python
# Compare outputs with mlx-lm
from mlx_lm import load, generate
model, tok = load('mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit')
text = generate(model, tok, prompt="What is 2+2?", max_tokens=20, verbose=False)
print(f"mlx-lm: {text}")
```

- [ ] **Step 5: Commit**

```bash
git add server/
git commit -m "feat(server): MoE per-layer dispatch + attention bias support"
```

---

### Task 5: Test + Benchmark

**Files:**
- Modify: `server/tests/test_engine.py`

- [ ] **Step 1: Add MoE model test**

Add to `server/tests/test_engine.py`:

```python
MOE_PATH = "<QWEN_MOE_PATH>"

class TestMoEEngine:
    @pytest.fixture(scope="class")
    def moe_engine(self):
        e = Engine(MOE_PATH, 2, 512)
        e.start()
        yield e
        e.stop()

    def test_moe_generates_tokens(self, moe_engine):
        moe_engine.submit_request("moe-test", list(range(1, 17)), 10, 0.0)
        tokens = collect_tokens(moe_engine, "moe-test")
        assert len(tokens) == 10
        assert all(isinstance(t, int) for t in tokens)

    def test_moe_concurrent(self, moe_engine):
        moe_engine.submit_request("moe-a", list(range(1, 9)), 5, 0.0)
        moe_engine.submit_request("moe-b", list(range(1, 9)), 5, 0.0)
        tokens_a = collect_tokens(moe_engine, "moe-a")
        tokens_b = collect_tokens(moe_engine, "moe-b")
        assert len(tokens_a) == 5
        assert len(tokens_b) == 5
```

- [ ] **Step 2: Run all tests (MoE + existing)**

```bash
source .venv/bin/activate
python -m pytest server/tests/test_engine.py -v --timeout=120
```

Expected: All tests pass (both Qwen3-0.6B and MoE tests).

- [ ] **Step 3: Benchmark MoE via HTTP**

Start the server and benchmark:

```bash
python -m server.run <QWEN_MOE_PATH> --port 8080 &
sleep 15
python server/bench_server.py --url http://localhost:8080/v1/chat/completions --max-tokens 64
kill %1
```

- [ ] **Step 4: Compare with mlx-lm baseline**

```python
import time
from mlx_lm import load, generate

model, tok = load('mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit')
generate(model, tok, prompt="Hello", max_tokens=20, verbose=False)  # warmup

t0 = time.perf_counter()
for _ in range(4):
    generate(model, tok, prompt="Explain attention", max_tokens=64, verbose=False)
t1 = time.perf_counter()
print(f"mlx-lm MoE: {256/(t1-t0):.1f} tok/s (4x64 sequential)")
```

- [ ] **Step 5: Commit**

```bash
git add server/tests/ server/
git commit -m "feat(server): MoE tests + benchmarks — Qwen1.5-MoE-A2.7B working"
```

---

### Task 6: Regression Test + Documentation

**Files:**
- Modify: `docs/model-configs.md`
- Modify: `README.md`

- [ ] **Step 1: Run existing dense model tests to verify no regression**

```bash
python -m pytest server/tests/test_engine.py::TestEngine -v --timeout=60
python -m pytest tests/ -x -q
```

Expected: All 45 engine tests + 4+ server tests pass.

- [ ] **Step 2: Update model-configs.md with MoE section**

Add a MoE section documenting the Qwen1.5-MoE performance and configuration.

- [ ] **Step 3: Update README.md supported models list**

Add MoE model types to the supported models section.

- [ ] **Step 4: Commit**

```bash
git add docs/ README.md
git commit -m "docs: add MoE model support documentation and benchmark results"
```
