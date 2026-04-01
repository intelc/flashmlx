# MoE Support for FlashMLX C++ Server — Design Spec

## Goal

Add Mixture-of-Experts layer support to the C++ inference server, enabling models like Qwen1.5-MoE-A2.7B and Qwen3-30B-A3B. Reuse existing attention/embedding/KV cache infrastructure — only add MoE routing and expert execution.

## Context

FlashMLX's C++ server currently supports the Llama architecture family (dense attention + SwiGLU MLP). MoE models use the same attention mechanism but replace the MLP in some or all layers with a router + multiple expert MLPs. This is the first of three planned extensions:
1. **MoE routing** (this spec) — unlocks Qwen-MoE, DeepSeek, Granite-MoE, Phi-MoE
2. Mamba-2 SSM layers (future) — unlocks Nemotron-H hybrid
3. Hybrid layer dispatch (future) — combines MoE + Mamba + Attention

## Architecture

### Layer Structure

Standard Llama layer:
```
x → RMSNorm → Attention → residual → RMSNorm → MLP → residual
```

MoE layer (replaces MLP only):
```
x → RMSNorm → Attention → residual → RMSNorm → MoE_Block → residual
                                                    ├── Router (top-k expert selection)
                                                    ├── Expert MLPs (gather-scatter execution)
                                                    └── Shared Expert (optional, always active)
```

Attention is unchanged. The MoE block is a drop-in replacement for the MLP sublayer.

### Model Detection

At load time, detect MoE models via:
1. Check `model_type` in config.json — known MoE types: `qwen2_moe`, `qwen3_moe`, `granitemoe`, `deepseek`, `deepseek_v2`, `phimoe`
2. Parse MoE-specific config fields: `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`
3. Detect which layers are MoE vs dense by checking for `experts.0` in weight keys

### Per-Layer Dispatch

The forward pass checks each layer's type:
```cpp
for (int i = 0; i < num_layers; i++) {
    auto normed = rms_norm(x, lw.input_norm_w);
    auto attn_out = attention(normed, i, cache_k, cache_v, offset);
    auto h = x + attn_out;
    auto normed2 = rms_norm(h, lw.post_norm_w);

    mx::array mlp_out;
    if (layer_is_moe_[i]) {
        mlp_out = moe_block(normed2, i);
    } else {
        mlp_out = mlp_fast(normed2, i);
    }
    x = h + mlp_out;
}
```

### MoE Block

```cpp
mx::array moe_block(const mx::array& x, int layer) {
    // x shape: [B, L, hidden_size]

    // 1. Router: project to expert logits, softmax, select top-k
    auto gate_logits = linear(x, router_weight);       // [B, L, num_experts]
    auto scores = mx::softmax(gate_logits, -1);        // [B, L, num_experts]
    auto top_k_indices = top_k_selection(scores, k);    // [B, L, k]
    auto top_k_scores = gather(scores, top_k_indices);  // [B, L, k]
    if (norm_topk_prob)
        top_k_scores = top_k_scores / sum(top_k_scores, -1, keepdim);

    // 2. Expert execution via gather-scatter
    //    For each of k active experts, run through that expert's SwiGLU MLP
    auto expert_out = mx::zeros_like(x);
    for (int e = 0; e < k; e++) {
        auto expert_idx = slice(top_k_indices, e);  // [B, L] — which expert
        auto weight = slice(top_k_scores, e);       // [B, L] — routing weight
        auto out = switch_mlp(x, expert_idx, layer);
        expert_out = expert_out + out * expand_dims(weight, -1);
    }

    // 3. Optional shared expert (always active, gated)
    if (has_shared_expert_[layer]) {
        auto shared = shared_expert_mlp(x, layer);
        auto gate = mx::sigmoid(shared_expert_gate(x, layer));
        expert_out = expert_out + gate * shared;
    }

    return expert_out;
}
```

### Switch MLP (Expert Execution)

Each expert is a standard SwiGLU MLP (gate_proj, up_proj, down_proj). Expert weights are stored stacked:
```
expert_gate_proj: [num_experts, moe_intermediate_size, hidden_size]  (quantized)
expert_up_proj:   [num_experts, moe_intermediate_size, hidden_size]  (quantized)
expert_down_proj: [num_experts, hidden_size, moe_intermediate_size]  (quantized)
```

The `switch_mlp` gathers the relevant expert weights for each token and executes via indexed quantized matmul:
```cpp
mx::array switch_mlp(const mx::array& x, const mx::array& expert_idx, int layer) {
    // expert_idx: [B, L] — index into [num_experts, ...] weight tensors
    // Use mx::take along expert dimension to select the right weights per token
    // Then run standard SwiGLU: down(silu(gate(x)) * up(x))
    auto& mw = moe_weights_[layer];
    auto gate = indexed_linear(x, expert_idx, mw.gate_w, mw.gate_s, mw.gate_b);
    auto up = indexed_linear(x, expert_idx, mw.up_w, mw.up_s, mw.up_b);
    auto activated = swiglu(gate, up);
    return indexed_linear(activated, expert_idx, mw.down_w, mw.down_s, mw.down_b);
}
```

The `indexed_linear` function selects the expert's weight slice and runs quantized_matmul. For quantized models, this uses the stacked weight tensors with index-based gather.

## Config Extensions

```cpp
struct ModelConfig {
    // ... existing fields ...

    // MoE parameters (0 = not MoE)
    int num_experts = 0;
    int num_experts_per_tok = 0;      // top-k routing
    int moe_intermediate_size = 0;     // per-expert intermediate size
    int num_shared_experts = 0;        // shared experts (always active)
    int shared_expert_intermediate_size = 0;
    bool norm_topk_prob = false;       // normalize top-k routing scores
};
```

## Weight Loading

MoE weight naming pattern (Qwen1.5-MoE style):
```
layers.N.mlp.gate.weight                    → router [num_experts, hidden_size]
layers.N.mlp.experts.E.gate_proj.weight     → expert E gate [moe_intermediate, hidden]
layers.N.mlp.experts.E.gate_proj.scales     → expert E gate scales
layers.N.mlp.experts.E.up_proj.weight       → expert E up
layers.N.mlp.experts.E.down_proj.weight     → expert E down
layers.N.mlp.shared_expert.gate_proj.weight → shared expert gate
layers.N.mlp.shared_expert.up_proj.weight   → shared expert up
layers.N.mlp.shared_expert.down_proj.weight → shared expert down
layers.N.mlp.shared_expert_gate.weight      → shared expert gating (sigmoid)
```

Detection: a layer is MoE if `layers.N.mlp.experts.0.gate_proj.weight` exists in the weight dict.

### MoE Weight Cache

Extend `LayerWeights` with MoE-specific cached references:
```cpp
struct MoEWeights {
    mx::array router_w;                        // [num_experts, hidden_size]
    // Stacked expert weights: [num_experts, dim1, dim2]
    mx::array gate_w, gate_s, up_w, up_s, down_w, down_s;
    std::optional<mx::array> gate_b, up_b, down_b;
    // Shared expert (optional)
    bool has_shared;
    mx::array shared_gate_w, shared_gate_s;
    std::optional<mx::array> shared_gate_b;
    mx::array shared_up_w, shared_up_s;
    std::optional<mx::array> shared_up_b;
    mx::array shared_down_w, shared_down_s;
    std::optional<mx::array> shared_down_b;
    mx::array shared_gate_proj_w;  // sigmoid gate for shared expert
};
```

Expert weights are stacked at load time: iterate over `experts.0` through `experts.N-1`, stack into single tensors `[num_experts, ...]` for efficient indexed access.

## Files Modified

| File | Change |
|------|--------|
| `server/include/flashmlx/model.h` | Add MoE config fields, `MoEWeights` struct, `moe_block()`, `switch_mlp()`, `indexed_linear()`, per-layer `is_moe` flag |
| `server/src/model.cpp` | Implement MoE config parsing, MoE weight loading + stacking, `moe_block()`, `switch_mlp()`, `indexed_linear()`, per-layer dispatch in `transformer_block()` |
| `server/src/engine.cpp` | No change |
| `server/src/scheduler.cpp` | No change |
| `server/src/kv_pool.cpp` | No change |

## Model Types Unlocked

| Model Type | Example | Active Params | Experts |
|-----------|---------|--------------|---------|
| `qwen2_moe` | Qwen1.5-MoE-A2.7B | 2.7B | 60 experts, top-4 |
| `qwen3_moe` | Qwen3-30B-A3B | 3B | 128 experts, top-8 |
| `granitemoe` | Granite-MoE | varies | varies |
| `deepseek` | DeepSeek-MoE | varies | group-limited routing |
| `phimoe` | Phi-3.5-MoE | varies | varies |

All use the same pattern: standard attention + MoE-MLP replacement.

## Test Plan

1. Download `mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit`
2. Load in C++ server — verify config parsing, all expert weights loaded
3. Generate tokens — verify coherent output (compare with mlx-lm)
4. Benchmark C=1 and C=4 throughput
5. Run existing Qwen3-0.6B and Llama-3-8B tests — verify no regression

## What's NOT in This Spec

- Mamba-2 SSM layers (next spec)
- Hybrid layer dispatch (next spec)
- Expert parallelism across devices
- MoE auxiliary loss / load balancing
- DeepSeek-V2/V3 group-limited routing (only standard top-k for V1)

## Success Criteria

1. Server loads Qwen1.5-MoE-A2.7B-Chat-4bit, generates coherent text
2. Performance within 20% of mlx-lm on same model
3. Existing Llama/Qwen3 dense models unaffected (all tests pass)
