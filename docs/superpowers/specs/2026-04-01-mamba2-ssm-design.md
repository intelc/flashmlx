# Mamba-2 SSM + Hybrid Layer Support — Design Spec

## Goal

Add Mamba-2 SSM layer support and hybrid layer dispatch to the C++ inference server, enabling Nemotron-3-Nano-4B (42 layers: ~38 Mamba-2 + 4 Attention). This completes the second of three planned architecture extensions (MoE done, Mamba-2 here, combined hybrid future).

## Context

FlashMLX C++ server supports Llama-family dense models and MoE models. Nemotron-H is a hybrid architecture with three layer types: Mamba-2 SSM, standard Attention, and MLP. The hybrid pattern is specified in config.json as a string like `M-M-M-MM-M-M*-M-M*-...` where `M` = Mamba mixer, `*` = Attention, `-` = MLP connector.

## Target Model

Nemotron-3-Nano-4B (`mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit`):
- model_type: `nemotron_h`
- 42 layers, hidden_size: 3136, vocab_size: 131072
- ~38 Mamba-2 layers, 4 Attention layers
- Mamba config: 96 heads, head_dim 80, ssm_state_size 128, conv_kernel 4, n_groups 8
- Attention config: 40 heads, 8 KV heads, head_dim 128

## Architecture

### Layer Types

Each layer in the hybrid model has a mixer type (Mamba-2 or Attention) and always has an MLP sublayer:

```
Mamba layer:     x → RMSNorm → Mamba2Mixer → residual → RMSNorm → MLP → residual
Attention layer: x → RMSNorm → Attention    → residual → RMSNorm → MLP → residual
```

The MLP sublayer is identical for both. Only the mixer differs.

### Per-Layer Dispatch

```cpp
enum class LayerType { ATTENTION, MAMBA };

for (int i = 0; i < num_layers; i++) {
    auto normed = rms_norm(x, lw.input_norm_w);
    
    mx::array mixer_out;
    if (layer_types_[i] == LayerType::MAMBA) {
        mixer_out = mamba_mixer(normed, i, mamba_states_[slot][i]);
    } else {
        mixer_out = attention(normed, i, cache_k, cache_v, offset);
    }
    auto h = x + mixer_out;
    
    auto normed2 = rms_norm(h, lw.post_norm_w);
    auto mlp_out = mlp_fast(normed2, i);
    x = h + mlp_out;
}
```

### Mamba-2 Mixer

The Mamba-2 SSM mixer for single-token decode:

```cpp
mx::array mamba_mixer(const mx::array& x, int layer, MambaState& state) {
    // 1. Input projection: x → combined projection
    //    Splits into: z (gate), x_conv (for conv1d), B, C, dt
    auto proj = linear(x, in_proj_w);
    auto [z, x_conv, B, C, dt] = split(proj, split_sizes);
    
    // 2. Conv1D with sliding window state
    //    conv_state is [1, conv_kernel-1, d_inner], shift left and append
    state.conv = concat(state.conv[:, 1:, :], x_conv, axis=1);
    auto x_after_conv = depthwise_conv1d(state.conv, conv_weight);
    x_after_conv = silu(x_after_conv);
    
    // 3. SSM update (custom Metal kernel)
    //    Computes: dA = exp(A * dt)
    //              state_new = dA * state_old + x * dt * B
    //              y = state_new · C + x * D
    auto [y, new_ssm] = ssm_kernel(x_after_conv, A_log, B, C, D, dt, state.ssm);
    state.ssm = new_ssm;
    
    // 4. Output gate: y * silu(z)
    auto out = y * silu(z);
    
    // 5. Norm + output projection
    out = rms_norm(out, norm_weight);
    return linear(out, out_proj_w);
}
```

### SSM Metal Kernel

The SSM selective scan is implemented as a custom Metal shader (same approach as mlx-lm). The kernel source string is embedded in C++ and registered via `mx::fast::metal_kernel`:

```cpp
static auto ssm_kernel = mx::fast::metal_kernel(
    "ssm_kernel",
    {"X", "A_log", "B", "C", "D", "dt", "state_in"},
    {"out", "state_out"},
    R"(
        // ... Metal shader source (30 lines) ...
        // Standard Mamba-2 selective state space update
        auto dA = fast::exp(A * dt_);
        auto state = dA * i_state[idx] + dB_by_x;
        o_state[idx] = static_cast<T>(state);
        acc += state * C_[s_idx];
    )"
);
```

For prefill (processing multiple tokens), the SSM runs sequentially token-by-token since the recurrence is inherently sequential. There is also an `ssm_attn` mode for prefill that uses attention-like parallel computation over the sequence — but for V1 we use the simpler sequential approach.

### State Management

Each Mamba layer needs two state arrays per request (instead of KV cache):

```cpp
struct MambaState {
    mx::array conv_state;  // [1, conv_kernel-1, d_inner]
    mx::array ssm_state;   // [1, n_groups, n_heads, head_dim, ssm_state_size]
};
```

Attention layers continue to use the existing KV cache (concat-based).

The KV cache pool is extended to manage per-layer heterogeneous state:

```cpp
struct SlotState {
    // Per-layer: either KV cache or Mamba state
    // Attention layers: cache_k, cache_v (concat-based, grow with sequence)
    // Mamba layers: conv_state, ssm_state (fixed size, updated in place)
    struct LayerState {
        LayerType type;
        mx::array cache_k, cache_v;      // for attention
        mx::array conv_state, ssm_state;  // for mamba
    };
    std::vector<LayerState> layers;
};
```

### Config Extensions

```cpp
struct ModelConfig {
    // ... existing fields ...
    
    // Hybrid layer dispatch
    std::string hybrid_pattern;  // raw pattern string from config
    // Parsed into layer_types vector during config loading
    
    // Mamba-2 parameters (all 0 for non-hybrid models)
    int mamba_num_heads = 0;
    int mamba_head_dim = 0;
    int ssm_state_size = 0;
    int conv_kernel = 4;
    int n_groups = 1;
    int mamba_d_inner = 0;  // intermediate_size for Mamba projection
};
```

### Hybrid Pattern Parsing

The pattern string `M-M-M-MM-M-M*-M-M*-...` is parsed character by character:
- `M` → MAMBA layer
- `*` → ATTENTION layer  
- `-` → separator (not a layer)

This produces a `vector<LayerType>` of length `num_hidden_layers`.

### Weight Structure (Nemotron-H)

Mamba layers have different weight keys than attention layers:

```
# Mamba layer weights:
layers.N.mixer.in_proj.weight       → [d_inner * 2 + 2*n_groups*state_size + n_heads, hidden_size]
layers.N.mixer.conv1d.weight        → [d_inner, 1, conv_kernel]
layers.N.mixer.A_log                → [n_heads]
layers.N.mixer.D                    → [n_heads]
layers.N.mixer.dt_bias              → [n_heads]
layers.N.mixer.norm.weight          → [d_inner]
layers.N.mixer.out_proj.weight      → [hidden_size, d_inner]

# Attention layer weights (same as Llama):
layers.N.mixer.q_proj.weight        → standard attention
layers.N.mixer.k_proj.weight
layers.N.mixer.v_proj.weight
layers.N.mixer.o_proj.weight

# MLP (both layer types):
layers.N.mlp.gate_proj.weight
layers.N.mlp.up_proj.weight
layers.N.mlp.down_proj.weight
```

Note: Nemotron-H uses `mixer` prefix instead of `self_attn` for the attention sublayer.

### Files Modified

| File | Change |
|------|--------|
| `server/include/flashmlx/model.h` | Add LayerType enum, Mamba config, MambaState, MambaWeights, mamba_mixer(), hybrid pattern parsing |
| `server/src/model.cpp` | Implement mamba_mixer(), SSM Metal kernel, conv1d, pattern parsing, per-layer dispatch, Mamba weight loading |
| `server/include/flashmlx/kv_pool.h` | Extend pool to manage per-layer heterogeneous state (KV + Mamba) |
| `server/src/kv_pool.cpp` | Allocate Mamba states for Mamba layers, KV for attention layers |
| `server/src/scheduler.cpp` | Pass layer states correctly to forward pass |

### What's NOT in This Spec

- Combined MoE + Mamba (Nemotron-30B-A3B — future, needs both)
- Parallel SSM prefill (ssm_attn mode) — sequential prefill for V1
- Mamba-1 support (different math, different kernel)
- Variable-length Mamba state (state size is fixed per model)

### Test Plan

1. Download `mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit`
2. Load in C++ server — verify pattern parsed, layers detected correctly
3. Generate tokens — verify coherent output vs mlx-lm
4. Benchmark: target within 20% of mlx-lm
5. Run existing tests for dense + MoE regression

### Success Criteria

1. Server loads Nemotron-3-Nano-4B, generates coherent text
2. Performance within 20% of mlx-lm
3. Existing Llama/Qwen/MoE models unaffected (all tests pass)
