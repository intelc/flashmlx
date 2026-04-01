#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/fast.h>

namespace mx = mlx::core;

namespace flashmlx {

struct ModelConfig {
    int hidden_size = 4096;
    int intermediate_size = 14336;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = 8;
    int head_dim = 0;  // 0 = hidden_size / num_attention_heads
    int vocab_size = 128256;
    int max_position_embeddings = 8192;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 500000.0f;
    bool tie_word_embeddings = false;

    // Activation dtype (detected from weights — float16 or bfloat16)
    mx::Dtype activation_dtype = mx::float16;

    // Quantization
    int quant_bits = 0;       // 0 = no quantization
    int quant_group_size = 64;

    // MoE parameters (num_experts == 0 means not MoE)
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    bool norm_topk_prob = false;
};

// MoE stacked expert weights for a single layer
// All fields use std::optional because mx::array has no default constructor.
struct MoEWeights {
    std::optional<mx::array> router_w;  // [num_experts, hidden_size] — may or may not be quantized
    std::optional<mx::array> router_s, router_b;  // scales/biases if router is quantized
    // Stacked expert weights: [num_experts, dim1, dim2] — quantized
    std::optional<mx::array> gate_w, gate_s, up_w, up_s, down_w, down_s;
    std::optional<mx::array> gate_b, up_b, down_b;
    // Shared expert (standard quantized linear)
    bool has_shared = false;
    std::optional<mx::array> shared_gate_w, shared_gate_s, shared_up_w, shared_up_s, shared_down_w, shared_down_s;
    std::optional<mx::array> shared_gate_b, shared_up_b, shared_down_b;
    // Shared expert gating [1, hidden_size]
    std::optional<mx::array> shared_expert_gate_w;
    std::optional<mx::array> shared_expert_gate_s, shared_expert_gate_b;
    bool shared_expert_gate_quantized = false;
};

/// Abstract base for all model architectures (LLaMA, Nemotron-H, etc.)
class ModelBase {
public:
    virtual ~ModelBase() = default;

    virtual mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        int cache_offset) = 0;

    virtual mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        const mx::array& cache_offsets) = 0;

    virtual const ModelConfig& config() const = 0;
    virtual std::vector<int> debug_forward(const std::vector<int>& token_ids) = 0;
    virtual std::vector<float> debug_embed(const std::vector<int>& token_ids) = 0;
};

class LlamaModel : public ModelBase {
public:
    explicit LlamaModel(const std::string& model_path);

    /// Full forward pass: input_ids [B, L] -> logits [B, L, vocab_size]
    mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        const mx::array& cache_offsets) override;

    /// Overload with integer offset (avoids mx::eval sync for homogeneous batching)
    mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        int cache_offset) override;

    const ModelConfig& config() const override { return config_; }

    /// Debug: run forward pass and return top-5 token IDs from last position
    std::vector<int> debug_forward(const std::vector<int>& token_ids) override;

    /// Debug: return embedding for given token IDs as flat float vector
    std::vector<float> debug_embed(const std::vector<int>& token_ids) override;

private:
    void load_config(const std::string& model_path);
    void load_weights(const std::string& model_path);

    // Building blocks
    mx::array rms_norm(const mx::array& x, const mx::array& weight);
    mx::array embed(const mx::array& input_ids);
    mx::array lm_head(const mx::array& x);
    mx::array linear(const mx::array& x, const std::string& prefix);
    mx::array transformer_block(
        const mx::array& x, int layer,
        mx::array& cache_k, mx::array& cache_v,
        const mx::array& cache_offsets);
    mx::array transformer_block(
        const mx::array& x, int layer,
        mx::array& cache_k, mx::array& cache_v,
        int cache_offset);
    mx::array attention(
        const mx::array& x, int layer,
        mx::array& cache_k, mx::array& cache_v,
        const mx::array& cache_offsets);
    mx::array attention(
        const mx::array& x, int layer,
        mx::array& cache_k, mx::array& cache_v,
        int cache_offset);
    mx::array mlp(const mx::array& x, int layer);
    mx::array mlp_fast(const mx::array& x, int layer);

    // MoE building blocks
    mx::array moe_block(const mx::array& x, int layer);
    mx::array switch_mlp(const mx::array& x, const mx::array& indices, int layer);
    mx::array shared_expert_mlp(const mx::array& x, int layer);

    // Fast linear using pre-resolved weight references (no hash lookup)
    mx::array linear_fast(const mx::array& x, const mx::array& w, const mx::array& s,
                          const std::optional<mx::array>& b);

    // Helpers
    mx::array get_weight(const std::string& name) const;
    bool has_weight(const std::string& name) const;

    ModelConfig config_;
    std::unordered_map<std::string, mx::array> weights_;
    int head_dim_ = 0;

    // Pre-built weight reference cache for fast lookup
    struct LayerWeights {
        // Attention
        mx::array q_w, q_s, k_w, k_s, v_w, v_s, o_w, o_s;
        std::optional<mx::array> q_b, k_b, v_b, o_b;
        // Attention linear biases (not quantization biases — e.g. Qwen2-MoE)
        std::optional<mx::array> q_bias, k_bias, v_bias;
        mx::array input_norm_w, post_norm_w;
        bool has_q_norm = false;
        mx::array q_norm_w, k_norm_w;
        // MLP (only used for dense layers)
        mx::array gate_w, gate_s, up_w, up_s, down_w, down_s;
        std::optional<mx::array> gate_b, up_b, down_b;
    };
    std::vector<LayerWeights> layer_weights_;
    std::optional<mx::array> norm_w_;   // cached final norm weight

    // MoE weight caches
    std::vector<bool> layer_is_moe_;
    std::vector<MoEWeights> moe_weights_;  // indexed by MoE layer index

    void build_weight_cache();
    void build_moe_weight_cache();
};

// Forward-declare block type enum (defined in nemotron_h.cpp)
enum class BlockType;

class NemotronHModel : public ModelBase {
public:
    explicit NemotronHModel(const std::string& model_path);

    mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        const mx::array& cache_offsets) override;

    mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        int cache_offset) override;

    const ModelConfig& config() const override { return config_; }
    std::vector<int> debug_forward(const std::vector<int>& token_ids) override;
    std::vector<float> debug_embed(const std::vector<int>& token_ids) override;

    /// Number of attention layers (for KV cache allocation)
    int num_attn_layers() const { return num_attn_layers_; }

private:
    void load_config(const std::string& model_path);
    void load_weights(const std::string& model_path);
    void parse_pattern();
    void build_ssm_kernel();
    void init_mamba_states(int batch_size);

    // Building blocks
    mx::array rms_norm(const mx::array& x, const mx::array& weight);
    mx::array embed(const mx::array& input_ids);
    mx::array lm_head_proj(const mx::array& x);
    mx::array linear(const mx::array& x, const std::string& prefix);

    mx::array mamba_block(const mx::array& x, int layer, int mamba_idx);
    mx::array attention_block(const mx::array& x, int layer, int attn_idx,
                              mx::array& cache_k, mx::array& cache_v, int cache_offset);
    mx::array attention_block(const mx::array& x, int layer, int attn_idx,
                              mx::array& cache_k, mx::array& cache_v,
                              const mx::array& cache_offsets);
    mx::array mlp_block(const mx::array& x, int layer);

    mx::array get_weight(const std::string& name) const;
    bool has_weight(const std::string& name) const;

    ModelConfig config_;
    std::unordered_map<std::string, mx::array> weights_;

    // Nemotron-H specific config
    int mamba_num_heads_ = 96;
    int mamba_head_dim_ = 80;
    int ssm_state_size_ = 128;
    int conv_kernel_ = 4;
    int n_groups_ = 8;
    float time_step_min_ = 0.001f;
    float time_step_max_ = 0.1f;
    bool use_conv_bias_ = true;

    // Derived dimensions
    int d_inner_ = 0;       // mamba_num_heads * mamba_head_dim
    int conv_dim_ = 0;      // d_inner + 2 * n_groups * ssm_state_size
    int attn_head_dim_ = 0;

    // Pattern
    std::string hybrid_override_pattern_;
    std::vector<BlockType> block_types_;
    std::vector<int> attn_layer_indices_;   // layer -> attn cache index (-1 if not attn)
    std::vector<int> mamba_layer_indices_;  // layer -> mamba state index (-1 if not mamba)
    int num_attn_layers_ = 0;
    int num_mamba_layers_ = 0;

    // SSM Metal kernel
    mx::fast::CustomKernelFunction ssm_kernel_;

    // Mamba states (managed internally, not in KV pool)
    std::vector<mx::array> mamba_conv_states_;   // one per Mamba block
    std::vector<mx::array> mamba_ssm_states_;    // one per Mamba block
    bool mamba_states_initialized_ = false;
};

} // namespace flashmlx
