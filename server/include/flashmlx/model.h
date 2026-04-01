#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

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

    // Quantization
    int quant_bits = 0;       // 0 = no quantization
    int quant_group_size = 64;
};

class LlamaModel {
public:
    explicit LlamaModel(const std::string& model_path);

    /// Full forward pass: input_ids [B, L] -> logits [B, L, vocab_size]
    mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        const mx::array& cache_offsets);

    /// Overload with integer offset (avoids mx::eval sync for homogeneous batching)
    mx::array forward(
        const mx::array& input_ids,
        std::vector<mx::array>& cache_keys,
        std::vector<mx::array>& cache_values,
        int cache_offset);

    const ModelConfig& config() const { return config_; }

    /// Debug: run forward pass and return top-5 token IDs from last position
    std::vector<int> debug_forward(const std::vector<int>& token_ids);

    /// Debug: return embedding for given token IDs as flat float vector
    std::vector<float> debug_embed(const std::vector<int>& token_ids);

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

    // Helpers
    mx::array get_weight(const std::string& name) const;
    bool has_weight(const std::string& name) const;

    ModelConfig config_;
    std::unordered_map<std::string, mx::array> weights_;
    int head_dim_ = 0;
};

} // namespace flashmlx
