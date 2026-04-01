#include "flashmlx/model.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace flashmlx {

// ---------------------------------------------------------------------------
// Minimal JSON value parser (handles ints, floats, bools, strings)
// ---------------------------------------------------------------------------

namespace {

// Trim whitespace
std::string trim(const std::string& s) {
    auto a = s.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return "";
    auto b = s.find_last_not_of(" \t\n\r");
    return s.substr(a, b - a + 1);
}

// Find the value string for a top-level key in JSON text.
// Handles nested objects by counting braces.
std::string json_value_for_key(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos++; // skip ':'
    // skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r'))
        pos++;
    if (pos >= json.size()) return "";

    // If it's a nested object, grab the whole thing
    if (json[pos] == '{') {
        int depth = 0;
        size_t start = pos;
        for (size_t i = pos; i < json.size(); i++) {
            if (json[i] == '{') depth++;
            else if (json[i] == '}') { depth--; if (depth == 0) return json.substr(start, i - start + 1); }
        }
        return "";
    }

    // Otherwise, grab until comma, closing brace, or end
    size_t start = pos;
    // Handle strings
    if (json[pos] == '"') {
        auto end = json.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return json.substr(pos + 1, end - pos - 1);
    }
    // Numbers, bools, null
    size_t end = json.find_first_of(",}\n\r", pos);
    if (end == std::string::npos) end = json.size();
    return trim(json.substr(start, end - start));
}

int json_int(const std::string& json, const std::string& key, int def) {
    auto v = json_value_for_key(json, key);
    if (v.empty()) return def;
    try { return std::stoi(v); } catch (...) { return def; }
}

float json_float(const std::string& json, const std::string& key, float def) {
    auto v = json_value_for_key(json, key);
    if (v.empty()) return def;
    try { return std::stof(v); } catch (...) { return def; }
}

bool json_bool(const std::string& json, const std::string& key, bool def) {
    auto v = json_value_for_key(json, key);
    if (v == "true") return true;
    if (v == "false") return false;
    return def;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

void LlamaModel::load_config(const std::string& model_path) {
    std::string config_file = model_path + "/config.json";
    std::ifstream f(config_file);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + config_file);
    std::stringstream buf;
    buf << f.rdbuf();
    std::string json = buf.str();

    config_.hidden_size = json_int(json, "hidden_size", 4096);
    config_.intermediate_size = json_int(json, "intermediate_size", 14336);
    config_.num_hidden_layers = json_int(json, "num_hidden_layers", 32);
    config_.num_attention_heads = json_int(json, "num_attention_heads", 32);
    config_.num_key_value_heads = json_int(json, "num_key_value_heads", 8);
    config_.head_dim = json_int(json, "head_dim", 0);  // 0 = compute from hidden_size/heads
    config_.vocab_size = json_int(json, "vocab_size", 128256);
    config_.max_position_embeddings = json_int(json, "max_position_embeddings", 8192);
    config_.rms_norm_eps = json_float(json, "rms_norm_eps", 1e-5f);
    config_.rope_theta = json_float(json, "rope_theta", 500000.0f);
    config_.tie_word_embeddings = json_bool(json, "tie_word_embeddings", false);

    // Parse nested quantization object
    auto quant_obj = json_value_for_key(json, "quantization");
    if (!quant_obj.empty() && quant_obj[0] == '{') {
        config_.quant_bits = json_int(quant_obj, "bits", 0);
        config_.quant_group_size = json_int(quant_obj, "group_size", 64);
    }

    // Use explicit head_dim from config if present, else compute from hidden_size
    head_dim_ = config_.head_dim > 0 ? config_.head_dim : config_.hidden_size / config_.num_attention_heads;
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

void LlamaModel::load_weights(const std::string& model_path) {
    // Find all safetensors files
    std::vector<std::string> safetensor_files;
    for (auto& entry : std::filesystem::directory_iterator(model_path)) {
        auto p = entry.path().string();
        if (p.size() >= 12 && p.substr(p.size() - 12) == ".safetensors") {
            safetensor_files.push_back(p);
        }
    }
    if (safetensor_files.empty())
        throw std::runtime_error("No .safetensors files found in " + model_path);

    std::sort(safetensor_files.begin(), safetensor_files.end());

    for (auto& file : safetensor_files) {
        std::cout << "[flashmlx] Loading " << file << std::endl;
        auto [w, meta] = mx::load_safetensors(file);
        for (auto& [key, arr] : w) {
            // Strip "model." prefix if present
            std::string clean_key = key;
            if (clean_key.substr(0, 6) == "model.") {
                clean_key = clean_key.substr(6);
            }
            weights_.insert_or_assign(clean_key, std::move(arr));
        }
    }

    // Eval all weights to materialize them
    std::vector<mx::array> to_eval;
    to_eval.reserve(weights_.size());
    for (auto& [k, v] : weights_) {
        to_eval.push_back(v);
    }
    mx::eval(to_eval);

    std::cout << "[flashmlx] Loaded " << weights_.size() << " weight tensors" << std::endl;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

LlamaModel::LlamaModel(const std::string& model_path) {
    load_config(model_path);
    std::cout << "[flashmlx] Config: hidden=" << config_.hidden_size
              << " layers=" << config_.num_hidden_layers
              << " heads=" << config_.num_attention_heads
              << " kv_heads=" << config_.num_key_value_heads
              << " vocab=" << config_.vocab_size
              << " quant=" << config_.quant_bits << "bit"
              << std::endl;
    load_weights(model_path);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

mx::array LlamaModel::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end())
        throw std::runtime_error("Weight not found: " + name);
    return it->second;
}

bool LlamaModel::has_weight(const std::string& name) const {
    return weights_.find(name) != weights_.end();
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

mx::array LlamaModel::rms_norm(const mx::array& x, const mx::array& weight) {
    return mx::fast::rms_norm(x, weight, config_.rms_norm_eps);
}

mx::array LlamaModel::embed(const mx::array& input_ids) {
    auto w = get_weight("embed_tokens.weight");
    if (has_weight("embed_tokens.scales")) {
        // Quantized embedding — dequantize then take
        auto scales = get_weight("embed_tokens.scales");
        std::optional<mx::array> biases = std::nullopt;
        if (has_weight("embed_tokens.biases")) {
            biases = get_weight("embed_tokens.biases");
        }
        auto dequant_w = mx::dequantize(w, scales, biases, config_.quant_group_size, config_.quant_bits);
        return mx::take(dequant_w, input_ids, 0);
    }
    return mx::take(w, input_ids, 0);
}

mx::array LlamaModel::lm_head(const mx::array& x) {
    if (config_.tie_word_embeddings) {
        auto w = get_weight("embed_tokens.weight");
        if (has_weight("embed_tokens.scales")) {
            // Quantized tied embedding — dequantize then matmul
            auto scales = get_weight("embed_tokens.scales");
            std::optional<mx::array> biases = std::nullopt;
            if (has_weight("embed_tokens.biases")) {
                biases = get_weight("embed_tokens.biases");
            }
            auto dequant_w = mx::dequantize(w, scales, biases, config_.quant_group_size, config_.quant_bits);
            return mx::matmul(x, mx::transpose(dequant_w, {1, 0}));
        }
        return mx::matmul(x, mx::transpose(w, {1, 0}));
    }
    // lm_head is NOT under "model." prefix, so it's stored as "lm_head.*"
    return linear(x, "lm_head");
}

mx::array LlamaModel::linear(const mx::array& x, const std::string& prefix) {
    if (config_.quant_bits > 0 && has_weight(prefix + ".scales")) {
        auto w = get_weight(prefix + ".weight");
        auto scales = get_weight(prefix + ".scales");
        std::optional<mx::array> biases = std::nullopt;
        if (has_weight(prefix + ".biases")) {
            biases = get_weight(prefix + ".biases");
        }
        return mx::quantized_matmul(
            x, w, scales, biases,
            /*transpose=*/true,
            config_.quant_group_size,
            config_.quant_bits);
    }
    // Non-quantized fallback
    auto w = get_weight(prefix + ".weight");
    return mx::matmul(x, mx::transpose(w, {1, 0}));
}

mx::array LlamaModel::attention(
    const mx::array& x, int layer,
    mx::array& cache_k, mx::array& cache_v,
    const mx::array& cache_offsets) {

    std::string prefix = "layers." + std::to_string(layer) + ".self_attn";

    int B = x.shape(0);
    int L = x.shape(1);
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int hd = head_dim_;

    // Q, K, V projections
    auto q = linear(x, prefix + ".q_proj");  // [B, L, n_heads * hd]
    auto k = linear(x, prefix + ".k_proj");  // [B, L, n_kv_heads * hd]
    auto v = linear(x, prefix + ".v_proj");  // [B, L, n_kv_heads * hd]

    // Reshape to [B, L, n_heads, hd] then transpose to [B, n_heads, L, hd]
    q = mx::transpose(mx::reshape(q, {B, L, n_heads, hd}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B, L, n_kv_heads, hd}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B, L, n_kv_heads, hd}), {0, 2, 1, 3});

    // Apply QK-norm if weights are present (e.g., Qwen3)
    std::string q_norm_key = prefix + ".q_norm.weight";
    std::string k_norm_key = prefix + ".k_norm.weight";
    if (has_weight(q_norm_key)) {
        q = rms_norm(q, get_weight(q_norm_key));
    }
    if (has_weight(k_norm_key)) {
        k = rms_norm(k, get_weight(k_norm_key));
    }

    // Apply RoPE
    // RoPE expects [B, n_heads, L, hd]; offset is per-sequence
    // Use the array-based offset overload for batched offsets
    q = mx::fast::rope(q, hd, /*traditional=*/false, config_.rope_theta, /*scale=*/1.0f, cache_offsets);
    k = mx::fast::rope(k, hd, /*traditional=*/false, config_.rope_theta, /*scale=*/1.0f, cache_offsets);

    // KV cache update
    // cache_k shape: [B, n_kv_heads, max_seq_len, hd]
    // We need to write k, v at positions [cache_offset : cache_offset + L] for each sequence
    // For simplicity with batched sequences, we use slice_update with uniform offset.
    // Since this is a batched server, each sequence in the batch may have a different offset.
    // We'll use concatenation approach: take existing cache up to max offset, append new tokens.
    // But with different offsets per sequence, we need scatter-based approach.
    //
    // For the initial implementation, we support the common case:
    // all sequences in the batch have the same offset (uniform batching).
    // We'll use slice_update with integer positions.

    // For now: get the maximum offset from cache_offsets and use that for slicing.
    // This works correctly for prefill (all offset=0) and single-sequence decode.
    // TODO: For truly heterogeneous batches, use scatter-based cache update.

    // k shape: [B, n_kv_heads, L, hd]
    // Write into cache at the offset position
    // cache_offsets is [B] int32 — for uniform batch, all values are the same
    // We use the first offset value for slice positioning
    mx::eval({cache_offsets});
    int offset_val = cache_offsets.item<int32_t>();  // take first element

    // slice_update: write k into cache_k at [0:B, 0:n_kv_heads, offset:offset+L, 0:hd]
    cache_k = mx::slice_update(
        cache_k, k,
        {0, 0, offset_val, 0},
        {B, n_kv_heads, offset_val + L, hd});
    cache_v = mx::slice_update(
        cache_v, v,
        {0, 0, offset_val, 0},
        {B, n_kv_heads, offset_val + L, hd});

    // Read back the relevant portion of the cache for attention
    int total_len = offset_val + L;
    auto full_k = mx::slice(cache_k, {0, 0, 0, 0}, {B, n_kv_heads, total_len, hd});
    auto full_v = mx::slice(cache_v, {0, 0, 0, 0}, {B, n_kv_heads, total_len, hd});

    // Scaled dot product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // Determine mask mode: causal for prefill (L > 1), none for single-token decode
    std::string mask_mode = (L > 1) ? "causal" : "";

    auto attn_out = mx::fast::scaled_dot_product_attention(
        q, full_k, full_v, scale, mask_mode);

    // attn_out: [B, n_heads, L, hd] -> [B, L, n_heads, hd] -> [B, L, hidden_size]
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, n_heads * hd});

    // Output projection
    return linear(attn_out, prefix + ".o_proj");
}

mx::array LlamaModel::mlp(const mx::array& x, int layer) {
    std::string prefix = "layers." + std::to_string(layer) + ".mlp";

    auto gate = linear(x, prefix + ".gate_proj");
    auto up = linear(x, prefix + ".up_proj");

    // SiLU(gate) * up
    auto activated = mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), up);

    return linear(activated, prefix + ".down_proj");
}

mx::array LlamaModel::transformer_block(
    const mx::array& x, int layer,
    mx::array& cache_k, mx::array& cache_v,
    const mx::array& cache_offsets) {

    std::string prefix = "layers." + std::to_string(layer);

    // Pre-attention norm
    auto normed = rms_norm(x, get_weight(prefix + ".input_layernorm.weight"));

    // Attention + residual
    auto attn_out = attention(normed, layer, cache_k, cache_v, cache_offsets);
    auto h = mx::add(x, attn_out);

    // Pre-MLP norm
    auto normed2 = rms_norm(h, get_weight(prefix + ".post_attention_layernorm.weight"));

    // MLP + residual
    auto mlp_out = mlp(normed2, layer);
    return mx::add(h, mlp_out);
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

mx::array LlamaModel::forward(
    const mx::array& input_ids,
    std::vector<mx::array>& cache_keys,
    std::vector<mx::array>& cache_values,
    const mx::array& cache_offsets) {

    // Embedding
    auto h = embed(input_ids);

    // Transformer blocks
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        h = transformer_block(h, i, cache_keys[i], cache_values[i], cache_offsets);
    }

    // Final norm
    h = rms_norm(h, get_weight("norm.weight"));

    // LM head
    return lm_head(h);
}

// ---------------------------------------------------------------------------
// Debug helpers
// ---------------------------------------------------------------------------

std::vector<float> LlamaModel::debug_embed(const std::vector<int>& token_ids) {
    auto input = mx::array(token_ids.data(), {1, (int)token_ids.size()}, mx::int32);
    auto h = embed(input);
    mx::eval({h});

    std::vector<float> result;
    // Cast to float32 to read data
    auto h_f32 = mx::astype(h, mx::float32);
    h_f32 = mx::reshape(h_f32, {-1});
    mx::eval({h_f32});
    const float* ptr = h_f32.data<float>();
    int total = h_f32.size();
    for (int i = 0; i < std::min(total, 20); i++) {
        result.push_back(ptr[i]);
    }
    return result;
}

std::vector<int> LlamaModel::debug_forward(const std::vector<int>& token_ids) {
    auto input = mx::array(token_ids.data(), {1, (int)token_ids.size()}, mx::int32);

    int n_layers = config_.num_hidden_layers;
    int n_kv = config_.num_key_value_heads;
    int hd = head_dim_;
    int ctx = 512;

    std::vector<mx::array> cache_k, cache_v;
    for (int i = 0; i < n_layers; i++) {
        cache_k.push_back(mx::zeros({1, n_kv, ctx, hd}, mx::float16));
        cache_v.push_back(mx::zeros({1, n_kv, ctx, hd}, mx::float16));
    }
    mx::eval(cache_k);
    mx::eval(cache_v);

    auto cache_offsets = mx::array({0}, mx::int32);
    auto logits = forward(input, cache_k, cache_v, cache_offsets);
    mx::eval({logits});

    // Get logits for last position
    int seq_len = logits.shape(1);
    auto last_logits = mx::slice(logits, {0, seq_len - 1, 0}, {1, seq_len, logits.shape(2)});
    last_logits = mx::reshape(last_logits, {-1});

    // Get top 10 tokens
    auto neg_logits = mx::negative(last_logits);
    auto top_indices = mx::argpartition(neg_logits, 10);
    mx::eval({top_indices});

    std::vector<int> result;
    const int32_t* ptr = top_indices.data<int32_t>();
    for (int i = 0; i < 10; i++) {
        result.push_back(ptr[i]);
    }
    return result;
}

} // namespace flashmlx
