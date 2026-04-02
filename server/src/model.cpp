#include "flashmlx/model.h"
#include <mlx/compile.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace flashmlx {

// Compiled SwiGLU activation: silu(gate) * up → single fused Metal kernel
static std::vector<mx::array> _swiglu_impl(const std::vector<mx::array>& inputs) {
    auto& gate = inputs[0];
    auto& up = inputs[1];
    return {mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), up)};
}

static auto compiled_swiglu = mx::compile(_swiglu_impl, /*shapeless=*/true);

static mx::array swiglu(const mx::array& gate, const mx::array& up) {
    return compiled_swiglu({gate, up})[0];
}

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
    config_.rope_theta = json_float(json, "rope_theta", 10000.0f);
    config_.tie_word_embeddings = json_bool(json, "tie_word_embeddings", true);

    // Parse nested quantization object
    auto quant_obj = json_value_for_key(json, "quantization");
    if (!quant_obj.empty() && quant_obj[0] == '{') {
        config_.quant_bits = json_int(quant_obj, "bits", 0);
        config_.quant_group_size = json_int(quant_obj, "group_size", 64);
    }

    // MoE config
    config_.num_experts = json_int(json, "num_experts", 0);
    config_.num_experts_per_tok = json_int(json, "num_experts_per_tok", 0);
    config_.moe_intermediate_size = json_int(json, "moe_intermediate_size", 0);
    config_.shared_expert_intermediate_size = json_int(json, "shared_expert_intermediate_size", 0);
    config_.norm_topk_prob = json_bool(json, "norm_topk_prob", false);

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
    if (config_.num_experts > 0) {
        std::cout << "[flashmlx] MoE: " << config_.num_experts << " experts, top-"
                  << config_.num_experts_per_tok
                  << ", moe_intermediate=" << config_.moe_intermediate_size
                  << ", shared_intermediate=" << config_.shared_expert_intermediate_size
                  << std::endl;
    }
    load_weights(model_path);

    // Detect activation dtype from first layer norm weight
    if (has_weight("layers.0.input_layernorm.weight")) {
        config_.activation_dtype = get_weight("layers.0.input_layernorm.weight").dtype();
        std::cout << "[flashmlx] Activation dtype: "
                  << (config_.activation_dtype == mx::bfloat16 ? "bfloat16" : "float16")
                  << std::endl;
    }

    build_weight_cache();

    // Pre-dequantize embedding for fast lookup (avoids per-call dequantize)
    if (has_weight("embed_tokens.scales")) {
        auto w = get_weight("embed_tokens.weight");
        auto scales = get_weight("embed_tokens.scales");
        std::optional<mx::array> biases = std::nullopt;
        if (has_weight("embed_tokens.biases")) {
            biases = get_weight("embed_tokens.biases");
        }
        auto dequant = mx::dequantize(w, scales, biases, config_.quant_group_size, config_.quant_bits);
        mx::eval({dequant});
        weights_.insert_or_assign("_embed_dequantized", dequant);
        std::cout << "[flashmlx] Pre-dequantized embedding: " << dequant.shape(0) << "x" << dequant.shape(1) << std::endl;
    }
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
// Weight cache
// ---------------------------------------------------------------------------

void LlamaModel::build_weight_cache() {
    bool is_moe_model = config_.num_experts > 0;

    // Initialize MoE layer tracking
    layer_is_moe_.resize(config_.num_hidden_layers, false);

    layer_weights_.reserve(config_.num_hidden_layers);
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        std::string p = "layers." + std::to_string(i);
        auto attn  = p + ".self_attn";
        auto mlp_p = p + ".mlp";

        // Check if this layer is MoE (has router/gate weight)
        bool this_layer_moe = has_weight(mlp_p + ".gate.weight");
        layer_is_moe_[i] = this_layer_moe;

        // Build norms + attention weights
        mx::array input_norm_w  = get_weight(p + ".input_layernorm.weight");
        mx::array post_norm_w   = get_weight(p + ".post_attention_layernorm.weight");

        mx::array q_w = get_weight(attn + ".q_proj.weight");
        mx::array q_s = get_weight(attn + ".q_proj.scales");
        std::optional<mx::array> q_b = has_weight(attn + ".q_proj.biases") ?
            std::optional<mx::array>(get_weight(attn + ".q_proj.biases")) : std::nullopt;

        mx::array k_w = get_weight(attn + ".k_proj.weight");
        mx::array k_s = get_weight(attn + ".k_proj.scales");
        std::optional<mx::array> k_b = has_weight(attn + ".k_proj.biases") ?
            std::optional<mx::array>(get_weight(attn + ".k_proj.biases")) : std::nullopt;

        mx::array v_w = get_weight(attn + ".v_proj.weight");
        mx::array v_s = get_weight(attn + ".v_proj.scales");
        std::optional<mx::array> v_b = has_weight(attn + ".v_proj.biases") ?
            std::optional<mx::array>(get_weight(attn + ".v_proj.biases")) : std::nullopt;

        mx::array o_w = get_weight(attn + ".o_proj.weight");
        mx::array o_s = get_weight(attn + ".o_proj.scales");
        std::optional<mx::array> o_b = has_weight(attn + ".o_proj.biases") ?
            std::optional<mx::array>(get_weight(attn + ".o_proj.biases")) : std::nullopt;

        // Attention linear biases (not quantization biases — e.g. Qwen2-MoE has q/k/v bias)
        std::optional<mx::array> q_bias = has_weight(attn + ".q_proj.bias") ?
            std::optional<mx::array>(get_weight(attn + ".q_proj.bias")) : std::nullopt;
        std::optional<mx::array> k_bias = has_weight(attn + ".k_proj.bias") ?
            std::optional<mx::array>(get_weight(attn + ".k_proj.bias")) : std::nullopt;
        std::optional<mx::array> v_bias = has_weight(attn + ".v_proj.bias") ?
            std::optional<mx::array>(get_weight(attn + ".v_proj.bias")) : std::nullopt;

        bool has_q_norm = has_weight(attn + ".q_norm.weight");

        // MLP weights — only for dense layers
        // For MoE layers, use dummy placeholders (MLP is handled by moe_block)
        mx::array gate_w_val = input_norm_w;  // dummy default
        mx::array gate_s_val = input_norm_w;
        mx::array up_w_val = input_norm_w;
        mx::array up_s_val = input_norm_w;
        mx::array down_w_val = input_norm_w;
        mx::array down_s_val = input_norm_w;
        std::optional<mx::array> gate_b_val = std::nullopt;
        std::optional<mx::array> up_b_val = std::nullopt;
        std::optional<mx::array> down_b_val = std::nullopt;

        if (!this_layer_moe) {
            gate_w_val = get_weight(mlp_p + ".gate_proj.weight");
            gate_s_val = get_weight(mlp_p + ".gate_proj.scales");
            gate_b_val = has_weight(mlp_p + ".gate_proj.biases") ?
                std::optional<mx::array>(get_weight(mlp_p + ".gate_proj.biases")) : std::nullopt;

            up_w_val = get_weight(mlp_p + ".up_proj.weight");
            up_s_val = get_weight(mlp_p + ".up_proj.scales");
            up_b_val = has_weight(mlp_p + ".up_proj.biases") ?
                std::optional<mx::array>(get_weight(mlp_p + ".up_proj.biases")) : std::nullopt;

            down_w_val = get_weight(mlp_p + ".down_proj.weight");
            down_s_val = get_weight(mlp_p + ".down_proj.scales");
            down_b_val = has_weight(mlp_p + ".down_proj.biases") ?
                std::optional<mx::array>(get_weight(mlp_p + ".down_proj.biases")) : std::nullopt;
        }

        mx::array dummy = input_norm_w;
        mx::array q_norm_w_val = dummy;
        mx::array k_norm_w_val = dummy;
        if (has_q_norm) {
            q_norm_w_val = get_weight(attn + ".q_norm.weight");
            k_norm_w_val = get_weight(attn + ".k_norm.weight");
        }

        layer_weights_.push_back(LayerWeights{
            std::move(q_w), std::move(q_s), std::move(k_w), std::move(k_s),
            std::move(v_w), std::move(v_s), std::move(o_w), std::move(o_s),
            std::move(q_b), std::move(k_b), std::move(v_b), std::move(o_b),
            std::move(q_bias), std::move(k_bias), std::move(v_bias),
            std::move(input_norm_w), std::move(post_norm_w),
            has_q_norm, std::move(q_norm_w_val), std::move(k_norm_w_val),
            std::move(gate_w_val), std::move(gate_s_val),
            std::move(up_w_val), std::move(up_s_val),
            std::move(down_w_val), std::move(down_s_val),
            std::move(gate_b_val), std::move(up_b_val), std::move(down_b_val)
        });
    }

    // Cache final norm weight
    norm_w_ = get_weight("norm.weight");

    std::cout << "[flashmlx] Built weight cache for " << config_.num_hidden_layers << " layers" << std::endl;

    // Build MoE weight cache if needed
    if (is_moe_model) {
        build_moe_weight_cache();
    }
}

// ---------------------------------------------------------------------------
// MoE weight stacking
// ---------------------------------------------------------------------------

void LlamaModel::build_moe_weight_cache() {
    int num_experts = config_.num_experts;
    int moe_count = 0;

    // Reserve space — we store one MoEWeights per layer (indexed by layer number)
    // Use a map-like approach: moe_weights_ is sized to num_hidden_layers,
    // but only MoE layers have valid data.
    moe_weights_.reserve(config_.num_hidden_layers);

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        if (!layer_is_moe_[i]) {
            // Push a default-ish MoEWeights (won't be accessed)
            moe_weights_.push_back(MoEWeights{});
            continue;
        }

        std::string prefix = "layers." + std::to_string(i) + ".mlp";
        MoEWeights mw;

        // Router weight — may or may not be quantized
        mw.router_w = get_weight(prefix + ".gate.weight");
        if (has_weight(prefix + ".gate.scales")) {
            mw.router_s = get_weight(prefix + ".gate.scales");
            if (has_weight(prefix + ".gate.biases")) {
                mw.router_b = get_weight(prefix + ".gate.biases");
            }
        }

        // Stack expert weights
        auto stack_component = [&](const std::string& proj, const std::string& component) {
            std::vector<mx::array> parts;
            parts.reserve(num_experts);
            for (int e = 0; e < num_experts; e++) {
                std::string key = prefix + ".experts." + std::to_string(e) + "." + proj + "." + component;
                parts.push_back(get_weight(key));
            }
            return mx::stack(parts, 0);
        };

        auto stack_optional = [&](const std::string& proj, const std::string& component) -> std::optional<mx::array> {
            std::string test_key = prefix + ".experts.0." + proj + "." + component;
            if (!has_weight(test_key)) return std::nullopt;
            return stack_component(proj, component);
        };

        mw.gate_w = stack_component("gate_proj", "weight");
        mw.gate_s = stack_component("gate_proj", "scales");
        mw.gate_b = stack_optional("gate_proj", "biases");

        mw.up_w = stack_component("up_proj", "weight");
        mw.up_s = stack_component("up_proj", "scales");
        mw.up_b = stack_optional("up_proj", "biases");

        mw.down_w = stack_component("down_proj", "weight");
        mw.down_s = stack_component("down_proj", "scales");
        mw.down_b = stack_optional("down_proj", "biases");

        // Shared expert
        std::string se_prefix = prefix + ".shared_expert";
        mw.has_shared = has_weight(se_prefix + ".gate_proj.weight");
        if (mw.has_shared) {
            mw.shared_gate_w = get_weight(se_prefix + ".gate_proj.weight");
            mw.shared_gate_s = get_weight(se_prefix + ".gate_proj.scales");
            mw.shared_gate_b = has_weight(se_prefix + ".gate_proj.biases") ?
                std::optional<mx::array>(get_weight(se_prefix + ".gate_proj.biases")) : std::nullopt;

            mw.shared_up_w = get_weight(se_prefix + ".up_proj.weight");
            mw.shared_up_s = get_weight(se_prefix + ".up_proj.scales");
            mw.shared_up_b = has_weight(se_prefix + ".up_proj.biases") ?
                std::optional<mx::array>(get_weight(se_prefix + ".up_proj.biases")) : std::nullopt;

            mw.shared_down_w = get_weight(se_prefix + ".down_proj.weight");
            mw.shared_down_s = get_weight(se_prefix + ".down_proj.scales");
            mw.shared_down_b = has_weight(se_prefix + ".down_proj.biases") ?
                std::optional<mx::array>(get_weight(se_prefix + ".down_proj.biases")) : std::nullopt;

            // Shared expert gate
            std::string seg_prefix = prefix + ".shared_expert_gate";
            mw.shared_expert_gate_w = get_weight(seg_prefix + ".weight");
            mw.shared_expert_gate_quantized = has_weight(seg_prefix + ".scales");
            if (mw.shared_expert_gate_quantized) {
                mw.shared_expert_gate_s = get_weight(seg_prefix + ".scales");
                mw.shared_expert_gate_b = has_weight(seg_prefix + ".biases") ?
                    std::optional<mx::array>(get_weight(seg_prefix + ".biases")) : std::nullopt;
            }
        }

        // Eval stacked weights to materialize them
        std::vector<mx::array> to_eval = {
            *mw.gate_w, *mw.gate_s, *mw.up_w, *mw.up_s, *mw.down_w, *mw.down_s
        };
        if (mw.gate_b) to_eval.push_back(*mw.gate_b);
        if (mw.up_b) to_eval.push_back(*mw.up_b);
        if (mw.down_b) to_eval.push_back(*mw.down_b);
        mx::eval(to_eval);

        moe_weights_.push_back(std::move(mw));
        moe_count++;
    }

    std::cout << "[flashmlx] Built MoE weight cache for " << moe_count << " MoE layers ("
              << num_experts << " experts each)" << std::endl;
}

// ---------------------------------------------------------------------------
// Fast linear (uses pre-resolved weight references — no hash lookup)
// ---------------------------------------------------------------------------

mx::array LlamaModel::linear_fast(const mx::array& x, const mx::array& w, const mx::array& s,
                                   const std::optional<mx::array>& b) {
    return mx::quantized_matmul(x, w, s, b, /*transpose=*/true,
                                config_.quant_group_size, config_.quant_bits);
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

mx::array LlamaModel::rms_norm(const mx::array& x, const mx::array& weight) {
    return mx::fast::rms_norm(x, weight, config_.rms_norm_eps);
}

mx::array LlamaModel::embed(const mx::array& input_ids) {
    // Use pre-dequantized embedding if available (set during load)
    if (has_weight("_embed_dequantized")) {
        return mx::take(get_weight("_embed_dequantized"), input_ids, 0);
    }
    auto w = get_weight("embed_tokens.weight");
    return mx::take(w, input_ids, 0);
}

mx::array LlamaModel::lm_head(const mx::array& x) {
    if (config_.tie_word_embeddings) {
        // Use quantized_matmul for tied embeddings (4x less bandwidth than dequantized)
        if (has_weight("embed_tokens.scales")) {
            auto w = get_weight("embed_tokens.weight");
            auto scales = get_weight("embed_tokens.scales");
            std::optional<mx::array> biases = std::nullopt;
            if (has_weight("embed_tokens.biases")) {
                biases = get_weight("embed_tokens.biases");
            }
            return mx::quantized_matmul(x, w, scales, biases,
                                        /*transpose=*/true,
                                        config_.quant_group_size, config_.quant_bits);
        }
        // Non-quantized: use pre-dequantized if available, else raw
        if (has_weight("_embed_dequantized")) {
            return mx::matmul(x, mx::transpose(get_weight("_embed_dequantized"), {1, 0}));
        }
        return mx::matmul(x, mx::transpose(get_weight("embed_tokens.weight"), {1, 0}));
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

    // Add attention linear biases if present (e.g. Qwen2-MoE)
    std::string q_bias_key = prefix + ".q_proj.bias";
    if (has_weight(q_bias_key)) q = mx::add(q, get_weight(q_bias_key));
    std::string k_bias_key = prefix + ".k_proj.bias";
    if (has_weight(k_bias_key)) k = mx::add(k, get_weight(k_bias_key));
    std::string v_bias_key = prefix + ".v_proj.bias";
    if (has_weight(v_bias_key)) v = mx::add(v, get_weight(v_bias_key));

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
    // For homogeneous batching all offsets are the same — extract first element
    auto first_offset = mx::slice(cache_offsets, {0}, {1});
    mx::eval({first_offset});
    int offset_val = first_offset.item<int32_t>();

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
    auto activated = swiglu(gate, up);

    return linear(activated, prefix + ".down_proj");
}

mx::array LlamaModel::mlp_fast(const mx::array& x, int layer) {
    const auto& lw = layer_weights_[layer];

    auto gate = linear_fast(x, lw.gate_w, lw.gate_s, lw.gate_b);
    auto up   = linear_fast(x, lw.up_w,   lw.up_s,   lw.up_b);

    // SiLU(gate) * up
    auto activated = swiglu(gate, up);

    return linear_fast(activated, lw.down_w, lw.down_s, lw.down_b);
}

// ---------------------------------------------------------------------------
// MoE block implementation
// ---------------------------------------------------------------------------

mx::array LlamaModel::switch_mlp(const mx::array& x, const mx::array& indices, int layer) {
    // x: [B, L, hidden_size], indices: [B, L, k] (top-k expert indices)
    const auto& mw = moe_weights_[layer];
    int group_size = config_.quant_group_size;
    int bits = config_.quant_bits;

    // expand_dims to [B, L, 1, 1, hidden_size] for gather_qmm
    auto xe = mx::expand_dims(x, {-2, -3});

    // Sort indices for cache locality when we have enough tokens
    bool do_sort = indices.size() >= 64;
    auto idx = indices;
    std::optional<mx::array> inv_order;
    auto sorted_x = xe;

    if (do_sort) {
        // _gather_sort: flatten batch dims, sort by expert index
        int k_val = indices.shape(-1);
        auto flat_indices = mx::flatten(indices);
        auto order = mx::argsort(flat_indices);
        inv_order = mx::argsort(order);

        // Reorder x: x.flatten(0, -3)[order // k]
        auto x_flat = mx::flatten(xe, 0, -3);  // [B*L, 1, 1, hidden_size]
        auto src_indices = mx::floor_divide(order, mx::array(k_val, mx::int32));
        sorted_x = mx::take(x_flat, src_indices, 0);  // [B*L*k, 1, 1, hidden_size]
        idx = mx::take(flat_indices, order);  // [B*L*k] sorted expert indices
    }

    // gate_proj: gather_qmm -> [*, 1, 1, moe_intermediate_size]
    auto x_gate = mx::gather_qmm(
        sorted_x, *mw.gate_w, *mw.gate_s, mw.gate_b,
        std::nullopt, idx,
        /*transpose=*/true, group_size, bits, "affine", do_sort);

    // up_proj
    auto x_up = mx::gather_qmm(
        sorted_x, *mw.up_w, *mw.up_s, mw.up_b,
        std::nullopt, idx,
        /*transpose=*/true, group_size, bits, "affine", do_sort);

    // SwiGLU: silu(gate) * up
    auto activated = swiglu(x_gate, x_up);

    // down_proj
    auto x_down = mx::gather_qmm(
        activated, *mw.down_w, *mw.down_s, mw.down_b,
        std::nullopt, idx,
        /*transpose=*/true, group_size, bits, "affine", do_sort);

    // Unsort if needed
    if (do_sort) {
        // x_down is [B*L*k, 1, 1, hidden_size], unsort then unflatten
        auto x_unsorted = mx::take(x_down, *inv_order, 0);
        // Unflatten back to [B, L, k, 1, hidden_size]
        auto orig_shape = indices.shape();  // [B, L, k]
        x_down = mx::unflatten(x_unsorted, 0, {orig_shape[0], orig_shape[1], orig_shape[2]});
    }

    // Squeeze the extra dims: result should be [B, L, k, hidden_size]
    return mx::squeeze(x_down, -2);
}

mx::array LlamaModel::shared_expert_mlp(const mx::array& x, int layer) {
    const auto& mw = moe_weights_[layer];
    int group_size = config_.quant_group_size;
    int bits = config_.quant_bits;

    auto gate = mx::quantized_matmul(x, *mw.shared_gate_w, *mw.shared_gate_s, mw.shared_gate_b,
                                      /*transpose=*/true, group_size, bits);
    auto up = mx::quantized_matmul(x, *mw.shared_up_w, *mw.shared_up_s, mw.shared_up_b,
                                    /*transpose=*/true, group_size, bits);
    auto activated = swiglu(gate, up);
    return mx::quantized_matmul(activated, *mw.shared_down_w, *mw.shared_down_s, mw.shared_down_b,
                                 /*transpose=*/true, group_size, bits);
}

mx::array LlamaModel::moe_block(const mx::array& x, int layer) {
    // x: [B, L, hidden_size]
    const auto& mw = moe_weights_[layer];
    int k = config_.num_experts_per_tok;

    // 1. Router: linear(x) -> softmax -> top-k
    // Router may or may not be quantized depending on the model
    mx::array gates(0.0f);
    if (mw.router_s) {
        gates = mx::quantized_matmul(x, *mw.router_w, *mw.router_s, mw.router_b,
                                      /*transpose=*/true, config_.quant_group_size, config_.quant_bits);
    } else {
        gates = mx::matmul(x, mx::transpose(*mw.router_w, {1, 0}));
    }
    gates = mx::softmax(gates, -1, /*precise=*/true);

    // Top-k selection via argpartition
    auto neg_gates = mx::negative(gates);
    auto inds = mx::argpartition(neg_gates, k - 1, -1);  // [B, L, num_experts]
    // Keep only top-k
    auto shape = inds.shape();
    inds = mx::slice(inds, {0, 0, 0}, {shape[0], shape[1], k});  // [B, L, k]
    inds = mx::stop_gradient(inds);

    // Get scores for the selected experts
    auto scores = mx::take_along_axis(gates, inds, -1);  // [B, L, k]

    // 2. Expert execution via switch_mlp
    auto y = switch_mlp(x, inds, layer);  // [B, L, k, hidden_size]

    // 3. Weight by routing scores and sum over experts
    // scores: [B, L, k] -> [B, L, k, 1] for broadcasting
    auto scores_expanded = mx::expand_dims(scores, -1);  // [B, L, k, 1]
    y = mx::multiply(y, scores_expanded);
    y = mx::sum(y, -2);  // [B, L, hidden_size]

    // 4. Add shared expert output (gated by sigmoid)
    if (mw.has_shared) {
        auto shared_out = shared_expert_mlp(x, layer);

        // Shared expert gate: sigmoid(linear(x)) * shared_out
        mx::array gate_val(0.0f);  // placeholder
        if (mw.shared_expert_gate_quantized) {
            gate_val = mx::quantized_matmul(
                x, *mw.shared_expert_gate_w, *mw.shared_expert_gate_s,
                mw.shared_expert_gate_b,
                /*transpose=*/true, config_.quant_group_size, config_.quant_bits);
        } else {
            gate_val = mx::matmul(x, mx::transpose(*mw.shared_expert_gate_w, {1, 0}));
        }
        gate_val = mx::sigmoid(gate_val);  // [B, L, 1]
        shared_out = mx::multiply(gate_val, shared_out);

        y = mx::add(y, shared_out);
    }

    return y;
}

// ---------------------------------------------------------------------------
// Transformer block (array-offset — used for prefill and heterogeneous batching)
// ---------------------------------------------------------------------------

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

    // MLP + residual — dispatch to MoE or dense
    auto mlp_out = [&]() -> mx::array {
        if (!layer_is_moe_.empty() && layer < (int)layer_is_moe_.size() && layer_is_moe_[layer]) {
            return moe_block(normed2, layer);
        } else {
            return mlp(normed2, layer);
        }
    }();
    return mx::add(h, mlp_out);
}

// Int-offset overloads (no mx::eval sync — critical for N-step batching)

mx::array LlamaModel::attention(
    const mx::array& x, int layer,
    mx::array& cache_k, mx::array& cache_v,
    int cache_offset) {

    const auto& lw = layer_weights_[layer];
    int B = x.shape(0);
    int L = x.shape(1);
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int hd = head_dim_;

    auto q = linear_fast(x, lw.q_w, lw.q_s, lw.q_b);
    auto k = linear_fast(x, lw.k_w, lw.k_s, lw.k_b);
    auto v = linear_fast(x, lw.v_w, lw.v_s, lw.v_b);

    // Add attention linear biases if present (e.g. Qwen2-MoE)
    if (lw.q_bias) q = mx::add(q, *lw.q_bias);
    if (lw.k_bias) k = mx::add(k, *lw.k_bias);
    if (lw.v_bias) v = mx::add(v, *lw.v_bias);

    q = mx::transpose(mx::reshape(q, {B, L, n_heads, hd}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B, L, n_kv_heads, hd}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B, L, n_kv_heads, hd}), {0, 2, 1, 3});

    // QK-norm (Qwen3) — use cached weight flags
    if (lw.has_q_norm) {
        q = rms_norm(q, lw.q_norm_w);
        k = rms_norm(k, lw.k_norm_w);
    }

    // Match the Python path: use the scalar-offset RoPE overload for homogeneous decode.
    q = mx::fast::rope(q, hd, false, config_.rope_theta, 1.0f, cache_offset);
    k = mx::fast::rope(k, hd, false, config_.rope_theta, 1.0f, cache_offset);

    // KV cache update — concat approach (simpler graph, better MLX optimization for small models)
    cache_k = mx::concatenate({cache_k, k}, 2);
    cache_v = mx::concatenate({cache_v, v}, 2);
    auto full_k = cache_k;
    auto full_v = cache_v;
    int total_len = cache_k.shape(2);
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::string mask_mode = (L > 1) ? "causal" : "";
    auto attn_out = mx::fast::scaled_dot_product_attention(q, full_k, full_v, scale, mask_mode);
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, n_heads * hd});
    return linear_fast(attn_out, lw.o_w, lw.o_s, lw.o_b);
}

mx::array LlamaModel::transformer_block(
    const mx::array& x, int layer,
    mx::array& cache_k, mx::array& cache_v,
    int cache_offset) {

    const auto& lw = layer_weights_[layer];
    auto normed = rms_norm(x, lw.input_norm_w);
    auto attn_out = attention(normed, layer, cache_k, cache_v, cache_offset);
    auto h = mx::add(x, attn_out);
    auto normed2 = rms_norm(h, lw.post_norm_w);

    // MLP + residual — dispatch to MoE or dense
    auto mlp_out = [&]() -> mx::array {
        if (!layer_is_moe_.empty() && layer < (int)layer_is_moe_.size() && layer_is_moe_[layer]) {
            return moe_block(normed2, layer);
        } else {
            return mlp_fast(normed2, layer);
        }
    }();
    return mx::add(h, mlp_out);
}

// ---------------------------------------------------------------------------
// Heterogeneous attention: per-sequence offsets, scatter KV, explicit mask
// No mx::eval() — everything stays in the lazy computation graph.
// ---------------------------------------------------------------------------

mx::array LlamaModel::attention_heterogeneous(
    const mx::array& x, int layer,
    mx::array& cache_k, mx::array& cache_v,
    const mx::array& offsets, int max_kv_len,
    const mx::array& mask) {

    const auto& lw = layer_weights_[layer];
    int B = x.shape(0);
    int L = x.shape(1);  // always 1 for decode
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int hd = head_dim_;

    // Q, K, V projections
    auto q = linear_fast(x, lw.q_w, lw.q_s, lw.q_b);
    auto k = linear_fast(x, lw.k_w, lw.k_s, lw.k_b);
    auto v = linear_fast(x, lw.v_w, lw.v_s, lw.v_b);

    // Attention linear biases (e.g. Qwen2-MoE)
    if (lw.q_bias) q = mx::add(q, *lw.q_bias);
    if (lw.k_bias) k = mx::add(k, *lw.k_bias);
    if (lw.v_bias) v = mx::add(v, *lw.v_bias);

    // Reshape to [B, n_heads, L, hd]
    q = mx::transpose(mx::reshape(q, {B, L, n_heads, hd}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B, L, n_kv_heads, hd}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B, L, n_kv_heads, hd}), {0, 2, 1, 3});

    // QK-norm (Qwen3)
    if (lw.has_q_norm) {
        q = rms_norm(q, lw.q_norm_w);
        k = rms_norm(k, lw.k_norm_w);
    }

    // RoPE with per-sequence offsets (array overload)
    q = mx::fast::rope(q, hd, false, config_.rope_theta, 1.0f, offsets);
    k = mx::fast::rope(k, hd, false, config_.rope_theta, 1.0f, offsets);

    // KV cache scatter: batched put_along_axis — single op for all B elements.
    auto idx = mx::broadcast_to(mx::reshape(offsets, {B, 1, 1, 1}),
                                {B, n_kv_heads, 1, hd});
    cache_k = mx::put_along_axis(cache_k, idx, k, 2);
    cache_v = mx::put_along_axis(cache_v, idx, v, 2);

    // SDPA with pre-built mask (built once in forward_heterogeneous)
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto attn_out = mx::fast::scaled_dot_product_attention(
        q, cache_k, cache_v, scale, /*mask_mode=*/"", /*mask=*/mask);

    // [B, n_heads, L, hd] -> [B, L, n_heads * hd]
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, n_heads * hd});
    return linear_fast(attn_out, lw.o_w, lw.o_s, lw.o_b);
}

mx::array LlamaModel::transformer_block_heterogeneous(
    const mx::array& x, int layer,
    mx::array& cache_k, mx::array& cache_v,
    const mx::array& offsets, int max_kv_len,
    const mx::array& mask) {

    const auto& lw = layer_weights_[layer];
    auto normed = rms_norm(x, lw.input_norm_w);
    auto attn_out = attention_heterogeneous(normed, layer, cache_k, cache_v, offsets, max_kv_len, mask);
    auto h = mx::add(x, attn_out);
    auto normed2 = rms_norm(h, lw.post_norm_w);

    // MLP + residual — dispatch to MoE or dense
    auto mlp_out = [&]() -> mx::array {
        if (!layer_is_moe_.empty() && layer < (int)layer_is_moe_.size() && layer_is_moe_[layer]) {
            return moe_block(normed2, layer);
        } else {
            return mlp_fast(normed2, layer);
        }
    }();
    return mx::add(h, mlp_out);
}

mx::array LlamaModel::forward_heterogeneous(
    const mx::array& input_ids,
    std::vector<mx::array>& cache_keys,
    std::vector<mx::array>& cache_values,
    const mx::array& offsets,
    int max_kv_len) {

    auto h = embed(input_ids);

    // Build mask once for all layers
    int B = input_ids.shape(0);
    auto positions = mx::arange(0, max_kv_len, mx::int32);
    auto offsets_4d = mx::reshape(offsets, {B, 1, 1, 1});
    auto dtype = config_.activation_dtype;
    auto mask = mx::where(
        mx::less_equal(positions, offsets_4d),
        mx::array(0.0f, dtype),
        mx::array(-1e9f, dtype));  // [B, 1, 1, max_kv_len]

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        h = transformer_block_heterogeneous(h, i, cache_keys[i], cache_values[i], offsets, max_kv_len, mask);
    }
    h = rms_norm(h, *norm_w_);
    return lm_head(h);
}

mx::array LlamaModel::forward(
    const mx::array& input_ids,
    std::vector<mx::array>& cache_keys,
    std::vector<mx::array>& cache_values,
    int cache_offset) {

    auto h = embed(input_ids);
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        h = transformer_block(h, i, cache_keys[i], cache_values[i], cache_offset);
    }
    h = rms_norm(h, *norm_w_);
    return lm_head(h);
}

// ---------------------------------------------------------------------------
// Forward pass (array-based offset — used for prefill and heterogeneous batching)
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
