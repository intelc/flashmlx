#include "flashmlx/model.h"
#include <mlx/compile.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <cmath>

namespace flashmlx {

// ---------------------------------------------------------------------------
// Minimal JSON helpers (same as model.cpp — duplicated to keep files independent)
// ---------------------------------------------------------------------------

namespace {

std::string trim(const std::string& s) {
    auto a = s.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return "";
    auto b = s.find_last_not_of(" \t\n\r");
    return s.substr(a, b - a + 1);
}

std::string json_value_for_key(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r'))
        pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '{') {
        int depth = 0;
        size_t start = pos;
        for (size_t i = pos; i < json.size(); i++) {
            if (json[i] == '{') depth++;
            else if (json[i] == '}') { depth--; if (depth == 0) return json.substr(start, i - start + 1); }
        }
        return "";
    }
    size_t start = pos;
    if (json[pos] == '"') {
        auto end = json.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return json.substr(pos + 1, end - pos - 1);
    }
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
// Block type enum
// ---------------------------------------------------------------------------

enum class BlockType { Mamba, Attention, MLP, MOE };

// ---------------------------------------------------------------------------
// NemotronHModel implementation
// ---------------------------------------------------------------------------

NemotronHModel::NemotronHModel(const std::string& model_path) {
    load_config(model_path);

    std::cout << "[NemotronH] Config: hidden=" << config_.hidden_size
              << " layers=" << config_.num_hidden_layers
              << " attn_heads=" << config_.num_attention_heads
              << " kv_heads=" << config_.num_key_value_heads
              << " head_dim=" << attn_head_dim_
              << " mamba_heads=" << mamba_num_heads_
              << " mamba_head_dim=" << mamba_head_dim_
              << " ssm_state=" << ssm_state_size_
              << " conv_kernel=" << conv_kernel_
              << " n_groups=" << n_groups_
              << " vocab=" << config_.vocab_size
              << " quant=" << config_.quant_bits << "bit"
              << std::endl;

    // Parse pattern
    parse_pattern();

    // Count block types
    int n_mamba = 0, n_attn = 0, n_mlp = 0, n_moe = 0;
    for (auto bt : block_types_) {
        if (bt == BlockType::Mamba) n_mamba++;
        else if (bt == BlockType::Attention) n_attn++;
        else if (bt == BlockType::MOE) n_moe++;
        else n_mlp++;
    }
    std::cout << "[NemotronH] Pattern: " << n_mamba << " Mamba, " << n_attn << " Attention, "
              << n_mlp << " MLP, " << n_moe << " MoE blocks" << std::endl;

    load_weights(model_path);
    build_ssm_kernel();

    // Detect activation dtype
    if (has_weight("layers.0.norm.weight")) {
        config_.activation_dtype = get_weight("layers.0.norm.weight").dtype();
        std::cout << "[NemotronH] Activation dtype: "
                  << (config_.activation_dtype == mx::bfloat16 ? "bfloat16" : "float16")
                  << std::endl;
    }

    // Pre-dequantize embedding
    if (has_weight("embeddings.scales")) {
        auto w = get_weight("embeddings.weight");
        auto scales = get_weight("embeddings.scales");
        std::optional<mx::array> biases = std::nullopt;
        if (has_weight("embeddings.biases")) {
            biases = get_weight("embeddings.biases");
        }
        auto dequant = mx::dequantize(w, scales, biases, config_.quant_group_size, config_.quant_bits);
        mx::eval({dequant});
        weights_.insert_or_assign("_embed_dequantized", dequant);
        std::cout << "[NemotronH] Pre-dequantized embedding: " << dequant.shape(0)
                  << "x" << dequant.shape(1) << std::endl;
    }

    // Initialize mamba states (empty — will be initialized on first forward)
    mamba_states_initialized_ = false;
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

void NemotronHModel::load_config(const std::string& model_path) {
    std::string config_file = model_path + "/config.json";
    std::ifstream f(config_file);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + config_file);
    std::stringstream buf;
    buf << f.rdbuf();
    std::string json = buf.str();

    config_.hidden_size = json_int(json, "hidden_size", 3136);
    config_.intermediate_size = json_int(json, "intermediate_size", 12544);
    config_.num_hidden_layers = json_int(json, "num_hidden_layers", 42);
    config_.num_attention_heads = json_int(json, "num_attention_heads", 40);
    config_.num_key_value_heads = json_int(json, "num_key_value_heads", 8);
    config_.head_dim = json_int(json, "head_dim", 128);
    config_.vocab_size = json_int(json, "vocab_size", 131072);
    config_.rms_norm_eps = json_float(json, "layer_norm_epsilon", 1e-5f);
    config_.tie_word_embeddings = json_bool(json, "tie_word_embeddings", false);

    // Quantization
    auto quant_obj = json_value_for_key(json, "quantization");
    if (!quant_obj.empty() && quant_obj[0] == '{') {
        config_.quant_bits = json_int(quant_obj, "bits", 0);
        config_.quant_group_size = json_int(quant_obj, "group_size", 64);
    }

    // Nemotron-H specific
    mamba_num_heads_ = json_int(json, "mamba_num_heads", 96);
    mamba_head_dim_ = json_int(json, "mamba_head_dim", 80);
    ssm_state_size_ = json_int(json, "ssm_state_size", 128);
    conv_kernel_ = json_int(json, "conv_kernel", 4);
    n_groups_ = json_int(json, "n_groups", 8);
    time_step_min_ = json_float(json, "time_step_min", 0.001f);
    time_step_max_ = json_float(json, "time_step_max", 0.1f);
    use_conv_bias_ = json_bool(json, "use_conv_bias", true);

    // MoE parameters
    config_.n_routed_experts = json_int(json, "n_routed_experts", 0);
    config_.n_shared_experts = json_int(json, "n_shared_experts", 0);
    config_.moe_shared_expert_intermediate_size = json_int(json, "moe_shared_expert_intermediate_size", 0);
    config_.routed_scaling_factor = json_float(json, "routed_scaling_factor", 1.0f);
    config_.n_group = json_int(json, "n_group", 1);
    config_.topk_group = json_int(json, "topk_group", 1);
    config_.num_experts_per_tok = json_int(json, "num_experts_per_tok", config_.num_experts_per_tok);
    config_.moe_intermediate_size = json_int(json, "moe_intermediate_size", config_.moe_intermediate_size);
    config_.norm_topk_prob = json_bool(json, "norm_topk_prob", config_.norm_topk_prob);
    // Sync n_routed_experts -> num_experts as fallback
    if (config_.num_experts == 0 && config_.n_routed_experts > 0) {
        config_.num_experts = config_.n_routed_experts;
    }

    // Pattern
    hybrid_override_pattern_ = json_value_for_key(json, "hybrid_override_pattern");

    // Derived dims
    d_inner_ = mamba_num_heads_ * mamba_head_dim_;  // 96 * 80 = 7680
    conv_dim_ = d_inner_ + 2 * n_groups_ * ssm_state_size_;  // 7680 + 2048 = 9728
    attn_head_dim_ = config_.head_dim > 0 ? config_.head_dim : config_.hidden_size / config_.num_attention_heads;
}

// ---------------------------------------------------------------------------
// Pattern parsing
// ---------------------------------------------------------------------------

void NemotronHModel::parse_pattern() {
    block_types_.clear();
    for (char c : hybrid_override_pattern_) {
        switch (c) {
            case 'M': block_types_.push_back(BlockType::Mamba); break;
            case '*': block_types_.push_back(BlockType::Attention); break;
            case '-': block_types_.push_back(BlockType::MLP); break;
            case 'E': block_types_.push_back(BlockType::MOE); break;
            default:
                throw std::runtime_error("Unknown block type in pattern: " + std::string(1, c));
        }
    }
    if ((int)block_types_.size() != config_.num_hidden_layers) {
        throw std::runtime_error("Pattern length " + std::to_string(block_types_.size()) +
                                 " != num_hidden_layers " + std::to_string(config_.num_hidden_layers));
    }

    // Build mapping: for each attention block, which index in the attention-only list it is
    // This is needed because KV cache only has slots for attention blocks
    int attn_idx = 0;
    int mamba_idx = 0;
    attn_layer_indices_.clear();
    mamba_layer_indices_.clear();
    for (int i = 0; i < (int)block_types_.size(); i++) {
        if (block_types_[i] == BlockType::Attention) {
            attn_layer_indices_.push_back(attn_idx++);
        } else {
            attn_layer_indices_.push_back(-1);
        }
        if (block_types_[i] == BlockType::Mamba) {
            mamba_layer_indices_.push_back(mamba_idx++);
        } else {
            mamba_layer_indices_.push_back(-1);
        }
    }
    num_attn_layers_ = attn_idx;
    num_mamba_layers_ = mamba_idx;
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

void NemotronHModel::load_weights(const std::string& model_path) {
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
        std::cout << "[NemotronH] Loading " << file << std::endl;
        auto [w, meta] = mx::load_safetensors(file);
        for (auto& [key, arr] : w) {
            // Strip "backbone." prefix if present
            std::string clean_key = key;
            if (clean_key.substr(0, 9) == "backbone.") {
                clean_key = clean_key.substr(9);
            }
            weights_.insert_or_assign(clean_key, std::move(arr));
        }
    }

    // Eval all weights
    std::vector<mx::array> to_eval;
    to_eval.reserve(weights_.size());
    for (auto& [k, v] : weights_) {
        to_eval.push_back(v);
    }
    mx::eval(to_eval);

    std::cout << "[NemotronH] Loaded " << weights_.size() << " weight tensors" << std::endl;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

mx::array NemotronHModel::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end())
        throw std::runtime_error("[NemotronH] Weight not found: " + name);
    return it->second;
}

bool NemotronHModel::has_weight(const std::string& name) const {
    return weights_.find(name) != weights_.end();
}

mx::array NemotronHModel::rms_norm(const mx::array& x, const mx::array& weight) {
    return mx::fast::rms_norm(x, weight, config_.rms_norm_eps);
}

mx::array NemotronHModel::linear(const mx::array& x, const std::string& prefix) {
    if (config_.quant_bits > 0 && has_weight(prefix + ".scales")) {
        auto w = get_weight(prefix + ".weight");
        auto scales = get_weight(prefix + ".scales");
        std::optional<mx::array> biases = std::nullopt;
        if (has_weight(prefix + ".biases")) {
            biases = get_weight(prefix + ".biases");
        }
        return mx::quantized_matmul(x, w, scales, biases, /*transpose=*/true,
                                    config_.quant_group_size, config_.quant_bits);
    }
    auto w = get_weight(prefix + ".weight");
    return mx::matmul(x, mx::transpose(w, {1, 0}));
}

mx::array NemotronHModel::embed(const mx::array& input_ids) {
    if (has_weight("_embed_dequantized")) {
        return mx::take(get_weight("_embed_dequantized"), input_ids, 0);
    }
    auto w = get_weight("embeddings.weight");
    return mx::take(w, input_ids, 0);
}

mx::array NemotronHModel::lm_head_proj(const mx::array& x) {
    // lm_head is NOT under "backbone." prefix — stored without prefix
    // But it was loaded from the safetensors file. Check if it's under the
    // backbone-stripped namespace or the raw key.
    // From the weight dump: "lm_head.weight", "lm_head.scales", "lm_head.biases"
    // (no backbone. prefix in the original file for lm_head)
    // After stripping backbone., these stay as-is if they didn't have it.
    // But we strip backbone. only when key starts with "backbone."
    // lm_head.* keys don't start with backbone., so they're stored as "lm_head.*"
    return linear(x, "lm_head");
}

// ---------------------------------------------------------------------------
// SSM Metal Kernel
// ---------------------------------------------------------------------------

void NemotronHModel::build_ssm_kernel() {
    std::string source = R"(
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

    ssm_kernel_ = mx::fast::metal_kernel(
        "ssm_kernel",
        {"X", "A_log", "B", "C", "D", "dt", "state_in"},
        {"out", "state_out"},
        source);
}

// ---------------------------------------------------------------------------
// Mamba-2 Mixer (single-token decode path)
// ---------------------------------------------------------------------------

mx::array NemotronHModel::mamba_block(
    const mx::array& x, int layer, int mamba_idx) {
    // x: [B, L, hidden_size]
    int B_dim = x.shape(0);
    int L = x.shape(1);

    std::string prefix = "layers." + std::to_string(layer) + ".mixer";

    // in_proj: [B, L, hidden] -> [B, L, gate + conv_input + dt]
    //   gate: d_inner (7680), conv_input: conv_dim (9728), dt: n_heads (96)
    auto proj = linear(x, prefix + ".in_proj");

    // Split into gate, conv_input, dt
    int gate_size = d_inner_;
    int dt_size = mamba_num_heads_;

    auto gate = mx::slice(proj, {0, 0, 0}, {B_dim, L, gate_size});
    auto conv_input = mx::slice(proj, {0, 0, gate_size}, {B_dim, L, gate_size + conv_dim_});
    auto dt = mx::slice(proj, {0, 0, gate_size + conv_dim_}, {B_dim, L, gate_size + conv_dim_ + dt_size});

    // Conv1d with state management
    // conv_state: [B, conv_kernel-1, conv_dim]
    // For token-by-token decode:
    //   - shift conv_state left, append conv_input
    //   - apply depthwise conv1d + silu
    auto& conv_state = mamba_conv_states_[mamba_idx];

    if (L == 1) {
        // Single token decode: update sliding window
        // conv_state: [B, conv_kernel-1, conv_dim] stores the last k-1 conv inputs.
        // For the convolution we need the last k inputs = concat(state, new_input).
        auto conv_w = get_weight(prefix + ".conv1d.weight");
        // conv_w shape: [conv_dim, conv_kernel, 1] -> squeeze to [conv_dim, conv_kernel]
        conv_w = mx::squeeze(conv_w, -1);  // [conv_dim, conv_kernel]

        // Build full convolution window: [B, conv_kernel, conv_dim]
        auto full_window = mx::concatenate({conv_state, conv_input}, 1);

        // Update state: drop oldest, keep last conv_kernel-1 for next step
        conv_state = mx::slice(full_window, {0, 1, 0}, {B_dim, conv_kernel_, conv_dim_});

        // Depthwise conv1d using MLX's built-in (faster Metal kernel)
        // full_window: [B, conv_kernel, conv_dim], conv_w: [conv_dim, conv_kernel]
        // conv1d expects: input [B, L, C_in], weight [C_out, K_w, C_in/groups]
        // For depthwise: groups=conv_dim, C_out=conv_dim, C_in/groups=1
        // Weight shape needs to be [conv_dim, conv_kernel, 1]
        auto conv_w_3d = mx::expand_dims(conv_w, -1);  // [conv_dim, conv_kernel, 1]
        auto conv_out = mx::conv1d(full_window, conv_w_3d, /*stride=*/1, /*padding=*/0,
                                   /*dilation=*/1, /*groups=*/conv_dim_);
        // conv_out: [B, 1, conv_dim]

        // Add conv bias
        if (use_conv_bias_) {
            auto conv_b = get_weight(prefix + ".conv1d.bias");  // [conv_dim]
            conv_out = mx::add(conv_out, conv_b);
        }

        // SiLU activation
        conv_out = mx::multiply(conv_out, mx::sigmoid(conv_out));

        // Split conv output: hidden_states, B_ssm, C_ssm
        auto hidden_states = mx::slice(conv_out, {0, 0, 0}, {B_dim, 1, d_inner_});
        int bc_size = n_groups_ * ssm_state_size_;
        auto B_ssm = mx::slice(conv_out, {0, 0, d_inner_}, {B_dim, 1, d_inner_ + bc_size});
        auto C_ssm = mx::slice(conv_out, {0, 0, d_inner_ + bc_size}, {B_dim, 1, d_inner_ + 2 * bc_size});

        // Reshape for SSM
        // hidden: [B, 1, n_heads, head_dim]
        hidden_states = mx::reshape(hidden_states, {B_dim, mamba_num_heads_, mamba_head_dim_});
        // B_ssm: [B, n_groups, state_size]
        B_ssm = mx::reshape(B_ssm, {B_dim, n_groups_, ssm_state_size_});
        // C_ssm: [B, n_groups, state_size]
        C_ssm = mx::reshape(C_ssm, {B_dim, n_groups_, ssm_state_size_});

        // Compute dt: softplus + clip
        auto dt_bias = get_weight(prefix + ".dt_bias");  // [n_heads]
        dt = mx::add(dt, dt_bias);
        // softplus: log(1 + exp(x))
        dt = mx::log1p(mx::exp(dt));
        // clip
        dt = mx::clip(dt, mx::array(time_step_min_), mx::array(time_step_max_));
        dt = mx::reshape(dt, {B_dim, mamba_num_heads_});  // [B, n_heads]

        // SSM kernel
        auto A_log = get_weight(prefix + ".A_log");  // [n_heads]
        auto D_param = get_weight(prefix + ".D");    // [n_heads]

        auto& ssm_state = mamba_ssm_states_[mamba_idx];
        // ssm_state: [B, n_heads, head_dim, state_size]

        // Call Metal kernel
        // Template args: T=bfloat16, Dh=head_dim, Ds=state_size, H=n_heads, G=heads_per_group
        int G = mamba_num_heads_ / n_groups_;  // heads per group

        // Grid: (32, head_dim, B*n_heads), Threadgroup: (32, 1, 1)
        // n_per_t = Ds / 32 = 128 / 32 = 4
        auto ssm_out_shape = mx::Shape{B_dim, mamba_num_heads_, mamba_head_dim_};
        auto ssm_state_shape = mx::Shape{B_dim, mamba_num_heads_, mamba_head_dim_, ssm_state_size_};

        auto kernel_results = ssm_kernel_(
            {hidden_states, A_log, B_ssm, C_ssm, D_param, dt, ssm_state},
            {ssm_out_shape, ssm_state_shape},
            {config_.activation_dtype, config_.activation_dtype},
            std::make_tuple(32, mamba_head_dim_, B_dim * mamba_num_heads_),
            std::make_tuple(32, 1, 1),
            {{"T", config_.activation_dtype},
             {"Dh", mamba_head_dim_},
             {"Ds", ssm_state_size_},
             {"H", mamba_num_heads_},
             {"G", G}},
            std::nullopt,
            false,
            {});

        auto y = kernel_results[0];       // [B, n_heads, head_dim]
        ssm_state = kernel_results[1];     // [B, n_heads, head_dim, state_size]

        // Reshape y: [B, n_heads, head_dim] -> [B, 1, d_inner]
        y = mx::reshape(y, {B_dim, 1, d_inner_});

        // MambaRMSNormGated: swiglu(gate, y) then group RMSNorm
        // gate: [B, 1, d_inner]
        // silu(gate) * y
        auto gated = mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), y);

        // Group RMS norm: unflatten to [B, 1, n_groups, d_inner/n_groups],
        // rms_norm per group, flatten back
        int group_dim = d_inner_ / n_groups_;
        gated = mx::reshape(gated, {B_dim, 1, n_groups_, group_dim});
        // RMS norm per group (last axis)
        auto sq = mx::mean(mx::square(gated), -1, /*keepdims=*/true);
        auto rms = mx::rsqrt(mx::add(sq, mx::array(config_.rms_norm_eps)));
        gated = mx::multiply(gated, rms);
        gated = mx::reshape(gated, {B_dim, 1, d_inner_});

        // Multiply by norm weight
        auto norm_w = get_weight(prefix + ".norm.weight");  // [d_inner]
        gated = mx::multiply(gated, norm_w);

        // Output projection
        return linear(gated, prefix + ".out_proj");
    } else {
        // Prefill path: process all tokens through conv1d
        // conv_input: [B, L, conv_dim]
        auto conv_w = get_weight(prefix + ".conv1d.weight");
        // conv_w: [conv_dim, conv_kernel, 1] -> need [conv_dim, 1, conv_kernel] for depthwise conv1d
        // MLX conv1d expects input: [B, L, C_in], weight: [C_out, kW, C_in/groups]
        // For depthwise: groups = conv_dim, C_in = conv_dim, C_out = conv_dim
        // weight shape: [conv_dim, conv_kernel, 1]
        // conv1d with groups=conv_dim, padding=conv_kernel-1 for causal

        auto conv_out = mx::conv1d(conv_input, conv_w,
                                   /*stride=*/1,
                                   /*padding=*/conv_kernel_ - 1,
                                   /*dilation=*/1,
                                   /*groups=*/conv_dim_);
        // Trim to causal: take first L elements (conv adds padding on both sides)
        conv_out = mx::slice(conv_out, {0, 0, 0}, {B_dim, L, conv_dim_});

        // Add conv bias
        if (use_conv_bias_) {
            auto conv_b = get_weight(prefix + ".conv1d.bias");
            conv_out = mx::add(conv_out, conv_b);
        }

        // SiLU
        conv_out = mx::multiply(conv_out, mx::sigmoid(conv_out));

        // Update conv state: store last conv_kernel-1 inputs for future decode
        if (L >= conv_kernel_ - 1) {
            conv_state = mx::slice(conv_input, {0, L - (conv_kernel_ - 1), 0},
                                   {B_dim, L, conv_dim_});
        } else {
            auto tail = mx::slice(conv_input, {0, 0, 0}, {B_dim, L, conv_dim_});
            auto pad = mx::zeros({B_dim, conv_kernel_ - 1 - L, conv_dim_}, config_.activation_dtype);
            conv_state = mx::concatenate({pad, tail}, 1);
        }
        mx::eval({conv_state});

        // Split conv output
        auto hidden_states = mx::slice(conv_out, {0, 0, 0}, {B_dim, L, d_inner_});
        int bc_size = n_groups_ * ssm_state_size_;
        auto B_ssm = mx::slice(conv_out, {0, 0, d_inner_}, {B_dim, L, d_inner_ + bc_size});
        auto C_ssm = mx::slice(conv_out, {0, 0, d_inner_ + bc_size}, {B_dim, L, d_inner_ + 2 * bc_size});

        // Compute dt
        auto dt_bias = get_weight(prefix + ".dt_bias");
        dt = mx::add(dt, dt_bias);
        dt = mx::log1p(mx::exp(dt));
        dt = mx::clip(dt, mx::array(time_step_min_), mx::array(time_step_max_));

        // For prefill, we run SSM sequentially over time steps
        auto A_log = get_weight(prefix + ".A_log");
        auto D_param = get_weight(prefix + ".D");
        int G = mamba_num_heads_ / n_groups_;

        auto& ssm_state = mamba_ssm_states_[mamba_idx];

        // Process each time step
        std::vector<mx::array> y_steps;
        for (int t = 0; t < L; t++) {
            auto hs_t = mx::slice(hidden_states, {0, t, 0}, {B_dim, t + 1, d_inner_});
            hs_t = mx::reshape(hs_t, {B_dim, mamba_num_heads_, mamba_head_dim_});

            auto B_t = mx::slice(B_ssm, {0, t, 0}, {B_dim, t + 1, bc_size});
            B_t = mx::reshape(B_t, {B_dim, n_groups_, ssm_state_size_});

            auto C_t = mx::slice(C_ssm, {0, t, 0}, {B_dim, t + 1, bc_size});
            C_t = mx::reshape(C_t, {B_dim, n_groups_, ssm_state_size_});

            auto dt_t = mx::slice(dt, {0, t, 0}, {B_dim, t + 1, mamba_num_heads_});
            dt_t = mx::reshape(dt_t, {B_dim, mamba_num_heads_});

            auto ssm_out_shape = mx::Shape{B_dim, mamba_num_heads_, mamba_head_dim_};
            auto ssm_state_shape = mx::Shape{B_dim, mamba_num_heads_, mamba_head_dim_, ssm_state_size_};

            auto kernel_results = ssm_kernel_(
                {hs_t, A_log, B_t, C_t, D_param, dt_t, ssm_state},
                {ssm_out_shape, ssm_state_shape},
                {config_.activation_dtype, config_.activation_dtype},
                std::make_tuple(32, mamba_head_dim_, B_dim * mamba_num_heads_),
                std::make_tuple(32, 1, 1),
                {{"T", config_.activation_dtype},
                 {"Dh", mamba_head_dim_},
                 {"Ds", ssm_state_size_},
                 {"H", mamba_num_heads_},
                 {"G", G}},
                std::nullopt,
                false,
                {});

            auto y_t = kernel_results[0];
            ssm_state = kernel_results[1];
            // Eval to prevent graph explosion
            mx::eval({y_t, ssm_state});
            y_steps.push_back(mx::expand_dims(y_t, 1));  // [B, 1, n_heads, head_dim]
        }

        // Stack all timesteps: [B, L, n_heads, head_dim]
        auto y = mx::concatenate(y_steps, 1);
        // Reshape: [B, L, d_inner]
        y = mx::reshape(y, {B_dim, L, d_inner_});

        // MambaRMSNormGated
        auto gated = mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), y);

        int group_dim = d_inner_ / n_groups_;
        gated = mx::reshape(gated, {B_dim, L, n_groups_, group_dim});
        auto sq = mx::mean(mx::square(gated), -1, /*keepdims=*/true);
        auto rms = mx::rsqrt(mx::add(sq, mx::array(config_.rms_norm_eps)));
        gated = mx::multiply(gated, rms);
        gated = mx::reshape(gated, {B_dim, L, d_inner_});

        auto norm_w = get_weight(prefix + ".norm.weight");
        gated = mx::multiply(gated, norm_w);

        return linear(gated, prefix + ".out_proj");
    }
}

// ---------------------------------------------------------------------------
// Attention block (no RoPE)
// ---------------------------------------------------------------------------

mx::array NemotronHModel::attention_block(
    const mx::array& x, int layer, int attn_idx,
    mx::array& cache_k, mx::array& cache_v,
    int cache_offset) {

    std::string prefix = "layers." + std::to_string(layer) + ".mixer";

    int B_dim = x.shape(0);
    int L = x.shape(1);
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int hd = attn_head_dim_;

    auto q = linear(x, prefix + ".q_proj");
    auto k = linear(x, prefix + ".k_proj");
    auto v = linear(x, prefix + ".v_proj");

    q = mx::transpose(mx::reshape(q, {B_dim, L, n_heads, hd}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B_dim, L, n_kv_heads, hd}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B_dim, L, n_kv_heads, hd}), {0, 2, 1, 3});

    // NO RoPE for Nemotron-H attention

    // KV cache update (concat approach for decode)
    cache_k = mx::concatenate({cache_k, k}, 2);
    cache_v = mx::concatenate({cache_v, v}, 2);

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::string mask_mode = (L > 1) ? "causal" : "";

    auto attn_out = mx::fast::scaled_dot_product_attention(
        q, cache_k, cache_v, scale, mask_mode);

    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B_dim, L, n_heads * hd});

    return linear(attn_out, prefix + ".o_proj");
}

mx::array NemotronHModel::attention_block(
    const mx::array& x, int layer, int attn_idx,
    mx::array& cache_k, mx::array& cache_v,
    const mx::array& cache_offsets) {

    std::string prefix = "layers." + std::to_string(layer) + ".mixer";

    int B_dim = x.shape(0);
    int L = x.shape(1);
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int hd = attn_head_dim_;

    auto q = linear(x, prefix + ".q_proj");
    auto k = linear(x, prefix + ".k_proj");
    auto v = linear(x, prefix + ".v_proj");

    q = mx::transpose(mx::reshape(q, {B_dim, L, n_heads, hd}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B_dim, L, n_kv_heads, hd}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B_dim, L, n_kv_heads, hd}), {0, 2, 1, 3});

    // NO RoPE

    // KV cache: slice_update approach for array offsets
    auto first_offset = mx::slice(cache_offsets, {0}, {1});
    mx::eval({first_offset});
    int offset_val = first_offset.item<int32_t>();

    cache_k = mx::slice_update(
        cache_k, k,
        {0, 0, offset_val, 0},
        {B_dim, n_kv_heads, offset_val + L, hd});
    cache_v = mx::slice_update(
        cache_v, v,
        {0, 0, offset_val, 0},
        {B_dim, n_kv_heads, offset_val + L, hd});

    int total_len = offset_val + L;
    auto full_k = mx::slice(cache_k, {0, 0, 0, 0}, {B_dim, n_kv_heads, total_len, hd});
    auto full_v = mx::slice(cache_v, {0, 0, 0, 0}, {B_dim, n_kv_heads, total_len, hd});

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::string mask_mode = (L > 1) ? "causal" : "";

    auto attn_out = mx::fast::scaled_dot_product_attention(
        q, full_k, full_v, scale, mask_mode);

    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B_dim, L, n_heads * hd});

    return linear(attn_out, prefix + ".o_proj");
}

// ---------------------------------------------------------------------------
// MLP block: up_proj -> relu^2 -> down_proj
// ---------------------------------------------------------------------------

mx::array NemotronHModel::mlp_block(const mx::array& x, int layer) {
    std::string prefix = "layers." + std::to_string(layer) + ".mixer";

    auto up = linear(x, prefix + ".up_proj");
    // relu squared
    auto activated = mx::square(mx::maximum(up, mx::array(0.0f, up.dtype())));
    return linear(activated, prefix + ".down_proj");
}

// ---------------------------------------------------------------------------
// MoE block: router -> switch_mlp (gather_qmm) -> shared expert
// ---------------------------------------------------------------------------

mx::array NemotronHModel::moe_block(const mx::array& x, int layer) {
    auto prefix = "layers." + std::to_string(layer) + ".mixer.";
    int k = config_.num_experts_per_tok;
    int B_dim = x.shape(0);
    int L = x.shape(1);

    // 1. Router
    auto gate_w = get_weight(prefix + "gate.weight");
    auto logits = mx::matmul(x, mx::transpose(gate_w, {1, 0}));
    if (has_weight(prefix + "gate.e_score_correction_bias")) {
        logits = mx::add(logits, get_weight(prefix + "gate.e_score_correction_bias"));
    }
    auto scores = mx::softmax(logits, -1);

    // 2. Top-k selection (simplified V1 — skip group selection)
    auto neg = mx::negative(scores);
    auto topk_inds = mx::argpartition(neg, k - 1, -1);
    topk_inds = mx::slice(topk_inds, {0, 0, 0}, {B_dim, L, k});
    auto topk_scores = mx::take_along_axis(scores, topk_inds, -1);

    // Scale
    topk_scores = mx::multiply(topk_scores, mx::array(config_.routed_scaling_factor));
    if (config_.norm_topk_prob) {
        topk_scores = mx::divide(topk_scores, mx::sum(topk_scores, -1, true));
    }

    // 3. SwitchMLP: fc1 -> relu^2 -> fc2 (via gather_qmm)
    auto x_exp = mx::expand_dims(x, {-2, -3});  // [B, L, 1, 1, hidden]

    auto fc1_w = get_weight(prefix + "switch_mlp.fc1.weight");
    auto fc1_s = get_weight(prefix + "switch_mlp.fc1.scales");
    std::optional<mx::array> fc1_b = std::nullopt;
    if (has_weight(prefix + "switch_mlp.fc1.biases")) {
        fc1_b = get_weight(prefix + "switch_mlp.fc1.biases");
    }

    auto fc2_w = get_weight(prefix + "switch_mlp.fc2.weight");
    auto fc2_s = get_weight(prefix + "switch_mlp.fc2.scales");
    std::optional<mx::array> fc2_b = std::nullopt;
    if (has_weight(prefix + "switch_mlp.fc2.biases")) {
        fc2_b = get_weight(prefix + "switch_mlp.fc2.biases");
    }

    auto h = mx::gather_qmm(x_exp, fc1_w, fc1_s, fc1_b,
                             std::nullopt, topk_inds, /*transpose=*/true,
                             config_.quant_group_size, config_.quant_bits, "affine", false);
    // relu^2
    h = mx::square(mx::maximum(h, mx::array(0.0f)));
    h = mx::gather_qmm(h, fc2_w, fc2_s, fc2_b,
                        std::nullopt, topk_inds, /*transpose=*/true,
                        config_.quant_group_size, config_.quant_bits, "affine", false);

    h = mx::squeeze(h, -2);  // [B, L, k, hidden]

    // 4. Weighted sum
    auto y = mx::multiply(h, mx::expand_dims(topk_scores, -1));
    y = mx::sum(y, -2);  // [B, L, hidden]

    // 5. Shared expert
    if (config_.n_shared_experts > 0 && has_weight(prefix + "shared_experts.up_proj.weight")) {
        auto shared = linear(x, prefix + "shared_experts.up_proj");
        shared = mx::square(mx::maximum(shared, mx::array(0.0f)));  // relu^2
        shared = linear(shared, prefix + "shared_experts.down_proj");
        y = mx::add(y, shared);
    }

    return y;
}

// ---------------------------------------------------------------------------
// Mamba state initialization
// ---------------------------------------------------------------------------

void NemotronHModel::init_mamba_states(int batch_size) {
    mamba_conv_states_.clear();
    mamba_ssm_states_.clear();

    for (int i = 0; i < num_mamba_layers_; i++) {
        mamba_conv_states_.push_back(
            mx::zeros({batch_size, conv_kernel_ - 1, conv_dim_}, config_.activation_dtype));
        mamba_ssm_states_.push_back(
            mx::zeros({batch_size, mamba_num_heads_, mamba_head_dim_, ssm_state_size_},
                      config_.activation_dtype));
    }

    // Eval to materialize
    std::vector<mx::array> to_eval;
    for (auto& s : mamba_conv_states_) to_eval.push_back(s);
    for (auto& s : mamba_ssm_states_) to_eval.push_back(s);
    mx::eval(to_eval);

    mamba_states_initialized_ = true;
}

// ---------------------------------------------------------------------------
// Forward pass (int offset — used for decode)
// ---------------------------------------------------------------------------

mx::array NemotronHModel::forward(
    const mx::array& input_ids,
    std::vector<mx::array>& cache_keys,
    std::vector<mx::array>& cache_values,
    int cache_offset) {

    int B_dim = input_ids.shape(0);

    // Initialize mamba states if needed
    if (!mamba_states_initialized_) {
        init_mamba_states(B_dim);
    }

    auto h = embed(input_ids);

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        auto norm_w = get_weight("layers." + std::to_string(i) + ".norm.weight");
        auto normed = rms_norm(h, norm_w);

        mx::array block_out = normed;  // placeholder, overwritten by each branch
        switch (block_types_[i]) {
            case BlockType::Mamba: {
                int mamba_idx = mamba_layer_indices_[i];
                block_out = mamba_block(normed, i, mamba_idx);
                break;
            }
            case BlockType::Attention: {
                int attn_idx = attn_layer_indices_[i];
                block_out = attention_block(normed, i, attn_idx,
                                            cache_keys[attn_idx], cache_values[attn_idx],
                                            cache_offset);
                break;
            }
            case BlockType::MLP: {
                block_out = mlp_block(normed, i);
                break;
            }
            case BlockType::MOE: {
                block_out = moe_block(normed, i);
                break;
            }
        }

        h = mx::add(h, block_out);

        // Mamba states are updated lazily — no per-block eval needed
    }

    // Final norm
    auto norm_f_w = get_weight("norm_f.weight");
    h = rms_norm(h, norm_f_w);

    return lm_head_proj(h);
}

// ---------------------------------------------------------------------------
// Forward pass (array offset — used for prefill)
// ---------------------------------------------------------------------------

mx::array NemotronHModel::forward(
    const mx::array& input_ids,
    std::vector<mx::array>& cache_keys,
    std::vector<mx::array>& cache_values,
    const mx::array& cache_offsets) {

    int B_dim = input_ids.shape(0);

    if (!mamba_states_initialized_) {
        init_mamba_states(B_dim);
    }

    auto h = embed(input_ids);

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        auto norm_w = get_weight("layers." + std::to_string(i) + ".norm.weight");
        auto normed = rms_norm(h, norm_w);


        mx::array block_out = normed;  // placeholder, overwritten by each branch
        switch (block_types_[i]) {
            case BlockType::Mamba: {
                int mamba_idx = mamba_layer_indices_[i];
                block_out = mamba_block(normed, i, mamba_idx);
                break;
            }
            case BlockType::Attention: {
                int attn_idx = attn_layer_indices_[i];
                block_out = attention_block(normed, i, attn_idx,
                                            cache_keys[attn_idx], cache_values[attn_idx],
                                            cache_offsets);
                break;
            }
            case BlockType::MLP: {
                block_out = mlp_block(normed, i);
                break;
            }
            case BlockType::MOE: {
                block_out = moe_block(normed, i);
                break;
            }
        }

        h = mx::add(h, block_out);
        mx::eval({h});
    }

    auto norm_f_w = get_weight("norm_f.weight");
    h = rms_norm(h, norm_f_w);

    return lm_head_proj(h);
}

// ---------------------------------------------------------------------------
// Debug helpers
// ---------------------------------------------------------------------------

std::vector<int> NemotronHModel::debug_forward(const std::vector<int>& token_ids) {
    auto input = mx::array(token_ids.data(), {1, (int)token_ids.size()}, mx::int32);

    // Reset mamba states for fresh inference
    mamba_states_initialized_ = false;

    // Create KV caches only for attention layers
    int n_kv = config_.num_key_value_heads;
    int hd = attn_head_dim_;
    int ctx = 512;

    std::vector<mx::array> cache_k, cache_v;
    for (int i = 0; i < num_attn_layers_; i++) {
        cache_k.push_back(mx::zeros({1, n_kv, ctx, hd}, mx::float16));
        cache_v.push_back(mx::zeros({1, n_kv, ctx, hd}, mx::float16));
    }
    mx::eval(cache_k);
    mx::eval(cache_v);

    auto cache_offsets = mx::array({0}, mx::int32);
    auto logits = forward(input, cache_k, cache_v, cache_offsets);
    mx::eval({logits});

    int seq_len = logits.shape(1);
    auto last_logits = mx::slice(logits, {0, seq_len - 1, 0}, {1, seq_len, logits.shape(2)});
    last_logits = mx::reshape(last_logits, {-1});

    auto neg_logits = mx::negative(last_logits);
    auto top_indices = mx::argpartition(neg_logits, 10);
    top_indices = mx::slice(top_indices, {0}, {10});
    mx::eval({top_indices});

    std::vector<int> result;
    const int32_t* ptr = top_indices.data<int32_t>();
    for (int i = 0; i < 10; i++) {
        result.push_back(ptr[i]);
    }
    return result;
}

std::vector<float> NemotronHModel::debug_embed(const std::vector<int>& token_ids) {
    auto input = mx::array(token_ids.data(), {1, (int)token_ids.size()}, mx::int32);
    auto h = embed(input);
    mx::eval({h});

    auto h_f32 = mx::astype(h, mx::float32);
    h_f32 = mx::reshape(h_f32, {-1});
    mx::eval({h_f32});
    const float* ptr = h_f32.data<float>();
    int total = h_f32.size();
    std::vector<float> result;
    for (int i = 0; i < std::min(total, 20); i++) {
        result.push_back(ptr[i]);
    }
    return result;
}

} // namespace flashmlx
