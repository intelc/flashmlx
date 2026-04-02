#include "flashmlx/model.h"
#include <mlx/compile.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <cmath>

namespace flashmlx {

// Compiled SwiGLU: silu(gate) * x → single fused Metal kernel
static std::vector<mx::array> _swiglu_impl(const std::vector<mx::array>& inputs) {
    auto& gate = inputs[0];
    auto& x = inputs[1];
    return {mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), x)};
}
static auto compiled_swiglu_nh = mx::compile(_swiglu_impl, /*shapeless=*/true);

static mx::array swiglu_nh(const mx::array& gate, const mx::array& up) {
    return compiled_swiglu_nh({gate, up})[0];
}

// Compiled MoE routing factory: creates a compiled function for specific k value
// (matches mlx-lm's @mx.compile group_expert_select which captures k at decoration time)
static std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
make_compiled_moe_route(int k) {
    auto impl = [k](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        // inputs: [gates, correction_bias, scaling_factor_arr]
        auto& gates = inputs[0];           // [B, L, num_experts]
        auto& correction = inputs[1];      // [num_experts]
        auto& scale = inputs[2];           // scalar

        auto orig_scores = mx::sigmoid(mx::astype(gates, mx::float32));
        auto selection_scores = mx::add(orig_scores, correction);

        auto topk_inds = mx::argpartition(mx::negative(selection_scores), k - 1, -1);
        int B = gates.shape(0), L = gates.shape(1);
        topk_inds = mx::slice(topk_inds, {0, 0, 0}, {B, L, k});
        auto topk_scores = mx::take_along_axis(orig_scores, topk_inds, -1);

        // Normalize
        auto denom = mx::add(mx::sum(topk_scores, -1, true), mx::array(1e-20f));
        topk_scores = mx::multiply(mx::divide(topk_scores, denom), scale);

        return {topk_inds, topk_scores};
    };
    return mx::compile(impl, /*shapeless=*/false);
}

// Compiled relu²: relu(x)² → single fused kernel
static std::vector<mx::array> _relu2_impl(const std::vector<mx::array>& inputs) {
    auto activated = mx::maximum(inputs[0], mx::array(0.0f));
    return {mx::square(activated)};
}
static auto compiled_relu2 = mx::compile(_relu2_impl, /*shapeless=*/true);

static mx::array relu2(const mx::array& x) {
    return compiled_relu2({x})[0];
}

// Compiled softplus + clip for Mamba dt, matching mlx_lm.models.ssm.compute_dt.
static std::vector<mx::array> _compute_dt_impl(const std::vector<mx::array>& inputs) {
    auto shifted = mx::add(inputs[0], inputs[1]);
    auto softplus = mx::logaddexp(shifted, mx::zeros_like(shifted));
    return {mx::clip(softplus, inputs[2], inputs[3])};
}
static auto compiled_compute_dt = mx::compile(_compute_dt_impl, /*shapeless=*/true);

static mx::array compute_dt(
    const mx::array& dt,
    const mx::array& dt_bias,
    float dt_min,
    float dt_max) {
    return compiled_compute_dt(
        {dt, dt_bias, mx::array(dt_min, mx::float32), mx::array(dt_max, mx::float32)})[0];
}

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

    // Skip pre-dequantize embedding to save 700MB GPU memory
    // Use quantized_matmul for embed instead (like mlx-lm)
    // if (has_weight("embeddings.scales")) { ... }

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

mx::array NemotronHModel::linear_fast(const mx::array& x, const mx::array& w, const mx::array& s,
                                      const std::optional<mx::array>& b) {
    return mx::quantized_matmul(x, w, s, b, /*transpose=*/true,
                                config_.quant_group_size, config_.quant_bits);
}

void NemotronHModel::build_weight_cache() {
    int n_layers = config_.num_hidden_layers;
    mamba_layer_weights_.resize(n_layers);
    attn_layer_weights_.resize(n_layers);
    moe_layer_weights_.resize(n_layers);
    mlp_layer_weights_.resize(n_layers);

    for (int i = 0; i < n_layers; i++) {
        std::string lp = "layers." + std::to_string(i);
        std::string mp = lp + ".mixer";

        switch (block_types_[i]) {
            case BlockType::Mamba: {
                MambaLayerWeights w;
                w.norm_w = get_weight(lp + ".norm.weight");
                w.in_proj_w = get_weight(mp + ".in_proj.weight");
                w.in_proj_s = get_weight(mp + ".in_proj.scales");
                if (has_weight(mp + ".in_proj.biases"))
                    w.in_proj_b = get_weight(mp + ".in_proj.biases");
                w.conv1d_w = get_weight(mp + ".conv1d.weight");
                if (use_conv_bias_ && has_weight(mp + ".conv1d.bias"))
                    w.conv1d_bias = get_weight(mp + ".conv1d.bias");
                w.dt_bias = get_weight(mp + ".dt_bias");
                w.A_log = get_weight(mp + ".A_log");
                w.D_param = get_weight(mp + ".D");
                w.mixer_norm_w = get_weight(mp + ".norm.weight");
                w.out_proj_w = get_weight(mp + ".out_proj.weight");
                w.out_proj_s = get_weight(mp + ".out_proj.scales");
                if (has_weight(mp + ".out_proj.biases"))
                    w.out_proj_b = get_weight(mp + ".out_proj.biases");
                mamba_layer_weights_[i] = std::move(w);
                break;
            }
            case BlockType::Attention: {
                AttnLayerWeights w;
                w.norm_w = get_weight(lp + ".norm.weight");
                w.q_w = get_weight(mp + ".q_proj.weight");
                w.q_s = get_weight(mp + ".q_proj.scales");
                w.k_w = get_weight(mp + ".k_proj.weight");
                w.k_s = get_weight(mp + ".k_proj.scales");
                w.v_w = get_weight(mp + ".v_proj.weight");
                w.v_s = get_weight(mp + ".v_proj.scales");
                w.o_w = get_weight(mp + ".o_proj.weight");
                w.o_s = get_weight(mp + ".o_proj.scales");
                if (has_weight(mp + ".q_proj.biases")) w.q_b = get_weight(mp + ".q_proj.biases");
                if (has_weight(mp + ".k_proj.biases")) w.k_b = get_weight(mp + ".k_proj.biases");
                if (has_weight(mp + ".v_proj.biases")) w.v_b = get_weight(mp + ".v_proj.biases");
                if (has_weight(mp + ".o_proj.biases")) w.o_b = get_weight(mp + ".o_proj.biases");
                attn_layer_weights_[i] = std::move(w);
                break;
            }
            case BlockType::MOE: {
                MoELayerWeights w;
                w.norm_w = get_weight(lp + ".norm.weight");
                w.gate_w = get_weight(mp + ".gate.weight");
                if (has_weight(mp + ".gate.e_score_correction_bias"))
                    w.gate_correction_bias = get_weight(mp + ".gate.e_score_correction_bias");
                w.fc1_w = get_weight(mp + ".switch_mlp.fc1.weight");
                w.fc1_s = get_weight(mp + ".switch_mlp.fc1.scales");
                w.fc2_w = get_weight(mp + ".switch_mlp.fc2.weight");
                w.fc2_s = get_weight(mp + ".switch_mlp.fc2.scales");
                if (has_weight(mp + ".switch_mlp.fc1.biases"))
                    w.fc1_b = get_weight(mp + ".switch_mlp.fc1.biases");
                if (has_weight(mp + ".switch_mlp.fc2.biases"))
                    w.fc2_b = get_weight(mp + ".switch_mlp.fc2.biases");
                if (config_.n_shared_experts > 0 && has_weight(mp + ".shared_experts.up_proj.weight")) {
                    w.has_shared_expert = true;
                    w.shared_up_w = get_weight(mp + ".shared_experts.up_proj.weight");
                    w.shared_up_s = get_weight(mp + ".shared_experts.up_proj.scales");
                    w.shared_down_w = get_weight(mp + ".shared_experts.down_proj.weight");
                    w.shared_down_s = get_weight(mp + ".shared_experts.down_proj.scales");
                    if (has_weight(mp + ".shared_experts.up_proj.biases"))
                        w.shared_up_b = get_weight(mp + ".shared_experts.up_proj.biases");
                    if (has_weight(mp + ".shared_experts.down_proj.biases"))
                        w.shared_down_b = get_weight(mp + ".shared_experts.down_proj.biases");
                }
                moe_layer_weights_[i] = std::move(w);
                break;
            }
            case BlockType::MLP: {
                MLPLayerWeights w;
                w.norm_w = get_weight(lp + ".norm.weight");
                w.up_w = get_weight(mp + ".up_proj.weight");
                w.up_s = get_weight(mp + ".up_proj.scales");
                w.down_w = get_weight(mp + ".down_proj.weight");
                w.down_s = get_weight(mp + ".down_proj.scales");
                if (has_weight(mp + ".up_proj.biases"))
                    w.up_b = get_weight(mp + ".up_proj.biases");
                if (has_weight(mp + ".down_proj.biases"))
                    w.down_b = get_weight(mp + ".down_proj.biases");
                mlp_layer_weights_[i] = std::move(w);
                break;
            }
        }
    }

    weight_cache_built_ = true;
    std::cout << "[NemotronH] Weight cache built for " << n_layers << " layers" << std::endl;
}

// ---------------------------------------------------------------------------
// Fast mamba block using cached weights (no hash lookups)
// ---------------------------------------------------------------------------

std::vector<mx::array> NemotronHModel::mamba_block_functional_fast(
    const mx::array& x, int layer,
    const mx::array& conv_state, const mx::array& ssm_state) {

    auto& w = *mamba_layer_weights_[layer];
    int B_dim = x.shape(0);

    // in_proj
    auto proj = linear_fast(x, *w.in_proj_w, *w.in_proj_s, w.in_proj_b);

    int gate_size = d_inner_;
    int dt_size = mamba_num_heads_;

    // Split proj into gate, conv_input, dt using mx::split (1 op vs 3 slices)
    auto proj_parts = mx::split(proj, {gate_size, gate_size + conv_dim_}, -1);
    auto& gate = proj_parts[0];
    auto& conv_input = proj_parts[1];
    auto& dt = proj_parts[2];

    // Conv1d
    auto full_window = mx::concatenate({conv_state, conv_input}, 1);
    auto new_conv_state = mx::slice(full_window, {0, 1, 0}, {B_dim, conv_kernel_, conv_dim_});
    auto conv_out = mx::conv1d(full_window, *w.conv1d_w, 1, 0, 1, conv_dim_);
    if (w.conv1d_bias) conv_out = mx::add(conv_out, *w.conv1d_bias);
    conv_out = mx::multiply(conv_out, mx::sigmoid(conv_out));

    // Split conv output using mx::split (1 op vs 3 slices)
    int bc_size = n_groups_ * ssm_state_size_;
    auto conv_parts = mx::split(conv_out, {d_inner_, d_inner_ + bc_size}, -1);
    auto& hidden_states_raw = conv_parts[0];
    auto& B_ssm_raw = conv_parts[1];
    auto& C_ssm_raw = conv_parts[2];

    auto hidden_states = mx::reshape(hidden_states_raw, {B_dim, mamba_num_heads_, mamba_head_dim_});
    auto B_ssm = mx::reshape(B_ssm_raw, {B_dim, n_groups_, ssm_state_size_});
    auto C_ssm = mx::reshape(C_ssm_raw, {B_dim, n_groups_, ssm_state_size_});

    dt = compute_dt(dt, *w.dt_bias, time_step_min_, time_step_max_);
    dt = mx::reshape(dt, {B_dim, mamba_num_heads_});

    int G = mamba_num_heads_ / n_groups_;
    auto ssm_out_shape = mx::Shape{B_dim, mamba_num_heads_, mamba_head_dim_};
    auto ssm_state_shape = mx::Shape{B_dim, mamba_num_heads_, mamba_head_dim_, ssm_state_size_};

    auto kernel_results = ssm_kernel_(
        {hidden_states, *w.A_log, B_ssm, C_ssm, *w.D_param, dt, ssm_state},
        {ssm_out_shape, ssm_state_shape},
        {config_.activation_dtype, config_.activation_dtype},
        std::make_tuple(32, mamba_head_dim_, B_dim * mamba_num_heads_),
        std::make_tuple(32, 8, 1),
        {{"T", config_.activation_dtype},
         {"Dh", mamba_head_dim_},
         {"Ds", ssm_state_size_},
         {"H", mamba_num_heads_},
         {"G", G}},
        std::nullopt, false, {});

    auto y = mx::reshape(kernel_results[0], {B_dim, 1, d_inner_});
    auto new_ssm_state = kernel_results[1];

    // MambaRMSNormGated
    auto gated = swiglu_nh(gate, y);
    int group_dim = d_inner_ / n_groups_;
    gated = mx::reshape(gated, {B_dim, 1, n_groups_, group_dim});
    gated = mx::fast::rms_norm(gated, std::nullopt, config_.rms_norm_eps);
    gated = mx::reshape(gated, {B_dim, 1, d_inner_});
    gated = mx::multiply(gated, *w.mixer_norm_w);

    auto output = linear_fast(gated, *w.out_proj_w, *w.out_proj_s, w.out_proj_b);
    return {output, new_conv_state, new_ssm_state};
}

mx::array NemotronHModel::embed(const mx::array& input_ids) {
    // Use nn.Embedding approach: take from quantized weights, then dequantize per-token
    if (has_weight("embeddings.scales")) {
        auto w = get_weight("embeddings.weight");
        auto scales = get_weight("embeddings.scales");
        std::optional<mx::array> biases = std::nullopt;
        if (has_weight("embeddings.biases"))
            biases = get_weight("embeddings.biases");
        // For single token, dequantize the single row
        auto quant_row = mx::take(w, input_ids, 0);
        auto scale_row = mx::take(scales, input_ids, 0);
        auto bias_row = biases ? std::optional<mx::array>(mx::take(*biases, input_ids, 0)) : std::nullopt;
        return mx::dequantize(quant_row, scale_row, bias_row,
                              config_.quant_group_size, config_.quant_bits);
    }
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

    // Split proj into gate, conv_input, dt (1 op vs 3 slices)
    auto proj_parts = mx::split(proj, {gate_size, gate_size + conv_dim_}, -1);
    auto gate = proj_parts[0];
    auto conv_input = proj_parts[1];
    auto dt = proj_parts[2];

    auto& conv_state = mamba_conv_states_[mamba_idx];

    if (L == 1) {
        // Single token decode: update sliding window
        // conv_state: [B, conv_kernel-1, conv_dim] stores the last k-1 conv inputs.
        // For the convolution we need the last k inputs = concat(state, new_input).
        auto conv_w = get_weight(prefix + ".conv1d.weight");
        // conv_w shape: [conv_dim, conv_kernel, 1] — ready for depthwise conv1d

        // Build full convolution window: [B, conv_kernel, conv_dim]
        auto full_window = mx::concatenate({conv_state, conv_input}, 1);

        // Update state: drop oldest, keep last conv_kernel-1 for next step
        conv_state = mx::slice(full_window, {0, 1, 0}, {B_dim, conv_kernel_, conv_dim_});

        // Depthwise conv1d
        auto conv_out = mx::conv1d(full_window, conv_w, /*stride=*/1, /*padding=*/0,
                                   /*dilation=*/1, /*groups=*/conv_dim_);
        // conv_out: [B, 1, conv_dim]

        // Add conv bias
        if (use_conv_bias_) {
            auto conv_b = get_weight(prefix + ".conv1d.bias");  // [conv_dim]
            conv_out = mx::add(conv_out, conv_b);
        }

        // SiLU activation
        conv_out = mx::multiply(conv_out, mx::sigmoid(conv_out));

        // Split conv output (1 op vs 3 slices)
        int bc_size = n_groups_ * ssm_state_size_;
        auto conv_parts = mx::split(conv_out, {d_inner_, d_inner_ + bc_size}, -1);

        auto hidden_states = mx::reshape(conv_parts[0], {B_dim, mamba_num_heads_, mamba_head_dim_});
        auto B_ssm = mx::reshape(conv_parts[1], {B_dim, n_groups_, ssm_state_size_});
        auto C_ssm = mx::reshape(conv_parts[2], {B_dim, n_groups_, ssm_state_size_});

        // Compute dt: softplus + clip
        auto dt_bias = get_weight(prefix + ".dt_bias");  // [n_heads]
        dt = compute_dt(dt, dt_bias, time_step_min_, time_step_max_);
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
            std::make_tuple(32, 8, 1),
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
        auto gated = swiglu_nh(gate, y);

        int group_dim = d_inner_ / n_groups_;
        gated = mx::reshape(gated, {B_dim, 1, n_groups_, group_dim});
        gated = mx::fast::rms_norm(gated, std::nullopt, config_.rms_norm_eps);
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
        dt = compute_dt(dt, dt_bias, time_step_min_, time_step_max_);

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
                std::make_tuple(32, 8, 1),
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
        auto gated = swiglu_nh(gate, y);

        int group_dim = d_inner_ / n_groups_;
        gated = mx::reshape(gated, {B_dim, L, n_groups_, group_dim});
        gated = mx::fast::rms_norm(gated, std::nullopt, config_.rms_norm_eps);
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

    cache_k = mx::slice_update(
        cache_k, k,
        {0, 0, cache_offset, 0},
        {B_dim, n_kv_heads, cache_offset + L, hd});
    cache_v = mx::slice_update(
        cache_v, v,
        {0, 0, cache_offset, 0},
        {B_dim, n_kv_heads, cache_offset + L, hd});

    int total_len = cache_offset + L;
    auto full_k = mx::slice(cache_k, {0, 0, 0, 0}, {B_dim, n_kv_heads, total_len, hd});
    auto full_v = mx::slice(cache_v, {0, 0, 0, 0}, {B_dim, n_kv_heads, total_len, hd});

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::string mask_mode = (L > 1) ? "causal" : "";

    auto attn_out = mx::fast::scaled_dot_product_attention(
        q, full_k, full_v, scale, mask_mode);

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
    auto activated = relu2(up);
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

    // 1. Router — compiled like mlx-lm's @mx.compile group_expert_select
    auto& w = *moe_layer_weights_[layer];
    auto logits = mx::matmul(x, mx::transpose(*w.gate_w, {1, 0}));

    auto topk_inds = mx::array(0);  // placeholder
    auto topk_scores = mx::array(0.0f);  // placeholder
    {
        // Routing using cached weight refs (no string lookups)
        auto orig_scores = mx::sigmoid(mx::astype(logits, mx::float32));
        auto selection_scores = orig_scores;
        if (w.gate_correction_bias.has_value()) {
            selection_scores = mx::add(selection_scores, *w.gate_correction_bias);
        }
        if (config_.n_group > 1) {
            selection_scores = mx::unflatten(selection_scores, -1, {config_.n_group, -1});
            auto group_scores = mx::sum(mx::topk(selection_scores, 2, -1), -1, true);
            int groups_to_zero = config_.n_group - config_.topk_group;
            if (groups_to_zero > 0) {
                auto group_idx = mx::argpartition(group_scores, groups_to_zero - 1, -2);
                group_idx = mx::slice(group_idx, {0, 0, 0, 0},
                    {B_dim, L, groups_to_zero, group_scores.shape(3)});
                selection_scores = mx::put_along_axis(selection_scores, group_idx,
                    mx::array(0.0f, selection_scores.dtype()), -2);
            }
            selection_scores = mx::flatten(selection_scores, -2, -1);
        }
        topk_inds = mx::argpartition(mx::negative(selection_scores), k - 1, -1);
        topk_inds = mx::slice(topk_inds, {0, 0, 0}, {B_dim, L, k});
        topk_scores = mx::take_along_axis(orig_scores, topk_inds, -1);
        if (config_.norm_topk_prob && k > 1) {
            auto denom = mx::add(mx::sum(topk_scores, -1, true), mx::array(1e-20f));
            topk_scores = mx::divide(topk_scores, denom);
        }
        topk_scores = mx::multiply(topk_scores, mx::array(config_.routed_scaling_factor));
    }

    // 3. SwitchMLP: fc1 -> relu^2 -> fc2 (via gather_qmm)
    auto x_exp = mx::expand_dims(x, {-2, -3});  // [B, L, 1, 1, hidden]

    // Use cached weight references (no hash lookup)
    auto& fc1_w = *w.fc1_w;
    auto& fc1_s = *w.fc1_s;
    auto& fc1_b = w.fc1_b;
    auto& fc2_w = *w.fc2_w;
    auto& fc2_s = *w.fc2_s;
    auto& fc2_b = w.fc2_b;

    bool do_sort = topk_inds.size() >= 64;
    auto h = [&]() {
        if (do_sort) {
            auto flat_x = mx::flatten(x_exp, 0, -3);
            auto flat_inds = mx::flatten(topk_inds);
            auto order = mx::argsort(flat_inds);
            auto inv_order = mx::argsort(order);
            auto routed_rows = mx::floor_divide(order, mx::array(k, order.dtype()));
            auto sorted_x = mx::take(flat_x, routed_rows, 0);
            auto sorted_inds = mx::take(flat_inds, order, 0);

            auto sorted_h = mx::gather_qmm(sorted_x, fc1_w, fc1_s, fc1_b,
                                           std::nullopt, sorted_inds, /*transpose=*/true,
                                           config_.quant_group_size, config_.quant_bits, "affine", true);
            sorted_h = relu2(sorted_h);
            sorted_h = mx::gather_qmm(sorted_h, fc2_w, fc2_s, fc2_b,
                                      std::nullopt, sorted_inds, /*transpose=*/true,
                                      config_.quant_group_size, config_.quant_bits, "affine", true);
            sorted_h = mx::take(sorted_h, inv_order, 0);
            sorted_h = mx::unflatten(sorted_h, 0, {B_dim, L, k});
            return mx::squeeze(sorted_h, -2);  // [B, L, k, hidden]
        }

        auto unsorted_h = mx::gather_qmm(x_exp, fc1_w, fc1_s, fc1_b,
                                         std::nullopt, topk_inds, /*transpose=*/true,
                                         config_.quant_group_size, config_.quant_bits, "affine", false);
        unsorted_h = relu2(unsorted_h);
        unsorted_h = mx::gather_qmm(unsorted_h, fc2_w, fc2_s, fc2_b,
                                    std::nullopt, topk_inds, /*transpose=*/true,
                                    config_.quant_group_size, config_.quant_bits, "affine", false);
        return mx::squeeze(unsorted_h, -2);  // [B, L, k, hidden]
    }();

    // 4. Weighted sum — cast back to model dtype (prevents float32 infection from sigmoid scores)
    auto y = mx::multiply(h, mx::expand_dims(topk_scores, -1));
    y = mx::astype(mx::sum(y, -2), x.dtype());  // [B, L, hidden]

    // 5. Shared expert (use cached weights)
    if (w.has_shared_expert) {
        auto shared = mx::quantized_matmul(x, *w.shared_up_w, *w.shared_up_s, w.shared_up_b,
            /*transpose=*/true, config_.quant_group_size, config_.quant_bits);
        shared = relu2(shared);
        shared = mx::quantized_matmul(shared, *w.shared_down_w, *w.shared_down_s, w.shared_down_b,
            /*transpose=*/true, config_.quant_group_size, config_.quant_bits);
        y = mx::add(y, shared);
    }

    return y;
}

// ---------------------------------------------------------------------------
// Functional Mamba block (stateless — for compiled forward)
// Returns: {output, new_conv_state, new_ssm_state}
// Only handles L==1 (single-token decode)
// ---------------------------------------------------------------------------

std::vector<mx::array> NemotronHModel::mamba_block_functional(
    const mx::array& x, int layer,
    const mx::array& conv_state, const mx::array& ssm_state) {

    int B_dim = x.shape(0);
    int L = 1;  // Always single-token decode

    std::string prefix = "layers." + std::to_string(layer) + ".mixer";

    // in_proj
    auto proj = linear(x, prefix + ".in_proj");

    int gate_size = d_inner_;
    int dt_size = mamba_num_heads_;

    auto gate = mx::slice(proj, {0, 0, 0}, {B_dim, L, gate_size});
    auto conv_input = mx::slice(proj, {0, 0, gate_size}, {B_dim, L, gate_size + conv_dim_});
    auto dt = mx::slice(proj, {0, 0, gate_size + conv_dim_}, {B_dim, L, gate_size + conv_dim_ + dt_size});

    // Conv1d with explicit state
    auto conv_w = get_weight(prefix + ".conv1d.weight");
    auto full_window = mx::concatenate({conv_state, conv_input}, 1);

    // New conv state: drop oldest, keep last conv_kernel-1
    auto new_conv_state = mx::slice(full_window, {0, 1, 0}, {B_dim, conv_kernel_, conv_dim_});

    // Depthwise conv1d
    auto conv_out = mx::conv1d(full_window, conv_w, /*stride=*/1, /*padding=*/0,
                               /*dilation=*/1, /*groups=*/conv_dim_);

    if (use_conv_bias_) {
        auto conv_b = get_weight(prefix + ".conv1d.bias");
        conv_out = mx::add(conv_out, conv_b);
    }

    // SiLU
    conv_out = mx::multiply(conv_out, mx::sigmoid(conv_out));

    // Split conv output
    auto hidden_states = mx::slice(conv_out, {0, 0, 0}, {B_dim, 1, d_inner_});
    int bc_size = n_groups_ * ssm_state_size_;
    auto B_ssm = mx::slice(conv_out, {0, 0, d_inner_}, {B_dim, 1, d_inner_ + bc_size});
    auto C_ssm = mx::slice(conv_out, {0, 0, d_inner_ + bc_size}, {B_dim, 1, d_inner_ + 2 * bc_size});

    // Reshape for SSM
    hidden_states = mx::reshape(hidden_states, {B_dim, mamba_num_heads_, mamba_head_dim_});
    B_ssm = mx::reshape(B_ssm, {B_dim, n_groups_, ssm_state_size_});
    C_ssm = mx::reshape(C_ssm, {B_dim, n_groups_, ssm_state_size_});

    // Compute dt
    auto dt_bias = get_weight(prefix + ".dt_bias");
    dt = compute_dt(dt, dt_bias, time_step_min_, time_step_max_);
    dt = mx::reshape(dt, {B_dim, mamba_num_heads_});

    // SSM kernel
    auto A_log = get_weight(prefix + ".A_log");
    auto D_param = get_weight(prefix + ".D");
    int G = mamba_num_heads_ / n_groups_;

    auto ssm_out_shape = mx::Shape{B_dim, mamba_num_heads_, mamba_head_dim_};
    auto ssm_state_shape = mx::Shape{B_dim, mamba_num_heads_, mamba_head_dim_, ssm_state_size_};

    auto kernel_results = ssm_kernel_(
        {hidden_states, A_log, B_ssm, C_ssm, D_param, dt, ssm_state},
        {ssm_out_shape, ssm_state_shape},
        {config_.activation_dtype, config_.activation_dtype},
        std::make_tuple(32, mamba_head_dim_, B_dim * mamba_num_heads_),
        std::make_tuple(32, 8, 1),
        {{"T", config_.activation_dtype},
         {"Dh", mamba_head_dim_},
         {"Ds", ssm_state_size_},
         {"H", mamba_num_heads_},
         {"G", G}},
        std::nullopt,
        false,
        {});

    auto y = kernel_results[0];
    auto new_ssm_state = kernel_results[1];

    // Reshape y: [B, n_heads, head_dim] -> [B, 1, d_inner]
    y = mx::reshape(y, {B_dim, 1, d_inner_});

    // MambaRMSNormGated
    auto gated = swiglu_nh(gate, y);

    int group_dim = d_inner_ / n_groups_;
    gated = mx::reshape(gated, {B_dim, 1, n_groups_, group_dim});
    gated = mx::fast::rms_norm(gated, std::nullopt, config_.rms_norm_eps);
    gated = mx::reshape(gated, {B_dim, 1, d_inner_});

    auto norm_w = get_weight(prefix + ".norm.weight");
    gated = mx::multiply(gated, norm_w);

    auto output = linear(gated, prefix + ".out_proj");
    return {output, new_conv_state, new_ssm_state};
}

// ---------------------------------------------------------------------------
// Functional Attention block (stateless — for compiled forward)
// Returns: {output, new_cache_k, new_cache_v}
// Only handles L==1 (single-token decode)
// ---------------------------------------------------------------------------

std::vector<mx::array> NemotronHModel::attention_block_functional(
    const mx::array& x, int layer,
    const mx::array& cache_k, const mx::array& cache_v, int cache_offset) {

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

    // NO RoPE for Nemotron-H

    auto new_cache_k = mx::slice_update(
        cache_k, k,
        {0, 0, cache_offset, 0},
        {B_dim, n_kv_heads, cache_offset + L, hd});
    auto new_cache_v = mx::slice_update(
        cache_v, v,
        {0, 0, cache_offset, 0},
        {B_dim, n_kv_heads, cache_offset + L, hd});

    int total_len = cache_offset + L;
    auto full_k = mx::slice(new_cache_k, {0, 0, 0, 0}, {B_dim, n_kv_heads, total_len, hd});
    auto full_v = mx::slice(new_cache_v, {0, 0, 0, 0}, {B_dim, n_kv_heads, total_len, hd});

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    auto attn_out = mx::fast::scaled_dot_product_attention(
        q, full_k, full_v, scale, "");

    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B_dim, L, n_heads * hd});

    auto output = linear(attn_out, prefix + ".o_proj");
    return {output, new_cache_k, new_cache_v};
}

// ---------------------------------------------------------------------------
// Compiled forward step initialization
// ---------------------------------------------------------------------------

void NemotronHModel::init_compiled_step() {
    compiled_step_initialized_ = true;
    // No-op: full-forward compilation not used due to KV cache offset issue.
    // Per-mamba-layer compilation is initialized in init_compiled_mamba_mixers().
}

// ---------------------------------------------------------------------------
// Compiled per-mamba-layer mixer functions
// Each compiled function: {x, conv_state, ssm_state} -> {output, new_conv, new_ssm}
// Shapes are constant for L=1, B=1 decode, so each traces once.
// ---------------------------------------------------------------------------

void NemotronHModel::init_compiled_mamba_mixers() {
    // Build weight cache first
    if (!weight_cache_built_) {
        build_weight_cache();
    }

    compiled_mamba_mixers_.clear();
    compiled_mamba_mixers_.reserve(config_.num_hidden_layers);

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        if (block_types_[i] != BlockType::Mamba) {
            compiled_mamba_mixers_.push_back(nullptr);
            continue;
        }

        auto& self = *this;
        int layer = i;

        std::function<std::vector<mx::array>(const std::vector<mx::array>&)> mixer_fn =
            [&self, layer](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                return self.mamba_block_functional_fast(inputs[0], layer, inputs[1], inputs[2]);
            };

        compiled_mamba_mixers_.push_back(mx::compile(mixer_fn, /*shapeless=*/false));
    }

    compiled_mamba_mixers_initialized_ = true;
    std::cout << "[NemotronH] Compiled " << num_mamba_layers_
              << " mamba mixer functions (with weight cache)" << std::endl;
}

// ---------------------------------------------------------------------------
// Compiled per-MoE-layer mixer functions
// Each compiled function: {x} -> {output}
// Captures all layer weights as constants for kernel fusion
// ---------------------------------------------------------------------------

void NemotronHModel::init_compiled_moe_mixers() {
    if (!weight_cache_built_) {
        build_weight_cache();
    }

    compiled_moe_mixers_.clear();
    compiled_moe_mixers_.resize(config_.num_hidden_layers);

    int k = config_.num_experts_per_tok;
    int gs = config_.quant_group_size;
    int bits = config_.quant_bits;
    float scale_factor = config_.routed_scaling_factor;
    bool do_norm = config_.norm_topk_prob;
    int n_group = config_.n_group;

    int num_moe = 0;
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        if (block_types_[i] != BlockType::MOE) {
            compiled_moe_mixers_[i] = nullptr;
            continue;
        }

        auto& self = *this;
        int layer = i;

        // Capture all weights by reference (they're stored in moe_layer_weights_)
        std::function<std::vector<mx::array>(const std::vector<mx::array>&)> moe_fn =
            [&self, layer, k, gs, bits, scale_factor, do_norm, n_group](
                const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                auto& x = inputs[0];  // [B, L, hidden]
                auto& w = *self.moe_layer_weights_[layer];
                int B = x.shape(0), L = x.shape(1);

                // Gate
                auto logits = mx::matmul(x, mx::transpose(*w.gate_w, {1, 0}));
                auto orig_scores = mx::sigmoid(mx::astype(logits, mx::float32));
                auto sel_scores = orig_scores;
                if (w.gate_correction_bias.has_value()) {
                    sel_scores = mx::add(sel_scores, *w.gate_correction_bias);
                }

                // Top-k selection (n_group=1 fast path)
                auto topk_inds = mx::argpartition(mx::negative(sel_scores), k - 1, -1);
                topk_inds = mx::slice(topk_inds, {0, 0, 0}, {B, L, k});
                auto topk_scores = mx::take_along_axis(orig_scores, topk_inds, -1);
                if (do_norm && k > 1) {
                    auto denom = mx::add(mx::sum(topk_scores, -1, true), mx::array(1e-20f));
                    topk_scores = mx::divide(topk_scores, denom);
                }
                topk_scores = mx::multiply(topk_scores, mx::array(scale_factor));

                // SwitchMLP
                auto x_exp = mx::expand_dims(x, {-2, -3});
                auto h = mx::gather_qmm(x_exp, *w.fc1_w, *w.fc1_s, w.fc1_b,
                    std::nullopt, topk_inds, true, gs, bits, "affine", false);
                h = relu2(h);
                h = mx::gather_qmm(h, *w.fc2_w, *w.fc2_s, w.fc2_b,
                    std::nullopt, topk_inds, true, gs, bits, "affine", false);
                h = mx::squeeze(h, -2);

                // Weighted sum — cast back to model dtype (critical: prevents float32 infection)
                auto y = mx::astype(
                    mx::sum(mx::multiply(h, mx::expand_dims(topk_scores, -1)), -2),
                    x.dtype());

                // Shared expert
                if (w.has_shared_expert) {
                    auto shared = mx::quantized_matmul(x, *w.shared_up_w, *w.shared_up_s, w.shared_up_b,
                        true, gs, bits);
                    shared = relu2(shared);
                    shared = mx::quantized_matmul(shared, *w.shared_down_w, *w.shared_down_s, w.shared_down_b,
                        true, gs, bits);
                    y = mx::add(y, shared);
                }
                return {y};
            };

        compiled_moe_mixers_[i] = mx::compile(moe_fn, /*shapeless=*/false);
        num_moe++;
    }

    compiled_moe_mixers_initialized_ = true;
    std::cout << "[NemotronH] Compiled " << num_moe << " MoE mixer functions" << std::endl;
}

// ---------------------------------------------------------------------------
// Compiled decode: functional forward with compiled mamba mixers
// ---------------------------------------------------------------------------

mx::array NemotronHModel::compiled_decode(
    const mx::array& input_ids,
    std::vector<mx::array>& cache_keys,
    std::vector<mx::array>& cache_values,
    int cache_offset) {

    if (!compiled_mamba_mixers_initialized_) {
        init_compiled_mamba_mixers();
    }

    if (!compiled_step_initialized_) {
        // Build the full-forward compiled function
        auto& self = *this;
        int n_heads = config_.num_attention_heads;
        int n_kv_heads = config_.num_key_value_heads;
        int hd = attn_head_dim_;
        int n_layers = config_.num_hidden_layers;

        std::function<std::vector<mx::array>(const std::vector<mx::array>&)> step_fn =
            [&self, n_heads, n_kv_heads, hd, n_layers](
                const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                // Input layout:
                // [0] = input_ids [B, 1]
                // [1] = offset_arr [1] (int32 scalar)
                // [2 .. 2+2*num_attn-1] = KV caches (k, v pairs) [B, n_kv, max_ctx, hd]
                // [2+2*num_attn .. +num_mamba-1] = conv states
                // [next .. +num_mamba-1] = ssm states

                int idx = 0;
                auto input_ids = inputs[idx++];
                auto offset_arr = inputs[idx++];  // [1] int32

                int kv_base = idx;
                int conv_base = kv_base + self.num_attn_layers_ * 2;
                int ssm_base = conv_base + self.num_mamba_layers_;

                auto h = self.embed(input_ids);
                int B_dim = h.shape(0);

                std::vector<mx::array> out_kv;       // updated kv caches
                std::vector<mx::array> out_conv;      // updated conv states
                std::vector<mx::array> out_ssm;       // updated ssm states

                int kv_idx = kv_base;
                int conv_idx = conv_base;
                int ssm_idx = ssm_base;

                // Pre-build index for put_along_axis: offset broadcast to [B, n_kv, 1, hd]
                auto kv_index = mx::broadcast_to(
                    mx::reshape(offset_arr, {1, 1, 1, 1}),
                    {B_dim, n_kv_heads, 1, hd});

                // Pre-build attention mask: valid positions <= offset have 0, rest -inf
                // mask input is at the end of the inputs vector
                auto attn_mask = inputs.back();  // [1, 1, 1, max_ctx]

                for (int i = 0; i < n_layers; i++) {
                    std::optional<mx::array> block_out;

                    switch (self.block_types_[i]) {
                        case BlockType::Mamba: {
                            auto& mw = *self.mamba_layer_weights_[i];
                            auto normed = self.rms_norm(h, *mw.norm_w);

                            auto conv_state = inputs[conv_idx];
                            auto ssm_state = inputs[ssm_idx];
                            auto results = self.mamba_block_functional_fast(
                                normed, i, conv_state, ssm_state);
                            block_out = results[0];
                            out_conv.push_back(results[1]);
                            out_ssm.push_back(results[2]);
                            conv_idx++;
                            ssm_idx++;
                            break;
                        }
                        case BlockType::Attention: {
                            auto& aw = *self.attn_layer_weights_[i];
                            auto normed = self.rms_norm(h, *aw.norm_w);

                            auto q = self.linear_fast(normed, *aw.q_w, *aw.q_s, aw.q_b);
                            auto k = self.linear_fast(normed, *aw.k_w, *aw.k_s, aw.k_b);
                            auto v = self.linear_fast(normed, *aw.v_w, *aw.v_s, aw.v_b);

                            q = mx::transpose(mx::reshape(q, {B_dim, 1, n_heads, hd}), {0, 2, 1, 3});
                            k = mx::transpose(mx::reshape(k, {B_dim, 1, n_kv_heads, hd}), {0, 2, 1, 3});
                            v = mx::transpose(mx::reshape(v, {B_dim, 1, n_kv_heads, hd}), {0, 2, 1, 3});

                            // KV cache update using put_along_axis (array-indexed, compilable)
                            auto ck = inputs[kv_idx];
                            auto cv = inputs[kv_idx + 1];
                            ck = mx::put_along_axis(ck, kv_index, k, 2);
                            cv = mx::put_along_axis(cv, kv_index, v, 2);

                            // SDPA with mask to ignore positions beyond offset
                            float scale = 1.0f / std::sqrt(static_cast<float>(hd));
                            auto attn_out = mx::fast::scaled_dot_product_attention(
                                q, ck, cv, scale, /*mask_mode=*/"", /*mask_arr=*/attn_mask);
                            attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}),
                                                   {B_dim, 1, n_heads * hd});
                            block_out = self.linear_fast(attn_out, *aw.o_w, *aw.o_s, aw.o_b);
                            out_kv.push_back(ck);
                            out_kv.push_back(cv);
                            kv_idx += 2;
                            break;
                        }
                        case BlockType::MLP: {
                            auto& mw = *self.mlp_layer_weights_[i];
                            auto normed = self.rms_norm(h, *mw.norm_w);
                            auto up = self.linear_fast(normed, *mw.up_w, *mw.up_s, mw.up_b);
                            block_out = self.linear_fast(
                                relu2(up), *mw.down_w, *mw.down_s, mw.down_b);
                            break;
                        }
                        case BlockType::MOE: {
                            auto& mw = *self.moe_layer_weights_[i];
                            auto normed = self.rms_norm(h, *mw.norm_w);
                            // Use compiled MoE mixer for kernel fusion
                            if (self.compiled_moe_mixers_initialized_ && self.compiled_moe_mixers_[i]) {
                                block_out = self.compiled_moe_mixers_[i]({normed})[0];
                            } else {
                                block_out = self.moe_block(normed, i);
                            }
                            break;
                        }
                    }

                    h = mx::add(h, *block_out);
                }

                auto norm_f_w = self.get_weight("norm_f.weight");
                h = self.rms_norm(h, norm_f_w);
                auto logits = self.lm_head_proj(h);

                // Pack: [logits, kv_0_k, kv_0_v, ..., conv_0, ..., ssm_0, ...]
                std::vector<mx::array> outputs;
                outputs.push_back(logits);
                for (auto& kv : out_kv) outputs.push_back(kv);
                for (auto& c : out_conv) outputs.push_back(c);
                for (auto& s : out_ssm) outputs.push_back(s);
                return outputs;
            };

        compiled_step_ = mx::compile(step_fn, /*shapeless=*/false);
        compiled_step_initialized_ = true;
        std::cout << "[NemotronH] Full forward compiled with array-indexed KV cache" << std::endl;
    }

    // Pack all state into input vector
    std::vector<mx::array> inputs;
    inputs.push_back(input_ids);
    inputs.push_back(mx::array({cache_offset}, mx::int32));
    for (size_t i = 0; i < cache_keys.size(); i++) {
        inputs.push_back(cache_keys[i]);
        inputs.push_back(cache_values[i]);
    }
    for (auto& c : mamba_conv_states_) inputs.push_back(c);
    for (auto& s : mamba_ssm_states_) inputs.push_back(s);

    // Build attention mask: positions 0..cache_offset are valid (0.0), rest are -inf
    int max_ctx = cache_keys[0].shape(2);
    auto positions = mx::arange(0, max_ctx, mx::int32);
    auto offset_arr = mx::array({cache_offset}, mx::int32);
    auto mask = mx::where(
        mx::less_equal(positions, offset_arr),
        mx::array(0.0f, mx::float32),
        mx::array(-1e9f, mx::float32));
    inputs.push_back(mx::reshape(mask, {1, 1, 1, max_ctx}));

    // Run compiled forward
    auto results = compiled_step_(inputs);

    // Unpack outputs
    auto logits = results[0];
    int ridx = 1;
    for (size_t i = 0; i < cache_keys.size(); i++) {
        cache_keys[i] = results[ridx++];
        cache_values[i] = results[ridx++];
    }
    for (size_t i = 0; i < mamba_conv_states_.size(); i++) {
        mamba_conv_states_[i] = results[ridx++];
    }
    for (size_t i = 0; i < mamba_ssm_states_.size(); i++) {
        mamba_ssm_states_[i] = results[ridx++];
    }

    return logits;
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
    int L = input_ids.shape(1);

    // Initialize mamba states if needed
    if (!mamba_states_initialized_) {
        init_mamba_states(B_dim);
    }

    // Ensure weight cache and compiled mixers are built
    if (!weight_cache_built_) {
        build_weight_cache();
    }
    if (!compiled_moe_mixers_initialized_) {
        init_compiled_moe_mixers();
    }

    // Direct forward with per-layer compiled MoE mixers (best for Nemotron-30B)
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
                // Use compiled MoE mixer for kernel fusion (elementwise ops fused)
                if (compiled_moe_mixers_initialized_ && compiled_moe_mixers_[i]) {
                    block_out = compiled_moe_mixers_[i]({normed})[0];
                } else {
                    block_out = moe_block(normed, i);
                }
                break;
            }
        }

        h = mx::add(h, block_out);
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
    if (!weight_cache_built_) {
        build_weight_cache();
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
// Benchmark helpers
// ---------------------------------------------------------------------------

mx::array NemotronHModel::embed_for_benchmark(const mx::array& input_ids) {
    if (!mamba_states_initialized_) init_mamba_states(1);
    if (!weight_cache_built_) build_weight_cache();
    return embed(input_ids);
}

mx::array NemotronHModel::forward_one_block(mx::array& h, int i,
    std::vector<mx::array>& cache_k, std::vector<mx::array>& cache_v, int offset) {
    auto norm_w = get_weight("layers." + std::to_string(i) + ".norm.weight");
    auto normed = rms_norm(h, norm_w);
    mx::array block_out = normed;
    switch (block_types_[i]) {
        case BlockType::Mamba: {
            int idx = mamba_layer_indices_[i];
            block_out = mamba_block(normed, i, idx);
            break;
        }
        case BlockType::Attention: {
            int idx = attn_layer_indices_[i];
            block_out = attention_block(normed, i, idx, cache_k[idx], cache_v[idx], offset);
            break;
        }
        case BlockType::MLP: block_out = mlp_block(normed, i); break;
        case BlockType::MOE: block_out = moe_block(normed, i); break;
    }
    h = mx::add(h, block_out);
    return h;
}

int NemotronHModel::block_type_int(int i) const {
    switch (block_types_[i]) {
        case BlockType::Mamba: return 0;
        case BlockType::Attention: return 1;
        case BlockType::MLP: return 2;
        case BlockType::MOE: return 3;
    }
    return -1;
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
