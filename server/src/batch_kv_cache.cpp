#include "flashmlx/batch_kv_cache.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace flashmlx {

static constexpr int kGrowChunk = 256;

void BatchKVCache::build(
    const std::vector<std::vector<mx::array>>& slot_keys,
    const std::vector<std::vector<mx::array>>& slot_values,
    const std::vector<int>& seq_lengths,
    int num_layers, int n_kv_heads, int head_dim, mx::Dtype dtype) {

    batch_size_ = static_cast<int>(seq_lengths.size());
    num_layers_ = num_layers;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    dtype_ = dtype;

    if (batch_size_ == 0) {
        valid_ = false;
        return;
    }

    // Find max sequence length
    int max_len = *std::max_element(seq_lengths.begin(), seq_lengths.end());
    write_pos_ = max_len;

    // Pre-allocate with growth room, rounded up to kGrowChunk
    buf_len_ = ((max_len + kGrowChunk - 1) / kGrowChunk) * kGrowChunk + kGrowChunk;

    // Compute left padding per sequence
    left_padding_.resize(batch_size_);
    for (int b = 0; b < batch_size_; ++b) {
        left_padding_[b] = max_len - seq_lengths[b];
    }

    // Allocate buffers per layer: [B, n_kv, buf_len, hd]
    keys_.clear();
    keys_.reserve(num_layers_);
    values_.clear();
    values_.reserve(num_layers_);

    std::vector<mx::array> to_eval;

    for (int l = 0; l < num_layers_; ++l) {
        auto k_buf = mx::zeros({batch_size_, n_kv_heads_, buf_len_, head_dim_}, dtype_);
        auto v_buf = mx::zeros({batch_size_, n_kv_heads_, buf_len_, head_dim_}, dtype_);

        // Copy each sequence's KV data right-justified into the buffer
        for (int b = 0; b < batch_size_; ++b) {
            int slen = seq_lengths[b];
            if (slen == 0) continue;

            int start = left_padding_[b];
            // slot_keys[b][l] is [1, n_kv, slen, hd]
            // Write into buffer at [b:b+1, 0:n_kv, start:start+slen, 0:hd]
            k_buf = mx::slice_update(
                k_buf, slot_keys[b][l],
                {b, 0, start, 0},
                {b + 1, n_kv_heads_, start + slen, head_dim_});
            v_buf = mx::slice_update(
                v_buf, slot_values[b][l],
                {b, 0, start, 0},
                {b + 1, n_kv_heads_, start + slen, head_dim_});
        }

        keys_.emplace_back(std::move(k_buf));
        values_.emplace_back(std::move(v_buf));
        to_eval.push_back(*keys_[l]);
        to_eval.push_back(*values_[l]);
    }

    // Build mask: [B, 1, 1, buf_len]
    // Position p is valid (0.0) iff p >= left_padding_[b], else -1e9
    std::vector<float> mask_data(batch_size_ * buf_len_);
    for (int b = 0; b < batch_size_; ++b) {
        for (int p = 0; p < buf_len_; ++p) {
            mask_data[b * buf_len_ + p] = (p >= left_padding_[b]) ? 0.0f : -1e9f;
        }
    }
    mask_ = mx::reshape(
        mx::astype(mx::array(mask_data.data(), {batch_size_, buf_len_}), dtype_),
        {batch_size_, 1, 1, buf_len_});
    to_eval.push_back(*mask_);

    mx::eval(to_eval);
    // Check if all sequences have the same left padding
    uniform_padding_ = true;
    for (int b = 1; b < batch_size_; b++) {
        if (left_padding_[b] != left_padding_[0]) {
            uniform_padding_ = false;
            break;
        }
    }

    valid_ = true;
}

void BatchKVCache::update(const mx::array& k, const mx::array& v, int layer) {
    if (!valid_) throw std::runtime_error("BatchKVCache::update: cache not valid");
    if (layer < 0 || layer >= num_layers_) throw std::out_of_range("BatchKVCache::update: bad layer");

    // k, v: [B, n_kv, 1, hd]
    // slice_update with std::move for buffer donation
    auto old_k = std::move(*keys_[layer]);
    keys_[layer] = mx::slice_update(old_k, k,
        {0, 0, write_pos_, 0},
        {batch_size_, n_kv_heads_, write_pos_ + 1, head_dim_});
    auto old_v = std::move(*values_[layer]);
    values_[layer] = mx::slice_update(old_v, v,
        {0, 0, write_pos_, 0},
        {batch_size_, n_kv_heads_, write_pos_ + 1, head_dim_});

    // Periodic eval to flatten graph depth (fewer syncs = better throughput)
    if (layer == num_layers_ - 1 && write_pos_ > 0 && write_pos_ % 32 == 0) {
        std::vector<mx::array> to_eval;
        to_eval.reserve(num_layers_ * 2);
        for (int l = 0; l < num_layers_; l++) {
            to_eval.push_back(*keys_[l]);
            to_eval.push_back(*values_[l]);
        }
        mx::eval(to_eval);
    }
}

mx::array BatchKVCache::get_keys(int layer) const {
    if (!valid_) throw std::runtime_error("BatchKVCache::get_keys: cache not valid");
    return mx::slice(*keys_[layer], {0, 0, 0, 0},
                     {batch_size_, n_kv_heads_, write_pos_, head_dim_});
}

mx::array BatchKVCache::get_values(int layer) const {
    if (!valid_) throw std::runtime_error("BatchKVCache::get_values: cache not valid");
    return mx::slice(*values_[layer], {0, 0, 0, 0},
                     {batch_size_, n_kv_heads_, write_pos_, head_dim_});
}

mx::array BatchKVCache::get_mask() const {
    if (!valid_) throw std::runtime_error("BatchKVCache::get_mask: cache not valid");
    return mx::slice(*mask_, {0, 0, 0, 0},
                     {batch_size_, 1, 1, write_pos_});
}

mx::array BatchKVCache::get_rope_offsets() const {
    if (!valid_) throw std::runtime_error("BatchKVCache::get_rope_offsets: cache not valid");
    // RoPE offset for sequence b = write_pos_ - left_padding_[b] = actual token count
    std::vector<int32_t> offsets(batch_size_);
    for (int b = 0; b < batch_size_; ++b) {
        offsets[b] = write_pos_ - left_padding_[b];
    }
    return mx::array(offsets.data(), {batch_size_}, mx::int32);
}

void BatchKVCache::advance() {
    if (!valid_) throw std::runtime_error("BatchKVCache::advance: cache not valid");
    write_pos_++;
    if (write_pos_ >= buf_len_) {
        grow_buffers();
    }
}

void BatchKVCache::grow_buffers() {
    int new_buf_len = buf_len_ + kGrowChunk;

    std::vector<mx::array> to_eval;

    // Grow each layer's K/V by concatenating zeros along the seq dimension (axis 2)
    auto pad = mx::zeros({batch_size_, n_kv_heads_, kGrowChunk, head_dim_}, dtype_);
    for (int l = 0; l < num_layers_; ++l) {
        keys_[l] = mx::concatenate({*keys_[l], pad}, 2);
        values_[l] = mx::concatenate({*values_[l], pad}, 2);
        to_eval.push_back(*keys_[l]);
        to_eval.push_back(*values_[l]);
    }

    // Grow mask: new positions are beyond all sequences, so they're "valid" (0.0)
    // because future tokens will be written there
    auto mask_pad = mx::zeros({batch_size_, 1, 1, kGrowChunk}, dtype_);
    mask_ = mx::concatenate({*mask_, mask_pad}, 3);
    to_eval.push_back(*mask_);

    mx::eval(to_eval);
    buf_len_ = new_buf_len;
}

void BatchKVCache::filter(const std::vector<int>& keep_indices) {
    if (!valid_) throw std::runtime_error("BatchKVCache::filter: cache not valid");

    int new_batch = static_cast<int>(keep_indices.size());
    if (new_batch == 0) {
        valid_ = false;
        return;
    }

    // Build index array for mx::take
    auto idx = mx::array(keep_indices.data(), {new_batch}, mx::int32);

    std::vector<mx::array> to_eval;
    for (int l = 0; l < num_layers_; ++l) {
        keys_[l] = mx::take(*keys_[l], idx, 0);
        values_[l] = mx::take(*values_[l], idx, 0);
        to_eval.push_back(*keys_[l]);
        to_eval.push_back(*values_[l]);
    }

    mask_ = mx::take(*mask_, idx, 0);
    to_eval.push_back(*mask_);

    // Update left_padding
    std::vector<int> new_padding(new_batch);
    for (int i = 0; i < new_batch; ++i) {
        new_padding[i] = left_padding_[keep_indices[i]];
    }
    left_padding_ = std::move(new_padding);
    batch_size_ = new_batch;

    mx::eval(to_eval);
}

void BatchKVCache::flush_to_slots(
    std::vector<std::vector<mx::array>>& slot_keys,
    std::vector<std::vector<mx::array>>& slot_values) const {

    if (!valid_) throw std::runtime_error("BatchKVCache::flush_to_slots: cache not valid");

    slot_keys.clear();
    slot_values.clear();

    for (int b = 0; b < batch_size_; ++b) {
        slot_keys.emplace_back();
        slot_values.emplace_back();
        slot_keys[b].reserve(num_layers_);
        slot_values[b].reserve(num_layers_);
        int start = left_padding_[b];

        for (int l = 0; l < num_layers_; ++l) {
            // Extract [b:b+1, :, start:write_pos_, :] -> [1, n_kv, actual_len, hd]
            slot_keys[b].push_back(mx::slice(*keys_[l],
                {b, 0, start, 0},
                {b + 1, n_kv_heads_, write_pos_, head_dim_}));
            slot_values[b].push_back(mx::slice(*values_[l],
                {b, 0, start, 0},
                {b + 1, n_kv_heads_, write_pos_, head_dim_}));
        }
    }
}

} // namespace flashmlx
