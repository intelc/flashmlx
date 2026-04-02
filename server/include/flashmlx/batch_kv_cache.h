#pragma once
#include <vector>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace flashmlx {

/// Batched KV cache with left-padding for heterogeneous sequence lengths.
///
/// Maintains a single [B, n_kv_heads, buf_len, head_dim] buffer per layer,
/// with all sequences right-justified so they share a common write position.
class BatchKVCache {
public:
    BatchKVCache() = default;

    /// Build from per-slot caches after prefill.
    /// slot_keys[b][l] has shape [1, n_kv, seq_len_b, hd].
    void build(const std::vector<std::vector<mx::array>>& slot_keys,
               const std::vector<std::vector<mx::array>>& slot_values,
               const std::vector<int>& seq_lengths,
               int num_layers, int n_kv_heads, int head_dim, mx::Dtype dtype);

    /// Write new K/V at write_pos for all B sequences.
    /// k, v shape: [B, n_kv, 1, hd]
    void update(const mx::array& k, const mx::array& v, int layer);

    /// Get cache sliced to [B, n_kv, write_pos, hd]
    mx::array get_keys(int layer) const;
    mx::array get_values(int layer) const;

    /// Get mask [B, 1, 1, write_pos] -- 0.0 for valid, -1e9 for padding
    mx::array get_mask() const;

    /// Get RoPE offsets [B] = write_pos - left_padding
    mx::array get_rope_offsets() const;

    /// Replace cache contents directly with concatenated full caches from forward_batched.
    /// new_keys[l] shape: [B, n_kv, write_pos+1, hd] (already includes new token via concat)
    void replace_caches(const std::vector<mx::array>& new_keys,
                        const std::vector<mx::array>& new_values);

    /// Advance write position by 1
    void advance();

    /// Remove completed sequences, keeping only those at keep_indices
    void filter(const std::vector<int>& keep_indices);

    /// Flush back to per-slot format.
    /// slot_keys[b][l] = [1, n_kv, actual_len_b, hd]
    void flush_to_slots(std::vector<std::vector<mx::array>>& slot_keys,
                        std::vector<std::vector<mx::array>>& slot_values) const;

    int batch_size() const { return batch_size_; }
    int write_pos() const { return write_pos_; }
    bool valid() const { return valid_; }
    void invalidate() { valid_ = false; }

private:
    void grow_buffers();

    int batch_size_ = 0;
    int num_layers_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    mx::Dtype dtype_ = mx::float16;

    int write_pos_ = 0;   // shared position where all sequences write next
    int buf_len_ = 0;     // allocated buffer length (grows in chunks of 256)

    std::vector<int> left_padding_;  // per-sequence padding count

    // keys_[layer] = [B, n_kv, buf_len, hd]
    std::vector<std::optional<mx::array>> keys_;
    std::vector<std::optional<mx::array>> values_;

    // mask_ = [B, 1, 1, buf_len] -- 0.0 valid, -1e9 padding
    std::optional<mx::array> mask_;

    bool valid_ = false;
};

} // namespace flashmlx
