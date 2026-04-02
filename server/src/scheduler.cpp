#include "flashmlx/scheduler.h"

#include <iostream>
#include <chrono>

namespace flashmlx {

BatchScheduler::BatchScheduler(ModelBase& model, KVCachePool& pool)
    : model_(model), pool_(pool) {}

void BatchScheduler::submit(Request req) {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    req.state = RequestState::QUEUED;
    pending_.push_back(std::move(req));
}

std::unordered_map<std::string, std::vector<int>> BatchScheduler::step() {
    std::unordered_map<std::string, std::vector<int>> new_tokens;

    // 1. Admit pending requests while KV slots are available
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        while (!pending_.empty() && pool_.num_free() > 0) {
            Request req = std::move(pending_.front());
            pending_.pop_front();
            std::string id = req.id;
            prefill_request(req);
            if (req.state == RequestState::DECODING) {
                // Record the first token generated during prefill
                new_tokens[id].push_back(req.output_tokens.back());
            }
            active_[id] = std::move(req);
        }
    }

    // 2. Decode: batch all decoding requests together
    std::vector<std::string> done_ids;

    std::vector<std::string> decoding_ids;
    for (auto& [id, req] : active_) {
        if (req.state == RequestState::DECODING) {
            decoding_ids.push_back(id);
        }
    }

    if (decoding_ids.size() == 1) {
        // Single request — N-step graph batching (fastest for C=1)
        auto& req = active_[decoding_ids[0]];
        int prev_count = req.generated_count;
        decode_request(req);
        for (int i = prev_count; i < req.generated_count; i++) {
            new_tokens[decoding_ids[0]].push_back(req.output_tokens[i]);
        }
        if (req.generated_count >= req.max_tokens) {
            req.state = RequestState::DONE;
            done_ids.push_back(decoding_ids[0]);
        }
    } else if (decoding_ids.size() > 1) {
        // Multiple requests — heterogeneous batched decode
        decode_batch_heterogeneous(decoding_ids, new_tokens, done_ids);
    }

    // 3. Cleanup completed requests
    {
        std::lock_guard<std::mutex> lock(completed_mutex_);
        for (const auto& id : done_ids) {
            auto it = active_.find(id);
            if (it != active_.end()) {
                pool_.free(it->second.kv_slot);
                completed_.push_back(id);
                active_.erase(it);
            }
        }
    }

    return new_tokens;
}

bool BatchScheduler::has_work() {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return !pending_.empty() || !active_.empty();
}

std::vector<std::string> BatchScheduler::drain_completed() {
    std::lock_guard<std::mutex> lock(completed_mutex_);
    std::vector<std::string> result;
    result.swap(completed_);
    return result;
}

int BatchScheduler::active_count() const {
    return static_cast<int>(active_.size());
}

void BatchScheduler::prefill_request(Request& req) {
    // Allocate a KV cache slot
    int slot = pool_.allocate();
    if (slot < 0) {
        // Should not happen — we checked num_free() > 0 before calling
        req.state = RequestState::QUEUED;
        return;
    }
    req.kv_slot = slot;
    req.state = RequestState::PREFILLING;

    int num_layers = pool_.num_layers();
    int prompt_len = static_cast<int>(req.prompt_tokens.size());
    int n_kv = model_.config().num_key_value_heads;
    int hd = model_.config().head_dim > 0 ? model_.config().head_dim
             : model_.config().hidden_size / model_.config().num_attention_heads;
    int max_ctx = pool_.max_context_len();

    // Build input_ids [1, prompt_len]
    auto input_ids = mx::array(req.prompt_tokens.data(),
                               {1, prompt_len}, mx::int32);

    // Create fresh full-size caches for prefill
    std::vector<mx::array> cache_keys, cache_values;
    cache_keys.reserve(num_layers);
    cache_values.reserve(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        cache_keys.push_back(mx::zeros({1, n_kv, max_ctx, hd}, model_.config().activation_dtype));
        cache_values.push_back(mx::zeros({1, n_kv, max_ctx, hd}, model_.config().activation_dtype));
    }

    // Cache offsets: all zero for fresh prefill
    auto cache_offsets = mx::array({0}, mx::int32);

    // Forward pass (array-offset path uses slice_update internally — caches stay at max_ctx)
    mx::array logits = model_.forward(input_ids, cache_keys, cache_values, cache_offsets);

    // Write prefilled caches back to pool (already max_ctx sized)
    for (int l = 0; l < num_layers; ++l) {
        pool_.keys(slot, l) = cache_keys[l];
        pool_.values(slot, l) = cache_values[l];
    }

    // Sample first token from last position logits
    // logits shape: [1, prompt_len, vocab_size] or [1, 1, vocab_size]
    // Take the last token's logits
    int logits_seq_len = logits.shape(1);
    mx::array last_logits = mx::slice(logits, {0, logits_seq_len - 1, 0},
                                       {1, logits_seq_len, logits.shape(2)});
    last_logits = mx::reshape(last_logits, {1, -1});

    mx::array token = sample_token(last_logits, req.temperature);
    mx::eval({token});
    int tok_id = token.item<int>();

    req.next_token = token;
    req.output_tokens.push_back(tok_id);
    req.generated_count = 1;
    req.cache_offset = prompt_len;
    req.state = RequestState::DECODING;
}

void BatchScheduler::decode_request(Request& req) {
    int slot = req.kv_slot;
    int num_layers = pool_.num_layers();
    bool is_moe = model_.config().n_routed_experts > 0;

    int n_kv = model_.config().num_key_value_heads;
    int hd_val = model_.config().head_dim > 0 ? model_.config().head_dim
             : model_.config().hidden_size / model_.config().num_attention_heads;
    int max_ctx = pool_.max_context_len();

    if (is_moe) {
        // MoE/hybrid: 1-step with async pipelining (matches mlx-lm pattern)
        // Build next step while GPU evaluates current step
        auto build_step = [&](mx::array& prev, int offset) -> mx::array {
            auto input_ids = mx::reshape(prev, {1, 1});

            // Slice pool caches to valid portion for concat-based forward
            std::vector<mx::array> ck, cv;
            ck.reserve(num_layers); cv.reserve(num_layers);
            for (int l = 0; l < num_layers; ++l) {
                ck.push_back(mx::slice(pool_.keys(slot, l),
                    {0, 0, 0, 0}, {1, n_kv, offset, hd_val}));
                cv.push_back(mx::slice(pool_.values(slot, l),
                    {0, 0, 0, 0}, {1, n_kv, offset, hd_val}));
            }
            mx::array logits = model_.forward(input_ids, ck, cv, offset);

            // Write grown caches back into pool's pre-allocated buffers
            int new_len = offset + 1;
            for (int l = 0; l < num_layers; ++l) {
                auto full_buf = mx::zeros({1, n_kv, max_ctx, hd_val}, model_.config().activation_dtype);
                pool_.keys(slot, l) = mx::slice_update(full_buf, ck[l],
                    {0, 0, 0, 0}, {1, n_kv, new_len, hd_val});
                full_buf = mx::zeros({1, n_kv, max_ctx, hd_val}, model_.config().activation_dtype);
                pool_.values(slot, l) = mx::slice_update(full_buf, cv[l],
                    {0, 0, 0, 0}, {1, n_kv, new_len, hd_val});
            }
            mx::array last_logits = mx::reshape(logits, {1, -1});
            return sample_token(last_logits, req.temperature);
        };

        int remaining = req.max_tokens - req.generated_count;
        mx::array cur_token = req.next_token;

        // Build first step
        auto next_token = build_step(cur_token, req.cache_offset);
        mx::async_eval({next_token});

        for (int s = 1; s < remaining; s++) {
            // Build step s while GPU evaluates step s-1
            auto next_next = build_step(next_token, req.cache_offset + s);
            mx::async_eval({next_next});

            // Extract previous token (should be ready by now)
            int tok_id = next_token.item<int>();
            req.output_tokens.push_back(tok_id);
            req.generated_count++;
            req.cache_offset++;

            if (req.generated_count >= req.max_tokens) {
                req.next_token = next_next;
                return;
            }
            next_token = next_next;
        }
        // Extract final token
        int tok_id = next_token.item<int>();
        req.output_tokens.push_back(tok_id);
        req.generated_count++;
        req.cache_offset++;
        req.next_token = next_token;
    } else {
        // Dense models: N-step graph batching
        int batch_n = 32;
        int N = std::min(batch_n, req.max_tokens - req.generated_count);
        int start_offset = req.cache_offset;

        // Slice pool caches to valid portion for concat-based decode
        std::vector<mx::array> cache_keys_local, cache_values_local;
        cache_keys_local.reserve(num_layers);
        cache_values_local.reserve(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            cache_keys_local.push_back(mx::slice(pool_.keys(slot, l),
                {0, 0, 0, 0}, {1, n_kv, start_offset, hd_val}));
            cache_values_local.push_back(mx::slice(pool_.values(slot, l),
                {0, 0, 0, 0}, {1, n_kv, start_offset, hd_val}));
        }

        std::vector<mx::array> step_tokens;
        mx::array prev_token = req.next_token;

        for (int s = 0; s < N; s++) {
            auto input_ids = mx::reshape(prev_token, {1, 1});
            mx::array logits = model_.forward(input_ids, cache_keys_local, cache_values_local,
                                              start_offset + s);
            mx::array last_logits = mx::reshape(logits, {1, -1});
            mx::array token = sample_token(last_logits, req.temperature);
            step_tokens.push_back(token);
            prev_token = token;
        }

        mx::async_eval(step_tokens);
        for (auto& tok : step_tokens) {
            int tok_id = tok.item<int>();
            req.output_tokens.push_back(tok_id);
            req.generated_count++;
            req.cache_offset++;
        }
        req.next_token = step_tokens.back();

        // Write grown caches back into pool's pre-allocated buffers
        int new_len = req.cache_offset;  // offset advanced by N
        for (int l = 0; l < num_layers; ++l) {
            auto full_buf = mx::zeros({1, n_kv, max_ctx, hd_val}, model_.config().activation_dtype);
            pool_.keys(slot, l) = mx::slice_update(full_buf, cache_keys_local[l],
                {0, 0, 0, 0}, {1, n_kv, new_len, hd_val});
            full_buf = mx::zeros({1, n_kv, max_ctx, hd_val}, model_.config().activation_dtype);
            pool_.values(slot, l) = mx::slice_update(full_buf, cache_values_local[l],
                {0, 0, 0, 0}, {1, n_kv, new_len, hd_val});
        }
    }
}

void BatchScheduler::decode_batch(
    const std::vector<std::string>& ids,
    std::unordered_map<std::string, std::vector<int>>& new_tokens,
    std::vector<std::string>& done_ids) {

    int B = static_cast<int>(ids.size());
    int num_layers = pool_.num_layers();

    // 1. Build batched input_ids [B, 1]
    std::vector<int> tok_ids;
    tok_ids.reserve(B);
    for (auto& id : ids) {
        tok_ids.push_back(active_[id].next_token.item<int>());
    }
    auto input_ids = mx::array(tok_ids.data(), {B, 1}, mx::int32);

    // 2. Build batched KV caches by concatenating slots along batch dim
    //    Each slot is [1, n_kv_heads, max_ctx, hd]
    //    Concatenated: [B, n_kv_heads, max_ctx, hd]
    std::vector<mx::array> batch_cache_k, batch_cache_v;
    batch_cache_k.reserve(num_layers);
    batch_cache_v.reserve(num_layers);

    for (int l = 0; l < num_layers; l++) {
        std::vector<mx::array> k_parts, v_parts;
        for (auto& id : ids) {
            int slot = active_[id].kv_slot;
            k_parts.push_back(pool_.keys(slot, l));
            v_parts.push_back(pool_.values(slot, l));
        }
        batch_cache_k.push_back(mx::concatenate(k_parts, 0));
        batch_cache_v.push_back(mx::concatenate(v_parts, 0));
    }

    // 3. All requests in the group share the same cache offset by construction.
    std::vector<int> offsets;
    for (auto& id : ids) {
        offsets.push_back(active_[id].cache_offset);
    }

    // 4. Single batched forward pass through the homogeneous decode path.
    mx::array logits = model_.forward(input_ids, batch_cache_k, batch_cache_v, offsets[0]);

    // 5. Split KV caches back to individual slots
    for (int l = 0; l < num_layers; l++) {
        for (int b = 0; b < B; b++) {
            int slot = active_[ids[b]].kv_slot;
            pool_.keys(slot, l) = mx::slice(batch_cache_k[l], {b, 0, 0, 0},
                {b + 1, batch_cache_k[l].shape(1), batch_cache_k[l].shape(2), batch_cache_k[l].shape(3)});
            pool_.values(slot, l) = mx::slice(batch_cache_v[l], {b, 0, 0, 0},
                {b + 1, batch_cache_v[l].shape(1), batch_cache_v[l].shape(2), batch_cache_v[l].shape(3)});
        }
    }

    // 6. Sample tokens per-request
    //    logits shape: [B, 1, vocab_size]
    for (int b = 0; b < B; b++) {
        auto& req = active_[ids[b]];
        auto req_logits = mx::slice(logits, {b, 0, 0}, {b + 1, 1, logits.shape(2)});
        req_logits = mx::reshape(req_logits, {1, -1});

        mx::array token = sample_token(req_logits, req.temperature);
        mx::eval({token});
        int tok_id = token.item<int>();

        req.next_token = token;
        req.output_tokens.push_back(tok_id);
        req.generated_count++;
        req.cache_offset++;

        new_tokens[ids[b]].push_back(tok_id);

        if (req.generated_count >= req.max_tokens) {
            req.state = RequestState::DONE;
            done_ids.push_back(ids[b]);
        }
    }
}

void BatchScheduler::decode_batch_heterogeneous(
    const std::vector<std::string>& ids,
    std::unordered_map<std::string, std::vector<int>>& new_tokens,
    std::vector<std::string>& done_ids) {

    int B = static_cast<int>(ids.size());
    int num_layers = pool_.num_layers();
    int n_kv = model_.config().num_key_value_heads;
    int hd = model_.config().head_dim > 0 ? model_.config().head_dim
             : model_.config().hidden_size / model_.config().num_attention_heads;
    // 1. Build batched input_ids [B, 1] and offsets [B]
    std::vector<int> tok_ids;
    std::vector<int> offset_vec;
    tok_ids.reserve(B);
    offset_vec.reserve(B);
    int max_offset = 0;
    for (auto& id : ids) {
        tok_ids.push_back(active_[id].next_token.item<int>());
        int off = active_[id].cache_offset;
        offset_vec.push_back(off);
        if (off > max_offset) max_offset = off;
    }
    auto input_ids = mx::array(tok_ids.data(), {B, 1}, mx::int32);
    auto offsets = mx::array(offset_vec.data(), {B}, mx::int32);

    // Use max(offsets) + 1 as effective KV length — avoids processing padding in SDPA
    int max_kv_len = max_offset + 1;

    // 2. Concatenate KV caches from pool slots, sliced to effective length
    //    Pool caches are [1, n_kv, pool_max_ctx, hd] — slice to [1, n_kv, max_kv_len, hd]
    int pool_max_ctx = pool_.max_context_len();
    std::vector<mx::array> batch_cache_k, batch_cache_v;
    batch_cache_k.reserve(num_layers);
    batch_cache_v.reserve(num_layers);

    for (int l = 0; l < num_layers; l++) {
        std::vector<mx::array> k_parts, v_parts;
        for (auto& id : ids) {
            int slot = active_[id].kv_slot;
            k_parts.push_back(mx::slice(pool_.keys(slot, l),
                {0, 0, 0, 0}, {1, n_kv, max_kv_len, hd}));
            v_parts.push_back(mx::slice(pool_.values(slot, l),
                {0, 0, 0, 0}, {1, n_kv, max_kv_len, hd}));
        }
        batch_cache_k.push_back(mx::concatenate(k_parts, 0));
        batch_cache_v.push_back(mx::concatenate(v_parts, 0));
    }

    // 4. Heterogeneous forward pass
    mx::array logits = model_.forward_heterogeneous(input_ids, batch_cache_k, batch_cache_v, offsets, max_kv_len);

    // 5. Split updated caches back to pool slots
    //    Batch caches are [B, n_kv, max_kv_len, hd] — write back into full pool buffers
    int new_kv_len = max_kv_len + 1;  // forward_heterogeneous wrote one new token
    for (int l = 0; l < num_layers; l++) {
        for (int b = 0; b < B; b++) {
            int slot = active_[ids[b]].kv_slot;
            auto slice_k = mx::slice(batch_cache_k[l], {b, 0, 0, 0},
                {b + 1, n_kv, batch_cache_k[l].shape(2), hd});
            auto slice_v = mx::slice(batch_cache_v[l], {b, 0, 0, 0},
                {b + 1, n_kv, batch_cache_v[l].shape(2), hd});
            pool_.keys(slot, l) = mx::slice_update(pool_.keys(slot, l), slice_k,
                {0, 0, 0, 0}, {1, n_kv, batch_cache_k[l].shape(2), hd});
            pool_.values(slot, l) = mx::slice_update(pool_.values(slot, l), slice_v,
                {0, 0, 0, 0}, {1, n_kv, batch_cache_v[l].shape(2), hd});
        }
    }

    // 6. Sample tokens per-request
    //    logits shape: [B, 1, vocab_size]
    for (int b = 0; b < B; b++) {
        auto& req = active_[ids[b]];
        auto req_logits = mx::slice(logits, {b, 0, 0}, {b + 1, 1, logits.shape(2)});
        req_logits = mx::reshape(req_logits, {1, -1});

        mx::array token = sample_token(req_logits, req.temperature);
        mx::eval({token});
        int tok_id = token.item<int>();

        req.next_token = token;
        req.output_tokens.push_back(tok_id);
        req.generated_count++;
        req.cache_offset++;

        new_tokens[ids[b]].push_back(tok_id);

        if (req.generated_count >= req.max_tokens) {
            req.state = RequestState::DONE;
            done_ids.push_back(ids[b]);
        }
    }
}

} // namespace flashmlx
