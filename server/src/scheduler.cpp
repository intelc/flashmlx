#include "flashmlx/scheduler.h"

#include <iostream>

namespace flashmlx {

BatchScheduler::BatchScheduler(LlamaModel& model, KVCachePool& pool)
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

    // 2. Decode one token for each active DECODING request
    std::vector<std::string> done_ids;
    for (auto& [id, req] : active_) {
        if (req.state != RequestState::DECODING) continue;

        decode_request(req);
        new_tokens[id].push_back(req.output_tokens.back());

        if (req.generated_count >= req.max_tokens) {
            req.state = RequestState::DONE;
            done_ids.push_back(id);
        }
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

    int num_layers = model_.config().num_hidden_layers;
    int prompt_len = static_cast<int>(req.prompt_tokens.size());

    // Build input_ids [1, prompt_len]
    auto input_ids = mx::array(req.prompt_tokens.data(),
                               {1, prompt_len}, mx::int32);

    // Gather cache arrays for this slot
    std::vector<mx::array> cache_keys;
    std::vector<mx::array> cache_values;
    cache_keys.reserve(num_layers);
    cache_values.reserve(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        cache_keys.push_back(pool_.keys(slot, l));
        cache_values.push_back(pool_.values(slot, l));
    }

    // Cache offsets: all zero for fresh prefill
    auto cache_offsets = mx::array({0}, mx::int32);

    // Forward pass
    mx::array logits = model_.forward(input_ids, cache_keys, cache_values, cache_offsets);

    // Write updated caches back to pool
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
    int num_layers = model_.config().num_hidden_layers;

    // Build input_ids [1, 1] from next_token
    int tok_id = req.next_token.item<int>();
    auto input_ids = mx::array(&tok_id, {1, 1}, mx::int32);

    // Gather cache arrays for this slot
    std::vector<mx::array> cache_keys;
    std::vector<mx::array> cache_values;
    cache_keys.reserve(num_layers);
    cache_values.reserve(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        cache_keys.push_back(pool_.keys(slot, l));
        cache_values.push_back(pool_.values(slot, l));
    }

    // Cache offset = current position
    auto cache_offsets = mx::array({req.cache_offset}, mx::int32);

    // Forward pass
    mx::array logits = model_.forward(input_ids, cache_keys, cache_values, cache_offsets);

    // Write updated caches back to pool
    for (int l = 0; l < num_layers; ++l) {
        pool_.keys(slot, l) = cache_keys[l];
        pool_.values(slot, l) = cache_values[l];
    }

    // Sample next token — logits shape [1, 1, vocab_size]
    mx::array last_logits = mx::reshape(logits, {1, -1});
    mx::array token = sample_token(last_logits, req.temperature);
    mx::eval({token});
    int new_tok_id = token.item<int>();

    req.next_token = token;
    req.output_tokens.push_back(new_tok_id);
    req.generated_count++;
    req.cache_offset++;
}

} // namespace flashmlx
