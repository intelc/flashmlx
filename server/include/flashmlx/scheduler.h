#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

#include "flashmlx/kv_pool.h"
#include "flashmlx/model.h"
#include "flashmlx/sampling.h"

namespace mx = mlx::core;

namespace flashmlx {

enum class RequestState { QUEUED, PREFILLING, DECODING, DONE };

struct Request {
    std::string id;
    std::vector<int> prompt_tokens;
    int max_tokens = 256;
    float temperature = 0.0f;

    // Managed by scheduler
    RequestState state = RequestState::QUEUED;
    int kv_slot = -1;
    int cache_offset = 0;
    int generated_count = 0;
    mx::array next_token = mx::array(0);
    std::vector<int> output_tokens;
};

class BatchScheduler {
public:
    BatchScheduler(LlamaModel& model, KVCachePool& pool);

    /// Thread-safe: enqueue a new request
    void submit(Request req);

    /// Run one scheduler iteration: admit pending, prefill, decode, cleanup.
    /// Returns map of request_id -> newly generated tokens this step.
    std::unordered_map<std::string, std::vector<int>> step();

    /// True if any requests are pending or active
    bool has_work();

    /// Returns and clears list of completed request IDs
    std::vector<std::string> drain_completed();

    /// Number of currently active (prefilling or decoding) requests
    int active_count() const;

private:
    void prefill_request(Request& req);
    void decode_request(Request& req);
    void decode_batch(const std::vector<std::string>& ids,
                      std::unordered_map<std::string, std::vector<int>>& new_tokens,
                      std::vector<std::string>& done_ids);


    LlamaModel& model_;
    KVCachePool& pool_;

    // Pending queue — accessed from submit() (Python thread) and step() (C++ thread)
    std::deque<Request> pending_;
    std::mutex pending_mutex_;

    // Active requests — only accessed from step()
    std::unordered_map<std::string, Request> active_;

    // Completed IDs — only accessed from step() and drain_completed()
    std::vector<std::string> completed_;
    std::mutex completed_mutex_;
};

} // namespace flashmlx
