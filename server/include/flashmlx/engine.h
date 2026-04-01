#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include "flashmlx/model.h"
#include "flashmlx/kv_pool.h"
#include "flashmlx/scheduler.h"

namespace flashmlx {

struct EngineStats {
    int active_requests;
    int total_requests;
    double avg_tok_s;
    int free_kv_slots;
};

struct TokenOutput {
    std::string request_id;
    std::vector<int> tokens;
    bool done;
};

class Engine {
public:
    Engine(const std::string& model_path, int max_batch_size = 8, int max_context_len = 2048);
    ~Engine();

    void submit_request(const std::string& request_id,
                       const std::vector<int>& prompt_tokens,
                       int max_tokens, float temperature);
    std::vector<TokenOutput> poll_tokens();
    EngineStats get_stats() const;
    std::string ping() const;
    void start();
    void stop();
    std::vector<int> debug_forward(const std::vector<int>& token_ids);
    std::vector<float> debug_embed(const std::vector<int>& token_ids);

private:
    std::unique_ptr<ModelBase> model_;
    std::unique_ptr<KVCachePool> kv_pool_;
    std::unique_ptr<BatchScheduler> scheduler_;
    std::string model_path_;
    int max_batch_size_;
    int max_context_len_;
    std::atomic<int> total_requests_{0};
    std::thread loop_thread_;
    std::atomic<bool> running_{false};
    void loop();
    std::queue<TokenOutput> output_queue_;
    std::mutex output_mutex_;
};

} // namespace flashmlx
