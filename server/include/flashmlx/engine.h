#pragma once

#include <string>
#include <vector>
#include <map>

namespace flashmlx {

struct EngineStats {
    int active_requests;
    int total_requests;
    double avg_tok_s;
};

class Engine {
public:
    Engine(const std::string& model_path, int max_batch_size, int max_context_len);
    ~Engine();

    std::string ping() const;
    EngineStats get_stats() const;

private:
    std::string model_path_;
    int max_batch_size_;
    int max_context_len_;
    int total_requests_ = 0;
};

} // namespace flashmlx
