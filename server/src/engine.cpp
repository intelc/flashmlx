#include "flashmlx/engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <chrono>

namespace py = pybind11;

namespace flashmlx {

Engine::Engine(const std::string& model_path, int max_batch_size, int max_context_len)
    : model_path_(model_path), max_batch_size_(max_batch_size), max_context_len_(max_context_len) {
    // Load model
    model_ = std::make_unique<LlamaModel>(model_path);

    // Create KV cache pool using model config
    const auto& cfg = model_->config();
    int head_dim = cfg.head_dim > 0 ? cfg.head_dim : cfg.hidden_size / cfg.num_attention_heads;
    kv_pool_ = std::make_unique<KVCachePool>(
        max_batch_size, max_context_len,
        cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim,
        cfg.activation_dtype);

    // Create scheduler
    scheduler_ = std::make_unique<BatchScheduler>(*model_, *kv_pool_);

    std::cout << "[Engine] Initialized: batch_size=" << max_batch_size
              << " context_len=" << max_context_len
              << " kv_slots=" << kv_pool_->num_free() << std::endl;
}

Engine::~Engine() {
    stop();
}

std::string Engine::ping() const {
    return "flashmlx engine ready";
}

EngineStats Engine::get_stats() const {
    int active = scheduler_ ? scheduler_->active_count() : 0;
    int free_slots = kv_pool_ ? kv_pool_->num_free() : 0;
    return {active, total_requests_.load(), 0.0, free_slots};
}

void Engine::submit_request(const std::string& request_id,
                           const std::vector<int>& prompt_tokens,
                           int max_tokens, float temperature) {
    Request req;
    req.id = request_id;
    req.prompt_tokens = prompt_tokens;
    req.max_tokens = max_tokens;
    req.temperature = temperature;

    scheduler_->submit(std::move(req));
    total_requests_.fetch_add(1);
}

std::vector<TokenOutput> Engine::poll_tokens() {
    std::vector<TokenOutput> results;
    std::lock_guard<std::mutex> lock(output_mutex_);
    while (!output_queue_.empty()) {
        results.push_back(std::move(output_queue_.front()));
        output_queue_.pop();
    }
    return results;
}

void Engine::start() {
    if (running_.load()) return;
    running_.store(true);
    loop_thread_ = std::thread(&Engine::loop, this);
    std::cout << "[Engine] Background loop started" << std::endl;
}

void Engine::stop() {
    if (!running_.load()) return;
    running_.store(false);
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    std::cout << "[Engine] Background loop stopped" << std::endl;
}

void Engine::loop() {
    while (running_.load()) {
        if (!scheduler_->has_work()) {
            // Sleep briefly to avoid busy-waiting when idle
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Run one scheduler step
        auto new_tokens = scheduler_->step();

        // Push token outputs to queue
        if (!new_tokens.empty()) {
            std::lock_guard<std::mutex> lock(output_mutex_);
            for (auto& [req_id, tokens] : new_tokens) {
                output_queue_.push(TokenOutput{req_id, std::move(tokens), false});
            }
        }

        // Check for completed requests
        auto completed = scheduler_->drain_completed();
        if (!completed.empty()) {
            std::lock_guard<std::mutex> lock(output_mutex_);
            for (const auto& req_id : completed) {
                output_queue_.push(TokenOutput{req_id, {}, true});
            }
        }
    }
}

std::vector<int> Engine::debug_forward(const std::vector<int>& token_ids) {
    return model_->debug_forward(token_ids);
}

std::vector<float> Engine::debug_embed(const std::vector<int>& token_ids) {
    return model_->debug_embed(token_ids);
}

} // namespace flashmlx

PYBIND11_MODULE(_flashmlx_engine, m) {
    m.doc() = "FlashMLX C++ inference engine";

    py::class_<flashmlx::EngineStats>(m, "EngineStats")
        .def_readonly("active_requests", &flashmlx::EngineStats::active_requests)
        .def_readonly("total_requests", &flashmlx::EngineStats::total_requests)
        .def_readonly("avg_tok_s", &flashmlx::EngineStats::avg_tok_s)
        .def_readonly("free_kv_slots", &flashmlx::EngineStats::free_kv_slots);

    py::class_<flashmlx::TokenOutput>(m, "TokenOutput")
        .def_readonly("request_id", &flashmlx::TokenOutput::request_id)
        .def_readonly("tokens", &flashmlx::TokenOutput::tokens)
        .def_readonly("done", &flashmlx::TokenOutput::done);

    py::class_<flashmlx::Engine>(m, "Engine")
        .def(py::init<const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("max_batch_size") = 8,
             py::arg("max_context_len") = 2048)
        .def("ping", &flashmlx::Engine::ping)
        .def("get_stats", &flashmlx::Engine::get_stats)
        .def("submit_request", &flashmlx::Engine::submit_request,
             py::arg("request_id"), py::arg("prompt_tokens"),
             py::arg("max_tokens"), py::arg("temperature"),
             py::call_guard<py::gil_scoped_release>())
        .def("poll_tokens", &flashmlx::Engine::poll_tokens,
             py::call_guard<py::gil_scoped_release>())
        .def("start", &flashmlx::Engine::start)
        .def("stop", &flashmlx::Engine::stop)
        .def("debug_forward", &flashmlx::Engine::debug_forward)
        .def("debug_embed", &flashmlx::Engine::debug_embed);
}
