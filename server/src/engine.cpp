#include "flashmlx/engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace flashmlx {

Engine::Engine(const std::string& model_path, int max_batch_size, int max_context_len)
    : model_path_(model_path), max_batch_size_(max_batch_size), max_context_len_(max_context_len) {}

Engine::~Engine() = default;

std::string Engine::ping() const {
    return "flashmlx engine ready";
}

EngineStats Engine::get_stats() const {
    return {0, total_requests_, 0.0};
}

} // namespace flashmlx

PYBIND11_MODULE(_flashmlx_engine, m) {
    m.doc() = "FlashMLX C++ inference engine";

    py::class_<flashmlx::EngineStats>(m, "EngineStats")
        .def_readonly("active_requests", &flashmlx::EngineStats::active_requests)
        .def_readonly("total_requests", &flashmlx::EngineStats::total_requests)
        .def_readonly("avg_tok_s", &flashmlx::EngineStats::avg_tok_s);

    py::class_<flashmlx::Engine>(m, "Engine")
        .def(py::init<const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("max_batch_size") = 8,
             py::arg("max_context_len") = 2048)
        .def("ping", &flashmlx::Engine::ping)
        .def("get_stats", &flashmlx::Engine::get_stats);
}
