#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <tuple>

#include "model/common.h"
#include "pipeline/chat.h"

namespace py = pybind11;

namespace {

DevMode parse_dev_mode(const std::string& mode) {
    if (mode == "cpu") {
        return DevMode::CPU_MODE;
    }
    if (mode == "gpu") {
        return DevMode::GPU_MODE;
    }
    if (mode == "auto") {
        return DevMode::AUTO_MODE;
    }
    throw std::invalid_argument("unsupported dev_mode: " + mode);
}

std::string dev_mode_to_string(DevMode mode) {
    switch (mode) {
        case DevMode::CPU_MODE:
            return "cpu";
        case DevMode::GPU_MODE:
            return "gpu";
        case DevMode::AUTO_MODE:
            return "auto";
    }
    return "cpu";
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "In-process pybind11 bindings for the ggml MoE ChatEngine.";

    py::class_<TimingStats>(m, "TimingStats")
        .def_readonly("prompt_tokens", &TimingStats::prompt_tokens)
        .def_readonly("generated_tokens", &TimingStats::generated_tokens)
        .def_readonly("queue_ms", &TimingStats::queue_ms)
        .def_readonly("prefill_ms", &TimingStats::prefill_ms)
        .def_readonly("decode_ms", &TimingStats::decode_ms)
        .def_readonly("inference_ms", &TimingStats::inference_ms)
        .def_readonly("total_ms", &TimingStats::total_ms)
        .def_readonly("ok", &TimingStats::ok)
        .def_readonly("timed", &TimingStats::timed);

    py::class_<CParam>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("model_path", &CParam::model_path)
        .def_readwrite("temperature", &CParam::temperature)
        .def_readwrite("top_p", &CParam::top_p)
        .def_readwrite("top_k", &CParam::top_k)
        .def_readwrite("n_threads", &CParam::n_threads)
        .def_readwrite("n_threads_batch", &CParam::n_threads_batch)
        .def_readwrite("ctx_size", &CParam::ctx_size)
        .def_readwrite("n_batch", &CParam::n_batch)
        .def_readwrite("n_ubatch", &CParam::n_ubatch)
        .def_readwrite("max_sequences", &CParam::max_sequences)
        .def_readwrite("gpu_layer", &CParam::gpu_layer)
        .def_readwrite("tensor_overrides", &CParam::tensor_overrides)
        .def_readwrite("use_chat", &CParam::use_chat)
        .def_readwrite("verbose", &CParam::verbose)
        .def_readwrite("repl_mode", &CParam::repl_mode)
        .def_readwrite("flash_attention", &CParam::flash_attention)
        .def_readwrite("enable_paged_kv", &CParam::enable_paged_kv)
        .def_readwrite("paged_kv_block_size", &CParam::paged_kv_block_size)
        .def_readwrite("rng_seed", &CParam::rng_seed)
        .def_readwrite("no_mmap", &CParam::no_mmap)
        .def_readwrite("gpu_id", &CParam::gpu_id)
        .def_property(
            "dev_mode",
            [](const CParam& param) { return dev_mode_to_string(param.dev_mode); },
            [](CParam& param, const std::string& mode) {
                param.dev_mode = parse_dev_mode(mode);
            });

    py::class_<ChatEngine>(m, "ChatEngine")
        .def(py::init<>())
        .def("init", &ChatEngine::init)
        .def(
            "run_complete",
            [](ChatEngine& engine,
               const std::string& prompt,
               int max_tokens) {
                std::string response;
                TimingStats timing;
                const bool ok =
                    engine.run_complete(prompt, max_tokens, response, timing);
                return py::make_tuple(ok, response, timing);
            },
            py::arg("prompt"),
            py::arg("max_tokens"));
}
