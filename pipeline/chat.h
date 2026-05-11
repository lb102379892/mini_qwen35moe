#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include "model/model.h"
#include "pipeline/sampling.h"
#include "pipeline/tokenizer.h"
#include "graph/graph.h"

class ChatEngine {
public:
    ChatEngine();
    ~ChatEngine();

    bool init(const std::string& model_path_, DevMode dev_mode = DevMode::CPU_MODE, int n_threads = 1, int max_seq_len = 2048, float top_p = -1.0f, int top_k = -1, float temperature = -1.0f, size_t gpu_layer = 0, bool flash_attention = false);
    bool run_complete(const std::string& prompt, const int max_tokens, std::string& response);

private:
    int max_seq_len_ = 2048;
    DevMode dev_mode_ = DevMode::CPU_MODE;
    std::shared_ptr<Qwen35moeModel> model_ = nullptr;
    std::shared_ptr<Tokenizer> tokenizer_ = nullptr;
    std::shared_ptr<Sampler> sampler_ = nullptr;  
    std::shared_ptr<Qwen35moeForwardPass> forward_pass_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
};
