#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <random>
#include "model/model.h"
#include "pipeline/sampling.h"
#include "pipeline/tokenizer.h"
#include "graph/graph.h"

class ChatEngine {
public:
    ChatEngine();
    ~ChatEngine();

    bool init(const CParam& param);
    bool run_complete(const std::string& prompt, const int max_tokens, std::string& response);

private:
    int sample_from_topk_candidates(const TopKSampleCandidates& candidates, float top_p);

private:
    DevMode dev_mode_ = DevMode::CPU_MODE;
    float top_p_ = 0.95f;
    int top_k_ = 40;
    float temperature_ = 0.7f;
    bool use_gpu_topk_sampling_ = false;
    int n_batch_ = 0;
    int n_ubatch_ = 0;
    std::mt19937 gpu_sampling_rng_;
    std::shared_ptr<Qwen35moeModel> model_ = nullptr;
    std::shared_ptr<Tokenizer> tokenizer_ = nullptr;
    std::shared_ptr<Sampler> sampler_ = nullptr;  
    std::shared_ptr<Qwen35moeForwardPass> forward_pass_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
};
