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

    bool init(const std::string& model_path_, DevMode dev_mode = DevMode::CPU_MODE, int n_threads = 1, int max_seq_len = 2048,
        float top_p = -1.0f, int top_k = -1, float temperature = -1.0f, size_t gpu_layer = 0, bool flash_attention = false,
        int n_batch = -1, int n_ubatch = -1, bool enable_paged_kv = false, uint32_t paged_kv_block_size = 16);
    bool run_complete(const std::string& prompt, const int max_tokens, std::string& response);

private:
    int sample_from_topk_candidates(const TopKSampleCandidates& candidates, float top_p);

private:
    int max_seq_len_ = 2048;
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
