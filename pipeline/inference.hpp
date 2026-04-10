// pipeline/inference.hpp
// Qwen3.5 MoE forward pass engine — CPU only, no KV cache
//
// Every call to forward() builds and executes a fresh GGML compute graph
// for the entire token sequence. Slow but correct.
//
#ifndef QWEN35MOE_PIPELINE_INFERENCE_HPP
#define QWEN35MOE_PIPELINE_INFERENCE_HPP

#include "model/model.hpp"
#include <vector>
#include <cstdint>
#include <memory>

// Forward declarations
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct ggml_gallocr;
typedef struct ggml_gallocr * ggml_gallocr_t;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

class InferenceEngine {
public:
    // model must outlive this object
    explicit InferenceEngine(const Qwen35moeModel& model, int n_threads = 4);
    ~InferenceEngine();

    // Not copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    // Run full forward pass on token sequence.
    // Returns logits for the LAST token: vector of size vocab_size.
    // Returns empty vector on error.
    std::vector<float> forward(const std::vector<int32_t>& tokens);

private:
    const Qwen35moeModel& model_;
    int n_threads_;

    ggml_backend_t   backend_  = nullptr;
    ggml_gallocr_t   galloc_   = nullptr;

    // Build the full compute graph into ctx, return the logits tensor
    ggml_tensor* build_graph(ggml_context* ctx, ggml_cgraph* gf,
                              ggml_tensor* inp_tokens, int n_tokens);

    // Sub-graph builders
    ggml_tensor* build_attn_layer(ggml_context* ctx, ggml_cgraph* gf,
                                   ggml_tensor* cur, ggml_tensor* inp_pos,
                                   ggml_tensor* kq_mask, int il, int n_tokens);

    ggml_tensor* build_ssm_layer(ggml_context* ctx, ggml_cgraph* gf,
                                  ggml_tensor* cur, int il, int n_tokens);

    ggml_tensor* build_moe_ffn(ggml_context* ctx, ggml_cgraph* gf,
                                ggml_tensor* cur, int il, int n_tokens);
};

#endif // QWEN35MOE_PIPELINE_INFERENCE_HPP
