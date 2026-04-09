// pipeline/inference.hpp
// Qwen3.5 MoE CPU inference engine.
//
// Responsibilities:
//   - Manages KV-cache (attention layers only; SSM layers are stateless here).
//   - Builds a per-step ggml compute graph and executes it on the CPU.
//   - Returns raw logit vector for the next-token distribution.
//
// Architecture notes:
//   - 40 transformer blocks.
//   - Blocks where (layer_idx % 4 == 3) use full GQA attention.
//   - All other blocks use a simplified gated-SSM transform.
//   - Every block has a MoE FFN (sparse experts + shared expert).
//     The sparse path uses ggml_mul_mat_id for efficient batched routing.
//
#ifndef FUNASR_PIPELINE_INFERENCE_HPP
#define FUNASR_PIPELINE_INFERENCE_HPP

#include "model/model.hpp"
#include "core/config.hpp"
#include <ggml.h>
#include <ggml-cpu.h>
#include <vector>
#include <memory>

class Qwen35moeInference {
public:
    Qwen35moeInference()  = default;
    ~Qwen35moeInference() { free_kv_cache(); }

    // Not copyable / movable (owns raw pointers)
    Qwen35moeInference(const Qwen35moeInference&) = delete;
    Qwen35moeInference& operator=(const Qwen35moeInference&) = delete;

    // Initialise inference engine.
    // max_ctx: maximum context length (KV-cache slots).
    // n_threads: CPU thread count.
    bool init(const Qwen35moeModel& model,
              int max_ctx   = 2048,
              int n_threads = 4);

    // Run one forward step for token `token_id` at position `pos`.
    // Returns logits over the full vocabulary [vocab_size].
    std::vector<float> forward(int token_id, int pos);

    int vocab_size() const { return vocab_size_; }

private:
    // ---- model reference (owned by caller) ----
    const Qwen35moeModel*  model_   = nullptr;
    const Qwen35moeConfig* cfg_     = nullptr;
    int vocab_size_ = 0;
    int n_layers_   = 0;

    // ---- runtime params ----
    int max_ctx_   = 2048;
    int n_threads_ = 4;

    // ---- KV cache (attention layers only) ----
    // Stored as F32.  Index: attention layer ordinal (0, 1, 2, …).
    struct KVCache {
        struct ggml_context*  ctx = nullptr;   // owns the memory
        struct ggml_tensor*   k   = nullptr;   // [head_dim, n_kv_heads, max_ctx]
        struct ggml_tensor*   v   = nullptr;   // [head_dim, n_kv_heads, max_ctx]
    };
    std::vector<KVCache> kv_;   // indexed by attention layer ordinal

    // ---- scratch buffer for compute graph ----
    std::vector<uint8_t> compute_buf_;

    // ---- helpers ----
    void  free_kv_cache();
    bool  alloc_kv_cache();

    // Build the full forward graph in ctx, return the logits tensor.
    // inp: [1] int32 tensor pre-filled with token_id.
    // pos: current position (for RoPE and KV-cache offset).
    struct ggml_tensor* build_graph(struct ggml_context* ctx,
                                    struct ggml_cgraph*  gf,
                                    struct ggml_tensor*  inp,
                                    int                  pos);

    // Build one attention layer (GQA).
    struct ggml_tensor* build_attn_layer(struct ggml_context* ctx,
                                         struct ggml_cgraph*  gf,
                                         struct ggml_tensor*  cur,
                                         int                  layer_idx,
                                         int                  attn_layer_ordinal,
                                         int                  pos);

    // Build one SSM layer (simplified gated-linear).
    struct ggml_tensor* build_ssm_layer(struct ggml_context* ctx,
                                        struct ggml_tensor*  cur,
                                        int                  layer_idx);

    // Build the MoE FFN (sparse + shared expert).
    struct ggml_tensor* build_moe_ffn(struct ggml_context* ctx,
                                      struct ggml_cgraph*  gf,
                                      struct ggml_tensor*  cur,
                                      int                  layer_idx);
};

#endif // FUNASR_PIPELINE_INFERENCE_HPP
