// pipeline/inference.hpp
// Qwen3.5 MoE CPU inference engine.
//
// Responsibilities:
//   - Manages KV-cache (attention layers only).
//   - Manages SSM recurrent state cache (DeltaNet layers).
//   - Builds a per-step ggml compute graph and executes it on the CPU.
//   - Returns raw logit vector for the next-token distribution.
//
// Architecture notes:
//   - 40 transformer blocks.
//   - Blocks where (layer_idx % 4 == 3) use full GQA attention.
//   - All other blocks use a gated DeltaNet SSM transform.
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
    ~Qwen35moeInference() { free_kv_cache(); free_ssm_states(); }

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
        struct ggml_tensor*   k   = nullptr;   // [head_dim, max_ctx, n_kv_heads]
        struct ggml_tensor*   v   = nullptr;   // [head_dim, max_ctx, n_kv_heads]
    };
    std::vector<KVCache> kv_;   // indexed by attention layer ordinal

    // ---- SSM recurrent state cache (DeltaNet layers) ----
    // Stored as F32.  Index: SSM layer ordinal (0..n_ssm-1).
    struct SSMState {
        struct ggml_context* ctx      = nullptr;
        // DeltaNet recurrent state: [head_v_dim*head_v_dim, num_v_heads] F32
        // reshaped to [head_v_dim, head_v_dim, num_v_heads] during compute
        struct ggml_tensor*  state    = nullptr;
        // Conv1d sliding-window buffer: [conv_kernel-1, conv_channels] F32
        // ne[0] = conv_kernel-1 = 3, ne[1] = conv_channels = 8192
        struct ggml_tensor*  conv_buf = nullptr;
    };
    std::vector<SSMState> ssm_states_;  // indexed by SSM layer ordinal

    // ---- scratch buffer for compute graph ----
    std::vector<uint8_t> compute_buf_;

    // ---- helpers ----
    void  free_kv_cache();
    bool  alloc_kv_cache();

    void  free_ssm_states();
    bool  alloc_ssm_states();

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

    // Build one SSM layer (DeltaNet with recurrent state).
    struct ggml_tensor* build_ssm_layer(struct ggml_context* ctx,
                                        struct ggml_cgraph*  gf,
                                        struct ggml_tensor*  cur,
                                        int                  layer_idx,
                                        int                  ssm_ord);

    // Build the MoE FFN (sparse + shared expert).
    struct ggml_tensor* build_moe_ffn(struct ggml_context* ctx,
                                      struct ggml_cgraph*  gf,
                                      struct ggml_tensor*  cur,
                                      int                  layer_idx);
};

#endif // FUNASR_PIPELINE_INFERENCE_HPP
