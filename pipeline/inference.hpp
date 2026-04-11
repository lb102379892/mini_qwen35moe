// pipeline/inference.hpp
// Qwen3.5 MoE forward pass engine — CPU, with KV cache and SSM state persistence.
//
// Incremental inference:
//   - First call to forward() processes the full prompt (any length).
//   - Subsequent calls pass a single new token; SSM states and the attention
//     KV cache are carried across calls automatically.
//   - Call reset_state() before each new conversation / prompt.
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
    explicit InferenceEngine(const Qwen35moeModel& model, int n_threads = 4,
                             int max_seq_len = 2048);
    ~InferenceEngine();

    // Not copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    // Run forward pass.
    // First call: pass the full prompt tokens → processes all of them, returns
    //             logits for the last token.
    // Subsequent calls: pass exactly one new token → uses cached SSM/KV state,
    //                   returns logits for that token.
    // Returns empty vector on error.
    std::vector<float> forward(const std::vector<int32_t>& tokens);

    // Reset all cached state (SSM conv, SSM recurrent, KV cache) and position
    // counter.  Must be called before processing a new prompt.
    void reset_state();

private:
    const Qwen35moeModel& model_;
    int n_threads_;
    int max_seq_len_;

    ggml_backend_t   backend_  = nullptr;
    ggml_gallocr_t   galloc_   = nullptr;

    // ---------------------------------------------------------------
    // Persistent incremental-inference state
    // ---------------------------------------------------------------

    // Current sequence position (= number of tokens processed so far).
    int pos_ = 0;

    // SSM conv state per SSM layer.
    // Layout: [conv_kernel-1, qkv_channels] contiguous F32.
    // Indexed by ssm_state_idx(il).
    std::vector<std::vector<float>> ssm_conv_states_;

    // SSM recurrent (GDN) state per SSM layer.
    // Layout: [head_v_dim, head_v_dim, dt_rank] = [S_v, S_v, H] contiguous F32
    // — exactly the format ggml_gated_delta_net expects for its state input.
    // Indexed by ssm_state_idx(il).
    std::vector<std::vector<float>> ssm_recurrent_states_;

    // KV cache per attention layer. Indexed by attn_cache_idx(il).
    struct KVCache {
        std::vector<float> k;  // [head_dim, max_seq_len, n_kv_heads] F32
        std::vector<float> v;  // [head_dim, max_seq_len, n_kv_heads] F32
        int len = 0;           // number of positions currently stored
    };
    std::vector<KVCache> kv_caches_;

    // ---------------------------------------------------------------
    // Temporary pointers collected during graph building; valid until
    // the next call to forward().  Used to read state back after compute.
    // ---------------------------------------------------------------
    struct SSMOutPtrs {
        ggml_tensor* gdn_out        = nullptr;  // full GDN output tensor
        ggml_tensor* conv_state_out = nullptr;  // new conv state (cont)
        int n_tokens                = 0;
    };
    std::vector<SSMOutPtrs> tmp_ssm_outs_;   // [n_ssm_layers]

    struct KVOutPtrs {
        ggml_tensor* k_new = nullptr;  // new K (after RoPE, permuted+cont)
        ggml_tensor* v_new = nullptr;  // new V (permuted+cont)
    };
    std::vector<KVOutPtrs> tmp_kv_outs_;    // [n_attn_layers]

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    // Allocate / zero all persistent state buffers from model config.
    void init_state();

    // Map layer index → SSM state slot (0-based among SSM layers).
    static int ssm_state_idx(int il) { return il - il / 4; }

    // Map layer index → attention cache slot (0-based among attn layers).
    static int attn_cache_idx(int il) { return il / 4; }

    // ---------------------------------------------------------------
    // Graph builders
    // ---------------------------------------------------------------

    // Build the full compute graph; returns the logits tensor.
    // pos  = absolute position of the first token in this batch.
    ggml_tensor* build_graph(ggml_context* ctx, ggml_cgraph* gf,
                              ggml_tensor* inp_tokens, int n_tokens, int pos);

    // Sub-graph builders (pos = absolute position of first token in batch)
    ggml_tensor* build_attn_layer(ggml_context* ctx, ggml_cgraph* gf,
                                   ggml_tensor* cur, ggml_tensor* inp_pos,
                                   ggml_tensor* kq_mask, int il,
                                   int n_tokens, int pos);

    ggml_tensor* build_ssm_layer(ggml_context* ctx, ggml_cgraph* gf,
                                  ggml_tensor* cur, int il,
                                  int n_tokens, int pos);

    ggml_tensor* build_moe_ffn(ggml_context* ctx, ggml_cgraph* gf,
                                ggml_tensor* cur, int il, int n_tokens);
};

#endif // QWEN35MOE_PIPELINE_INFERENCE_HPP
