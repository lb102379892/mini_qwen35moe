// pipeline/inference.hpp
// Qwen3.5 MoE forward pass engine — CPU, with KV cache and SSM state persistence.
//
// Incremental inference:
//   - First call to forward() processes the full prompt (any length).
//   - Subsequent calls pass a single new token; SSM states and the attention
//     KV cache are carried across calls automatically.
//   - Call reset_state() before each new conversation / prompt.
//
// Architecture: per-layer execution with CPU-side MoE routing.
// Each transformer layer is executed as separate ggml sub-graphs.
// MoE expert routing (softmax + top-k + normalize) is computed in pure C++
// on the CPU; only the FFN compute uses ggml.
//
#ifndef QWEN35MOE_PIPELINE_INFERENCE_HPP
#define QWEN35MOE_PIPELINE_INFERENCE_HPP

#include "core/gguf_reader.hpp"
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
    explicit InferenceEngine(const Qwen35moeModel& model, GGUFReader* reader = nullptr,
                             int n_threads = 4, int max_seq_len = 2048, bool use_gpu = false);
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
    GGUFReader* reader_ = nullptr;
    int n_threads_;
    int max_seq_len_;

    ggml_backend_t   backend_  = nullptr;
    bool             use_gpu_  = false;

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
    // Per-layer MoE routing results (computed on CPU, fed as leaf tensors)
    // Both vectors use ggml column-major layout: index = k + t * n_top_k
    //   (k = expert slot 0..n_top_k-1, t = token 0..n_tokens-1)
    // Total size per layer: n_top_k * n_tokens  (resized each forward call)
    // ---------------------------------------------------------------
    struct MoERoute {
        std::vector<int32_t> selected; // expert indices,   size: n_top_k * n_tokens
        std::vector<float>   weights;  // normalized probs,  size: n_top_k * n_tokens
    };
    std::vector<MoERoute> moe_routes_; // [n_layer]

    // ---------------------------------------------------------------
    // Temporary pointers set during sub-graph building; valid only
    // between ggml_gallocr_alloc_graph() and ggml_free() for that sub-ctx.
    // Must be read before ggml_free() is called.
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
    // CPU-side MoE routing (pure C++, no ggml)
    // W: ffn_gate_inp->data F32 [n_embd, n_expert] column-major
    // X: ffn_in F32 [n_embd, n_tokens] column-major
    // Results written to moe_routes_[il].
    // ---------------------------------------------------------------
    void compute_moe_routing_cpu(const float* W, const float* X,
                                  int il, int n_tokens);

    // ---------------------------------------------------------------
    // Per-layer execution helpers
    // Each creates a small ggml context, runs it, returns F32 data.
    // ---------------------------------------------------------------

    // Token embedding lookup: returns [n_embd * n_tokens] F32
    std::vector<float> exec_token_embd(const std::vector<int32_t>& tokens,
                                        int n_tokens);

    // attn_norm(cur) + attn-or-SSM sub-layer.
    // Handles KV cache / SSM state I/O internally.
    // Returns attn/SSM output [n_embd * n_tokens] F32 (no residual).
    std::vector<float> exec_attn_or_ssm(const std::vector<float>& cur,
                                         int il, int n_tokens, int pos);

    // RMS norm + weight multiply: returns [n_embd * n_tokens] F32
    std::vector<float> exec_rms_norm(const std::vector<float>& cur,
                                      ggml_tensor* norm_weight, int n_tokens);

    // MoE FFN using CPU-precomputed routing in moe_routes_[il].
    // Returns [n_embd * n_tokens] F32.
    std::vector<float> exec_moe_ffn(const std::vector<float>& ffn_in,
                                     int il, int n_tokens);

    // final_norm(cur[last_token]) + lm_head → logits [vocab_size] F32
    std::vector<float> exec_lm_head(const std::vector<float>& cur, int n_tokens);

    // ---------------------------------------------------------------
    // Sub-graph builders (called from exec_attn_or_ssm)
    // ---------------------------------------------------------------
    ggml_tensor* build_attn_layer(ggml_context* ctx, ggml_cgraph* gf,
                                   ggml_tensor* cur, ggml_tensor* inp_pos,
                                   ggml_tensor* kq_mask, int il,
                                   int n_tokens, int pos);

    ggml_tensor* build_ssm_layer(ggml_context* ctx, ggml_cgraph* gf,
                                  ggml_tensor* cur, int il,
                                  int n_tokens, int pos);
};

#endif // QWEN35MOE_PIPELINE_INFERENCE_HPP
