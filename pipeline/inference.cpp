// pipeline/inference.cpp
// Qwen3.5 MoE forward pass — CPU, with SSM state and KV cache persistence.
//
// Architecture recap (from GGUF metadata + llama.cpp reference):
//   40 blocks total
//   Block type: layer % 4 == 3  → full GQA attention
//               otherwise       → DeltaNet SSM (linear attention)
//   All blocks have MoE FFN (256 experts, top-8) + shared expert
//
// Key dimensions:
//   n_embd        = 2048
//   n_heads       = 16,  n_kv_heads = 2,  head_dim = 256
//   rope_dim      = 64,  rope_base  = 1e7
//   n_expert      = 256, top_k      = 8,   ff_dim = 512
//   ssm: d_inner=4096, d_state(head_k_dim)=128, n_group=16, dt_rank(num_v_heads)=32,
//        head_v_dim=128, conv_kernel=4
#include "pipeline/inference.hpp"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <limits>

// ============================================================
// Helpers
// ============================================================

static inline bool is_attn_layer(int il) { return (il % 4) == 3; }

// RMS norm helper
static ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x,
                              ggml_tensor* w, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    return x;
}

// ============================================================
// InferenceEngine — constructor / destructor
// ============================================================

InferenceEngine::InferenceEngine(const Qwen35moeModel& model, int n_threads,
                                 int max_seq_len)
    : model_(model), n_threads_(n_threads), max_seq_len_(max_seq_len) {
    backend_ = ggml_backend_cpu_init();
    if (!backend_) {
        fprintf(stderr, "[Inference] ERROR: failed to init CPU backend\n");
        return;
    }
    ggml_backend_cpu_set_n_threads(backend_, n_threads_);
    galloc_ = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    fprintf(stderr, "[Inference] CPU backend ready (%d threads)\n", n_threads_);
    init_state();
}

InferenceEngine::~InferenceEngine() {
    if (galloc_)  ggml_gallocr_free(galloc_);
    if (backend_) ggml_backend_free(backend_);
}

// ============================================================
// State management
// ============================================================

void InferenceEngine::init_state() {
    const auto& cfg = model_.config.qwen35moe;
    const int n_layer      = (int)cfg.block_count;          // 40
    const int64_t d_inner  = cfg.inner_size;                // 4096
    const int64_t d_state  = cfg.state_size;                // 128  (head_k_dim)
    const int64_t n_group  = cfg.group_count;               // 16
    const int64_t dt_rank  = cfg.time_step_rank;            // 32  (num_v_heads = H)
    const int64_t head_v   = d_inner / dt_rank;             // 128 (S_v)
    const int64_t conv_k   = cfg.conv_kernel;               // 4
    const int64_t qkv_ch   = d_inner + 2 * n_group * d_state; // 8192
    const int64_t head_dim = cfg.key_length;                // 256
    const int64_t n_kv_h   = cfg.head_count_kv;            // 2

    // Count layer types
    int n_ssm = 0, n_attn = 0;
    for (int il = 0; il < n_layer; il++) {
        if (is_attn_layer(il)) n_attn++; else n_ssm++;
    }

    // SSM conv state: [conv_k-1, qkv_ch] per layer
    ssm_conv_states_.assign(n_ssm,
        std::vector<float>((conv_k - 1) * qkv_ch, 0.0f));

    // SSM recurrent state: [head_v, head_v, dt_rank] per layer  (= S_v * S_v * H)
    ssm_recurrent_states_.assign(n_ssm,
        std::vector<float>(head_v * head_v * dt_rank, 0.0f));

    // KV cache: [head_dim * max_seq_len * n_kv_h] per attention layer
    kv_caches_.resize(n_attn);
    for (auto& c : kv_caches_) {
        c.k.assign(head_dim * max_seq_len_ * n_kv_h, 0.0f);
        c.v.assign(head_dim * max_seq_len_ * n_kv_h, 0.0f);
        c.len = 0;
    }

    // Temporary pointer vectors
    tmp_ssm_outs_.resize(n_ssm);
    tmp_kv_outs_.resize(n_attn);

    pos_ = 0;
}

void InferenceEngine::reset_state() {
    for (auto& s : ssm_conv_states_)       std::fill(s.begin(), s.end(), 0.0f);
    for (auto& s : ssm_recurrent_states_)  std::fill(s.begin(), s.end(), 0.0f);
    for (auto& c : kv_caches_) {
        std::fill(c.k.begin(), c.k.end(), 0.0f);
        std::fill(c.v.begin(), c.v.end(), 0.0f);
        c.len = 0;
    }
    pos_ = 0;
}

// ============================================================
// forward()
// ============================================================
std::vector<float> InferenceEngine::forward(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return {};
    if (!backend_ || !galloc_) return {};

    const auto& cfg = model_.config.qwen35moe;
    const int n_tokens   = (int)tokens.size();
    const int vocab_size = (int)model_.weights.token_embd->ne[1];
    const int pos        = pos_;  // absolute position of the first token in this batch

    // Safecheck: don't exceed the KV cache capacity
    if (pos + n_tokens > max_seq_len_) {
        fprintf(stderr, "[Inference] ERROR: sequence length %d exceeds max_seq_len %d\n",
                pos + n_tokens, max_seq_len_);
        return {};
    }

    // Validate single-token constraint for incremental steps
    if (pos_ > 0 && n_tokens != 1) {
        fprintf(stderr, "[Inference] ERROR: incremental mode requires exactly 1 token \
"
                "(got %d; call reset_state() before a new prompt)\n", n_tokens);
        return {};
    }

    // Clear temporary output-pointer vectors from previous call
    for (auto& s : tmp_ssm_outs_) { s.gdn_out = nullptr; s.conv_state_out = nullptr; s.n_tokens = 0; }
    for (auto& k : tmp_kv_outs_)  { k.k_new = nullptr;   k.v_new = nullptr; }

    // --------------------------------------------------
    // 1. Create compute context for graph nodes
    // --------------------------------------------------
    const size_t ctx_size = 512 * 1024 * 1024; // 512 MB for node descriptors
    struct ggml_init_params init_params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context* ctx = ggml_init(init_params);
    if (!ctx) {
        fprintf(stderr, "[Inference] ERROR: ggml_init failed\n");
        return {};
    }

    // --------------------------------------------------
    // 2. Build the compute graph
    // --------------------------------------------------
    const int graph_size = 65536;
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, graph_size, false);

    struct ggml_tensor* inp_tokens_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens_t, "inp_tokens");

    struct ggml_tensor* logits = build_graph(ctx, gf, inp_tokens_t, n_tokens, pos);
    if (!logits) {
        fprintf(stderr, "[Inference] ERROR: build_graph failed\n");
        ggml_free(ctx);
        return {};
    }
    ggml_build_forward_expand(gf, logits);

    // --------------------------------------------------
    // 3. Allocate compute buffers via gallocr
    // --------------------------------------------------
    if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
        fprintf(stderr, "[Inference] ERROR: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return {};
    }

    // --------------------------------------------------
    // 4. Fill inputs: iterate leaf tensors in the graph
    // --------------------------------------------------
    const int64_t d_inner  = cfg.inner_size;
    const int64_t d_state  = cfg.state_size;
    const int64_t n_group  = cfg.group_count;
    const int64_t dt_rank  = cfg.time_step_rank;
    const int64_t head_v   = d_inner / dt_rank;
    const int64_t qkv_ch   = d_inner + 2 * n_group * d_state;
    const int64_t conv_k   = cfg.conv_kernel;
    const int64_t head_dim = cfg.key_length;
    const int64_t n_kv_h   = cfg.head_count_kv;

    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        struct ggml_tensor* t = ggml_graph_node(gf, i);
        if (!t || !t->data || t->op != GGML_OP_NONE) continue;
        const char* nm = t->name;
        if (!nm) continue;

        if (strcmp(nm, "inp_tokens") == 0) {
            memcpy(t->data, tokens.data(), n_tokens * sizeof(int32_t));

        } else if (strcmp(nm, "inp_pos") == 0) {
            // mrope: 4 sections, each containing the absolute token positions
            int32_t* pd = (int32_t*)t->data;
            for (int s = 0; s < 4; s++) {
                for (int p = 0; p < n_tokens; p++) {
                    pd[s * n_tokens + p] = pos + p;
                }
            }
        } else if (strcmp(nm, "kq_mask") == 0) {
            // For full-prompt (pos==0): standard causal mask [n_tokens, n_tokens]
            // For single-token (pos>0): all-zero mask [kv_len, 1]
            ggml_fp16_t* md = (ggml_fp16_t*)t->data;
            if (pos == 0) {
                // Causal: mask[k, q] = 0 if k <= q, else -inf
                for (int q = 0; q < n_tokens; q++) {
                    for (int k = 0; k < n_tokens; k++) {
                        float val = (k <= q) ? 0.0f : -INFINITY;
                        md[q * n_tokens + k] = ggml_fp32_to_fp16(val);
                    }
                }
            } else {
                // Single new token can attend to all kv_len = pos+1 positions
                int kv_len = pos + 1;
                for (int k = 0; k < kv_len; k++) {
                    md[k] = ggml_fp32_to_fp16(0.0f);
                }
            }
        } else if (strncmp(nm, "ssm_conv_pad_", 13) == 0) {
            // Fill with persistent conv state (zeros on first call)
            int il = atoi(nm + 13);
            int si = ssm_state_idx(il);
            size_t sz = (size_t)(conv_k - 1) * (size_t)qkv_ch * sizeof(float);
            memcpy(t->data, ssm_conv_states_[si].data(), sz);

        } else if (strncmp(nm, "ssm_state_in_", 13) == 0) {
            // Fill with persistent recurrent state (zeros on first call)
            int il = atoi(nm + 13);
            int si = ssm_state_idx(il);
            size_t sz = (size_t)head_v * (size_t)head_v * (size_t)dt_rank * sizeof(float);
            memcpy(t->data, ssm_recurrent_states_[si].data(), sz);

        } else if (strncmp(nm, "gate_zero", 9) == 0) {
            memset(t->data, 0, ggml_nbytes(t));

        } else if (strncmp(nm, "beta_ones", 9) == 0) {
            float* fd = (float*)t->data;
            int64_t n = ggml_nelements(t);
            for (int64_t j = 0; j < n; j++) fd[j] = 1.0f;

        } else if (strncmp(nm, "kv_k_cache_", 11) == 0) {
            // Fill KV-K cache tensor for incremental attention.
            // K_cache tensor shape: [head_dim, pos, n_kv_heads, 1]
            // Buffer layout:        [head_dim, max_seq_len, n_kv_heads]
            // (h's stride: head_dim * max_seq_len in buffer, head_dim * pos in tensor)
            int il = atoi(nm + 11);
            int ai = attn_cache_idx(il);
            float* dst = (float*)t->data;
            const float* buf = kv_caches_[ai].k.data();
            for (int64_t h = 0; h < n_kv_h; h++) {
                memcpy(dst + h * head_dim * pos,
                       buf + h * head_dim * max_seq_len_,
                       (size_t)head_dim * pos * sizeof(float));
            }

        } else if (strncmp(nm, "kv_v_cache_", 11) == 0) {
            // Fill KV-V cache tensor for incremental attention.
            int il = atoi(nm + 11);
            int ai = attn_cache_idx(il);
            float* dst = (float*)t->data;
            const float* buf = kv_caches_[ai].v.data();
            for (int64_t h = 0; h < n_kv_h; h++) {
                memcpy(dst + h * head_dim * pos,
                       buf + h * head_dim * max_seq_len_,
                       (size_t)head_dim * pos * sizeof(float));
            }
        }
    }

    // --------------------------------------------------
    // 5. Execute
    // --------------------------------------------------
    ggml_backend_graph_compute(backend_, gf);

    // --------------------------------------------------
    // 6. Save SSM states back to persistent buffers
    // --------------------------------------------------
    {
        const int64_t sv_h = head_v * dt_rank;  // S_v * H = 128*32 = 4096
        for (int il = 0; il < (int)cfg.block_count; il++) {
            if (is_attn_layer(il)) continue;
            int si = ssm_state_idx(il);
            auto& sout = tmp_ssm_outs_[si];

            // Conv state: sout.conv_state_out is a cont tensor of shape
            // [conv_k-1, qkv_ch, 1]
            if (sout.conv_state_out && sout.conv_state_out->data) {
                size_t sz = (size_t)(conv_k - 1) * (size_t)qkv_ch * sizeof(float);
                memcpy(ssm_conv_states_[si].data(), sout.conv_state_out->data, sz);
            }

            // Recurrent state: raw bytes from gdn_out starting at
            // offset n_tokens * sv_h * sizeof(float).
            // The GDN writes state in the same layout as the input expects:
            // flat [H, S_v, S_v] i.e. h*S_v*S_v + j*S_v + i.
            if (sout.gdn_out && sout.gdn_out->data) {
                const float* src = (const float*)sout.gdn_out->data
                                   + (size_t)sout.n_tokens * sv_h;
                size_t sz = (size_t)head_v * (size_t)head_v * (size_t)dt_rank * sizeof(float);
                memcpy(ssm_recurrent_states_[si].data(), src, sz);
            }
        }
    }

    // --------------------------------------------------
    // 7. Save KV cache entries
    // --------------------------------------------------
    {
        for (int il = 0; il < (int)cfg.block_count; il++) {
            if (!is_attn_layer(il)) continue;
            int ai = attn_cache_idx(il);
            auto& kout = tmp_kv_outs_[ai];

            if (kout.k_new && kout.k_new->data) {
                // k_new shape (contiguous): [head_dim, n_tokens, n_kv_heads, 1]
                // Buffer layout:            [head_dim, max_seq_len, n_kv_heads]
                // Stride of h in k_new:     head_dim * n_tokens
                // Stride of h in buffer:    head_dim * max_seq_len
                const float* src = (const float*)kout.k_new->data;
                float* buf = kv_caches_[ai].k.data();
                for (int64_t h = 0; h < n_kv_h; h++) {
                    memcpy(buf + h * head_dim * max_seq_len_ + (size_t)pos * head_dim,
                           src + h * head_dim * n_tokens,
                           (size_t)head_dim * n_tokens * sizeof(float));
                }
            }
            if (kout.v_new && kout.v_new->data) {
                const float* src = (const float*)kout.v_new->data;
                float* buf = kv_caches_[ai].v.data();
                for (int64_t h = 0; h < n_kv_h; h++) {
                    memcpy(buf + h * head_dim * max_seq_len_ + (size_t)pos * head_dim,
                           src + h * head_dim * n_tokens,
                           (size_t)head_dim * n_tokens * sizeof(float));
                }
            }
            kv_caches_[ai].len = pos + n_tokens;
        }
    }

    // Advance position counter
    pos_ += n_tokens;

    // --------------------------------------------------
    // 8. Read logits for the last token
    // --------------------------------------------------
    std::vector<float> result(vocab_size);
    if (logits->data) {
        memcpy(result.data(), logits->data, vocab_size * sizeof(float));
    }

    ggml_free(ctx);
    return result;
}

// ============================================================
// build_graph()
// ============================================================
ggml_tensor* InferenceEngine::build_graph(ggml_context* ctx, ggml_cgraph* gf,
                                           ggml_tensor* inp_tokens, int n_tokens,
                                           int pos) {
    const auto& cfg = model_.config.qwen35moe;
    const auto& w   = model_.weights;

    const float rms_eps = cfg.layer_norm_rms_epsilon > 0.0f
                          ? cfg.layer_norm_rms_epsilon : 1e-6f;
    const int   n_layer = (int)cfg.block_count;

    // --------------------------------------------------
    // Token embedding: E[tokens] → [n_embd, n_tokens]
    // --------------------------------------------------
    struct ggml_tensor* cur = ggml_get_rows(ctx, w.token_embd, inp_tokens);
    ggml_set_name(cur, "token_embd");

    // --------------------------------------------------
    // Position tensor (I32) for RoPE (mrope needs 4 * n_tokens)
    // For text-only: all 4 sections use the same positions
    // --------------------------------------------------
    struct ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(inp_pos, "inp_pos");

    // --------------------------------------------------
    // Causal mask (F16) for flash attention.
    // Full-prompt (pos==0): [n_tokens, n_tokens] causal mask.
    // Incremental  (pos>0): [pos+1, 1] all-zero mask.
    // --------------------------------------------------
    struct ggml_tensor* kq_mask;
    if (pos == 0) {
        kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_tokens, n_tokens);
    } else {
        kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, pos + n_tokens, n_tokens);
    }
    ggml_set_name(kq_mask, "kq_mask");

    // --------------------------------------------------
    // Layer loop
    // --------------------------------------------------
    for (int il = 0; il < n_layer; il++) {
        const auto& lyr = w.layers[il];
        if (!lyr.attn_norm) continue; // skip uninitialised layers (shouldn't happen)

        // Attention norm
        struct ggml_tensor* normed = rms_norm(ctx, cur, lyr.attn_norm, rms_eps);

        // Sub-layer: attention or SSM
        struct ggml_tensor* attn_out;
        if (is_attn_layer(il)) {
            attn_out = build_attn_layer(ctx, gf, normed, inp_pos, kq_mask, il, n_tokens, pos);
        } else {
            attn_out = build_ssm_layer(ctx, gf, normed, il, n_tokens, pos);
        }
        if (!attn_out) return nullptr;

        // Residual
        cur = ggml_add(ctx, cur, attn_out);

        // Post-attention norm
        if (!lyr.post_attention_norm) {
            fprintf(stderr, "[Inference] layer %d missing post_attention_norm\n", il);
            return nullptr;
        }
        struct ggml_tensor* ffn_in = rms_norm(ctx, cur, lyr.post_attention_norm, rms_eps);

        // MoE FFN
        struct ggml_tensor* ffn_out = build_moe_ffn(ctx, gf, ffn_in, il, n_tokens);
        if (!ffn_out) return nullptr;

        // Residual
        cur = ggml_add(ctx, cur, ffn_out);
    }

    // --------------------------------------------------
    // Final norm
    // --------------------------------------------------
    cur = rms_norm(ctx, cur, w.output_norm, rms_eps);

    // --------------------------------------------------
    // Extract last token: cur has shape [n_embd, n_tokens]
    // We only need the logits for the last position.
    // --------------------------------------------------
    struct ggml_tensor* last = ggml_view_1d(ctx, cur, (int64_t)cfg.embedding_length,
                                             (int64_t)(n_tokens - 1) * cfg.embedding_length *
                                             ggml_element_size(cur));
    ggml_set_name(last, "last_token_embd");

    // --------------------------------------------------
    // LM head: output.weight [n_embd, vocab_size]
    // --------------------------------------------------
    struct ggml_tensor* logits = ggml_mul_mat(ctx, w.output, last);
    ggml_set_name(logits, "logits");

    return logits;
}

// ============================================================
// build_attn_layer() — Full GQA attention (layer % 4 == 3)
// ============================================================
ggml_tensor* InferenceEngine::build_attn_layer(ggml_context* ctx, ggml_cgraph* gf,
                                                ggml_tensor* cur, ggml_tensor* inp_pos,
                                                ggml_tensor* kq_mask, int il,
                                                int n_tokens, int pos) {
    const auto& cfg = model_.config.qwen35moe;
    const auto& lyr = model_.weights.layers[il];
    const float rms_eps = cfg.layer_norm_rms_epsilon > 0.0f ? cfg.layer_norm_rms_epsilon : 1e-6f;

    if (!lyr.attn_q || !lyr.attn_k || !lyr.attn_v || !lyr.attn_output) {
        fprintf(stderr, "[Inference] layer %d (attn): missing weights\n", il);
        return nullptr;
    }
    // Note: gf is passed for API consistency with other sub-builders but is
    // not needed in this function because all ops are connected via return value.
    (void)gf;
    const int64_t n_heads     = cfg.head_count;             // 16
    const int64_t n_kv_heads  = cfg.head_count_kv;          // 2
    const int64_t head_dim    = cfg.key_length;             // 256
    const int64_t rope_dim    = cfg.dimension_count;        // 64
    const float   freq_base   = cfg.freq_base > 0.0f ? cfg.freq_base : 1e7f;
    const float   kq_scale    = 1.0f / sqrtf((float)head_dim);

    // RoPE sections: [11, 11, 10, 0] from GGUF
    int sections[4] = {11, 11, 10, 0};
    for (int s = 0; s < 4 && s < (int)cfg.dimension_sections.size(); s++) {
        sections[s] = cfg.dimension_sections[s];
    }

    // --------------------------------------------------
    // Q projection: attn_q.weight [n_embd, n_heads * head_dim * 2]
    // (interleaved Q and gate per head)
    // --------------------------------------------------
    struct ggml_tensor* Qfull = ggml_mul_mat(ctx, lyr.attn_q, cur);
    // Qfull: [n_heads * head_dim * 2, n_tokens]
    // Reshape to [head_dim*2, n_heads, n_tokens] then extract Q and gate via strided view

    // Reshape: [head_dim*2, n_heads, n_tokens]
    Qfull = ggml_reshape_3d(ctx, Qfull, head_dim * 2, n_heads, n_tokens);

    // Q: stride = head_dim*2*sizeof(float), ne[0]=head_dim, offset=0
    struct ggml_tensor* Qcur = ggml_view_3d(ctx, Qfull,
        head_dim, n_heads, n_tokens,
        ggml_element_size(Qfull) * head_dim * 2,
        ggml_element_size(Qfull) * head_dim * 2 * n_heads,
        0);
    Qcur = ggml_cont(ctx, Qcur); // make contiguous for norm

    // Gate: offset = head_dim * sizeof(float)
    struct ggml_tensor* gate_q = ggml_view_3d(ctx, Qfull,
        head_dim, n_heads, n_tokens,
        ggml_element_size(Qfull) * head_dim * 2,
        ggml_element_size(Qfull) * head_dim * 2 * n_heads,
        ggml_element_size(Qfull) * head_dim);
    gate_q = ggml_cont_3d(ctx, gate_q, head_dim, n_heads, n_tokens);

    // Q RMS norm (per head)
    if (lyr.attn_q_norm) {
        Qcur = ggml_rms_norm(ctx, Qcur, rms_eps);
        Qcur = ggml_mul(ctx, Qcur, lyr.attn_q_norm);
    }

    // --------------------------------------------------
    // K projection: attn_k.weight [n_embd, n_kv_heads * head_dim]
    // --------------------------------------------------
    struct ggml_tensor* Kcur = ggml_mul_mat(ctx, lyr.attn_k, cur);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_kv_heads, n_tokens);

    if (lyr.attn_k_norm) {
        Kcur = ggml_rms_norm(ctx, Kcur, rms_eps);
        Kcur = ggml_mul(ctx, Kcur, lyr.attn_k_norm);
    }

    // --------------------------------------------------
    // V projection: attn_v.weight [n_embd, n_kv_heads * head_dim]
    // --------------------------------------------------
    struct ggml_tensor* Vcur = ggml_mul_mat(ctx, lyr.attn_v, cur);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_kv_heads, n_tokens);

    // --------------------------------------------------
    // RoPE on Q and K (multi-section rotary for Qwen3.5)
    // --------------------------------------------------
    const int rope_type = GGML_ROPE_TYPE_MROPE; // multi-section rotary for Qwen3.5
    Qcur = ggml_rope_multi(ctx, Qcur, inp_pos, nullptr,
                           (int)rope_dim, sections, rope_type,
                           (int)cfg.context_length, freq_base, 1.0f,
                           0.0f, 1.0f, 0.0f, 0.0f);

    Kcur = ggml_rope_multi(ctx, Kcur, inp_pos, nullptr,
                           (int)rope_dim, sections, rope_type,
                           (int)cfg.context_length, freq_base, 1.0f,
                           0.0f, 1.0f, 0.0f, 0.0f);

    // flash_attn_ext expects: q [head_dim, n_tokens, n_heads, 1]
    //                         k [head_dim, kv_len,  n_kv_heads, 1]
    //                         v [head_dim, kv_len,  n_kv_heads, 1]
    // Our current shape: [head_dim, n_heads/n_kv_heads, n_tokens] (3D)
    // Need to permute: [head_dim, n_tokens, n_heads, 1]
    Qcur = ggml_permute(ctx, Qcur, 0, 2, 1, 3); // [head_dim, n_tokens, n_heads, 1]
    Kcur = ggml_permute(ctx, Kcur, 0, 2, 1, 3); // [head_dim, n_tokens, n_kv_heads, 1]
    Vcur = ggml_cont(ctx, ggml_permute(ctx, Vcur, 0, 2, 1, 3));

    // Make K contiguous and save pointer for KV cache update
    Kcur = ggml_cont(ctx, Kcur);  // [head_dim, n_tokens, n_kv_heads, 1] contiguous

    // Save new K/V pointers so forward() can copy them into the KV cache
    int ai = attn_cache_idx(il);
    tmp_kv_outs_[ai].k_new = Kcur;
    tmp_kv_outs_[ai].v_new = Vcur;

    // --------------------------------------------------
    // Incremental KV cache: prepend cached K/V when pos > 0
    // --------------------------------------------------
    struct ggml_tensor* K_for_attn = Kcur;
    struct ggml_tensor* V_for_attn = Vcur;

    if (pos > 0) {
        // K_cache: leaf tensor [head_dim, pos, n_kv_heads, 1] filled from cache
        char nm_kc[64]; snprintf(nm_kc, sizeof(nm_kc), "kv_k_cache_%d", il);
        struct ggml_tensor* K_cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                          head_dim, pos, n_kv_heads, 1);
        ggml_set_name(K_cache, nm_kc);

        char nm_vc[64]; snprintf(nm_vc, sizeof(nm_vc), "kv_v_cache_%d", il);
        struct ggml_tensor* V_cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                          head_dim, pos, n_kv_heads, 1);
        ggml_set_name(V_cache, nm_vc);

        // Concat along dim 1 (sequence): [head_dim, pos+n_tokens, n_kv_heads, 1]
        K_for_attn = ggml_concat(ctx, K_cache, Kcur, 1);
        V_for_attn = ggml_cont(ctx, ggml_concat(ctx, V_cache, Vcur, 1));
    }

    // --------------------------------------------------
    // Flash attention (GQA: K/V broadcast from n_kv_heads to n_heads)
    // --------------------------------------------------
    struct ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, Qcur, K_for_attn, V_for_attn,
                                                        kq_mask, kq_scale, 0.0f, 0.0f);
    // attn_out: [head_dim, n_tokens, n_heads, 1]

    // Merge heads: [head_dim * n_heads, n_tokens]
    attn_out = ggml_cont_2d(ctx, attn_out, head_dim * n_heads, n_tokens);

    // --------------------------------------------------
    // Apply attention gate: sigmoid(gate_q) * attn_out
    // gate_q: [head_dim, n_heads, n_tokens] → [head_dim * n_heads, n_tokens]
    // --------------------------------------------------
    gate_q = ggml_cont_2d(ctx, gate_q, head_dim * n_heads, n_tokens);
    struct ggml_tensor* gate_sig = ggml_sigmoid(ctx, gate_q);
    attn_out = ggml_mul(ctx, attn_out, gate_sig);

    // --------------------------------------------------
    // Output projection: attn_output.weight [head_dim * n_heads, n_embd]
    // --------------------------------------------------
    struct ggml_tensor* out = ggml_mul_mat(ctx, lyr.attn_output, attn_out);
    // out: [n_embd, n_tokens]
    return out;
}

// ============================================================
// build_ssm_layer() — DeltaNet SSM (layer % 4 != 3)
// ============================================================
ggml_tensor* InferenceEngine::build_ssm_layer(ggml_context* ctx, ggml_cgraph* gf,
                                               ggml_tensor* cur, int il,
                                               int n_tokens, int pos) {
    (void)pos;  // pos not used directly in SSM; state carries history
    const auto& cfg = model_.config.qwen35moe;
    const auto& lyr = model_.weights.layers[il];
    const float rms_eps = cfg.layer_norm_rms_epsilon > 0.0f ? cfg.layer_norm_rms_epsilon : 1e-6f;

    if (!lyr.attn_qkv || !lyr.attn_gate || !lyr.ssm_out) {
        fprintf(stderr, "[Inference] layer %d (ssm): missing required weights\n", il);
        return nullptr;
    }

    // Dimensions
    const int64_t n_embd      = cfg.embedding_length;        // 2048
    const int64_t d_inner     = cfg.inner_size;              // 4096
    const int64_t d_state     = cfg.state_size;              // 128
    const int64_t n_group     = cfg.group_count;             // 16  (num_k_heads)
    const int64_t dt_rank     = cfg.time_step_rank;          // 32  (num_v_heads)
    const int64_t head_v_dim  = d_inner / dt_rank;           // 128 (head_v_dim)
    const int64_t conv_kernel  = cfg.conv_kernel;             // 4

    // qkv_channels = d_inner + 2 * n_group * d_state = 4096 + 2*16*128 = 8192
    const int64_t qkv_channels = d_inner + 2 * n_group * d_state;

    // --------------------------------------------------
    // QKV projection: attn_qkv.weight [n_embd, qkv_channels]
    // cur: [n_embd, n_tokens] → qkv: [qkv_channels, n_tokens]
    // --------------------------------------------------
    struct ggml_tensor* qkv = ggml_mul_mat(ctx, lyr.attn_qkv, cur);
    // qkv: [qkv_channels=8192, n_tokens]

    // --------------------------------------------------
    // Gate (Z) projection: attn_gate.weight [n_embd, d_inner]
    // --------------------------------------------------
    struct ggml_tensor* z = ggml_mul_mat(ctx, lyr.attn_gate, cur);
    // z: [d_inner=4096, n_tokens]

    // --------------------------------------------------
    // Alpha (time-step) projection: ssm_alpha.weight [n_embd, dt_rank]
    // --------------------------------------------------
    struct ggml_tensor* alpha = nullptr;
    if (lyr.ssm_alpha) {
        alpha = ggml_mul_mat(ctx, lyr.ssm_alpha, cur);
        // alpha: [dt_rank=32, n_tokens]

        // Add dt bias
        if (lyr.ssm_dt_b) {
            // ssm_dt_b: [dt_rank=32] → broadcast to [dt_rank, n_tokens]
            alpha = ggml_add(ctx, alpha, lyr.ssm_dt_b);
        }
        // softplus: log(1 + exp(x))
        alpha = ggml_softplus(ctx, alpha);
        // multiply by ssm_a: [dt_rank=32]
        if (lyr.ssm_a) {
            alpha = ggml_mul(ctx, alpha, lyr.ssm_a);
        }
        // alpha: [dt_rank=32, n_tokens] — this is the per-head decay gate
    }

    // --------------------------------------------------
    // Beta projection: ssm_beta.weight [n_embd, dt_rank]
    // --------------------------------------------------
    struct ggml_tensor* beta = nullptr;
    if (lyr.ssm_beta) {
        beta = ggml_mul_mat(ctx, lyr.ssm_beta, cur);
        // beta: [dt_rank=32, n_tokens]
        beta = ggml_sigmoid(ctx, beta);
    }

    // --------------------------------------------------
    // Conv: persistent conv state pad + apply 1D depthwise conv over the QKV channels
    // conv1d.weight: [conv_kernel=4, qkv_channels=8192]
    // --------------------------------------------------
    // Reshape qkv: [qkv_channels, n_tokens] → [qkv_channels, n_tokens, 1]
    struct ggml_tensor* qkv_3d = ggml_reshape_3d(ctx, qkv, qkv_channels, n_tokens, 1);

    // Transpose dim 0 and 1: [n_tokens, qkv_channels, 1]
    qkv_3d = ggml_permute(ctx, qkv_3d, 1, 0, 2, 3);
    qkv_3d = ggml_cont_3d(ctx, qkv_3d, n_tokens, qkv_channels, 1);

    // Persistent conv state pad: [conv_kernel-1, qkv_channels, 1] = [3, 8192, 1]
    // Filled from ssm_conv_states_[si] (zeros on first call).
    struct ggml_tensor* conv_pad = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                       conv_kernel - 1, qkv_channels, 1);
    {
        char nm[64];
        snprintf(nm, sizeof(nm), "ssm_conv_pad_%d", il);
        ggml_set_name(conv_pad, nm);
    }

    // Concatenate: [conv_kernel-1+n_tokens, qkv_channels, 1]
    struct ggml_tensor* conv_in = ggml_concat(ctx, conv_pad, qkv_3d, 0);

    // Save the new conv state = last conv_kernel-1 rows of conv_in
    // (i.e. the last n_tokens portion of qkv_3d shifted in, exactly what the
    //  next call needs as its conv pad).
    {
        // View of shape [conv_kernel-1, qkv_channels, 1] starting at row n_tokens
    // within conv_in (which has n_tokens rows from qkv_3d at the end).
    struct ggml_tensor* new_cs = ggml_view_3d(ctx, conv_in,
        conv_kernel - 1, qkv_channels, 1,
        conv_in->nb[1],    // stride between qkv_channels rows
        conv_in->nb[2],    // stride between batch dim
        (size_t)n_tokens * sizeof(float)); // skip the first n_tokens in dim-0
        new_cs = ggml_cont(ctx, new_cs);
        char nm[64]; snprintf(nm, sizeof(nm), "ssm_conv_state_out_%d", il);
        ggml_set_name(new_cs, nm);
        ggml_build_forward_expand(gf, new_cs);
        tmp_ssm_outs_[ssm_state_idx(il)].conv_state_out = new_cs;
    }

    // ggml_ssm_conv expects sx: [d_conv-1+n_t, d_inner, n_s]
    // But conv1d.weight is [conv_kernel, qkv_channels], so d_inner = qkv_channels
    struct ggml_tensor* conv_out = nullptr;
    if (lyr.ssm_conv1d) {
        conv_out = ggml_ssm_conv(ctx, conv_in, lyr.ssm_conv1d);
        // conv_out: [qkv_channels, n_tokens, 1]
        conv_out = ggml_silu(ctx, conv_out);
    } else {
        // Fallback: no conv, just use qkv directly
        conv_out = ggml_reshape_3d(ctx, qkv_3d, qkv_channels, n_tokens, 1);
    }
    // conv_out: [qkv_channels=8192, n_tokens, 1]

    // --------------------------------------------------
    // Split conv_out into Q, K, V
    //   q_offset = 0,                  q_size = n_group * d_state = 16*128 = 2048
    //   k_offset = n_group*d_state,    k_size = 2048
    //   v_offset = 2*n_group*d_state,  v_size = d_inner = 4096
    // --------------------------------------------------
    const int64_t qk_size = n_group * d_state;  // 2048

    // We need [head_k_dim, num_heads, n_tokens, 1] shaped tensors
    // q_conv: [d_state=128, n_group=16, n_tokens, 1]
    // k_conv: [d_state=128, n_group=16, n_tokens, 1]
    // v_conv: [head_v_dim=128, dt_rank=32, n_tokens, 1]

    const int64_t nb1    = ggml_row_size(conv_out->type, qkv_channels);

    struct ggml_tensor* q_conv = ggml_view_4d(ctx, conv_out,
        d_state, n_group, n_tokens, 1,
        ggml_row_size(conv_out->type, d_state),
        nb1,
        nb1 * n_tokens,
        0);  // offset 0

    struct ggml_tensor* k_conv = ggml_view_4d(ctx, conv_out,
        d_state, n_group, n_tokens, 1,
        ggml_row_size(conv_out->type, d_state),
        nb1,
        nb1 * n_tokens,
        ggml_row_size(conv_out->type, qk_size)); // offset after q

    struct ggml_tensor* v_conv = ggml_view_4d(ctx, conv_out,
        head_v_dim, dt_rank, n_tokens, 1,
        ggml_row_size(conv_out->type, head_v_dim),
        nb1,
        nb1 * n_tokens,
        ggml_row_size(conv_out->type, 2 * qk_size)); // offset after q+k

    // Make contiguous for l2_norm
    q_conv = ggml_cont(ctx, q_conv);
    k_conv = ggml_cont(ctx, k_conv);
    v_conv = ggml_cont(ctx, v_conv);

    // L2 normalize Q and K (per head row)
    q_conv = ggml_l2_norm(ctx, q_conv, rms_eps);
    k_conv = ggml_l2_norm(ctx, k_conv, rms_eps);

    // Repeat q and k from n_group=16 heads to dt_rank=32 heads
    // (each head group is broadcast 2x to match num_v_heads)
    q_conv = ggml_repeat_4d(ctx, q_conv, d_state, dt_rank, n_tokens, 1);
    k_conv = ggml_repeat_4d(ctx, k_conv, d_state, dt_rank, n_tokens, 1);

    // --------------------------------------------------
    // Gate and beta: reshape to [1, dt_rank, n_tokens, 1]
    // --------------------------------------------------
    struct ggml_tensor* gate_4d = nullptr;
    struct ggml_tensor* beta_4d = nullptr;

    if (alpha) {
        gate_4d = ggml_reshape_4d(ctx, alpha, 1, dt_rank, n_tokens, 1);
    } else {
        // Create zero gate if alpha weights are missing
        gate_4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, dt_rank, n_tokens, 1);
        char nm[64]; snprintf(nm, sizeof(nm), "gate_zero_%d", il);
        ggml_set_name(gate_4d, nm);
    }

    if (beta) {
        beta_4d = ggml_reshape_4d(ctx, beta, 1, dt_rank, n_tokens, 1);
    } else {
        // Create ones beta if beta weights are missing
        beta_4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, dt_rank, n_tokens, 1);
        char nm[64]; snprintf(nm, sizeof(nm), "beta_ones_%d", il);
        ggml_set_name(beta_4d, nm);
    }

    // --------------------------------------------------
    // Persistent SSM recurrent state: [head_v_dim, head_v_dim, dt_rank, 1]
    // Filled from ssm_recurrent_states_[si] (zeros on first call).
    // --------------------------------------------------
    struct ggml_tensor* ssm_state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                        head_v_dim, head_v_dim, dt_rank, 1);
    {
        char nm[64];
        snprintf(nm, sizeof(nm), "ssm_state_in_%d", il);
        ggml_set_name(ssm_state, nm);
    }

    // --------------------------------------------------
    // GDN (Gated Delta Net)
    // q,k: [d_state=128, dt_rank=32, n_tokens, 1]
    // v:   [head_v_dim=128, dt_rank=32, n_tokens, 1]
    // g:   [1, dt_rank=32, n_tokens, 1]
    // beta:[1, dt_rank=32, n_tokens, 1]
    // state: [head_v_dim=128, head_v_dim=128, dt_rank=32, 1]
    // --------------------------------------------------
    struct ggml_tensor* gdn_out = ggml_gated_delta_net(ctx,
        q_conv, k_conv, v_conv, gate_4d, beta_4d, ssm_state);
    // gdn_out: [head_v_dim*dt_rank=4096, n_tokens + head_v_dim, 1, 1]

    // Save gdn_out pointer so forward() can memcpy the updated recurrent state
    {
        int si = ssm_state_idx(il);
        tmp_ssm_outs_[si].gdn_out   = gdn_out;
        tmp_ssm_outs_[si].n_tokens  = n_tokens;
    }
    // Ensure gdn_out is computed (it's a source for gdn_view below, but let's be explicit)
    ggml_build_forward_expand(gf, gdn_out);

    // Extract output (first n_tokens "columns")
    // Shape: [4096, n_tokens]
    struct ggml_tensor* gdn_view = ggml_view_2d(ctx, gdn_out,
        head_v_dim * dt_rank, n_tokens,
        gdn_out->nb[1],
        0); // offset 0 (output comes first)
    gdn_view = ggml_cont_2d(ctx, gdn_view, head_v_dim * dt_rank, n_tokens);

    // Reshape to [head_v_dim, dt_rank, n_tokens, 1] for gated norm
    struct ggml_tensor* gdn_4d = ggml_reshape_4d(ctx, gdn_view, head_v_dim, dt_rank, n_tokens, 1);

    // --------------------------------------------------
    // Z gate: reshape to [head_v_dim, dt_rank, n_tokens, 1]
    // z: [d_inner=4096, n_tokens] → [head_v_dim=128, dt_rank=32, n_tokens, 1]
    // --------------------------------------------------
    struct ggml_tensor* z_4d = ggml_reshape_4d(ctx, z, head_v_dim, dt_rank, n_tokens, 1);

    // --------------------------------------------------
    // Gated norm: rms_norm(gdn_4d, ssm_norm) * silu(z_4d)
    // ssm_norm.weight: [head_v_dim=128]
    // --------------------------------------------------
    struct ggml_tensor* normed_gdn = gdn_4d;
    if (lyr.ssm_norm) {
        normed_gdn = ggml_rms_norm(ctx, gdn_4d, rms_eps);
        normed_gdn = ggml_mul(ctx, normed_gdn, lyr.ssm_norm);
    }
    struct ggml_tensor* z_silu = ggml_silu(ctx, z_4d);
    struct ggml_tensor* gated  = ggml_mul(ctx, normed_gdn, z_silu);

    // Reshape back: [d_inner=4096, n_tokens]
    struct ggml_tensor* gated_2d = ggml_reshape_2d(ctx, gated, d_inner, n_tokens);

    // --------------------------------------------------
    // Output projection: ssm_out.weight [d_inner=4096, n_embd=2048]
    // --------------------------------------------------
    struct ggml_tensor* out = ggml_mul_mat(ctx, lyr.ssm_out, gated_2d);
    // out: [n_embd=2048, n_tokens]

    return out;
}

// ============================================================
// build_moe_ffn() — MoE FFN + shared expert (all layers)
// ============================================================
ggml_tensor* InferenceEngine::build_moe_ffn(ggml_context* ctx, ggml_cgraph* gf,
                                              ggml_tensor* cur, int il, int n_tokens) {
    const auto& cfg = model_.config.qwen35moe;
    const auto& lyr = model_.weights.layers[il];

    if (!lyr.ffn_gate_inp) {
        fprintf(stderr, "[Inference] layer %d (moe): missing ffn_gate_inp\n", il);
        return nullptr;
    }

    const int64_t n_embd      = cfg.embedding_length;  // 2048
    const int64_t n_expert    = cfg.expert_count;       // 256
    const int64_t n_top_k     = cfg.expert_used_count;  // 8
    const int64_t ff_dim      = cfg.expert_feed_forward_length; // 512

    // --------------------------------------------------
    // Router: ffn_gate_inp.weight [n_embd, n_expert]
    // logits: [n_expert, n_tokens]
    // --------------------------------------------------
    struct ggml_tensor* logits = ggml_mul_mat(ctx, lyr.ffn_gate_inp, cur);
    // logits: [n_expert=256, n_tokens]

    // Softmax over experts for each token
    struct ggml_tensor* probs = ggml_soft_max(ctx, logits);
    // probs: [n_expert=256, n_tokens]

    // Top-k selection: returns [n_top_k=8, n_tokens] I32
    struct ggml_tensor* selected = ggml_argsort_top_k(ctx, probs, (int)n_top_k);
    // selected: [n_top_k=8, n_tokens] I32

    // --------------------------------------------------
    // Gather expert probabilities for the selected top-k experts.
    //
    // Use ggml_get_rows to gather the scalar probability for each selected
    // expert per token.  ggml_get_rows(a, b) indexes into a->ne[1] using
    // values from b, where a->ne[2] (batch dimension in source) must equal
    // b->ne[1] (batch dimension in indices).
    //
    // probs_3d: [1, n_expert=256, n_tokens]   (ne[0]=1 row-width scalar,
    //                                          ne[1]=n_expert is the indexed dim,
    //                                          ne[2]=n_tokens is the batch dim)
    // selected: [n_top_k=8, n_tokens]          (ne[1]=n_tokens matches a->ne[2])
    // weights:  [1, n_top_k, n_tokens, 1]      nelements = 1*n_top_k*n_tokens*1 = n_top_k*n_tokens ✓
    //
    // Semantics: weights[0, i, j] = probs_3d[0, selected[i, j], j]
    //          = probs[selected[i, j], j]  — the probability of the i-th
    //            selected expert for token j.
    // --------------------------------------------------
    struct ggml_tensor* probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);

    // weights: [1, n_top_k, n_tokens, 1]
    struct ggml_tensor* weights = ggml_get_rows(ctx, probs_3d, selected);

    // Normalize expert weights to sum to 1.
    // ggml_get_rows returns a 4D tensor; ensure contiguous before reshape.
    weights = ggml_cont(ctx, weights);
    struct ggml_tensor* weights_2d = ggml_reshape_2d(ctx, weights, n_top_k, n_tokens);
    // Sum along expert dim
    struct ggml_tensor* w_sum = ggml_sum_rows(ctx, weights_2d); // [1, n_tokens]
    // Clamp to avoid division by zero
    w_sum = ggml_clamp(ctx, w_sum, 6.103515625e-5f, INFINITY);
    // Normalize
    weights_2d = ggml_div(ctx, weights_2d, w_sum); // [n_top_k, n_tokens]
    weights = ggml_reshape_3d(ctx, weights_2d, 1, n_top_k, n_tokens);

    // Ensure weights are computed before MoE ops
    ggml_build_forward_expand(gf, weights);

    // Reshape cur for mul_mat_id: [n_embd, 1, n_tokens]
    struct ggml_tensor* cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

    // --------------------------------------------------
    // Expert computation via ggml_mul_mat_id
    // gate_exps: [n_embd, ff_dim, n_expert] → [ff_dim, n_top_k, n_tokens]
    // up_exps:   [n_embd, ff_dim, n_expert] → [ff_dim, n_top_k, n_tokens]
    // --------------------------------------------------
    struct ggml_tensor* gate_out = ggml_mul_mat_id(ctx, lyr.ffn_gate_exps, cur_3d, selected);
    struct ggml_tensor* up_out   = ggml_mul_mat_id(ctx, lyr.ffn_up_exps,   cur_3d, selected);
    // gate_out, up_out: [ff_dim=512, n_top_k=8, n_tokens]

    // SwiGLU activation: silu(gate) * up
    struct ggml_tensor* hidden = ggml_silu(ctx, gate_out);
    hidden = ggml_mul(ctx, hidden, up_out);
    // hidden: [ff_dim=512, n_top_k=8, n_tokens]

    // --------------------------------------------------
    // Down projection via ggml_mul_mat_id
    // down_exps: [ff_dim, n_embd, n_expert]
    // hidden: [ff_dim, n_top_k, n_tokens]
    // selected: [n_top_k, n_tokens]
    // --------------------------------------------------
    struct ggml_tensor* experts = ggml_mul_mat_id(ctx, lyr.ffn_down_exps, hidden, selected);
    // experts: [n_embd=2048, n_top_k=8, n_tokens]

    // Multiply by expert weights: [1, n_top_k, n_tokens]
    experts = ggml_mul(ctx, experts, weights);
    // experts: [n_embd=2048, n_top_k=8, n_tokens]

    // Sum over top-k experts: build views per expert slot, then add
    // Each view: [n_embd, n_tokens] with offset i * experts->nb[1]
    struct ggml_tensor* moe_out = ggml_view_2d(ctx, experts, n_embd, n_tokens,
                                                experts->nb[2], 0);
    ggml_build_forward_expand(gf, moe_out);

    for (int i = 1; i < (int)n_top_k; i++) {
        struct ggml_tensor* slot = ggml_view_2d(ctx, experts, n_embd, n_tokens,
                                                 experts->nb[2], i * experts->nb[1]);
        ggml_build_forward_expand(gf, slot);
        moe_out = ggml_add(ctx, moe_out, slot);
    }
    // moe_out: [n_embd, n_tokens]

    // --------------------------------------------------
    // Shared expert (present in all layers)
    // --------------------------------------------------
    if (lyr.ffn_up_shexp && lyr.ffn_gate_shexp && lyr.ffn_down_shexp) {
        // Shared expert FFN (SwiGLU)
        // ffn_gate_shexp: [n_embd, ff_shexp_dim]
        // ffn_up_shexp:   [n_embd, ff_shexp_dim]
        // ffn_down_shexp: [ff_shexp_dim, n_embd]
        struct ggml_tensor* sh_gate = ggml_mul_mat(ctx, lyr.ffn_gate_shexp, cur);
        struct ggml_tensor* sh_up   = ggml_mul_mat(ctx, lyr.ffn_up_shexp,   cur);
        struct ggml_tensor* sh_act  = ggml_mul(ctx, ggml_silu(ctx, sh_gate), sh_up);
        struct ggml_tensor* sh_out  = ggml_mul_mat(ctx, lyr.ffn_down_shexp, sh_act);
        // sh_out: [n_embd, n_tokens]

        // Apply shared expert gate (sigmoid)
        if (lyr.ffn_gate_inp_shexp) {
            struct ggml_tensor* sh_g = ggml_mul_mat(ctx, lyr.ffn_gate_inp_shexp, cur);
            sh_g   = ggml_sigmoid(ctx, sh_g);
            sh_out = ggml_mul(ctx, sh_out, sh_g);
        }

        moe_out = ggml_add(ctx, moe_out, sh_out);
    }

    return moe_out;
}