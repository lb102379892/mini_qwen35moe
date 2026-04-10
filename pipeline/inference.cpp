// pipeline/inference.cpp
// Qwen3.5 MoE forward-pass implementation using ggml.
#include "pipeline/inference.hpp"
#include "compute/ops_cpu.hpp"
#include <ggml.h>
#include <ggml-cpu.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr size_t SZF32 = sizeof(float);

static inline bool is_attn_layer(int L) { return (L % 4) == 3; }

static void dump_t(const char * name, ggml_tensor * t) {
    printf("[%s] type=%d ne=[%lld %lld %lld %lld] nb=[%lld %lld %lld %lld] name=%s\n",
        name,
        (int)t->type,
        t->ne[0], t->ne[1], t->ne[2], t->ne[3],
        (long long)t->nb[0], (long long)t->nb[1], (long long)t->nb[2], (long long)t->nb[3],
        t->name
    );
}

static ggml_tensor* mul_dbg(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, const char* tag) {
    if (a->type != b->type) {
        printf("\n[MUL TYPE MISMATCH] %s\n", tag);
        dump_t("A", a);
        dump_t("B", b);
        printf("\n");
    }
    return ggml_mul(ctx, a, b);
}

ggml_tensor* ops_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    ggml_tensor* cur = ggml_rms_norm(ctx, x, eps);
    if (weight) {
        cur = mul_dbg(ctx, cur, weight, "ops_rms_norm: cur * weight");
    }
    return cur;
}

ggml_tensor* ops_layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, float eps) {
    ggml_tensor* cur = ggml_norm(ctx, x, eps);
    if (weight) {
        cur = mul_dbg(ctx, cur, weight, "ops_layer_norm: cur * weight");
    }
    if (bias) {
        cur = ggml_add(ctx, cur, bias);
    }
    return cur;
}

ggml_tensor* ops_swiglu_ffn(ggml_context* ctx, ggml_tensor* x,
                              ggml_tensor* w_gate, ggml_tensor* w_up, ggml_tensor* w_down) {
    ggml_tensor* gate   = ggml_mul_mat(ctx, w_gate, x);
    ggml_tensor* up     = ggml_mul_mat(ctx, w_up,   x);
    ggml_tensor* hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    return ggml_mul_mat(ctx, w_down, hidden);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool Qwen35moeInference::init(const Qwen35moeModel& model,
                               int max_ctx, int n_threads) {
    model_     = &model;
    cfg_       = &model.config.qwen35moe;
    n_layers_  = (int)cfg_->block_count;
    vocab_size_ = (int)model.config.tokenizer.ggml_tokens.size();
    max_ctx_   = max_ctx;
    n_threads_ = n_threads;

    if (n_layers_ == 0) {
        printf("[Inference] ERROR: block_count is 0\n");
        return false;
    }

    if (!alloc_kv_cache()) return false;
    if (!alloc_ssm_states()) return false;

    // Pre-allocate scratch buffer: 512 MB for all intermediate tensors
    // (increased from 256 MB to accommodate larger SSM intermediate tensors)
    compute_buf_.resize(512ull * 1024 * 1024);

    printf("[Inference] Initialised: layers=%d, max_ctx=%d, threads=%d, \
           embed=%u, heads=%u/%u, head_dim=%u\n",
           n_layers_, max_ctx_, n_threads_,
           cfg_->embedding_length, cfg_->head_count, cfg_->head_count_kv,
           cfg_->key_length);
    return true;
}

std::vector<float> Qwen35moeInference::forward(int token_id, int pos) {
    if (!model_) {
        printf("[Inference] ERROR: not initialised\n");
        return {};
    }
    if (pos >= max_ctx_) {
        printf("[Inference] ERROR: pos %d >= max_ctx %d\n", pos, max_ctx_);
        return {};
    }

    struct ggml_init_params p = {
        /* .mem_size   = */ compute_buf_.size(),
        /* .mem_buffer = */ compute_buf_.data(),
        /* .no_alloc   = */ false,
    };
    struct ggml_context* ctx = ggml_init(p);

    // Input token (1 int32)
    struct ggml_tensor* inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ((int32_t*)inp->data)[0] = token_id;

    // Build compute graph
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, /*grads=*/false);
    struct ggml_tensor* logits = build_graph(ctx, gf, inp, pos);
    if (!logits) {
        printf("[Inference] ERROR: build_graph failed (returned nullptr)\n");
        ggml_free(ctx);
        return {};
    }
    // 关键：确保 logits 连续
    logits = ggml_cont(ctx, logits);
    ggml_build_forward_expand(gf, logits);

    //ggml_graph_print(gf);

    // Execute on CPU
    ggml_graph_compute_with_ctx(ctx, gf, n_threads_);

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(logits) == vocab_size_);
    GGML_ASSERT(logits->nb[0] == sizeof(float));

    // Copy logits out
    // logits tensor is F32 by construction
    std::vector<float> result(vocab_size_);
    memcpy(result.data(), logits->data, (size_t)vocab_size_ * SZF32);

    // Debug: print top-5 logits for the first few positions
    if (pos < 5) {
        const float* data = result.data();
        int top_n = std::min(5, vocab_size_);
        std::vector<int> idx(vocab_size_);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(),
                          [&](int a, int b){ return data[a] > data[b]; });
        printf("[DBG_LOGITS] pos=%d top5: ", pos);
        for (int i = 0; i < top_n; i++) printf("[id=%d val=%.3f] ", idx[i], data[idx[i]]);
        printf("\n");
        fflush(stdout);
    }

    ggml_free(ctx);
    return result;
}

// ---------------------------------------------------------------------------
// KV cache management
// ---------------------------------------------------------------------------

bool Qwen35moeInference::alloc_kv_cache() {
    free_kv_cache();

    int n_attn = 0;
    for (int L = 0; L < n_layers_; L++) {
        if (is_attn_layer(L)) n_attn++;
    }

    int head_dim   = (int)cfg_->key_length;    // 256
    int n_kv_heads = (int)cfg_->head_count_kv; // 2

    // KV cache layout per attention layer: [head_dim, max_ctx, n_kv_heads]
    // This matches ggml_flash_attn_ext's required K/V layout: [head_dim, n_kv, n_kv_heads]
    // Strides: nb[0]=F32, nb[1]=head_dim*F32 (per position), nb[2]=head_dim*max_ctx*F32 (per kv-head)
    kv_.resize(n_attn);
    for (int i = 0; i < n_attn; i++) {
        size_t bytes_per_kv = (size_t)head_dim * max_ctx_ * n_kv_heads * SZF32;
        size_t buf_size = 2 * bytes_per_kv + 8 * ggml_tensor_overhead();

        struct ggml_init_params p = {
            /* .mem_size   = */ buf_size,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        kv_[i].ctx = ggml_init(p);
        if (!kv_[i].ctx) {
            printf("[Inference] ERROR: KV cache alloc failed for attn layer %d\n", i);
            return false;
        }
        // [head_dim, max_ctx, n_kv_heads] F32
        // ne[0]=head_dim (fastest), ne[1]=max_ctx, ne[2]=n_kv_heads
        kv_[i].k = ggml_new_tensor_3d(kv_[i].ctx, GGML_TYPE_F32,
                                       head_dim, max_ctx_, n_kv_heads);
        kv_[i].v = ggml_new_tensor_3d(kv_[i].ctx, GGML_TYPE_F32,
                                       head_dim, max_ctx_, n_kv_heads);
        memset(kv_[i].k->data, 0, ggml_nbytes(kv_[i].k));
        memset(kv_[i].v->data, 0, ggml_nbytes(kv_[i].v));
    }
    printf("[Inference] KV cache: %d attn layers × %zu MB each\n",
           n_attn,
           2 * (size_t)head_dim * max_ctx_ * n_kv_heads * SZF32 / (1024*1024));
    return true;
}

void Qwen35moeInference::free_kv_cache() {
    for (auto& kv : kv_) {
        if (kv.ctx) { ggml_free(kv.ctx); kv.ctx = nullptr; }
    }
    kv_.clear();
}

// ---------------------------------------------------------------------------
// SSM state management (DeltaNet recurrent state + conv1d sliding window)
// ---------------------------------------------------------------------------

bool Qwen35moeInference::alloc_ssm_states() {
    free_ssm_states();

    int n_ssm = 0;
    for (int L = 0; L < n_layers_; L++) {
        if (!is_attn_layer(L)) n_ssm++;
    }

    // SSM dimensions from config:
    // In the DeltaNet architecture:
    //   time_step_rank = number of value heads (num_v_heads = 32)
    //   group_count    = number of key heads (num_k_heads = 16)
    //   state_size     = key/value head dimension (head_k_dim = head_v_dim = 128)
    //   inner_size     = total hidden dimension = num_v_heads * head_v_dim = 4096
    int num_v_heads    = (int)cfg_->time_step_rank;   // 32  (= ssm.time_step_rank in GGUF)
    int inner_size     = (int)cfg_->inner_size;        // 4096
    int head_v_dim     = inner_size / num_v_heads;     // 128
    int num_k_heads    = (int)cfg_->group_count;       // 16
    int head_k_dim     = (int)cfg_->state_size;        // 128
    int conv_kernel    = (int)cfg_->conv_kernel;       // 4
    int conv_channels  = inner_size + 2 * num_k_heads * head_k_dim; // 4096 + 2*16*128 = 8192

    // Per-layer sizes:
    //   state:    [head_v_dim*head_v_dim, num_v_heads] = [16384, 32]  F32
    //   conv_buf: [conv_kernel-1, conv_channels]        = [3,    8192] F32
    size_t state_bytes    = (size_t)head_v_dim * head_v_dim * num_v_heads * SZF32;
    size_t conv_buf_bytes = (size_t)(conv_kernel - 1) * conv_channels * SZF32;
    size_t buf_per_layer  = state_bytes + conv_buf_bytes + 8 * ggml_tensor_overhead();

    ssm_states_.resize(n_ssm);
    for (int i = 0; i < n_ssm; i++) {
        struct ggml_init_params p = {
            /* .mem_size   = */ buf_per_layer,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        ssm_states_[i].ctx = ggml_init(p);
        if (!ssm_states_[i].ctx) {
            printf("[Inference] ERROR: SSM state alloc failed for SSM layer %d\n", i);
            return false;
        }
        // state: [head_v_dim*head_v_dim, num_v_heads] F32
        ssm_states_[i].state = ggml_new_tensor_2d(ssm_states_[i].ctx, GGML_TYPE_F32,
                                                    (int64_t)head_v_dim * head_v_dim,
                                                    num_v_heads);
        // conv_buf: [conv_kernel-1, conv_channels] F32
        // ne[0] = conv_kernel-1 (fastest-varying: conv position k)
        // ne[1] = conv_channels (slower-varying: channel index)
        ssm_states_[i].conv_buf = ggml_new_tensor_2d(ssm_states_[i].ctx, GGML_TYPE_F32,
                                                       conv_kernel - 1,
                                                       conv_channels);
        memset(ssm_states_[i].state->data,    0, ggml_nbytes(ssm_states_[i].state));
        memset(ssm_states_[i].conv_buf->data, 0, ggml_nbytes(ssm_states_[i].conv_buf));
    }
    printf("[Inference] SSM states: %d SSM layers × (state=%.1f MB + conv_buf=%.1f KB)\n",
           n_ssm,
           (float)state_bytes / (1024*1024),
           (float)conv_buf_bytes / 1024);
    return true;
}

void Qwen35moeInference::free_ssm_states() {
    for (auto& s : ssm_states_) {
        if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    }
    ssm_states_.clear();
}

// ---------------------------------------------------------------------------
// build_graph: full forward pass
// ---------------------------------------------------------------------------

struct ggml_tensor* Qwen35moeInference::build_graph(
        struct ggml_context* ctx,
        struct ggml_cgraph*  gf,
        struct ggml_tensor*  inp,
        int                  pos) {

    const Qwen35moeWeights& w = model_->weights;
    float eps = cfg_->layer_norm_rms_epsilon;

    // 1. Token embedding: [embed_dim, vocab] row lookup → [embed_dim, 1]
    struct ggml_tensor* cur = ggml_get_rows(ctx, w.token_embd, inp);
    // Squeeze to 1D [embed_dim]
    cur = ggml_reshape_1d(ctx, cur, cur->ne[0]);

    // Debug: log input token id and position
    printf("[DBG_GRAPH] pos=%d token_id=%d\n", pos, ((int32_t*)inp->data)[0]);

    int attn_ord = 0;
    int ssm_ord  = 0;

    for (int L = 0; L < n_layers_; L++) {
        const Qwen35moeLayer& layer = w.layers[L];

        if (pos == 0) {
            printf("[DBG_LAYER] L=%d type=%s\n", L, is_attn_layer(L) ? "ATTN" : "SSM");
        }

        // 2a. Pre-attn/SSM RMSNorm
        struct ggml_tensor* normed = ops_rms_norm(ctx, cur, layer.attn_norm, eps);

        // 2b. Attn or SSM sublayer
        struct ggml_tensor* sublayer_out;
        if (is_attn_layer(L)) {
            sublayer_out = build_attn_layer(ctx, gf, normed, L, attn_ord, pos);
            if (!sublayer_out) {
                printf("[Inference] ERROR: build_attn_layer returned nullptr for layer %d\n", L);
                return nullptr;
            }
            attn_ord++;
        } else {
            sublayer_out = build_ssm_layer(ctx, gf, normed, L, ssm_ord);
            ssm_ord++;
        }

        // Residual
        cur = ggml_add(ctx, cur, sublayer_out);

        // 2c. Post-attn RMSNorm
        struct ggml_tensor* normed2 = ops_rms_norm(ctx, cur, layer.post_attention_norm, eps);

        // 2d. MoE FFN
        struct ggml_tensor* ffn_out = build_moe_ffn(ctx, gf, normed2, L);

        // Residual
        cur = ggml_add(ctx, cur, ffn_out);
    }

    // 3. Final RMSNorm
    cur = ops_rms_norm(ctx, cur, w.output_norm, eps);

    // 4. LM head → [vocab_size]
    // output.weight: [embed_dim, vocab_size]
    struct ggml_tensor* logits = ggml_mul_mat(ctx, w.output, cur);
    // logits is [vocab_size, 1] from mul_mat; reshape to 1D
    int64_t vsz = logits->ne[0] * logits->ne[1];
    logits = ggml_reshape_1d(ctx, logits, vsz);

    return logits;
}

// ---------------------------------------------------------------------------
// build_attn_layer: GQA attention with RoPE and KV cache
//
// Tensor layouts (ggml ne[0..3], ne[0] is fastest-varying):
//   Q:   [head_dim, 1, n_q_heads]    after reshape
//   K,V: [head_dim, 1, n_kv_heads]   after reshape
//   KV cache (k/v): [head_dim, max_ctx, n_kv_heads]
//   flash_attn_ext inputs:
//     q:  [n_embd_k, n_batch, n_head,    1]  = [head_dim, 1, n_q_heads, 1]
//     k:  [n_embd_k, n_kv,    n_head_kv, 1]  = [head_dim, pos+1, n_kv_heads, 1]
//     v:  [n_embd_v, n_kv,    n_head_kv, 1]  = [head_dim, pos+1, n_kv_heads, 1]
//   Result: [head_dim, 1, n_q_heads]  → flatten to [head_dim * n_q_heads]
// ---------------------------------------------------------------------------

struct ggml_tensor* Qwen35moeInference::build_attn_layer(
        struct ggml_context* ctx,
        struct ggml_cgraph*  gf,
        struct ggml_tensor*  normed,
        int                  layer_idx,
        int                  attn_ord,
        int                  pos) {

    const Qwen35moeLayer& lw = model_->weights.layers[layer_idx];
    float eps = cfg_->layer_norm_rms_epsilon;

    // Guard: all four tensors are required for an attention layer.
    // If any is missing, report an error and return nullptr to the caller.
    if (!lw.attn_q || !lw.attn_k || !lw.attn_v || !lw.attn_output) {
        printf("[Inference] ERROR: layer %d is an attention layer but is missing required \
               tensors: attn_q=%s attn_k=%s attn_v=%s attn_output=%s\n",
               layer_idx,
               lw.attn_q      ? "OK" : "nullptr",
               lw.attn_k      ? "OK" : "nullptr",
               lw.attn_v      ? "OK" : "nullptr",
               lw.attn_output ? "OK" : "nullptr");
        return nullptr;
    }

    int embed_dim  = (int)cfg_->embedding_length;
    int n_q_heads  = (int)cfg_->head_count;          // 16
    int n_kv_heads = (int)cfg_->head_count_kv;        // 2
    int head_dim   = (int)cfg_->key_length;            // 256
    int rope_dim   = (int)cfg_->dimension_count;       // 64

    printf("[DBG_ATTN] layer=%d embed_dim=%d n_q_heads=%d n_kv_heads=%d head_dim=%d\n",
           layer_idx, embed_dim, n_q_heads, n_kv_heads, head_dim);
    if (lw.attn_q) {
        printf("[DBG_ATTN] attn_q shape: ne=[%lld %lld %lld]\n",
               lw.attn_q->ne[0], lw.attn_q->ne[1], lw.attn_q->ne[2]);
    }
    // RoPE frequency base: Qwen3.5 MoE uses 10 000 000 (10M) per the model card
    float freq_base = (cfg_->freq_base > 0.0f) ? cfg_->freq_base : 10000000.0f;

    // ggml_rope_ext extended parameters (YaRN-style, standard values for inference):
    //   freq_scale=1.0  (no frequency scaling)
    //   ext_factor=0.0  (no YaRN extrapolation)
    //   attn_factor=1.0 (no attention scale adjustment)
    //   beta_fast=32.0  (YaRN high-frequency boundary)
    //   beta_slow=1.0   (YaRN low-frequency boundary)

    // Q projection: [2048] → [8192] = (n_embd_head * 2) * n_head
    // The output contains [Q_head0, Gate_head0, Q_head1, Gate_head1, ...]
    // interleaved per head: each head contributes head_dim Q dims + head_dim gate dims.
    struct ggml_tensor* q_full = ggml_mul_mat(ctx, lw.attn_q, normed);

    // Extract Q with stride: pick first head_dim of every (2*head_dim) block
    // Shape: [head_dim, n_q_heads, 1] with nb1 = 2 * head_dim * sizeof(float)
    struct ggml_tensor* q = ggml_view_3d(ctx, q_full,
        head_dim, n_q_heads, 1,
        SZF32 * head_dim * 2,                          // stride between heads (skip Q+Gate)
        SZF32 * head_dim * 2 * (int64_t)n_q_heads,     // stride between tokens
        0);                                              // Q starts at offset 0
    q = ggml_cont(ctx, q);  // make contiguous

    // Extract Gate with stride: pick second head_dim of every (2*head_dim) block
    struct ggml_tensor* attn_gate = ggml_view_3d(ctx, q_full,
        head_dim, n_q_heads, 1,
        SZF32 * head_dim * 2,
        SZF32 * head_dim * 2 * (int64_t)n_q_heads,
        SZF32 * head_dim);                              // Gate starts at head_dim offset
    attn_gate = ggml_cont(ctx, attn_gate);
    // Flatten gate: [head_dim * n_q_heads]
    attn_gate = ggml_reshape_1d(ctx, attn_gate, (int64_t)head_dim * n_q_heads);

    // Reshape Q: [head_dim, 1, n_q_heads] for RoPE
    q = ggml_reshape_3d(ctx, q, head_dim, 1, n_q_heads);

    // K projection: [2048] → [512]
    struct ggml_tensor* k = ggml_mul_mat(ctx, lw.attn_k, normed);
    // Reshape: [head_dim, 1, n_kv_heads]
    k = ggml_reshape_3d(ctx, k, head_dim, 1, n_kv_heads);

    // V projection: [2048] → [512]
    struct ggml_tensor* v = ggml_mul_mat(ctx, lw.attn_v, normed);
    v = ggml_reshape_3d(ctx, v, head_dim, 1, n_kv_heads);

    // Per-head RMSNorm for Q and K (weight shape [head_dim] must broadcast over heads).
    // q: [head_dim, 1, n_q_heads]  — norm weight is [head_dim], reshape to [head_dim, 1, 1]
    // so ggml_mul broadcasts it correctly across all heads.
    if (lw.attn_q_norm) {
        q = ggml_rms_norm(ctx, q, eps);
        struct ggml_tensor* q_norm_w = ggml_reshape_3d(ctx, lw.attn_q_norm, head_dim, 1, 1);
        q = mul_dbg(ctx, q, q_norm_w, "attn: q * attn_q_norm");
    }
    if (lw.attn_k_norm) {
        k = ggml_rms_norm(ctx, k, eps);
        struct ggml_tensor* k_norm_w = ggml_reshape_3d(ctx, lw.attn_k_norm, head_dim, 1, 1);
        k = mul_dbg(ctx, k, k_norm_w, "attn: k * attn_k_norm");
    }

    // RoPE: position tensor must be [n_tokens] = [1] for single-token inference
    struct ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ((int32_t*)pos_t->data)[0] = pos;

    q = ggml_rope_ext(ctx, q, pos_t, nullptr,
                    rope_dim, 0, max_ctx_,
                    freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, pos_t, nullptr,
                    rope_dim, 0, max_ctx_,
                    freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    // ------------------------------------------------------------------
    // Update KV cache at position `pos`
    // ------------------------------------------------------------------
    // KV cache layout: [head_dim, max_ctx, n_kv_heads]
    //   nb[0] = SZF32
    //   nb[1] = head_dim * SZF32          (stride between KV positions)
    //   nb[2] = head_dim * max_ctx * SZF32 (stride between KV heads)

    size_t nb1_kv = (size_t)head_dim * SZF32;           // stride: positions
    size_t nb2_kv = (size_t)head_dim * max_ctx_ * SZF32; // stride: kv heads

    {
        // k_slot: view into cache for position `pos` — shape [head_dim, 1, n_kv_heads]
        struct ggml_tensor* k_slot = ggml_view_3d(ctx, kv_[attn_ord].k,
            head_dim, 1, n_kv_heads,
            nb1_kv,
            nb2_kv,
            (size_t)pos * nb1_kv);  // offset to position pos
        
        struct ggml_tensor* v_slot = ggml_view_3d(ctx, kv_[attn_ord].v,
            head_dim, 1, n_kv_heads,
            nb1_kv,
            nb2_kv,
            (size_t)pos * nb1_kv);

        // k and v are already [head_dim, 1, n_kv_heads] but may not be contiguous
        // after rope_ext; make contiguous before copy
        struct ggml_tensor* k_cont = ggml_cont(ctx, k);
        struct ggml_tensor* v_cont = ggml_cont(ctx, v);

        // Add copy operations to the graph (they run before the attention read)
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k_cont, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v_cont, v_slot));
    }

    // ------------------------------------------------------------------
    // Read K, V from cache for all positions 0..pos (inclusive)
    // ------------------------------------------------------------------
    // View: [head_dim, pos+1, n_kv_heads] — contiguous strides inherited from cache
    struct ggml_tensor* k_cache_view = ggml_view_3d(ctx, kv_[attn_ord].k,
        head_dim, pos + 1, n_kv_heads,
        nb1_kv,
        nb2_kv,
        0);  // from the start

    struct ggml_tensor* v_cache_view = ggml_view_3d(ctx, kv_[attn_ord].v,
        head_dim, pos + 1, n_kv_heads,
        nb1_kv,
        nb2_kv,
        0);

    // ------------------------------------------------------------------
    // Flash Attention (GQA: n_q_heads / n_kv_heads handled internally)
    // ------------------------------------------------------------------
    // float scale = 1.0f / sqrtf((float)head_dim);
    // struct ggml_tensor* attn_out =
    //     ggml_flash_attn_ext(ctx, q, k_cache_view, v_cache_view,
    //                         nullptr, scale, 0.0f, 0.0f);
    // // Result: [head_dim, 1, n_q_heads, 1] → flatten
    // attn_out = ggml_cont(ctx, attn_out);
    // int64_t flat_sz = attn_out->ne[0] * attn_out->ne[1] * attn_out->ne[2];
    // attn_out = ggml_reshape_1d(ctx, attn_out, flat_sz);

    // -------- Naive attention (debug correctness, single-token) --------
    // Make contiguous views
    ggml_tensor * q_c = ggml_cont(ctx, q);
    ggml_tensor * k_c = ggml_cont(ctx, k_cache_view);
    ggml_tensor * v_c = ggml_cont(ctx, v_cache_view);

    // q_c: [head_dim, 1, n_q_heads]
    // k_c: [head_dim, pos+1, n_kv_heads]
    // v_c: [head_dim, pos+1, n_kv_heads]

    // 1) Repeat kv heads to q heads (GQA)
    // We'll build k_rep/v_rep: [head_dim, pos+1, n_q_heads]
    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    ggml_tensor * k_rep = ggml_repeat(ctx, k_c, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, pos + 1, n_q_heads));
    ggml_tensor * v_rep = ggml_repeat(ctx, v_c, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, pos + 1, n_q_heads));

    // 2) Reshape to treat heads as "batch" for mul_mat
    // For scores we want: for each head h: (K_h^T [pos+1, head_dim]) @ (Q_h [head_dim, 1]) -> [pos+1, 1]
    // ggml_mul_mat expects A:[K, M], B:[K, N] -> [M, N].
    // So set A = K_h  with shape [head_dim, pos+1], B = Q_h with shape [head_dim, 1].

    ggml_tensor * q2 = ggml_reshape_3d(ctx, q_c, head_dim, 1, n_q_heads);              // [head_dim, 1, heads]
    ggml_tensor * k2 = ggml_reshape_3d(ctx, k_rep, head_dim, pos + 1, n_q_heads);       // [head_dim, pos+1, heads]

    // permute K to [head_dim, pos+1, heads] stays same for mul_mat (A's ne[0]=head_dim, ne[1]=pos+1)
    ggml_tensor * scores = ggml_mul_mat(ctx, k2, q2);  // -> [pos+1, 1, heads]

    // scale
    float scale = 1.0f / sqrtf((float)head_dim);
    scores = ggml_scale(ctx, scores, scale);

    // softmax over seq dimension (ne[0] = pos+1)
    ggml_tensor * probs = ggml_soft_max(ctx, scores);  // [pos+1, 1, heads]

    // 3) Compute weighted sum: V_h [head_dim, pos+1] @ probs_h [pos+1, 1] -> [head_dim, 1]
    //
    // We need A:[K, M] = probs_h with K=pos+1, M=1 ? No: mul_mat uses shared K at ne[0].
    // So we transpose V to make ne[0]=pos+1, then mul_mat with probs (ne[0]=pos+1).

    ggml_tensor * v2 = ggml_reshape_3d(ctx, v_rep, head_dim, pos + 1, n_q_heads);       // [head_dim, pos+1, heads]
    ggml_tensor * v_t = ggml_permute(ctx, v2, 1, 0, 2, 3);                               // [pos+1, head_dim, heads, 1] (but ggml treats missing dims as 1)
    v_t = ggml_cont(ctx, v_t);
    v_t = ggml_reshape_3d(ctx, v_t, pos + 1, head_dim, n_q_heads);                       // [pos+1, head_dim, heads]

    ggml_tensor * attn_t = ggml_mul_mat(ctx, v_t, probs);                                // [head_dim, 1, heads]

    // Now attn_t is [head_dim, 1, heads] which matches earlier flash output (except the extra dim)
    ggml_tensor * attn = ggml_cont(ctx, attn_t);
    int64_t flat_sz = attn->ne[0] * attn->ne[1] * attn->ne[2];
    ggml_tensor * attn_out = ggml_reshape_1d(ctx, attn, flat_sz);

    // Gate the attention output: attn_out = attn_out * sigmoid(gate)
    // The Q projection produced both Q and a gate tensor; the gate controls
    // the final attention contribution (like in Qwen3.5's gated attention).
    struct ggml_tensor* gate_sigmoid = ggml_sigmoid(ctx, attn_gate);
    attn_out = ggml_mul(ctx, attn_out, gate_sigmoid);

    // Output projection: [n_q_heads * head_dim, embed_dim] @ [flat_sz] → [embed_dim]
    struct ggml_tensor* out = ggml_mul_mat(ctx, lw.attn_output, attn_out);
    out = ggml_reshape_1d(ctx, out, embed_dim);

    return out;
}

// ---------------------------------------------------------------------------
// build_ssm_layer: Gated DeltaNet (recurrent SSM with persistent state)
//
// Reference: llama.cpp qwen35moe.cpp build_layer_attn_linear
//
// Algorithm (single token):
//  1. qkv_mixed = attn_qkv  @ normed          → [conv_channels=8192]
//  2. z         = attn_gate @ normed          → [inner_size=4096]
//  3. beta      = sigmoid(ssm_beta @ normed)  → [num_v_heads=32]
//  4. gate_a    = -abs(softplus(ssm_alpha @ normed + ssm_dt_b) * ssm_a)
//                                              → [num_v_heads=32]  (negative)
//  5. Conv1d with sliding window state (persistent conv_buf):
//     sx        = concat(conv_buf, qkv_mixed_row, dim=0) → [4, 8192]
//     conv_out  = ggml_ssm_conv(sx_3d, ssm_conv1d)       → [8192]
//     conv_silu = silu(conv_out)
//     Update conv_buf ← last 3 rows of sx
//  6. Split conv_silu: q_conv[2048], k_conv[2048], v_conv[4096]
//  7. L2-normalize q_conv, k_conv per head
//  8. Expand q_conv, k_conv from num_k_heads=16 to num_v_heads=32
//  9. DeltaNet recurrent update (persistent state):
//     S: [head_v_dim, head_v_dim, num_v_heads]
//     o = S @ k; corr = v - o; outer = k ⊗ corr
//     S_new = exp(gate_a) * S + beta * outer
//     output = S_new @ q   → [head_v_dim, num_v_heads]
//     Update state ← S_new
// 10. Gated RMS norm: rms_norm(output, ssm_norm) * silu(z)
// 11. Output projection: ssm_out @ flat → [embed_dim]
// ---------------------------------------------------------------------------

struct ggml_tensor* Qwen35moeInference::build_ssm_layer(
        struct ggml_context* ctx,
        struct ggml_cgraph*  gf,
        struct ggml_tensor*  normed,
        int                  layer_idx,
        int                  ssm_ord) {

    const Qwen35moeLayer& lw = model_->weights.layers[layer_idx];
    float eps = cfg_->layer_norm_rms_epsilon;

    // SSM dimensions
    int embed_dim      = (int)cfg_->embedding_length;   // 2048
    int inner_size     = (int)cfg_->inner_size;          // 4096
    int num_v_heads    = (int)cfg_->time_step_rank;      // 32
    int head_v_dim     = inner_size / num_v_heads;       // 128
    int num_k_heads    = (int)cfg_->group_count;         // 16
    int head_k_dim     = (int)cfg_->state_size;          // 128
    int conv_kernel    = (int)cfg_->conv_kernel;         // 4
    int conv_channels  = inner_size + 2 * num_k_heads * head_k_dim; // 8192

    // -----------------------------------------------------------------------
    // Step 1: Projections
    // -----------------------------------------------------------------------

    // qkv_mixed = attn_qkv @ normed → [conv_channels=8192]
    // attn_qkv: [embed_dim=2048, conv_channels=8192]
    struct ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, lw.attn_qkv, normed);
    qkv_mixed = ggml_reshape_1d(ctx, qkv_mixed, conv_channels);

    // z = attn_gate @ normed → [inner_size=4096]
    // attn_gate: [embed_dim=2048, inner_size=4096]
    struct ggml_tensor* z = ggml_mul_mat(ctx, lw.attn_gate, normed);
    z = ggml_reshape_1d(ctx, z, inner_size);

    // -----------------------------------------------------------------------
    // Step 3-4: Compute beta and gate_a
    // -----------------------------------------------------------------------

    // beta = sigmoid(ssm_beta @ normed) → [num_v_heads=32]
    // ssm_beta: [embed_dim=2048, num_v_heads=32]
    struct ggml_tensor* beta_raw = ggml_mul_mat(ctx, lw.ssm_beta, normed);
    beta_raw = ggml_reshape_1d(ctx, beta_raw, num_v_heads);
    struct ggml_tensor* beta = ggml_sigmoid(ctx, beta_raw);  // [32]

    // alpha = ssm_alpha @ normed → [num_v_heads=32]
    // ssm_alpha: [embed_dim=2048, num_v_heads=32]
    struct ggml_tensor* alpha = ggml_mul_mat(ctx, lw.ssm_alpha, normed);
    alpha = ggml_reshape_1d(ctx, alpha, num_v_heads);

    // alpha_biased = alpha + ssm_dt_b
    // ssm_dt_b: [num_v_heads=32] F32
    struct ggml_tensor* alpha_biased = ggml_add(ctx, alpha, lw.ssm_dt_b);

    // alpha_softplus = softplus(alpha_biased) → [32]
    struct ggml_tensor* alpha_sp = ggml_softplus(ctx, alpha_biased);

    // gate_a = -abs(alpha_softplus * ssm_a)
    // ssm_a: [num_v_heads=32] F32  (negative values: -A_log.exp())
    struct ggml_tensor* gate_a = ggml_mul(ctx, alpha_sp, lw.ssm_a);
    gate_a = ggml_neg(ctx, ggml_abs(ctx, gate_a));  // [32], negative

    // -----------------------------------------------------------------------
    // Step 5: Conv1d with persistent sliding-window state
    // -----------------------------------------------------------------------

    // Access persistent conv_buf: [conv_kernel-1, conv_channels] = [3, 8192]
    // ne[0]=3 (k position, fastest), ne[1]=8192 (channel)
    struct ggml_tensor* conv_buf = ssm_states_[ssm_ord].conv_buf;

    // qkv_row: reshape qkv_mixed [8192] → [ne0=1, ne1=8192]
    // ne[0]=1 = conv position (fastest-varying), ne[1]=8192 = channel axis.
    // This allows ggml_concat with conv_buf [ne0=3, ne1=8192] along dim=0
    // to produce sx [ne0=4, ne1=8192] where each channel gets 4 time-steps:
    //   sx[k=0..2, c] = conv_buf[k, c]  (history)
    //   sx[k=3,    c] = qkv_mixed[c]    (current token)
    struct ggml_tensor* qkv_row = ggml_reshape_2d(ctx, qkv_mixed, 1, conv_channels);  // ne=[1, 8192]

    // sx = concat(conv_buf [ne0=3,ne1=8192], qkv_row [ne0=1,ne1=8192], dim=0)
    // → [ne0=4, ne1=8192]: all 4 time-steps for each channel
    struct ggml_tensor* sx = ggml_concat(ctx, conv_buf, qkv_row, 0);

    // Make sx contiguous before ggml_ssm_conv (required by assertions)
    struct ggml_tensor* sx_cont = ggml_cont(ctx, sx);  // [4, 8192] contiguous

    // Reshape to 3D for ggml_ssm_conv: [d_conv-1+n_t, d_inner, n_s] = [4, 8192, 1]
    struct ggml_tensor* sx_3d = ggml_reshape_3d(ctx, sx_cont,
                                                  conv_kernel,          // ne[0] = 4
                                                  conv_channels,        // ne[1] = 8192
                                                  1);                    // ne[2] = 1 (single seq)

    // conv_out = ggml_ssm_conv(sx_3d, ssm_conv1d) → [conv_channels, 1, 1] = [8192, 1, 1]
    // ssm_conv1d: [conv_kernel=4, conv_channels=8192] F32
    struct ggml_tensor* conv_out_3d = ggml_ssm_conv(ctx, sx_3d, lw.ssm_conv1d);
    // Reshape to [8192]
    struct ggml_tensor* conv_out = ggml_reshape_1d(ctx, conv_out_3d, conv_channels);
    // Apply SiLU
    struct ggml_tensor* conv_silu = ggml_silu(ctx, conv_out);  // [8192]

    // Update persistent conv_buf with last (conv_kernel-1) rows of sx_cont
    // sx_cont is [4, 8192] with ne[0]=4. We want rows k=1,2,3 → new conv_buf [3, 8192].
    // View: ne=[3, 8192], nb[0]=sizeof(float), nb[1]=4*sizeof(float), offset=1*sizeof(float)
    // This gives: view[k', c] = sx_cont[1+k', c] for k'=0..2 ✓
    struct ggml_tensor* new_conv_buf_view = ggml_view_2d(ctx, sx_cont,
                                                           conv_kernel - 1,      // ne[0] = 3
                                                           conv_channels,         // ne[1] = 8192
                                                           (size_t)conv_kernel * SZF32,  // nb[1]: stride in bytes
                                                           SZF32);                // offset: skip k=0
    struct ggml_tensor* new_conv_buf = ggml_cont(ctx, new_conv_buf_view);  // [3, 8192] contiguous
    // Schedule write-back to persistent conv_buf
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_conv_buf, conv_buf));

    // -----------------------------------------------------------------------
    // Step 6: Split conv_silu into q_conv, k_conv, v_conv
    // -----------------------------------------------------------------------
    // Layout: [q(2048) | k(2048) | v(4096)]
    int qk_size = head_k_dim * num_k_heads;  // 128 * 16 = 2048
    int v_size  = head_v_dim * num_v_heads;  // 128 * 32 = 4096

    struct ggml_tensor* q_conv_raw = ggml_view_1d(ctx, conv_silu, qk_size,
                                                    0);
    struct ggml_tensor* k_conv_raw = ggml_view_1d(ctx, conv_silu, qk_size,
                                                    (size_t)qk_size * SZF32);
    struct ggml_tensor* v_conv     = ggml_view_1d(ctx, conv_silu, v_size,
                                                    (size_t)2 * qk_size * SZF32);

    // Reshape to [head_k_dim, num_k_heads] = [128, 16]
    struct ggml_tensor* q_conv = ggml_reshape_2d(ctx, q_conv_raw, head_k_dim, num_k_heads);
    struct ggml_tensor* k_conv = ggml_reshape_2d(ctx, k_conv_raw, head_k_dim, num_k_heads);
    // Reshape v to [head_v_dim, num_v_heads] = [128, 32]
    struct ggml_tensor* v_2d   = ggml_reshape_2d(ctx, v_conv, head_v_dim, num_v_heads);

    // -----------------------------------------------------------------------
    // Step 7: L2-normalize q_conv and k_conv per head (along ne[0] = head_k_dim)
    // -----------------------------------------------------------------------
    static constexpr float L2_EPS = 1e-12f;
    struct ggml_tensor* q_norm = ggml_l2_norm(ctx, q_conv, L2_EPS);  // [128, 16]
    struct ggml_tensor* k_norm = ggml_l2_norm(ctx, k_conv, L2_EPS);  // [128, 16]

    // -----------------------------------------------------------------------
    // Step 8: Expand q/k from num_k_heads=16 to num_v_heads=32
    // -----------------------------------------------------------------------
    struct ggml_tensor* q_tmpl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_k_dim, num_v_heads);
    struct ggml_tensor* k_tmpl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_k_dim, num_v_heads);
    struct ggml_tensor* q_exp  = ggml_repeat(ctx, q_norm, q_tmpl);  // [128, 32]
    struct ggml_tensor* k_exp  = ggml_repeat(ctx, k_norm, k_tmpl);  // [128, 32]

    // -----------------------------------------------------------------------
    // Step 9: DeltaNet recurrent update (persistent state)
    //
    // State S: [head_v_dim, head_v_dim, num_v_heads] = [128, 128, 32]
    //   S[j, i, h] = element at head h, row i, col j  (j = ne[0], fastest)
    //
    // For each head h:
    //   o_h    = S_h @ k_h                         (S_h=[128,128], k_h=[128])
    //   corr_h = v_h - o_h
    //   S_new_h = exp(gate_a[h]) * S_h + beta[h] * outer(corr_h, k_h)
    //   out_h   = S_new_h @ q_h
    //
    // Batched via ggml ops (all 32 heads in parallel).
    // ggml_mul supports broadcasting: b is repeated if ggml_can_repeat(b, a).
    // -----------------------------------------------------------------------

    // Access persistent state: [head_v_dim*head_v_dim, num_v_heads] = [128, 128, 32]
    struct ggml_tensor* state_persistent = ssm_states_[ssm_ord].state;

    // View as 3D: [head_v_dim, head_v_dim, num_v_heads] = [128, 128, 32]
    // This view IS contiguous (strides match the natural layout).
    // nb[0]=4, nb[1]=head_v_dim*4=512, nb[2]=head_v_dim^2*4=65536
    struct ggml_tensor* S = ggml_view_3d(ctx, state_persistent,
                                          head_v_dim,                              // ne[0] = col
                                          head_v_dim,                              // ne[1] = row
                                          num_v_heads,                             // ne[2] = head
                                          (size_t)head_v_dim * SZF32,              // nb[1]
                                          (size_t)head_v_dim * head_v_dim * SZF32, // nb[2]
                                          0);

    // Reshape k_exp, v_2d, q_exp to 3D: [head_v_dim, 1, num_v_heads]
    struct ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_exp,  head_v_dim, 1, num_v_heads);
    struct ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_2d,   head_v_dim, 1, num_v_heads);
    struct ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_exp,  head_v_dim, 1, num_v_heads);

    // --- o = S @ k → [128, 1, 32] ---
    // ggml_mul_mat(A=[128,128,32], B=[128,1,32]) → [128,1,32]
    // result[m,n,h] = sum_j A[j,m,h] * B[j,n,h] = (S_mat @ k)[m,h]
    struct ggml_tensor* o = ggml_mul_mat(ctx, S, k_3d);  // [128, 1, 32]

    // --- corr = v - o: [128, 1, 32] ---
    struct ggml_tensor* corr = ggml_sub(ctx, v_3d, o);

    // --- outer = k ⊗ corr: [128, 128, 32] ---
    // outer[col=m, row=n, h] = k[m, h] * corr[n, h]
    // Reshape k to [1, head_v_dim, num_v_heads] and corr to [1, head_v_dim, num_v_heads]
    struct ggml_tensor* k_col   = ggml_reshape_3d(ctx, k_exp,  1, head_v_dim, num_v_heads);
    struct ggml_tensor* corr_2d = ggml_reshape_2d(ctx, corr,   head_v_dim, num_v_heads);
    struct ggml_tensor* corr_col = ggml_reshape_3d(ctx, corr_2d, 1, head_v_dim, num_v_heads);
    // ggml_mul_mat(A=[1,128,32], B=[1,128,32]) → [128,128,32]
    // result[m,n,h] = A[0,m,h]*B[0,n,h] = k[m,h]*corr[n,h]
    struct ggml_tensor* outer = ggml_mul_mat(ctx, k_col, corr_col);  // [128, 128, 32]

    // --- S_new = exp(gate_a) * S + beta * outer ---
    // Use ggml_mul broadcasting: [1,1,32] broadcasts to [128,128,32]
    struct ggml_tensor* exp_gate = ggml_exp(ctx, gate_a);            // [32]
    struct ggml_tensor* eg_3d    = ggml_reshape_3d(ctx, exp_gate, 1, 1, num_v_heads); // [1,1,32]
    // ggml_mul(S=[128,128,32], eg_3d=[1,1,32]) broadcasts eg_3d (ggml_can_repeat is true)
    struct ggml_tensor* S_decayed = ggml_mul(ctx, S, eg_3d);         // [128, 128, 32]

    struct ggml_tensor* beta_3d = ggml_reshape_3d(ctx, beta, 1, 1, num_v_heads); // [1,1,32]
    // ggml_mul(outer=[128,128,32], beta_3d=[1,1,32]) broadcasts beta_3d
    struct ggml_tensor* outer_sc = ggml_mul(ctx, outer, beta_3d);   // [128, 128, 32]

    struct ggml_tensor* S_new = ggml_add(ctx, S_decayed, outer_sc);  // [128, 128, 32]

    // --- output = S_new @ q → [128, 1, 32] ---
    struct ggml_tensor* deltanet_out = ggml_mul_mat(ctx, S_new, q_3d);  // [128, 1, 32]

    // Write S_new back to persistent state via 3D target view (same shape as S_new)
    struct ggml_tensor* state_3d_target = ggml_view_3d(ctx, state_persistent,
                                                         head_v_dim,
                                                         head_v_dim,
                                                         num_v_heads,
                                                         (size_t)head_v_dim * SZF32,
                                                         (size_t)head_v_dim * head_v_dim * SZF32,
                                                         0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, S_new, state_3d_target));

    // -----------------------------------------------------------------------
    // Step 10: Gated RMS norm
    // -----------------------------------------------------------------------
    // Reshape output to [head_v_dim, num_v_heads] = [128, 32]
    struct ggml_tensor* out_2d = ggml_reshape_2d(ctx, deltanet_out, head_v_dim, num_v_heads);

    // RMS norm per head (over head_v_dim=128 dimension)
    struct ggml_tensor* out_normed = ggml_rms_norm(ctx, out_2d, eps);
    // Scale by ssm_norm weights: ssm_norm is [128] — broadcast over num_v_heads
    if (lw.ssm_norm) {
        out_normed = mul_dbg(ctx, out_normed, lw.ssm_norm, "ssm: out_normed * ssm_norm");
    }

    // z: [inner_size=4096] → reshape to [head_v_dim, num_v_heads] = [128, 32]
    struct ggml_tensor* z_2d = ggml_reshape_2d(ctx, z, head_v_dim, num_v_heads);
    // Gated output: out_normed * silu(z)
    struct ggml_tensor* output = ggml_mul(ctx, out_normed, ggml_silu(ctx, z_2d));

    // -----------------------------------------------------------------------
    // Step 11: Flatten and output projection
    // -----------------------------------------------------------------------
    output = ggml_reshape_1d(ctx, output, inner_size);
    // ssm_out: [inner_size=4096, embed_dim=2048]
    struct ggml_tensor* out = ggml_mul_mat(ctx, lw.ssm_out, output);
    out = ggml_reshape_1d(ctx, out, embed_dim);

    return out;
}