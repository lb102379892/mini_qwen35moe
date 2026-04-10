// pipeline/inference.cpp
// Qwen3.5 MoE forward-pass implementation using ggml.
//
#include "pipeline/inference.hpp"
#include "compute/ops_cpu.hpp"
#include <ggml.h>
#include <ggml-cpu.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>

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

    // Pre-allocate scratch buffer: generous 256 MB for all intermediate tensors
    compute_buf_.resize(256ull * 1024 * 1024);

    printf("[Inference] Initialised: layers=%d, max_ctx=%d, threads=%d, "
           "embed=%u, heads=%u/%u, head_dim=%u\n",
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
        // [head_dim, max_ctx, n_kv_heads]
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

    int attn_ord = 0;

    for (int L = 0; L < n_layers_; L++) {
        const Qwen35moeLayer& layer = w.layers[L];

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
            sublayer_out = build_ssm_layer(ctx, normed, L);
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
        printf("[Inference] ERROR: layer %d is an attention layer but is missing required "
               "tensors: attn_q=%s attn_k=%s attn_v=%s attn_output=%s\n",
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

    // Per-head RMSNorm for Q and K (weight broadcasts over heads)
    if (lw.attn_q_norm) {
        q = ggml_rms_norm(ctx, q, eps);
        q = mul_dbg(ctx, q, lw.attn_q_norm, "attn: q * attn_q_norm");
    }
    if (lw.attn_k_norm) {
        k = ggml_rms_norm(ctx, k, eps);
        k = mul_dbg(ctx, k, lw.attn_k_norm, "attn: k * attn_k_norm");
    }

    // RoPE: ggml 要求 b 是 int32 向量，长度必须等于 a->ne[2]（head 维）
    struct ggml_tensor* pos_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_q_heads);
    for (int i = 0; i < n_q_heads; ++i) {
        ((int32_t*)pos_q->data)[i] = pos;
    }

    struct ggml_tensor* pos_k = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv_heads);
    for (int i = 0; i < n_kv_heads; ++i) {
        ((int32_t*)pos_k->data)[i] = pos;
    }

    q = ggml_rope_ext(ctx, q, pos_q, nullptr,
                    rope_dim, 0, max_ctx_,
                    freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, pos_k, nullptr,
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
// build_ssm_layer: Gated DeltaNet (simplified approximation)
//
// Reference (llama.cpp qwen35moe.cpp):
//   qkv = attn_qkv @ normed    →  [8192]   (QKV mixed projection)
//   z   = attn_gate @ normed   →  [4096]   (gate for normalization)
//   qkv_act = conv1d(qkv) + silu   (conv1d with state, then silu)
//   q, k, v = split(qkv_act)   →  [2048], [2048], [4096]
//   output = delta_net(q, k, v, ...)  (recurrent SSM computation)
//   output = rms_norm(output, ssm_norm) * silu(z)
//   out = ssm_out @ output      →  [embed_dim]
//
// Simplified: without conv1d state and delta net, we approximate by
// applying silu to the QKV projection and using V directly.
// ---------------------------------------------------------------------------

struct ggml_tensor* Qwen35moeInference::build_ssm_layer(
        struct ggml_context* ctx,
        struct ggml_tensor*  normed,
        int                  layer_idx) {

    const Qwen35moeLayer& lw = model_->weights.layers[layer_idx];
    float eps = cfg_->layer_norm_rms_epsilon;

    int embed_dim   = (int)cfg_->embedding_length;   // 2048
    int inner_size  = (int)cfg_->inner_size;          // 4096
    int num_v_heads = (int)cfg_->time_step_rank;      // 32
    int head_v_dim  = inner_size / num_v_heads;        // 128

    // QKV mixed projection → [8192]
    struct ggml_tensor* qkv = ggml_mul_mat(ctx, lw.attn_qkv, normed);

    // Z (gate for normalization) → [4096]
    struct ggml_tensor* z = ggml_mul_mat(ctx, lw.attn_gate, normed);

    // Apply SiLU activation (approximates conv1d + silu in the full implementation)
    struct ggml_tensor* qkv_act = ggml_silu(ctx, qkv);

    // Extract V portion: the last inner_size elements
    // Full split would be: Q[0:2048], K[2048:4096], V[4096:8192]
    // Without delta net, we use V directly as the approximate output
    struct ggml_tensor* v = ggml_view_1d(ctx, qkv_act, inner_size,
                                          (size_t)inner_size * SZF32);

    // Gated normalization: rms_norm(v, ssm_norm) * silu(z)
    // Reshape for per-head normalization: [head_v_dim, num_v_heads]
    struct ggml_tensor* v_2d = ggml_reshape_2d(ctx, v, head_v_dim, num_v_heads);
    struct ggml_tensor* z_2d = ggml_reshape_2d(ctx, z, head_v_dim, num_v_heads);

    // RMS norm on each head_v_dim-sized vector, then scale by ssm_norm weight
    struct ggml_tensor* v_normed = ggml_rms_norm(ctx, v_2d, eps);
    if (lw.ssm_norm) {
        v_normed = mul_dbg(ctx, v_normed, lw.ssm_norm, "ssm: v_normed * ssm_norm");
    }

    // Gated output: normed_v * silu(z)
    struct ggml_tensor* output = ggml_mul(ctx, v_normed, ggml_silu(ctx, z_2d));

    // Flatten back to [inner_size]
    output = ggml_reshape_1d(ctx, output, inner_size);

    // Output projection → [embed_dim]
    struct ggml_tensor* out = ggml_mul_mat(ctx, lw.ssm_out, output);
    out = ggml_reshape_1d(ctx, out, embed_dim);

    return out;
}

// ---------------------------------------------------------------------------
// build_moe_ffn: Mixture-of-Experts FFN
//
// Sparse:
//   logits = ffn_gate_inp   @ cur                → [n_experts]
//   probs  = softmax(logits)
//   ids    = argsort_top_k(probs, k)              → [k] int32
//   weights= top_k(probs, k)                      → [k] float
//   input_tiled: repeat cur → [embed_dim, k]
//   gate_out = mul_mat_id(ffn_gate_exps, tiled, ids) → [ff_dim, k]
//   up_out   = mul_mat_id(ffn_up_exps,   tiled, ids) → [ff_dim, k]
//   hidden   = silu(gate_out) * up_out
//   down_out = mul_mat_id(ffn_down_exps, hidden, ids) → [embed_dim, k]
//   expert_out = sum_j( down_out[:, j] * weights[j] )
//
// Shared:
//   sh_scale = sigmoid( ffn_gate_inp_shexp ⋅ cur )   (scalar)
//   sh_out   = down_shexp( silu(gate_shexp @ cur) * up_shexp @ cur )
//   sh_out  *= sh_scale
//
// FFN = expert_out + sh_out
// ---------------------------------------------------------------------------

struct ggml_tensor* Qwen35moeInference::build_moe_ffn(
        struct ggml_context* ctx,
        struct ggml_cgraph*  gf,
        struct ggml_tensor*  cur,
        int                  layer_idx) {

    const Qwen35moeLayer& lw = model_->weights.layers[layer_idx];

    int embed_dim = (int)cfg_->embedding_length;
    int ff_dim    = (int)cfg_->expert_feed_forward_length;   // 512
    int n_experts = (int)cfg_->expert_count;                  // 256
    int expert_k  = (int)cfg_->expert_used_count;             // 8
    int ff_dim_sh = (int)cfg_->expert_shared_feed_forward_length; // 512

    // ------------------------------------------------------------------
    // Router
    // ------------------------------------------------------------------
    // ffn_gate_inp: [embed_dim, n_experts] (F32)
    struct ggml_tensor* router_logits = ggml_mul_mat(ctx, lw.ffn_gate_inp, cur);
    router_logits = ggml_reshape_1d(ctx, router_logits, n_experts);

    struct ggml_tensor* router_probs = ggml_soft_max(ctx, router_logits);

    // Top-k indices [k] and weights [k]
    struct ggml_tensor* top_k_ids   = ggml_argsort_top_k(ctx, router_probs, expert_k);
    struct ggml_tensor* top_k_probs = ggml_top_k(ctx, router_probs, expert_k);
    if (top_k_probs->type != GGML_TYPE_F32) {
        top_k_probs = ggml_cast(ctx, top_k_probs, GGML_TYPE_F32);
    }
    // Normalize top-k weights so they sum to 1.0
    // (softmax was over all 256 experts; after selecting top-8, renormalize)
    {
        struct ggml_tensor* wsum = ggml_sum(ctx, top_k_probs); // [1]
        struct ggml_tensor* wsum_bc = ggml_repeat(ctx, wsum,
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, expert_k));
        top_k_probs = ggml_div(ctx, top_k_probs, wsum_bc);
    }
    // ------------------------------------------------------------------
    // Sparse expert path (batched via mul_mat_id)
    // ------------------------------------------------------------------
    // Tile input so each of the k experts gets a copy: [embed_dim, k]
    struct ggml_tensor* cur_2d = ggml_reshape_2d(ctx, cur, embed_dim, 1);
    struct ggml_tensor* tmpl   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, expert_k);
    struct ggml_tensor* cur_tiled = ggml_repeat(ctx, cur_2d, tmpl);   // [embed_dim, k]

    // ids must be [k, 1] for mul_mat_id
    struct ggml_tensor* ids_2d = ggml_reshape_2d(ctx, top_k_ids, expert_k, 1);

    // Gate / up projections: [ff_dim, k]
    // ffn_gate_exps: [embed_dim, ff_dim, n_experts]
    struct ggml_tensor* gate_out =
        ggml_mul_mat_id(ctx, lw.ffn_gate_exps, cur_tiled, ids_2d);
    struct ggml_tensor* up_out =
        ggml_mul_mat_id(ctx, lw.ffn_up_exps,   cur_tiled, ids_2d);

    // SwiGLU activation: [ff_dim, k]
    struct ggml_tensor* hidden = mul_dbg(ctx, ggml_silu(ctx, gate_out), up_out, "silu:gete_out * up_out");

    // Down projection: [embed_dim, k]
    // ffn_down_exps: [ff_dim, embed_dim, n_experts]
    struct ggml_tensor* down_out =
        ggml_mul_mat_id(ctx, lw.ffn_down_exps, hidden, ids_2d);

    // Weighted sum: multiply each expert column by its router probability
    // top_k_probs [k] → broadcast to [embed_dim, k]
    struct ggml_tensor* probs_2d = ggml_reshape_2d(ctx, top_k_probs, 1, expert_k);
    struct ggml_tensor* probs_bc = ggml_repeat(ctx, probs_2d, down_out); // [embed_dim, k]
    struct ggml_tensor* scaled   = mul_dbg(ctx, down_out, probs_bc, "down_out * probs_bc");    // [embed_dim, k]

    // Accumulate the k expert outputs column by column
    struct ggml_tensor* expert_out =
        ggml_view_1d(ctx, scaled, embed_dim, 0);
    for (int j = 1; j < expert_k; j++) {
        struct ggml_tensor* col =
            ggml_view_1d(ctx, scaled, embed_dim, (size_t)j * embed_dim * SZF32);
        expert_out = ggml_add(ctx, expert_out, col);
    }

    // ------------------------------------------------------------------
    // Shared expert
    // ------------------------------------------------------------------
    // ffn_gate_inp_shexp: [embed_dim] F32 → scalar gate for shared expert
    // Reshape to a [1, embed_dim] "row" matrix so mul_mat gives a scalar
    struct ggml_tensor* shexp_row = ggml_reshape_2d(ctx, lw.ffn_gate_inp_shexp,
                                                     embed_dim, 1);
    struct ggml_tensor* sh_scale_raw = ggml_mul_mat(ctx, shexp_row, cur); // [1]
    sh_scale_raw = ggml_reshape_1d(ctx, sh_scale_raw, 1);
    struct ggml_tensor* sh_scale = ggml_sigmoid(ctx, sh_scale_raw);        // [1]

    // Shared expert SwiGLU
    struct ggml_tensor* sh_gate   = ggml_mul_mat(ctx, lw.ffn_gate_shexp, cur); // [ff_dim_sh]
    struct ggml_tensor* sh_up     = ggml_mul_mat(ctx, lw.ffn_up_shexp,   cur); // [ff_dim_sh]
    struct ggml_tensor* sh_hidden = mul_dbg(ctx, ggml_silu(ctx, sh_gate), sh_up, "silu:sh_gate * sh_up");
    struct ggml_tensor* sh_out    = ggml_mul_mat(ctx, lw.ffn_down_shexp, sh_hidden);
    sh_out = ggml_reshape_1d(ctx, sh_out, embed_dim);

    // Scale by shared-expert gate scalar (broadcast scalar over embed_dim)
    struct ggml_tensor* sh_scale_bc =
        ggml_repeat(ctx, sh_scale,
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim));
    sh_out = mul_dbg(ctx, sh_out, sh_scale_bc, "sh_out * sh_scale_bc");

    // ------------------------------------------------------------------
    // Combine
    // ------------------------------------------------------------------
    expert_out = ggml_reshape_1d(ctx, expert_out, embed_dim);
    return ggml_add(ctx, expert_out, sh_out);
}

