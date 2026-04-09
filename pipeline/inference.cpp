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
    ggml_build_forward_expand(gf, logits);

    // Execute on CPU
    ggml_graph_compute_with_ctx(ctx, gf, n_threads_);

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

    // Q projection: [2048] → [8192]; we use first 4096 = n_q_heads * head_dim
    struct ggml_tensor* q_full = ggml_mul_mat(ctx, lw.attn_q, normed);
    struct ggml_tensor* q = ggml_view_1d(ctx, q_full,
                                          (int64_t)n_q_heads * head_dim, 0);
    // Reshape: [head_dim, 1, n_q_heads] — flash_attn_ext requires ne[2]=n_q_heads
    q = ggml_reshape_3d(ctx, q, head_dim, 1, n_q_heads);

    // K projection: [2048] → [512]
    struct ggml_tensor* k = ggml_mul_mat(ctx, lw.attn_k, normed);
    // Reshape: [head_dim, 1, n_kv_heads]
    k = ggml_reshape_3d(ctx, k, head_dim, 1, n_kv_heads);

    // V projection: [2048] → [512]
    struct ggml_tensor* v = ggml_mul_mat(ctx, lw.attn_v, normed);
    v = ggml_reshape_3d(ctx, v, head_dim, 1, n_kv_heads);

    // Per-head RMSNorm for Q and K (weight broadcasts over heads)
    // q: [head_dim, 1, n_q_heads] — rms_norm normalises each row of head_dim
    if (lw.attn_q_norm) {
        q = ggml_rms_norm(ctx, q, eps);
        q = ggml_mul(ctx, q, lw.attn_q_norm);  // weight [head_dim] broadcasts
    }
    if (lw.attn_k_norm) {
        k = ggml_rms_norm(ctx, k, eps);
        k = ggml_mul(ctx, k, lw.attn_k_norm);
    }

    // RoPE
    struct ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ((int32_t*)pos_t->data)[0] = pos;

    q = ggml_rope_ext(ctx, q, pos_t, nullptr,
                      rope_dim, 0,
                      max_ctx_,
                      freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, pos_t, nullptr,
                      rope_dim, 0,
                      max_ctx_,
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
    float scale = 1.0f / sqrtf((float)head_dim);
    struct ggml_tensor* attn_out =
        ggml_flash_attn_ext(ctx, q, k_cache_view, v_cache_view,
                            nullptr, scale, 0.0f, 0.0f);
    // Result: [head_dim, 1, n_q_heads, 1] → flatten
    attn_out = ggml_cont(ctx, attn_out);
    int64_t flat_sz = attn_out->ne[0] * attn_out->ne[1] * attn_out->ne[2];
    attn_out = ggml_reshape_1d(ctx, attn_out, flat_sz);

    // Output projection: [n_q_heads * head_dim, embed_dim] @ [flat_sz] → [embed_dim]
    struct ggml_tensor* out = ggml_mul_mat(ctx, lw.attn_output, attn_out);
    out = ggml_reshape_1d(ctx, out, embed_dim);

    return out;
}

// ---------------------------------------------------------------------------
// build_ssm_layer: simplified gated-linear transform (Mamba2 approximation)
//
//   xz  = attn_qkv   @ normed  →  [2 * inner_size]
//   x,z = split(xz)            →  [inner_size] each
//   g   = attn_gate  @ normed  →  [inner_size]
//   act = silu(x) * silu(z)
//   act = act * sigmoid(g)
//   out = ssm_out    @ act     →  [embed_dim]
// ---------------------------------------------------------------------------

struct ggml_tensor* Qwen35moeInference::build_ssm_layer(
        struct ggml_context* ctx,
        struct ggml_tensor*  normed,
        int                  layer_idx) {

    const Qwen35moeLayer& lw = model_->weights.layers[layer_idx];

    int embed_dim  = (int)cfg_->embedding_length;
    int inner_size = (int)cfg_->inner_size;         // 4096

    // Input projection → [2 * inner_size]
    struct ggml_tensor* xz = ggml_mul_mat(ctx, lw.attn_qkv, normed);

    // Split
    struct ggml_tensor* x = ggml_view_1d(ctx, xz, inner_size, 0);
    struct ggml_tensor* z = ggml_view_1d(ctx, xz, inner_size,
                                          (size_t)inner_size * SZF32);

    // Gate path
    struct ggml_tensor* g = ggml_mul_mat(ctx, lw.attn_gate, normed);

    // Gated activation: silu(x) * silu(z) * sigmoid(g)
    struct ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, x), ggml_silu(ctx, z));
    act = ggml_mul(ctx, act, ggml_sigmoid(ctx, g));

    // Output projection → [embed_dim]
    struct ggml_tensor* out = ggml_mul_mat(ctx, lw.ssm_out, act);
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
    struct ggml_tensor* hidden = ggml_mul(ctx, ggml_silu(ctx, gate_out), up_out);

    // Down projection: [embed_dim, k]
    // ffn_down_exps: [ff_dim, embed_dim, n_experts]
    struct ggml_tensor* down_out =
        ggml_mul_mat_id(ctx, lw.ffn_down_exps, hidden, ids_2d);

    // Weighted sum: multiply each expert column by its router probability
    // top_k_probs [k] → broadcast to [embed_dim, k]
    struct ggml_tensor* probs_2d = ggml_reshape_2d(ctx, top_k_probs, 1, expert_k);
    struct ggml_tensor* probs_bc = ggml_repeat(ctx, probs_2d, down_out); // [embed_dim, k]
    struct ggml_tensor* scaled   = ggml_mul(ctx, down_out, probs_bc);    // [embed_dim, k]

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
    struct ggml_tensor* sh_hidden = ggml_mul(ctx, ggml_silu(ctx, sh_gate), sh_up);
    struct ggml_tensor* sh_out    = ggml_mul_mat(ctx, lw.ffn_down_shexp, sh_hidden);
    sh_out = ggml_reshape_1d(ctx, sh_out, embed_dim);

    // Scale by shared-expert gate scalar (broadcast scalar over embed_dim)
    struct ggml_tensor* sh_scale_bc =
        ggml_repeat(ctx, sh_scale,
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim));
    sh_out = ggml_mul(ctx, sh_out, sh_scale_bc);

    // ------------------------------------------------------------------
    // Combine
    // ------------------------------------------------------------------
    expert_out = ggml_reshape_1d(ctx, expert_out, embed_dim);
    return ggml_add(ctx, expert_out, sh_out);
}

