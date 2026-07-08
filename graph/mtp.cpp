/**
 * @file mtp.cpp
 * @brief MTP (Multi-Token Prediction / NextN) speculative decoding for the
 *        Qwen3.5-MoE engine.
 *
 * The GGUF ships one extra decoder block (`blk.<trunk_layer_count>`, the "nextn"
 * block) trained as a lightweight draft head. Each MTP step:
 *   1. Drafts up to `spec_draft_n_max` tokens autoregressively with the nextn
 *      block (one full-attention layer + MoE FFN), seeded by the last committed
 *      token and the trunk hidden state that pairs with it.
 *   2. Verifies the drafted tokens in a single batched trunk forward, taking the
 *      longest greedy-matching prefix plus one bonus (correction) token.
 *   3. Rolls back the trunk state to the accepted boundary. Attention KV uses
 *      O(1) truncation; the DeltaNet recurrent state is restored from a
 *      device-resident shadow and the committed tokens are replayed.
 *
 * Correctness is guaranteed by verification (the emitted tokens always equal the
 * trunk's greedy output); MTP only changes throughput. This path is greedy-only
 * (temp<=0 or top_k==1), single-slot (--parallel 1), and cpu/gpu (not mixed).
 *
 * See engines/ggml_moe/README.md and the reference build in
 * llama.cpp/src/models/qwen35moe.cpp (graph_mtp) for the block structure.
 */
#include "graph/graph.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
inline uint64_t now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// Thread-pool scope guard mirrors the private CpuThreadScope in graph.cpp; MTP
// forwards are batch-like (multi-token), so use the batch thread pool.
struct MtpThreadScope {
    std::shared_ptr<Qwen35moeModel> model;
    explicit MtpThreadScope(const std::shared_ptr<Qwen35moeModel>& m) : model(m) {
        if (model) {
            model->apply_cpu_thread_pool(true);
        }
    }
};
} // namespace

void Qwen35moeForwardPass::configure_mtp(bool enabled, int spec_draft_n_max) {
    mtp_enabled_ = enabled;
    if (spec_draft_n_max > 0) {
        mtp_n_max_ = spec_draft_n_max;
    }
    // If init() already ran, materialise resources now; otherwise init() will.
    if (model_) {
        init_mtp();
    }
}

void Qwen35moeForwardPass::init_mtp() {
    if (!model_ || !mtp_enabled_) {
        return;
    }
    mtp_supported_ = model_->has_mtp();
    if (!mtp_supported_) {
        std::fprintf(stderr,
            "[mtp] --mtp requested but model has no nextn block "
            "(nextn_predict_layers=0); falling back to standard decode.\n");
        return;
    }
    if (mtp_kv_ && dn_shadow_) {
        return;  // already initialised
    }

    auto& m = model_->meta_->qwen35moe;
    ggml_backend_t cache_backend = model_->get_curr_backend();

    // Single-layer dense KV cache for the nextn attention block. The nextn block
    // is a full-attention layer (no DeltaNet), so one KV layer with the same K/V
    // geometry as the trunk attention layers is sufficient. Paged layout is off
    // for the draft head (its sequences are short and repeatedly truncated).
    const uint32_t n_embd_k = static_cast<uint32_t>(m.head_count_kv * m.key_length);
    const uint32_t n_embd_v = static_cast<uint32_t>(m.head_count_kv * m.value_length);
    std::vector<ggml_backend_t> mtp_kv_backends(1, cache_backend);
    mtp_kv_ = std::make_unique<simple_kv_cache>(
        1u, context_len_, max_batch_size_, n_embd_k, n_embd_v,
        GGML_TYPE_F16, GGML_TYPE_F16, cache_backend, mtp_kv_backends,
        PagedKVConfig{false, 16u, false});

    // Device-resident shadow of the trunk DeltaNet state for O(1) snapshot /
    // restore around verify forwards (a host round-trip of the ~60MB state would
    // dominate the step; a same-backend copy is effectively free).
    const uint32_t d_inner       = m.inner_size;
    const uint32_t num_v_heads   = m.time_step_rank;
    const uint32_t head_v_dim    = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * m.group_count * m.state_size;

    const uint32_t trunk_layers = model_->trunk_layer_count();
    int n_dn_layers = 0;
    for (uint32_t il = 0; il < trunk_layers; ++il) {
        if (!is_full_attention_layer(il)) {
            ++n_dn_layers;
        }
    }
    std::vector<ggml_backend_t> dn_layer_backends(static_cast<size_t>(n_dn_layers), cache_backend);
    int dn_idx = 0;
    for (uint32_t il = 0; il < trunk_layers; ++il) {
        if (!is_full_attention_layer(il)) {
            dn_layer_backends[static_cast<size_t>(dn_idx++)] = model_->get_layer_backend(il);
        }
    }
    DeltaNetStateParams dn_state_hp{
        static_cast<uint32_t>(n_dn_layers),
        max_batch_size_,
        head_v_dim,
        m.state_size,
        num_v_heads,
        conv_channels,
        m.conv_kernel,
        cache_backend,
        dn_layer_backends
    };
    dn_shadow_ = std::make_unique<DeltaNetState>(dn_state_hp);

    // Cached trunk graphs indexed by token-count: verify uses D+1, replay uses
    // the committed length (1..D). Index directly by n_tok.
    const size_t cache_slots = static_cast<size_t>(mtp_n_max_) + 2;
    mtp_graph_ctx_.assign(cache_slots, nullptr);
    mtp_graph_buf_.assign(cache_slots, {});
    mtp_graph_allocr_.assign(cache_slots, nullptr);
    mtp_graph_.assign(cache_slots, nullptr);

    // Snapshot-based rollback: capture per-token DeltaNet state during verify so a
    // partial acceptance needs no replay forward. This is strictly faster than the
    // shadow+replay path (measured ~246 vs ~213 tok/s), so it is ON by default;
    // set QWEN35MOE_MTP_SNAPSHOT=0 to fall back to shadow+replay for A/B.
    mtp_use_snapshots_ = [] {
        const char* v = std::getenv("QWEN35MOE_MTP_SNAPSHOT");
        return !(v && v[0] == '0');  // default ON; disable only with explicit 0
    }();
    mtp_grow_bucket_ = [] {
        const char* v = std::getenv("QWEN35MOE_MTP_GROW_BUCKET");
        return v && v[0] != '\0' && v[0] != '0';
    }();
    if (mtp_use_snapshots_) {
        mtp_snap_k_ = static_cast<uint32_t>(mtp_n_max_) + 1;
        const int64_t rec_slot_floats =
            static_cast<int64_t>(head_v_dim) * head_v_dim * num_v_heads;
        const int64_t conv_slot_floats =
            static_cast<int64_t>(m.conv_kernel > 0 ? m.conv_kernel - 1 : 0) * conv_channels;
        const size_t n_snap_tensors = 2 * static_cast<size_t>(n_dn_layers);
        ggml_init_params sp = {
            /* .mem_size   = */ (n_snap_tensors + 8) * ggml_tensor_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        mtp_snap_ctx_ = ggml_init(sp);
        mtp_rec_snap_.assign(static_cast<size_t>(n_dn_layers), nullptr);
        mtp_conv_snap_.assign(static_cast<size_t>(n_dn_layers), nullptr);
        for (int dn = 0; dn < n_dn_layers; ++dn) {
            mtp_rec_snap_[dn] = ggml_new_tensor_2d(
                mtp_snap_ctx_, GGML_TYPE_F32, rec_slot_floats, mtp_snap_k_);
            mtp_conv_snap_[dn] = ggml_new_tensor_2d(
                mtp_snap_ctx_, GGML_TYPE_F32, conv_slot_floats, mtp_snap_k_);
        }
        // All DN layers share one backend here (MTP is non-mixed), so a single
        // buffer suffices.
        mtp_snap_buf_ = ggml_backend_alloc_ctx_tensors_from_buft(
            mtp_snap_ctx_, ggml_backend_get_default_buffer_type(cache_backend));
        if (!mtp_snap_buf_) {
            throw std::runtime_error("init_mtp: failed to allocate snapshot buffer");
        }
    }

    std::fprintf(stderr,
        "[mtp] nextn draft head enabled: mtp_layer=%u spec_draft_n_max=%d snapshots=%s\n",
        model_->mtp_layer_index(), mtp_n_max_, mtp_use_snapshots_ ? "on" : "off");
}

// ── Graph builders ────────────────────────────────────────────────────────────

void Qwen35moeForwardPass::build_mtp_head(
    ggml_cgraph* gf,
    ggml_tensor* hidden_all,
    bool need_argmax,
    bool need_hidden
) {
    // Post-trunk RMSNorm produces the "h_nextn" hidden that (a) feeds the LM head
    // and (b) pairs with the next token when seeding the draft head.
    ggml_tensor* normed = build_norm(gf, hidden_all, model_->get_output_norm_weight(), -1);

    if (need_hidden) {
        // Materialise a persistent copy so it is still readable after compute
        // (the LM head consumes `normed`, so it is not itself a terminal node).
        ggml_tensor* hidden_copy = ggml_cont(ctx_, normed);
        set_tensor_name(hidden_copy, "mtp_hidden");
        ggml_build_forward_expand(gf, hidden_copy);
    }

    ggml_tensor* out_w = model_->get_output_weight();
    ggml_tensor* logits = ggml_mul_mat(
        ctx_, out_w ? out_w : model_->get_token_embedding_weight(), normed);
    set_tensor_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);

    if (need_argmax) {
        // Greedy: argmax along the vocab dimension for each column → I32[n_tok].
        ggml_tensor* am = ggml_argmax(ctx_, logits);
        set_tensor_name(am, "mtp_argmax");
        ggml_build_forward_expand(gf, am);
    }
}

ggml_cgraph* Qwen35moeForwardPass::build_mtp_verify_graph(
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx
) {
    reset_context();

    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    if (n_tok == 0) {
        throw std::runtime_error("build_mtp_verify_graph: empty tokens");
    }

    ggml_cgraph* gf = new_graph();

    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(inpL, "inpL");

    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp_pos);
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    inpL = build_layer_range(
        gf, inpL, inp_pos, n_tok, slot_idx, 0, model_->trunk_layer_count(), 0, nullptr);

    // Full per-column head: verify needs the greedy token AND the paired hidden
    // for every column (not just the last, as build_prefill_graph does).
    build_mtp_head(gf, inpL, /*need_argmax=*/true, /*need_hidden=*/true);
    return gf;
}

ggml_cgraph* Qwen35moeForwardPass::build_mtp_block_graph(
    uint32_t n_tok,
    uint32_t slot_idx,
    bool need_argmax,
    bool need_hidden_out
) {
    reset_context();

    auto& m = model_->meta_->qwen35moe;
    const int il = static_cast<int>(model_->mtp_layer_index());
    const int64_t n_embd = static_cast<int64_t>(m.embedding_length);

    ggml_cgraph* gf = new_graph();

    ggml_tensor* tokens_t = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(tokens_t);
    set_tensor_name(tokens_t, "tokens");
    ggml_build_forward_expand(gf, tokens_t);

    ggml_tensor* h_in = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, n_embd, n_tok);
    ggml_set_input(h_in);
    set_tensor_name(h_in, "mtp_h_in");
    ggml_build_forward_expand(gf, h_in);

    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp_pos);
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // NextN input: eh_proj( concat( enorm(embed(token)), hnorm(h) ) )
    ggml_tensor* tok_embd = ggml_get_rows(ctx_, model_->get_token_embedding_weight(), tokens_t);
    ggml_tensor* e_norm = build_rms_norm(
        ctx_, tok_embd, model_->get_nextn_enorm_weight(il), m.layer_norm_rms_epsilon, il);
    ggml_tensor* h_norm = build_rms_norm(
        ctx_, h_in, model_->get_nextn_hnorm_weight(il), m.layer_norm_rms_epsilon, il);
    ggml_tensor* concat = ggml_concat(ctx_, e_norm, h_norm, /*dim=*/0);  // [2*n_embd, n_tok]
    ggml_tensor* cur = ggml_mul_mat(ctx_, model_->get_nextn_eh_proj_weight(il), concat);
    set_tensor_name(cur, "mtp_eh_proj", il);

    ggml_tensor* inpSA = cur;

    cur = build_norm(gf, cur, model_->get_attn_norm_weight(il), il);

    // Causal mask shared by the single MTP attention layer. Width = existing MTP
    // KV positions + this batch. build_gated_attention consumes it and
    // set_mtp_block_inputs fills it (via fill_dynamic_prefill_mask).
    const uint32_t cache_pos = mtp_kv_->get_pos(slot_idx);
    const uint32_t n_kv = cache_pos + n_tok;
    const bool layer_fa = use_flash_attention_ && layer_allows_flash_attn(il);
    ggml_tensor* shared_kq_mask = ggml_new_tensor_2d(
        ctx_, layer_fa ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv, n_tok);
    set_tensor_name(shared_kq_mask, "kq_mask_shared");
    ggml_build_forward_expand(gf, shared_kq_mask);

    cur = build_gated_attention(
        ctx_, gf, mtp_kv_.get(), cur, inp_pos,
        /*kv_cache_layer=*/0, n_tok, slot_idx, il,
        model_->get_attn_q_weight(il), model_->get_attn_q_norm_weight(il),
        model_->get_attn_k_weight(il), model_->get_attn_k_norm_weight(il),
        model_->get_attn_v_weight(il), model_->get_attn_output_weight(il),
        m.key_length, m.head_count, m.head_count_kv, m.dimension_count, m.freq_base,
        static_cast<int>(context_len_), m.layer_norm_rms_epsilon, shared_kq_mask,
        /*kv_write_row=*/nullptr, /*fixed_read_n_kv=*/0);

    cur = ggml_add(ctx_, cur, inpSA);

    ggml_tensor* ffn_residual = cur;
    cur = build_norm(gf, cur, model_->get_post_attention_norm_weight(il), il);
    cur = build_moe_layer(ctx_, gf, cur, il);
    cur = ggml_add(ctx_, cur, ffn_residual);

    // Head norm uses the nextn shared_head_norm; output projection reuses the
    // trunk LM head (this GGUF has no dedicated nextn.shared_head.head).
    ggml_tensor* head_norm_w = model_->get_nextn_shared_head_norm_weight(il);
    if (!head_norm_w) {
        head_norm_w = model_->get_output_norm_weight();
    }
    ggml_tensor* normed = build_norm(gf, cur, head_norm_w, -1);

    if (need_hidden_out) {
        ggml_tensor* h_out = ggml_cont(ctx_, normed);
        set_tensor_name(h_out, "mtp_out_h");
        ggml_build_forward_expand(gf, h_out);
    }

    ggml_tensor* out_w = model_->get_output_weight();
    ggml_tensor* logits = ggml_mul_mat(
        ctx_, out_w ? out_w : model_->get_token_embedding_weight(), normed);
    set_tensor_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);

    if (need_argmax) {
        ggml_tensor* am = ggml_argmax(ctx_, logits);
        set_tensor_name(am, "mtp_argmax");
        ggml_build_forward_expand(gf, am);
    }
    return gf;
}

void Qwen35moeForwardPass::set_mtp_block_inputs(
    ggml_cgraph* gf,
    const std::vector<int32_t>& tokens,
    const std::vector<float>& hiddens,
    int pos,
    uint32_t /*slot_idx*/
) {
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());

    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) throw std::runtime_error("mtp block: 'tokens' missing");
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, n_tok * sizeof(int32_t));

    ggml_tensor* h_t = ggml_graph_get_tensor(gf, "mtp_h_in");
    if (!h_t) throw std::runtime_error("mtp block: 'mtp_h_in' missing");
    if (hiddens.size() != static_cast<size_t>(h_t->ne[0]) * n_tok) {
        throw std::runtime_error("mtp block: hidden buffer size mismatch");
    }
    ggml_backend_tensor_set(h_t, hiddens.data(), 0, hiddens.size() * sizeof(float));

    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) throw std::runtime_error("mtp block: 'inp_pos' missing");
    std::vector<int32_t> pos_data(n_tok);
    for (uint32_t i = 0; i < n_tok; ++i) {
        pos_data[i] = pos + static_cast<int>(i);
    }
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, n_tok * sizeof(int32_t));

    ggml_tensor* mask = ggml_graph_get_tensor(gf, "kq_mask_shared");
    if (mask) {
        const uint32_t graph_n_kv = static_cast<uint32_t>(mask->ne[0]);
        std::vector<float> mask_f32;
        fill_dynamic_prefill_mask(mask_f32, graph_n_kv, n_tok, pos);
        upload_decode_mask_tensor(mask, mask_f32);
    }
}

// ── Runners ─────────────────────────────────────────────────────────────────

ggml_cgraph* Qwen35moeForwardPass::ensure_mtp_trunk_graph(uint32_t n_tok) {
    const uint32_t bucket = context_len_;  // lock-ctx bucket, matches gpu decode policy
    // Grow the cache if n_tok is unexpectedly large (should not happen: n_tok is
    // always <= mtp_n_max_+1).
    if (n_tok >= mtp_graph_.size()) {
        const size_t need = n_tok + 1;
        mtp_graph_ctx_.resize(need, nullptr);
        mtp_graph_buf_.resize(need);
        mtp_graph_allocr_.resize(need, nullptr);
        mtp_graph_.resize(need, nullptr);
    }
    // A change of context length invalidates all cached graphs (bucket baked in).
    if (mtp_graph_bucket_ != bucket) {
        for (size_t i = 0; i < mtp_graph_.size(); ++i) {
            if (mtp_graph_allocr_[i]) { ggml_gallocr_free(mtp_graph_allocr_[i]); mtp_graph_allocr_[i] = nullptr; }
            if (mtp_graph_ctx_[i]) { ggml_free(mtp_graph_ctx_[i]); mtp_graph_ctx_[i] = nullptr; }
            mtp_graph_[i] = nullptr;
        }
        mtp_graph_bucket_ = bucket;
    }
    if (mtp_graph_[n_tok]) {
        return mtp_graph_[n_tok];
    }

    if (mtp_graph_buf_[n_tok].empty()) {
        mtp_graph_buf_[n_tok].resize(FP_GRAPH_SIZE_METADATA);
    }
    ggml_init_params params = {
        /* .mem_size   = */ mtp_graph_buf_[n_tok].size(),
        /* .mem_buffer = */ mtp_graph_buf_[n_tok].data(),
        /* .no_alloc   = */ true,
    };
    mtp_graph_ctx_[n_tok] = ggml_init(params);

    // Build the trunk graph into the dedicated context. The build helpers use the
    // ctx_ member, so swap it temporarily. Non-paged: the KV write row is inp_pos
    // itself and the read span is the fixed bucket, so the same graph is reusable
    // across steps (only the input contents change).
    ggml_context* saved = ctx_;
    ctx_ = mtp_graph_ctx_[n_tok];
    ggml_cgraph* gf = ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);

    std::vector<int32_t> dummy(n_tok, 0);
    ggml_tensor* inpL = embedding(gf, dummy);
    set_tensor_name(inpL, "inpL");

    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp_pos);
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    inpL = build_layer_range(
        gf, inpL, inp_pos, n_tok, /*slot_idx=*/0, 0, model_->trunk_layer_count(),
        /*fixed_shared_n_kv=*/bucket, /*kv_write_row=*/inp_pos);
    build_mtp_head(gf, inpL, /*need_argmax=*/true, /*need_hidden=*/true);

    ctx_ = saved;

    ggml_backend_t backend = model_->get_curr_backend();
    mtp_graph_allocr_[n_tok] = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(mtp_graph_allocr_[n_tok], gf)) {
        throw std::runtime_error("ensure_mtp_trunk_graph: gallocr alloc failed");
    }
    mtp_graph_[n_tok] = gf;
    return gf;
}

void Qwen35moeForwardPass::run_mtp_advance(
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx
) {
    // Advance trunk KV + DeltaNet state over `tokens` using a cached graph. Used
    // to replay the committed prefix after a partial-acceptance verify. Head
    // outputs are computed but ignored. Only valid on the cacheable (non-paged,
    // single-backend) path; callers guarantee that.
    (void)slot_idx;
    ggml_cgraph* gf = ensure_mtp_trunk_graph(static_cast<uint32_t>(tokens.size()));
    set_inputs(gf, tokens, pos);
    ggml_backend_t backend = model_->get_curr_backend();
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
}

void Qwen35moeForwardPass::ensure_mtp_snap_verify_graph(uint32_t n_tok, uint32_t bucket) {
    if (mtp_snap_verify_graph_ && mtp_snap_verify_n_tok_ == n_tok &&
        mtp_snap_verify_bucket_ == bucket) {
        return;
    }
    if (mtp_snap_verify_allocr_) { ggml_gallocr_free(mtp_snap_verify_allocr_); mtp_snap_verify_allocr_ = nullptr; }
    if (mtp_snap_verify_ctx_) { ggml_free(mtp_snap_verify_ctx_); mtp_snap_verify_ctx_ = nullptr; }
    if (mtp_snap_verify_buf_.empty()) {
        mtp_snap_verify_buf_.resize(FP_GRAPH_SIZE_METADATA);
    }
    ggml_init_params params = {
        /* .mem_size   = */ mtp_snap_verify_buf_.size(),
        /* .mem_buffer = */ mtp_snap_verify_buf_.data(),
        /* .no_alloc   = */ true,
    };
    mtp_snap_verify_ctx_ = ggml_init(params);

    ggml_context* saved = ctx_;
    ctx_ = mtp_snap_verify_ctx_;
    mtp_snapshot_build_ = true;  // build_deltanet_layer captures per-token snapshots
    ggml_cgraph* gf = ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);

    std::vector<int32_t> dummy(n_tok, 0);
    ggml_tensor* inpL = embedding(gf, dummy);
    set_tensor_name(inpL, "inpL");
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp_pos);
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);
    inpL = build_layer_range(
        gf, inpL, inp_pos, n_tok, /*slot_idx=*/0, 0, model_->trunk_layer_count(),
        /*fixed_shared_n_kv=*/bucket, /*kv_write_row=*/inp_pos);
    build_mtp_head(gf, inpL, /*need_argmax=*/true, /*need_hidden=*/true);

    mtp_snapshot_build_ = false;
    ctx_ = saved;

    ggml_backend_t backend = model_->get_curr_backend();
    mtp_snap_verify_allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(mtp_snap_verify_allocr_, gf)) {
        throw std::runtime_error("ensure_mtp_snap_verify_graph: gallocr alloc failed");
    }
    mtp_snap_verify_graph_ = gf;
    mtp_snap_verify_n_tok_ = n_tok;
    mtp_snap_verify_bucket_ = bucket;
}

void Qwen35moeForwardPass::run_mtp_verify_snapshot(
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx,
    std::vector<int32_t>& argmax_out,
    std::vector<float>& hidden_out
) {
    (void)slot_idx;
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    const int64_t n_embd = model_->meta_->qwen35moe.embedding_length;

    // KV bucket: fixed ctx_len by default; optionally grow with position so early
    // attention reads fewer KV rows (256,512,...,ctx_len).
    uint32_t bucket = context_len_;
    if (mtp_grow_bucket_) {
        const uint32_t need = static_cast<uint32_t>(pos) + n_tok;
        bucket = 256;
        while (bucket < need && bucket < context_len_) {
            bucket *= 2;
        }
        if (bucket > context_len_) {
            bucket = context_len_;
        }
    }
    ensure_mtp_snap_verify_graph(n_tok, bucket);
    ggml_cgraph* gf = mtp_snap_verify_graph_;
    set_inputs(gf, tokens, pos);
    ggml_backend_t backend = model_->get_curr_backend();
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    ggml_tensor* am = ggml_graph_get_tensor(gf, "mtp_argmax");
    if (!am || am->type != GGML_TYPE_I32) {
        throw std::runtime_error("run_mtp_verify_snapshot: mtp_argmax missing");
    }
    argmax_out.resize(n_tok);
    ggml_backend_tensor_get(am, argmax_out.data(), 0, n_tok * sizeof(int32_t));

    ggml_tensor* hid = ggml_graph_get_tensor(gf, "mtp_hidden");
    if (!hid) {
        throw std::runtime_error("run_mtp_verify_snapshot: mtp_hidden missing");
    }
    hidden_out.resize(static_cast<size_t>(n_embd) * n_tok);
    ggml_backend_tensor_get(hid, hidden_out.data(), 0, hidden_out.size() * sizeof(float));
}

void Qwen35moeForwardPass::ensure_mtp_commit_graph() {
    if (mtp_commit_graph_) {
        return;
    }
    if (mtp_commit_buf_.empty()) {
        mtp_commit_buf_.resize(16u * 1024u * 1024u);  // metadata only; ample for ~2*n_dn ops
    }
    ggml_init_params params = {
        /* .mem_size   = */ mtp_commit_buf_.size(),
        /* .mem_buffer = */ mtp_commit_buf_.data(),
        /* .no_alloc   = */ true,
    };
    mtp_commit_ctx_ = ggml_init(params);

    ggml_context* saved = ctx_;
    ctx_ = mtp_commit_ctx_;
    ggml_cgraph* gf = ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);

    mtp_commit_idx_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, 1);
    ggml_set_input(mtp_commit_idx_);
    set_tensor_name(mtp_commit_idx_, "mtp_commit_idx");
    ggml_build_forward_expand(gf, mtp_commit_idx_);

    const uint32_t n_dn = dn_state_->n_dn_layers();
    const int64_t rec_sf = static_cast<int64_t>(dn_state_->rec_slot_floats());
    const int64_t conv_sf = static_cast<int64_t>(dn_state_->conv_slot_floats());
    for (uint32_t dn = 0; dn < n_dn; ++dn) {
        // recurrent: select accepted column of the snapshot → dn_state slot 0.
        ggml_tensor* rec_col = ggml_get_rows(ctx_, mtp_rec_snap_[dn], mtp_commit_idx_);
        rec_col = ggml_reshape_1d(ctx_, rec_col, rec_sf);
        ggml_tensor* rec_dst = ggml_view_1d(ctx_, dn_state_->recurrent_tensor(dn), rec_sf, 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx_, rec_col, rec_dst));
        // conv:
        if (conv_sf > 0) {
            ggml_tensor* conv_col = ggml_get_rows(ctx_, mtp_conv_snap_[dn], mtp_commit_idx_);
            conv_col = ggml_reshape_1d(ctx_, conv_col, conv_sf);
            ggml_tensor* conv_dst = ggml_view_1d(ctx_, dn_state_->conv_tensor(dn), conv_sf, 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx_, conv_col, conv_dst));
        }
    }

    ctx_ = saved;

    ggml_backend_t backend = model_->get_curr_backend();
    mtp_commit_allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(mtp_commit_allocr_, gf)) {
        throw std::runtime_error("ensure_mtp_commit_graph: gallocr alloc failed");
    }
    mtp_commit_graph_ = gf;
}

void Qwen35moeForwardPass::run_mtp_commit(int accepted_col) {
    ensure_mtp_commit_graph();
    const int32_t idx = accepted_col;
    ggml_backend_tensor_set(mtp_commit_idx_, &idx, 0, sizeof(int32_t));
    ggml_backend_t backend = model_->get_curr_backend();
    ggml_backend_graph_compute(backend, mtp_commit_graph_);
    ggml_backend_synchronize(backend);
}

void Qwen35moeForwardPass::run_mtp_verify(
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx,
    ggml_backend_sched_t scheduler,
    std::vector<int32_t>& argmax_out,
    std::vector<float>& hidden_out
) {
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    const int64_t n_embd = model_->meta_->qwen35moe.embedding_length;

    ggml_cgraph* gf = nullptr;

    // Fast path: single-backend, non-paged, small token-count (decode-loop verify
    // and replay) → reuse a persistent trunk graph and compute directly on the
    // backend (no per-step rebuild / sched realloc). Long prompt chunks fall back
    // to the eager scheduler path so we don't cache huge graphs.
    const bool cacheable = !paged_kv_enabled_ && !model_->is_mixed_mode() &&
                           n_tok <= static_cast<uint32_t>(mtp_n_max_) + 1;
    if (cacheable) {
        gf = ensure_mtp_trunk_graph(n_tok);
        set_inputs(gf, tokens, pos);
        ggml_backend_t compute_backend = model_->get_curr_backend();
        ggml_backend_graph_compute(compute_backend, gf);
        ggml_backend_synchronize(compute_backend);
    } else {
        // Fallback: rebuild each step via the scheduler (paged / mixed).
        ggml_backend_sched_reset(scheduler);
        gf = build_mtp_verify_graph(tokens, pos, slot_idx);
        if (!ggml_backend_sched_alloc_graph(scheduler, gf)) {
            throw std::runtime_error("run_mtp_verify: failed to allocate graph");
        }
        set_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);
        ggml_backend_sched_synchronize(scheduler);
    }

    ggml_tensor* am = ggml_graph_get_tensor(gf, "mtp_argmax");
    if (!am || am->type != GGML_TYPE_I32) {
        throw std::runtime_error("run_mtp_verify: mtp_argmax missing/typemismatch");
    }
    argmax_out.resize(n_tok);
    ggml_backend_tensor_get(am, argmax_out.data(), 0, n_tok * sizeof(int32_t));

    ggml_tensor* hid = ggml_graph_get_tensor(gf, "mtp_hidden");
    if (!hid) {
        throw std::runtime_error("run_mtp_verify: mtp_hidden missing");
    }
    hidden_out.resize(static_cast<size_t>(n_embd) * n_tok);
    ggml_backend_tensor_get(hid, hidden_out.data(), 0, hidden_out.size() * sizeof(float));
}

void Qwen35moeForwardPass::run_mtp_process(
    const std::vector<int32_t>& tokens,
    const std::vector<float>& hiddens_shifted,
    int pos,
    uint32_t slot_idx,
    ggml_backend_sched_t scheduler
) {
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    if (n_tok == 0) {
        return;
    }
    ggml_backend_sched_reset(scheduler);
    ggml_cgraph* gf = build_mtp_block_graph(n_tok, slot_idx, /*need_argmax=*/false, /*need_hidden_out=*/false);
    if (!ggml_backend_sched_alloc_graph(scheduler, gf)) {
        throw std::runtime_error("run_mtp_process: failed to allocate graph");
    }
    set_mtp_block_inputs(gf, tokens, hiddens_shifted, pos, slot_idx);
    ggml_backend_sched_graph_compute(scheduler, gf);
    ggml_backend_sched_synchronize(scheduler);
    mtp_kv_->advance(n_tok, slot_idx);
}

// ── Public entry points ───────────────────────────────────────────────────────

int32_t Qwen35moeForwardPass::mtp_prefill(
    const std::vector<int32_t>& prompt_tokens,
    uint32_t slot_idx,
    std::vector<float>& seed_h,
    int& pos_out,
    ggml_backend_sched_t scheduler
) {
    if (!mtp_active()) {
        throw std::runtime_error("mtp_prefill called but MTP is not active");
    }
    MtpThreadScope thread_scope(model_);

    reset_sequence(slot_idx);

    const uint32_t L = static_cast<uint32_t>(prompt_tokens.size());
    const int64_t n_embd = model_->meta_->qwen35moe.embedding_length;
    const uint32_t ubatch = std::max<uint32_t>(1, effective_prefill_ubatch_limit());

    // Process the prompt through the trunk in ubatch chunks via the verify graph,
    // capturing all per-position hiddens so the nextn KV can be seeded correctly.
    std::vector<float> all_hidden(static_cast<size_t>(n_embd) * L);
    std::vector<int32_t> last_argmax;
    for (uint32_t start = 0; start < L; start += ubatch) {
        const uint32_t chunk = std::min(ubatch, L - start);
        std::vector<int32_t> chunk_tokens(
            prompt_tokens.begin() + start, prompt_tokens.begin() + start + chunk);
        std::vector<int32_t> chunk_argmax;
        std::vector<float> chunk_hidden;
        run_mtp_verify(chunk_tokens, static_cast<int>(start), slot_idx,
                       scheduler, chunk_argmax, chunk_hidden);
        advance_cache(chunk, slot_idx);
        std::memcpy(all_hidden.data() + static_cast<size_t>(start) * n_embd,
                    chunk_hidden.data(), chunk_hidden.size() * sizeof(float));
        last_argmax = std::move(chunk_argmax);
    }

    const int32_t first_tok = last_argmax.empty() ? 0 : last_argmax.back();
    seed_h.assign(all_hidden.end() - n_embd, all_hidden.end());  // hidden at pos L-1

    // Seed the nextn KV over the prompt: token@q is paired with trunk hidden@(q-1).
    // Position 0 has no predecessor hidden → pair with zeros.
    std::vector<float> proc_hidden(static_cast<size_t>(n_embd) * L, 0.0f);
    if (L > 1) {
        std::memcpy(proc_hidden.data() + n_embd, all_hidden.data(),
                    static_cast<size_t>(n_embd) * (L - 1) * sizeof(float));
    }
    for (uint32_t start = 0; start < L; start += ubatch) {
        const uint32_t chunk = std::min(ubatch, L - start);
        std::vector<int32_t> chunk_tokens(
            prompt_tokens.begin() + start, prompt_tokens.begin() + start + chunk);
        std::vector<float> chunk_hidden(
            proc_hidden.begin() + static_cast<size_t>(start) * n_embd,
            proc_hidden.begin() + static_cast<size_t>(start + chunk) * n_embd);
        run_mtp_process(chunk_tokens, chunk_hidden, static_cast<int>(start), slot_idx, scheduler);
    }

    pos_out = static_cast<int>(L);
    return first_tok;
}

int Qwen35moeForwardPass::mtp_step(
    int32_t& seed_tok,
    int& pos,
    std::vector<float>& seed_h,
    uint32_t slot_idx,
    std::vector<int32_t>& out,
    ggml_backend_sched_t scheduler
) {
    if (!mtp_active()) {
        throw std::runtime_error("mtp_step called but MTP is not active");
    }
    MtpThreadScope thread_scope(model_);

    const int P = pos;
    const int64_t n_embd = model_->meta_->qwen35moe.embedding_length;

    // Bound the draft length so every candidate position stays inside the context
    // (need room for pos..P+D verify tokens plus the bonus at P+D+1).
    int D = mtp_n_max_;
    const int max_D = static_cast<int>(context_len_) - P - 2;
    if (D > max_D) D = max_D;
    if (D < 0) D = 0;

    // 1) DRAFT: autoregressive nextn head, one token per step.
    std::vector<int32_t> drafts;
    drafts.reserve(static_cast<size_t>(D));
    const uint64_t t_draft0 = now_us();
    if (D > 0) {
        std::vector<float> h = seed_h;   // trunk hidden paired with the current token
        int32_t tok = seed_tok;
        mtp_kv_->set_pos(static_cast<uint32_t>(P), slot_idx);
        for (int i = 0; i < D; ++i) {
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf = build_mtp_block_graph(
                1u, slot_idx, /*need_argmax=*/true, /*need_hidden_out=*/true);
            if (!ggml_backend_sched_alloc_graph(scheduler, gf)) {
                throw std::runtime_error("mtp_step draft: failed to allocate graph");
            }
            std::vector<int32_t> tok_vec{tok};
            set_mtp_block_inputs(gf, tok_vec, h, P + i, slot_idx);
            ggml_backend_sched_graph_compute(scheduler, gf);
            ggml_backend_sched_synchronize(scheduler);

            ggml_tensor* am = ggml_graph_get_tensor(gf, "mtp_argmax");
            ggml_tensor* hout = ggml_graph_get_tensor(gf, "mtp_out_h");
            if (!am || !hout) {
                throw std::runtime_error("mtp_step draft: outputs missing");
            }
            int32_t d = 0;
            ggml_backend_tensor_get(am, &d, 0, sizeof(int32_t));
            ggml_backend_tensor_get(hout, h.data(), 0, static_cast<size_t>(n_embd) * sizeof(float));
            mtp_kv_->advance(1, slot_idx);

            drafts.push_back(d);
            tok = d;
        }
    }
    mtp_draft_us_ += now_us() - t_draft0;

    // 2) VERIFY the [seed_tok, drafts...] window in a single trunk forward.
    std::vector<int32_t> verify_tokens;
    verify_tokens.reserve(static_cast<size_t>(D) + 1);
    verify_tokens.push_back(seed_tok);
    verify_tokens.insert(verify_tokens.end(), drafts.begin(), drafts.end());

    // Snapshot path: capture per-token DeltaNet state in verify → no replay. Only
    // for D>=1 (D==0 would use the 1-token DeltaNet fast path, which cannot
    // snapshot); the shadow+replay path is the D==0 / snapshots-off fallback.
    const bool use_snap = mtp_use_snapshots_ && D >= 1 &&
                          !paged_kv_enabled_ && !model_->is_mixed_mode();

    std::vector<int32_t> argmax_cols;
    std::vector<float> hidden_cols;
    int a = 0;
    int32_t bonus = 0;

    if (use_snap) {
        const uint64_t t_verify0 = now_us();
        // Verify does NOT write dn_state (snapshots captured instead); KV is
        // written at [P..P+D] but get_pos is not advanced.
        run_mtp_verify_snapshot(verify_tokens, P, slot_idx, argmax_cols, hidden_cols);
        mtp_verify_us_ += now_us() - t_verify0;

        while (a < D && argmax_cols[static_cast<size_t>(a)] == drafts[static_cast<size_t>(a)]) {
            ++a;
        }
        bonus = argmax_cols[static_cast<size_t>(a)];

        // Commit: copy the accepted DeltaNet snapshot column (state after token a)
        // into dn_state, then advance KV to P+a+1 (rows [P..P+a] already written).
        const uint64_t t_commit0 = now_us();
        run_mtp_commit(a);
        advance_cache(static_cast<uint32_t>(a + 1), slot_idx);
        mtp_replay_us_ += now_us() - t_commit0;
        if (a < D) { ++mtp_replay_count_; }
    } else {
        // Non-snapshot path: shadow + replay.
        const uint64_t t_verify0 = now_us();
        dn_shadow_->copy_all_from(*dn_state_);  // shadow = committed state [0..P-1]
        run_mtp_verify(verify_tokens, P, slot_idx, scheduler, argmax_cols, hidden_cols);
        mtp_verify_us_ += now_us() - t_verify0;

        while (a < D && argmax_cols[static_cast<size_t>(a)] == drafts[static_cast<size_t>(a)]) {
            ++a;
        }
        bonus = argmax_cols[static_cast<size_t>(a)];

        if (a == D) {
            // All drafts accepted: verify state is exactly what we want.
            advance_cache(static_cast<uint32_t>(verify_tokens.size()), slot_idx);
        } else {
            // Partial acceptance: discard the speculative tail. Attention KV is
            // truncated (O(1)); DeltaNet is restored from the shadow and the
            // committed tokens are replayed through the trunk.
            const uint64_t t_replay0 = now_us();
            dn_state_->copy_all_from(*dn_shadow_);      // restore [0..P-1]
            kv_cache_->truncate_to_position(P, slot_idx);
            std::vector<int32_t> committed(verify_tokens.begin(), verify_tokens.begin() + (a + 1));
            // Replay the committed prefix. The cached-graph advance avoids the
            // ~3ms/step rebuild that run_prefill pays; fall back on paged/mixed.
            if (!paged_kv_enabled_ && !model_->is_mixed_mode()) {
                run_mtp_advance(committed, P, slot_idx);
                advance_cache(static_cast<uint32_t>(committed.size()), slot_idx);
            } else {
                run_prefill(committed, P, slot_idx, scheduler);
            }
            mtp_replay_us_ += now_us() - t_replay0;
            ++mtp_replay_count_;
        }
    }

    // 5) Refresh the nextn KV for the committed span using the true trunk hiddens
    //    (token@q paired with hidden@(q-1)); replaces the speculative draft KV.
    {
        const int committed_len = a + 1;  // seed_tok + a accepted drafts
        std::vector<int32_t> proc_tokens(verify_tokens.begin(), verify_tokens.begin() + committed_len);
        std::vector<float> proc_hidden(static_cast<size_t>(n_embd) * committed_len);
        // row 0 (seed_tok@P) pairs with hidden@(P-1) == the seed_h we drafted from
        std::memcpy(proc_hidden.data(), seed_h.data(), static_cast<size_t>(n_embd) * sizeof(float));
        for (int i = 1; i < committed_len; ++i) {
            // token at column i (position P+i) pairs with verify hidden col (i-1)
            std::memcpy(proc_hidden.data() + static_cast<size_t>(i) * n_embd,
                        hidden_cols.data() + static_cast<size_t>(i - 1) * n_embd,
                        static_cast<size_t>(n_embd) * sizeof(float));
        }
        const uint64_t t_proc0 = now_us();
        mtp_kv_->truncate_to_position(P, slot_idx);
        run_mtp_process(proc_tokens, proc_hidden, P, slot_idx, scheduler);
        mtp_process_us_ += now_us() - t_proc0;
    }

    // 6) Emit accepted drafts + bonus; set up the next seed.
    for (int i = 0; i < a; ++i) {
        out.push_back(drafts[static_cast<size_t>(i)]);
    }
    out.push_back(bonus);

    seed_tok = bonus;
    pos = P + a + 1;
    // seed_h for the next step = trunk hidden at position P+a (verify column a).
    seed_h.assign(hidden_cols.begin() + static_cast<size_t>(a) * n_embd,
                  hidden_cols.begin() + static_cast<size_t>(a + 1) * n_embd);

    // stats
    ++mtp_step_count_;
    mtp_draft_total_ += static_cast<uint64_t>(D);
    mtp_accept_total_ += static_cast<uint64_t>(a);
    mtp_emit_total_ += static_cast<uint64_t>(a + 1);

    return a + 1;
}

void Qwen35moeForwardPass::mtp_log_stats() const {
    if (mtp_step_count_ == 0) {
        return;
    }
    const double acc_rate = mtp_draft_total_ > 0
        ? static_cast<double>(mtp_accept_total_) / static_cast<double>(mtp_draft_total_)
        : 0.0;
    const double tok_per_step =
        static_cast<double>(mtp_emit_total_) / static_cast<double>(mtp_step_count_);
    const double steps = static_cast<double>(mtp_step_count_);
    std::fprintf(stderr,
        "[mtp] steps=%llu draft=%llu accepted=%llu emitted=%llu "
        "accept_rate=%.3f tokens/step=%.2f\n",
        static_cast<unsigned long long>(mtp_step_count_),
        static_cast<unsigned long long>(mtp_draft_total_),
        static_cast<unsigned long long>(mtp_accept_total_),
        static_cast<unsigned long long>(mtp_emit_total_),
        acc_rate, tok_per_step);
    // Per-step averages (ms). replay is amortised over ALL steps but only occurs
    // on partial acceptance, so replay_when_used shows its true per-occurrence cost.
    const double replay_when_used = mtp_replay_count_ > 0
        ? static_cast<double>(mtp_replay_us_) / static_cast<double>(mtp_replay_count_) / 1000.0
        : 0.0;
    std::fprintf(stderr,
        "[mtp] per-step ms: draft=%.2f verify=%.2f process=%.2f replay_avg=%.2f "
        "(replay hit %llu/%llu steps, %.2f ms each)\n",
        static_cast<double>(mtp_draft_us_) / steps / 1000.0,
        static_cast<double>(mtp_verify_us_) / steps / 1000.0,
        static_cast<double>(mtp_process_us_) / steps / 1000.0,
        static_cast<double>(mtp_replay_us_) / steps / 1000.0,
        static_cast<unsigned long long>(mtp_replay_count_),
        static_cast<unsigned long long>(mtp_step_count_),
        replay_when_used);
}
