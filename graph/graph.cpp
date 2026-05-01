#include "graph/graph.h"

Qwen35moeForwardPass::Qwen35moeForwardPass() {
}

Qwen35moeForwardPass::~Qwen35moeForwardPass() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

int Qwen35moeForwardPass::init(const uint32_t context_len, const uint32_t max_batch_size, std::shared_ptr<Qwen35moeModel> model) {
    model_ = model;
    auto& m = model_->meta_->qwen35moe;

    // Pre-allocate persistent buffer for graph metadata
    ctx_buffer_.resize(FP_GRAPH_SIZE_METADATA);

    struct ggml_init_params params = {
        .mem_size   = ctx_buffer_.size(),
        .mem_buffer = ctx_buffer_.data(),
        .no_alloc   = true,
    };
    ctx_ = ggml_init(params);

    kv_layer_map_.assign(m.block_count, -1);
    dn_layer_map_.assign(m.block_count, -1);
    int kv_idx = 0, dn_idx = 0;
    for (uint32_t il = 0; il < m.block_count; ++il) {
        if (is_full_attention_layer(il))
            kv_layer_map_[il] = kv_idx++;
        else
            dn_layer_map_[il] = dn_idx++;
    }

    // KV cache — 10 attention layers, F32, on Metal if available.
    ggml_backend_t cache_backend = model_->get_curr_backend();

    const int n_kv_layers = kv_idx;  // 10
    const uint32_t n_embd_k = static_cast<uint32_t>(m.head_count_kv * m.key_length);
    const uint32_t n_embd_v = static_cast<uint32_t>(m.head_count_kv * m.value_length);
    kv_cache_ = std::make_unique<simple_kv_cache>(
        static_cast<uint32_t>(n_kv_layers),
        context_len,
        max_batch_size,
        n_embd_k, n_embd_v,
        GGML_TYPE_F32, GGML_TYPE_F32,
        cache_backend
    );

    const int n_dn_layers = dn_idx;  // 30
    // DeltaNet state — 30 DeltaNet layers, backend-backed.
    const uint32_t d_inner       = m.inner_size;     // 4096
    const uint32_t num_v_heads   = m.time_step_rank; // 32
    const uint32_t num_k_heads   = m.group_count;    // 16
    const uint32_t head_v_dim    = d_inner / num_v_heads; // 128
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.state_size; // 8192

    DeltaNetStateParams dn_state_hp {
        static_cast<uint32_t>(n_dn_layers),
        max_batch_size,
        head_v_dim,
        m.state_size,  // head_k_dim = 128
        num_v_heads,
        conv_channels,
        m.conv_kernel, // 4
        cache_backend
    };
    dn_state_ = std::make_unique<DeltaNetState>(dn_state_hp);

    return 0;
}

void Qwen35moeForwardPass::reset_context() {
    if (ctx_) {
        ggml_free(ctx_);
    }
    struct ggml_init_params params = {
        .mem_size   = ctx_buffer_.size(),
        .mem_buffer = ctx_buffer_.data(),
        .no_alloc   = true, 
    };
    ctx_ = ggml_init(params);
}

ggml_cgraph* Qwen35moeForwardPass::build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx) {
    reset_context();

    ggml_cgraph* gf = new_graph();

    auto& m = model_->meta_->qwen35moe;
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    const uint32_t d_inner = m.inner_size;
    const uint32_t num_v_heads = m.time_step_rank;
    const uint32_t num_k_heads = m.group_count;
    const uint32_t head_v_dim = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.state_size;
    DeltaNetStateParams dn_hp {
        0, 0,             // n_dn_layers / n_slots not used in helper
        head_v_dim,
        m.state_size,
        num_v_heads,
        conv_channels,
        m.conv_kernel,
        nullptr
    };

    // 1. Token embedding
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(inpL, "inpL");

    // 2. Position tensor (shared by all attention layers)
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp_pos);
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. Transformer loop
    for (uint32_t il = 0; il < m.block_count; ++il) {
        ggml_tensor* inpSA = inpL;

        // ── Pre-attention norm ──────────────────────────────────────────────
        struct ggml_tensor* attn_norm_weight = model_->get_attn_norm_weight(il);
        ggml_tensor* cur = build_norm(gf, inpL, attn_norm_weight, il);

        // ── Attention or DeltaNet ───────────────────────────────────────────
        if (is_full_attention_layer(il)) {
            int kv_idx = kv_layer_map_[il];

            // Gated attention: joint Q+Gate projection, Q weight outputs
            // [(n_embd_head*2)*n_head, n_tokens]. build_gated_attention
            // handles the strided view split, sigmoid gating, and out-proj.
            struct ggml_tensor* attn_q_weight = model_->get_attn_q_weight(il);
            struct ggml_tensor* attn_q_norm_weight = model_->get_attn_q_norm_weight(il);
            struct ggml_tensor* attn_k_weight = model_->get_attn_k_weight(il);
            struct ggml_tensor* attn_k_norm_weight = model_->get_attn_k_norm_weight(il);
            struct ggml_tensor* attn_v_weight = model_->get_attn_v_weight(il);
            struct ggml_tensor* attn_output_weight = model_->get_attn_output_weight(il);
            cur = build_gated_attention(
                ctx_, gf, kv_cache_.get(), cur, inp_pos,
                kv_idx, n_tok, slot_idx, il, attn_q_weight, 
                attn_q_norm_weight, attn_k_weight, attn_k_norm_weight, attn_v_weight, attn_output_weight,
                m.key_length, m.head_count, m.head_count_kv, m.dimension_count, m.freq_base,
                static_cast<int>(m.context_length), m.layer_norm_rms_epsilon
            );
        } else {
            // DeltaNet layer
            uint32_t dn_idx = static_cast<uint32_t>(dn_layer_map_[il]);
            cur = build_dn_layer(ctx_, gf, cur, dn_state_.get(), dn_hp, num_k_heads, 
                m.embedding_length, dn_idx, n_tok, slot_idx, m.layer_norm_rms_epsilon, il
            );
        }

        // ── Residual 1 (attention / DeltaNet) ──────────────────────────────
        cur = ggml_add(ctx_, cur, inpSA);

        // ── Pre-FFN norm ────────────────────────────────────────────────────
        ggml_tensor* ffn_inp = cur;
        struct ggml_tensor* post_attention_norm = model_->get_post_attention_norm_weight(il);
        cur = build_norm(gf, cur, post_attention_norm, il);

        // ── MoE FFN ─────────────────────────────────────────────────────────
        cur = build_moe_layer(ctx_, gf, cur, il);

        // ── Residual 2 (FFN) ─────────────────────────────────────────────────
        cur = ggml_add(ctx_, cur, ffn_inp);
        set_tensor_name(cur, "layer_out", il);

        inpL = cur;
    }

    // 4. Final norm + LM head
    build_output_head(gf, inpL);

    return gf;
}

ggml_cgraph* Qwen35moeForwardPass::build_decoding_graph(const std::vector<int32_t>& tokens, const std::vector<uint32_t>& slots, const std::vector<int32_t>&  positions) {
    reset_context();

    ggml_cgraph* gf = new_graph();

    auto& m = model_->meta_->qwen35moe;
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());

    // Derive DeltaNet state DeltaNetParams from metadata for the helper.
    const uint32_t d_inner = m.inner_size;
    const uint32_t num_v_heads = m.time_step_rank;
    const uint32_t num_k_heads = m.group_count;
    const uint32_t head_v_dim = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.state_size;
    DeltaNetParams dn_hp {
        static_cast<int>(m.embedding_length),
        static_cast<int>(head_v_dim * num_v_heads),
        static_cast<int>(m.state_size),
        static_cast<int>(num_k_heads),
        static_cast<int>(num_v_heads),
        static_cast<int>(head_v_dim),
        static_cast<int>(conv_channels),
        static_cast<int>(m.conv_kernel),
        m.layer_norm_rms_epsilon
    };

    // 1. Token embedding
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(inpL, "inpL");

    // 2. Position tensor (one per batch element)
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_batch);
    ggml_set_input(inp_pos);
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. KV gather mask — shared across all attention layers.
    uint32_t max_physical = 0;
    for (uint32_t s : slots) {
        uint32_t phys = get_physical_cache_pos(s);
        if (phys > max_physical) 
            max_physical = phys;
    }
    const uint32_t n_kv_len = max_physical + 1;  // +1 for token being written
    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, n_kv_len, 1, 1, n_batch);
    ggml_set_input(kq_mask);
    set_tensor_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    ggml_tensor* gather_indices = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, static_cast<int64_t>(n_batch * n_kv_len));
    ggml_set_input(gather_indices);
    set_tensor_name(gather_indices, "gather_indices");

    // 4. Transformer loop
    for (uint32_t il = 0; il < m.block_count; ++il) {
        ggml_tensor* inpSA = inpL;

        // Pre-attention norm
        struct ggml_tensor* attn_norm_weight = model_->get_attn_norm_weight(il);
        ggml_tensor* cur = build_norm(gf, inpL, attn_norm_weight, il);

        if (is_full_attention_layer(il)) {
            int kv_idx = kv_layer_map_[il];

            struct ggml_tensor* attn_q_weight = model_->get_attn_q_weight(il);
            struct ggml_tensor* attn_q_norm_weight = model_->get_attn_q_norm_weight(il);
            struct ggml_tensor* attn_k_weight = model_->get_attn_k_weight(il);
            struct ggml_tensor* attn_k_norm_weight = model_->get_attn_k_norm_weight(il);
            struct ggml_tensor* attn_v_weight = model_->get_attn_v_weight(il);
            struct ggml_tensor* attn_output_weight = model_->get_attn_output_weight(il);
            cur = build_gated_batched_attention(
                ctx_, gf, kv_cache_.get(), cur, inp_pos,
                kq_mask, gather_indices,
                kv_idx, slots, positions, il,
                attn_q_weight, attn_q_norm_weight,
                attn_k_weight, attn_k_norm_weight,
                attn_v_weight, attn_output_weight,
                m.key_length, m.head_count, m.head_count_kv,
                m.dimension_count, m.freq_base,
                static_cast<int>(m.context_length),
                m.layer_norm_rms_epsilon
            );
        } else {
            uint32_t dn_idx = static_cast<uint32_t>(dn_layer_map_[il]);
            // One token per slot: pass slots vector to DeltaNet decode path.
            DecodeArgs da{slots};
            PrefillArgs pa_unused{1, 0};

            cur = build_all_deltanet_layer(ctx_, gf, cur, dn_idx, Phase::Decode, pa_unused, &da, dn_state_.get(), dn_hp, il);
        }

        // Residual 1
        cur = ggml_add(ctx_, cur, inpSA);

        // Pre-FFN norm + MoE
        ggml_tensor* ffn_inp = cur;
        struct ggml_tensor* post_attention_norm = model_->get_post_attention_norm_weight(il);
        cur = build_norm(gf, cur, post_attention_norm, il);
        cur = build_moe_layer(ctx_, gf, cur, il);

        // Residual 2
        cur = ggml_add(ctx_, cur, ffn_inp);
        inpL = cur;
    }

    build_output_head(gf, inpL);
    return gf;
}

void Qwen35moeForwardPass::set_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens, int pos) {
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());

    // Tokens
    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) 
        throw std::runtime_error("qwen36: 'tokens' tensor missing from graph");
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, n_tok * sizeof(int32_t));

    // Position IDs
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) 
        throw std::runtime_error("qwen36: 'inp_pos' tensor missing from graph");

    std::vector<int32_t> pos_data(n_tok);
    for (uint32_t i = 0; i < n_tok; ++i) 
        pos_data[i] = pos + static_cast<int>(i);
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, n_tok * sizeof(int32_t));

    // Causal masks — only for attention layers.
    // build_attention() names each layer's mask "kq_mask.{physical_il}".
    for (uint32_t il = 0; il < model_->meta_->qwen35moe.block_count; ++il) {
        if (!is_full_attention_layer(il)) 
            continue;

        char name[32];
        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, name);
        if (!kq_mask) 
            continue;  // mask may not exist if kv_cache was empty

        const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        std::vector<float> mask(n_kv * n_tok);
        for (uint32_t t = 0; t < n_tok; ++t) {
            const uint32_t q_pos = static_cast<uint32_t>(pos) + t;
            for (uint32_t j = 0; j < n_kv; ++j)
                mask[t * n_kv + j] = (j <= q_pos) ? 0.0f : -INFINITY;
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
    }
}

void Qwen35moeForwardPass::set_batched_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots, const std::vector<int32_t>&  positions) {
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());

    // Tokens
    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) throw std::runtime_error("qwen36: 'tokens' tensor missing from graph");
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, n_batch * sizeof(int32_t));

    // Position IDs (one per batch slot)
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) throw std::runtime_error("qwen36: 'inp_pos' tensor missing from graph");
    ggml_backend_tensor_set(pos_t, positions.data(), 0, n_batch * sizeof(int32_t));

    // Shared KV mask [n_kv_len, 1, 1, n_batch]
    ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, "kq_mask_b");
    if (kq_mask) {
        const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        std::vector<float> mask(n_kv * n_batch, -INFINITY);
        for (uint32_t b = 0; b < n_batch; ++b) {
            const uint32_t q_pos = static_cast<uint32_t>(positions[b]);
            for (uint32_t j = 0; j <= q_pos && j < n_kv; ++j)
                mask[b * n_kv + j] = 0.0f;
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
    }

    // Gather indices [n_batch * n_kv_len]
    ggml_tensor* gi = ggml_graph_get_tensor(gf, "gather_indices");
    if (gi) {
        const uint32_t n_kv   = static_cast<uint32_t>(gi->ne[0]) / n_batch;
        std::vector<int32_t> idx(n_batch * n_kv);
        for (uint32_t b = 0; b < n_batch; ++b) {
            const uint32_t slot = slots[b];
            for (uint32_t j = 0; j < n_kv; ++j)
                idx[b * n_kv + j] = static_cast<int32_t>(slot * n_kv + j);
        }
        ggml_backend_tensor_set(gi, idx.data(), 0, idx.size() * sizeof(int32_t));
    }
}

std::vector<float> Qwen35moeForwardPass::run_prefill(const std::vector<int32_t>& tokens, int pos, 
    uint32_t slot_idx, ggml_backend_sched_t scheduler) {
    // Default: monolithic path (subclasses override for TQ)
    ggml_backend_sched_reset(scheduler);
    ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
    ggml_backend_sched_alloc_graph(scheduler, gf);
    set_inputs(gf, tokens, pos);
    ggml_backend_sched_graph_compute(scheduler, gf);
    advance_cache(tokens.size(), slot_idx);
    return get_output_logits(gf);
}

uint32_t Qwen35moeForwardPass::get_cache_pos(uint32_t slot_idx) const {
    uint32_t seq = snapkv_get_seq_pos(slot_idx);
    if (seq > 0) return seq;
    return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
}

ggml_cgraph* Qwen35moeForwardPass::new_graph() {
    return ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);
}

ggml_tensor* Qwen35moeForwardPass::embedding(ggml_cgraph* gf, const std::vector<int32_t>& tokens) {
    const size_t n_tokens = tokens.size();

    // 1. Create a 1D tensor from the input token IDs
    struct ggml_tensor* tokens_tensor = ggml_new_tensor_1d(
        ctx_,
        GGML_TYPE_I32,
        n_tokens
    );
    
    ggml_set_input(tokens_tensor);
    set_tensor_name(tokens_tensor, "tokens");
    ggml_build_forward_expand(gf, tokens_tensor);
    // memcpy(tokens_tensor->data, tokens.data(), ggml_nbytes(tokens_tensor));

    // 2. Perform the embedding lookup using ggml_get_rows
    struct ggml_tensor* token_embedding = model_->get_token_embedding_weight();
    ggml_tensor * cur = ggml_get_rows(
        ctx_,
        token_embedding,
        tokens_tensor
    );

    set_tensor_name(cur, "embed_lookup");
    return cur;
}

// Inline MoE FFN for one physical layer, after the pre-FFN norm has been applied.
// Returns the FFN output (before residual). il is the physical layer index.
ggml_tensor* Qwen35moeForwardPass::build_moe_layer(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  input,
    int il
)
{
    struct ggml_tensor* ffn_gate_shexp = model_->get_ffn_gate_shexp_weight(il);
    struct ggml_tensor* ffn_up_shexp = model_->get_ffn_up_shexp_weight(il);
    struct ggml_tensor* ffn_down_shexp = model_->get_ffn_down_shexp_weight(il);
    if (!ffn_gate_shexp || !ffn_up_shexp || !ffn_down_shexp) {
        throw std::runtime_error("moe_layer: has_shared_expert=true but shared expert weights are null");
    }
    
    struct ggml_tensor* ffn_gate_inp = model_->get_ffn_gate_inp_weight(il);
    struct ggml_tensor* ffn_gate_exps = model_->get_ffn_gate_exps_weight(il);
    struct ggml_tensor* ffn_up_exps = model_->get_ffn_up_exps_weight(il);
    struct ggml_tensor* ffn_down_exps = model_->get_ffn_down_exps_weight(il);
    struct ggml_tensor* ffn_gate_inp_shexp = model_->get_ffn_gate_inp_shexp_weight(il);

    auto& m = model_->meta_->qwen35moe;
    // input: [n_embd, n_tokens]
    const int64_t n_embd   = input->ne[0];
    const int64_t n_tokens = input->ne[1];
    const int     n_exp    = m.expert_count;
    const int     top_k    = m.expert_used_count;
    const int64_t ffn_dim  = m.expert_feed_forward_length;

    // ── 1. Routing logits and top-k gating ────────────────────────────────────
    // logits: [n_experts, n_tokens]
    ggml_tensor* logits = ggml_mul_mat(ctx, ffn_gate_inp, input);
    set_tensor_name(logits, "moe_logits", il);

    // Get indices of top-k experts
    // sorted_idx: [n_experts, n_tokens] I32
    ggml_tensor* sorted_idx = ggml_argsort(ctx, logits, GGML_SORT_ORDER_DESC);
    // expert_idx: [top_k, n_tokens] I32
    ggml_tensor* expert_idx = ggml_view_2d(ctx, sorted_idx,
        top_k, n_tokens,
        sorted_idx->nb[1],
        0
    );
    set_tensor_name(expert_idx, "moe_idx", il);

    // Gather the actual logit values for the top-k experts
    // To use ggml_get_rows per token, we reshape logits to [1, n_experts, n_tokens]
    // so that ggml_get_rows picks from the n_experts dimension (ne[1]).
    ggml_tensor* logits_3d = ggml_reshape_3d(ctx, logits, 1, n_exp, n_tokens);
    ggml_tensor* expert_logits = ggml_get_rows(ctx, logits_3d, expert_idx);
    // expert_logits is [1, top_k, n_tokens], reshape back to 2D for softmax
    expert_logits = ggml_reshape_2d(ctx, expert_logits, top_k, n_tokens);

    // Apply softmax over top-k weights per token to normalize routing weights
    ggml_tensor* expert_weights = ggml_soft_max(ctx, expert_logits);
    set_tensor_name(expert_weights, "moe_weights", il);

    // ── 2. Expert dispatch via ggml_mul_mat_id (QINF_MOE_FALLBACK path) ───────
    //
    // ggml_mul_mat_id: batched matmul where each token uses a different expert
    // weight matrix. Signature: (W [in, out, n_exp], x [in, n_tok], idx [top_k, n_tok])
    // Returns: [out, top_k, n_tok]

    // Reshape input for ggml_mul_mat_id: [in, n_tok] -> [in, 1, n_tok]
    // This aligns b->ne[2] with ids->ne[1] (n_tokens).
    ggml_tensor* input_3d = ggml_reshape_3d(ctx, input, n_embd, 1, n_tokens);

    // Gate projection: each token × its top_k expert gate weights
    ggml_tensor* exp_gate_out = ggml_mul_mat_id(ctx, ffn_gate_exps, input_3d, expert_idx);
    set_tensor_name(exp_gate_out, "moe_exp_gate", il);
    // [ffn_dim, top_k, n_tokens]

    // Up projection
    ggml_tensor* exp_up_out = ggml_mul_mat_id(ctx, ffn_up_exps, input_3d, expert_idx);
    set_tensor_name(exp_up_out, "moe_exp_up", il);

    // SwiGLU activation: silu(gate) * up
    ggml_tensor* exp_act = ggml_mul(ctx, ggml_silu(ctx, exp_gate_out), exp_up_out);
    set_tensor_name(exp_act, "moe_exp_act", il);
    // [ffn_dim, top_k, n_tokens]

    // Down projection
    // exp_act: [ffn_dim, top_k, n_tokens] — need to reshape for ggml_mul_mat_id
    // which expects x as [in_dim, n_tokens] with index [top_k, n_tokens].
    // We reshape exp_act to treat each (token, topk) independently:
    // ggml_mul_mat_id with w_exp_down [ffn_dim, n_embd, n_exp], input [ffn_dim, top_k, n_tokens]
    // This correctly picks the down-weight for each expert per token.
    ggml_tensor* exp_down_out = ggml_mul_mat_id(ctx, ffn_down_exps, exp_act, expert_idx);
    set_tensor_name(exp_down_out, "moe_exp_down", il);
    // [n_embd, top_k, n_tokens]

    // ── 3. Weighted sum of expert outputs ─────────────────────────────────────

    // expert_weights: [top_k, n_tokens] — reshape to [1, top_k, n_tokens] for broadcast
    ggml_tensor* w_expanded = ggml_reshape_3d(ctx, expert_weights, 1, top_k, n_tokens);

    // exp_down_out: [n_embd, top_k, n_tokens]
    // Multiply each expert output by its routing weight
    ggml_tensor* weighted = ggml_mul(ctx, exp_down_out, w_expanded);
    set_tensor_name(weighted, "moe_weighted", il);

    // Sum over top_k dimension → [n_embd, n_tokens].
    // weighted: [n_embd, top_k, n_tokens]; slice each expert via view and accumulate.
    ggml_tensor* routed_out = ggml_view_2d(ctx, weighted, n_embd, n_tokens, weighted->nb[2], 0);
    for (int k = 1; k < top_k; ++k) {
        ggml_tensor* expert_k = ggml_view_2d(ctx, weighted,
            n_embd, n_tokens, weighted->nb[2],
            static_cast<size_t>(k) * weighted->nb[1]
        );
        routed_out = ggml_add(ctx, routed_out, expert_k);
    }
    set_tensor_name(routed_out, "moe_routed_out", il);

    // ── 4. Shared expert (optional) ───────────────────────────────────────────
    // Shared expert: standard SwiGLU FFN on all tokens
    ggml_tensor* sh_gate_out = ggml_mul_mat(ctx, ffn_gate_shexp, input);
    ggml_tensor* sh_up_out   = ggml_mul_mat(ctx, ffn_up_shexp,   input);
    ggml_tensor* sh_act      = ggml_mul(ctx, ggml_silu(ctx, sh_gate_out), sh_up_out);
    ggml_tensor* sh_down_out = ggml_mul_mat(ctx, ffn_down_shexp, sh_act);
    set_tensor_name(sh_down_out, "moe_shared_out", il);

    // Per-token scalar gate: ffn_gate_inp_shexp is [n_embd], mul_mat gives [1, n_tokens],
    // sigmoid maps to (0,1), ggml_mul broadcasts over [n_embd, n_tokens].
    ggml_tensor* sh_gate_logit = ggml_mul_mat(ctx, ffn_gate_inp_shexp, input);
    ggml_tensor* sh_gate       = ggml_sigmoid(ctx, sh_gate_logit);
    ggml_tensor* sh_contribution = ggml_mul(ctx, sh_down_out, sh_gate);
    set_tensor_name(sh_contribution, "moe_shared_contrib", il);

    ggml_tensor* combined = ggml_add(ctx, routed_out, sh_contribution);
    set_tensor_name(combined, "moe_combined", il);

    return combined;
}

ggml_tensor* Qwen35moeForwardPass::build_rms_norm(
    ggml_context* ctx,
    ggml_tensor*  cur,
    ggml_tensor*  weight,
    float         eps,
    int           il
)
{
    cur = ggml_rms_norm(ctx, cur, eps);
    set_tensor_name(cur, "cur_rms_normed", il);
    cur = ggml_mul(ctx, cur, weight);
    return cur;
}

// Extracted from ForwardPassBase::ffn_swiglu (src/models/forward_pass_base.cpp).
// Logic is identical — only ctx_ → ctx parameter.
ggml_tensor* Qwen35moeForwardPass::build_ffn_swiglu(
    ggml_context* ctx,
    ggml_cgraph*  /*gf*/,
    ggml_tensor*  cur,
    ggml_tensor*  gate,
    ggml_tensor*  up,
    ggml_tensor*  down,
    int           il
)
{
    char name[128];

    ggml_tensor* tmp = ggml_mul_mat(ctx, up, cur);
    snprintf(name, sizeof(name), "ffn_up.%d", il);
    set_tensor_name(tmp, name);

    cur = ggml_mul_mat(ctx, gate, cur);
    snprintf(name, sizeof(name), "ffn_gate.%d", il);
    set_tensor_name(cur, name);

    cur = ggml_swiglu_split(ctx, cur, tmp);
    snprintf(name, sizeof(name), "ffn_swiglu.%d", il);
    set_tensor_name(cur, name);

    cur = ggml_mul_mat(ctx, down, cur);
    return cur;
}

ggml_tensor* Qwen35moeForwardPass::build_norm(
    ggml_cgraph* gf,
    ggml_tensor* cur,
    ggml_tensor* mw,
    int il
)
{
    return build_rms_norm(ctx_, cur, mw, model_->meta_->qwen35moe.layer_norm_rms_epsilon, il);
}

// ── build_attn_mha ───────────────────────────────────────────────────────────
// Extracted from ForwardPassBase::build_attn_mha (src/models/forward_pass_base.cpp).
// Logic is identical — only ctx_ → ctx parameter.
ggml_tensor* Qwen35moeForwardPass::build_attn_mha(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  q,
    ggml_tensor*  k,
    ggml_tensor*  v,
    ggml_tensor*  kq_mask,
    ggml_tensor*  sinks,
    float         kq_scale,
    uint32_t      pos,
    int           il
)
{
    (void)gf; (void)pos; // gf/pos unused directly; kept for API symmetry with callers

    const bool v_trans = v->nb[1] > v->nb[2];
    (void)v_trans;

    const auto n_stream = k->ne[3];

    q = ggml_reshape_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream);
    set_tensor_name(q, "q_reshaped", il);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    set_tensor_name(q, "q_permuted", il);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    set_tensor_name(k, "k_permuted", il);
    v = ggml_permute(ctx, v, 1, 2, 0, 3);
    set_tensor_name(v, "v_permuted", il);
    v = ggml_cont(ctx, v);
    set_tensor_name(v, "v_cont", il);

    ggml_tensor* cur;
    {
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
        set_tensor_name(kq, "kq", il);

        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0);
        set_tensor_name(kq, "kq_soft", il);

        ggml_soft_max_add_sinks(kq, sinks);

        ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);
        set_tensor_name(kqv, "kqv", il);

        cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        set_tensor_name(cur, "kqv_permuted", il);

        cur = ggml_cont_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        set_tensor_name(cur, "attn_recombined", il);
    }

    return cur;
}

// ── build_gated_attention ─────────────────────────────────────────────────────
// Gated attention variant used by Qwen3.5 and Qwen3.6: joint Q+Gate projection,
// Q/K RMS norms, partial RoPE, sigmoid gating on the output.
ggml_tensor* Qwen35moeForwardPass::build_gated_attention(
    ggml_context*    ctx,
    ggml_cgraph*     gf,
    simple_kv_cache* kv_cache,
    ggml_tensor*     cur,
    ggml_tensor*     inp_pos,
    int              kv_cache_layer,
    uint32_t         n_tokens,
    uint32_t         slot_idx,
    int              il,
    ggml_tensor*     w_q,
    ggml_tensor*     w_q_norm,
    ggml_tensor*     w_k,
    ggml_tensor*     w_k_norm,
    ggml_tensor*     w_v,
    ggml_tensor*     w_out,
    int              n_embd_head,
    int              n_head,
    int              n_head_kv,
    int              n_rot,
    float            freq_base,
    int              context_length,
    float            rms_norm_eps
)
{
    // A. Joint Q+Gate projection
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx, w_q, cur);
    set_tensor_name(Qcur_full, "Qcur_full", il);

    // B. Extract Q via strided view (every other n_embd_head block)
    ggml_tensor* Qcur = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    set_tensor_name(Qcur, "Qcur", il);

    Qcur = build_rms_norm(ctx, Qcur, w_q_norm, rms_norm_eps, il);
    set_tensor_name(Qcur, "Qcur_normed", il);

    // C. K and V projections
    ggml_tensor* Kcur = ggml_mul_mat(ctx, w_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx, w_v, cur);

    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    Kcur = build_rms_norm(ctx, Kcur, w_k_norm, rms_norm_eps, il);
    set_tensor_name(Kcur, "Kcur_normed", il);

    // D. Extract Gate (offset by n_embd_head within each interleaved pair)
    ggml_tensor* gate = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);
    set_tensor_name(gate, "gate", il);

    // E. Partial RoPE
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // F. KV cache write + full-history read
    const float    kq_scale  = 1.0f / sqrtf(float(n_embd_head));
    const uint32_t cache_pos = kv_cache->get_pos(slot_idx);
    const uint32_t n_kv      = cache_pos + n_tokens;

    ggml_build_forward_expand(gf, kv_cache->cpy_k(ctx, Kcur, kv_cache_layer, slot_idx));
    ggml_build_forward_expand(gf, kv_cache->cpy_v(ctx, Vcur, kv_cache_layer, slot_idx));

    ggml_tensor* k_full = kv_cache->get_k(ctx, kv_cache_layer, n_kv, slot_idx);
    ggml_tensor* v_full = kv_cache->get_v(ctx, kv_cache_layer, n_kv, slot_idx);

    const int n_embd_kv = n_head_kv * n_embd_head;
    ggml_tensor* k_view = ggml_view_3d(ctx, k_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * sizeof(float), n_embd_kv * sizeof(float), 0);
    ggml_tensor* v_view = ggml_view_3d(ctx, v_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * sizeof(float), n_embd_kv * sizeof(float), 0);

    ggml_tensor* kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    set_tensor_name(kq_mask, "kq_mask", il);
    ggml_build_forward_expand(gf, kq_mask);

    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, cache_pos, il);

    // G. Sigmoid gating
    cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));
    set_tensor_name(cur, "attn_gated", il);

    // H. Output projection
    cur = ggml_mul_mat(ctx, w_out, cur);
    set_tensor_name(cur, "attn_output", il);

    return cur;
}

// Batched decode variant of build_gated_attention: same projections/norms/gating,
// operates on a batch of slots with pre-built kq_mask and gather_indices.
ggml_tensor* Qwen35moeForwardPass::build_gated_batched_attention(
    ggml_context*                ctx,
    ggml_cgraph*                 gf,
    simple_kv_cache*             kv_cache,
    ggml_tensor*                 cur,
    ggml_tensor*                 inp_pos,
    ggml_tensor*                 kq_mask,
    ggml_tensor*                 gather_indices,
    int                          kv_cache_layer,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>&  positions,
    int                          il,
    ggml_tensor*                 w_q,
    ggml_tensor*                 w_q_norm,
    ggml_tensor*                 w_k,
    ggml_tensor*                 w_k_norm,
    ggml_tensor*                 w_v,
    ggml_tensor*                 w_out,
    int                          n_embd_head,
    int                          n_head,
    int                          n_head_kv,
    int                          n_rot,
    float                        freq_base,
    int                          context_length,
    float                        rms_norm_eps
)
{
    const size_t n_batch = slots.size();

    // A. Joint Q+Gate projection → [(n_embd_head*2)*n_head, n_batch]
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx, w_q, cur);

    // B. Extract Q via strided view → [n_embd_head, n_head, n_batch]
    ggml_tensor* Qcur = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_batch,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0
    );

    Qcur = build_rms_norm(ctx, Qcur, w_q_norm, rms_norm_eps, il);

    // C. K and V projections
    ggml_tensor* Kcur = ggml_mul_mat(ctx, w_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx, w_v, cur);

    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_batch);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_batch);

    Kcur = build_rms_norm(ctx, Kcur, w_k_norm, rms_norm_eps, il);

    // D. Extract Gate → [n_embd_head*n_head, n_batch]
    ggml_tensor* gate = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_batch,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head
    );
    gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_batch);

    // E. Partial RoPE
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f
    );
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f
    );

    // F. Per-slot KV cache write
    const int n_embd_k = n_head_kv * n_embd_head;
    const int n_embd_v = n_head_kv * n_embd_head;

    ggml_tensor* k_storage_fmt = ggml_reshape_3d(ctx, Kcur, n_embd_k, 1, n_batch);
    ggml_tensor* v_storage_fmt = ggml_reshape_3d(ctx, Vcur, n_embd_v, 1, n_batch);

    for (size_t b = 0; b < n_batch; ++b) {
        ggml_tensor* k_slice = ggml_view_2d(ctx, k_storage_fmt, n_embd_k, 1, k_storage_fmt->nb[1], b * k_storage_fmt->nb[2]);
        ggml_tensor* v_slice = ggml_view_2d(ctx, v_storage_fmt, n_embd_v, 1, v_storage_fmt->nb[1], b * v_storage_fmt->nb[2]);

        ggml_build_forward_expand(gf, kv_cache->cpy_k(ctx, k_slice, kv_cache_layer, slots[b]));
        ggml_build_forward_expand(gf, kv_cache->cpy_v(ctx, v_slice, kv_cache_layer, slots[b]));
    }

    // G. Gather KV for all slots
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p > (int32_t)max_pos) 
            max_pos = (uint32_t)p;
    }
    uint32_t n_kv_len = max_pos + 1;
    ggml_tensor* k_gathered = kv_cache->gather_k(ctx, gf, kv_cache_layer, gather_indices, n_batch, n_kv_len);
    ggml_tensor* v_gathered = kv_cache->gather_v(ctx, gf, kv_cache_layer, gather_indices, n_batch, n_kv_len);

    ggml_tensor* k_view = ggml_view_4d(ctx, k_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_k    * sizeof(float),
        n_embd_k    * n_kv_len * sizeof(float), 0
    );
    ggml_tensor* v_view = ggml_view_4d(ctx, v_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_v    * sizeof(float),
        n_embd_v    * n_kv_len * sizeof(float), 0
    );

    // H. Attention
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, 0, il);

    // I. Sigmoid gating
    cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));

    // J. Output projection
    cur = ggml_mul_mat(ctx, w_out, cur);

    return cur;
}

ggml_tensor* Qwen35moeForwardPass::build_deltanet_layer(
    ggml_context*   ctx,
    ggml_cgraph*    gf,
    ggml_tensor*    cur,
    DeltaNetState*  dn_state,
    uint32_t        dn_idx,
    uint32_t        slot_idx,
    uint32_t        n_tokens,
    ggml_tensor*    w_qkv,
    ggml_tensor*    w_gate,
    ggml_tensor*    w_beta,
    ggml_tensor*    w_a,
    ggml_tensor*    w_dt_bias,
    ggml_tensor*    w_a_log,
    ggml_tensor*    w_conv,
    ggml_tensor*    w_norm,
    ggml_tensor*    w_out,
    int             n_embd,
    int             d_inner,
    int             head_k_dim,
    int             num_k_heads,
    int             num_v_heads,
    int             head_v_dim,
    int             conv_channels,
    int             conv_kernel,
    float           rms_norm_eps,
    int             il)
{
    const int64_t n_seq_tokens = static_cast<int64_t>(n_tokens);
    const int64_t n_seqs = 1;

    // ── 1. Input projections ─────────────────────────────────────────────────
    // QKV mixed: [n_embd, conv_channels] @ [n_embd, n_tokens]^T → [conv_channels, n_tokens]
    ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, w_qkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    set_tensor_name(qkv_mixed, "dn_qkv", il);

    // Z (output gate): [n_embd, d_inner] @ cur → [d_inner, n_tokens]
    ggml_tensor* z = ggml_mul_mat(ctx, w_gate, cur);
    set_tensor_name(z, "dn_z", il);

    // ── 2. Beta and decay-gate projections ───────────────────────────────────
    // Beta: sigmoid([n_embd, num_v_heads] @ cur) → [1, num_v_heads, n_tokens, 1]
    ggml_tensor* beta = ggml_mul_mat(ctx, w_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);
    set_tensor_name(beta, "dn_beta", il);

    // Alpha → decay gate: softplus(alpha @ cur + dt_bias) * A_log
    ggml_tensor* alpha = ggml_mul_mat(ctx, w_a, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* alpha_biased = ggml_add(ctx, alpha, w_dt_bias);
    ggml_tensor* alpha_sp     = ggml_softplus(ctx, alpha_biased);
    ggml_tensor* decay_gate   = ggml_mul(ctx, alpha_sp, w_a_log);
    decay_gate = ggml_reshape_4d(ctx, decay_gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    set_tensor_name(decay_gate, "dn_decay", il);

    // ── 3. Causal conv1d ─────────────────────────────────────────────────────
    ggml_tensor* conv_all = dn_state->conv_tensor(dn_idx);
    const int64_t conv_state_elems = static_cast<int64_t>(conv_kernel - 1) * conv_channels;

    // Extract the sliding window for this slot
    ggml_tensor* conv_states = ggml_view_1d(ctx, conv_all, conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    conv_states = ggml_reshape_3d(ctx, conv_states, conv_kernel - 1, conv_channels, n_seqs);
    set_tensor_name(conv_states, "dn_conv_st", il);

    // Concatenate conv window with new QKV tokens (along time axis = dim 0)
    ggml_tensor* qkv_t = ggml_transpose(ctx, qkv_mixed);
    ggml_tensor* conv_input = ggml_concat(ctx, conv_states, qkv_t, 0);
    set_tensor_name(conv_input, "dn_conv_in", il);

    // Update conv state: keep last (conv_kernel-1) tokens
    ggml_tensor* last_conv = ggml_view_3d(ctx, conv_input,
        conv_kernel - 1, conv_channels, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        static_cast<size_t>(conv_input->ne[0] - (conv_kernel - 1)) * ggml_element_size(conv_input)
    );
    ggml_tensor* conv_dst = ggml_view_1d(ctx, conv_all, conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last_conv, conv_dst));

    // Depthwise conv1d + SiLU activation
    ggml_tensor* conv_out = ggml_ssm_conv(ctx, conv_input, w_conv);
    conv_out = ggml_silu(ctx, conv_out);
    set_tensor_name(conv_out, "dn_conv_out", il);

    // ── 4. Split conv output into Q, K, V ────────────────────────────────────
    const int64_t qkv_dim  = static_cast<int64_t>(head_k_dim) * num_k_heads * 2 + static_cast<int64_t>(head_v_dim) * num_v_heads;
    const int64_t nb1_qkv  = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor* q_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens, 0
    );

    ggml_tensor* k_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        static_cast<size_t>(head_k_dim) * num_k_heads * ggml_element_size(conv_out)
    );

    ggml_tensor* v_conv = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_v_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        ggml_row_size(conv_out->type, 2LL * head_k_dim * num_k_heads)
    );

    // ── 5. L2-normalise Q and K ───────────────────────────────────────────────
    q_conv = ggml_l2_norm(ctx, q_conv, rms_norm_eps);
    k_conv = ggml_l2_norm(ctx, k_conv, rms_norm_eps);

    // Repeat K heads to match V heads if needed (GQA-style grouping)
    if (num_k_heads != num_v_heads) {
        q_conv = ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // ── 6. Gated delta-net recurrence (fused op) ──────────────────────────────
    ggml_tensor* rec_all = dn_state->recurrent_tensor(dn_idx);
    const int64_t rec_slot_floats = static_cast<int64_t>(head_v_dim) * head_k_dim * num_v_heads;

    ggml_tensor* S = ggml_view_1d(ctx, rec_all, rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    S = ggml_reshape_4d(ctx, S, head_v_dim, head_k_dim, num_v_heads, n_seqs);
    set_tensor_name(S, "dn_state_in", il);

    // ggml_gated_delta_net: fused gated-delta-rule forward pass.
    // Returns a tensor that packs both the per-token output AND the final state:
    //   output:    [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]   (first part)
    //   new_state: [head_v_dim, head_k_dim,  num_v_heads,  n_seqs]  (second part)
    ggml_tensor* result = ggml_gated_delta_net(ctx, q_conv, k_conv, v_conv, decay_gate, beta, S);
    set_tensor_name(result, "dn_gdn_result", il);

    // Extract per-token output view
    ggml_tensor* output = ggml_view_4d(ctx, result,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * num_v_heads),
        ggml_row_size(result->type, head_v_dim * num_v_heads * n_seq_tokens),
        0
    );
    set_tensor_name(output, "dn_delta_out", il);

    // Extract and write back the new recurrent state
    ggml_tensor* new_state = ggml_view_4d(ctx, result,
        head_v_dim, head_k_dim, num_v_heads, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim * num_v_heads),
        ggml_row_size(result->type, static_cast<int64_t>(head_v_dim) * num_v_heads * n_seq_tokens * n_seqs)
    );

    ggml_tensor* rec_dst = ggml_view_1d(ctx, rec_all, rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, new_state, rec_slot_floats), rec_dst));

    // ── 7. Gated RMSNorm ─────────────────────────────────────────────────────
    // output is [head_v_dim, num_v_heads, n_seq_tokens, n_seqs].
    // Normalize each head_v_dim vector independently (per-head, not d_inner-wide).
    // w_norm is [head_v_dim]; ggml broadcasts it over all heads and tokens.
    ggml_tensor* z_4d   = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* normed = ggml_rms_norm(ctx, output, rms_norm_eps);
    normed = ggml_mul(ctx, normed, w_norm);
    ggml_tensor* z_silu = ggml_silu(ctx, z_4d);
    ggml_tensor* gated  = ggml_mul(ctx, normed, z_silu);
    set_tensor_name(gated, "dn_gated", il);

    // ── 8. Output projection ─────────────────────────────────────────────────
    ggml_tensor* flat = ggml_reshape_3d(ctx, gated, static_cast<int64_t>(head_v_dim) * num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* out_proj = ggml_mul_mat(ctx, w_out, flat);
    set_tensor_name(out_proj, "dn_output", il);
    out_proj = ggml_reshape_2d(ctx, out_proj, n_embd, n_seq_tokens * n_seqs);

    return out_proj;
}

// Build the DeltaNet subgraph for physical layer il (DeltaNet index dn_idx).
ggml_tensor* Qwen35moeForwardPass::build_dn_layer(
    ggml_context*   ctx,
    ggml_cgraph*    gf,
    ggml_tensor*    cur,
    DeltaNetState*  dn_state,
    const DeltaNetStateParams& state_hp,
    uint32_t num_k_heads,
    uint32_t n_embd,
    uint32_t dn_idx,
    uint32_t n_tokens,
    uint32_t slot_idx,
    float    rms_norm_eps,
    int      il
)
{
    struct ggml_tensor* attn_qkv_weight = model_->get_attn_qkv_weight(il);
    struct ggml_tensor* attn_gate_weight = model_->get_attn_gate_weight(il);
    struct ggml_tensor* ssm_beta_weight = model_->get_ssm_beta_weight(il);
    struct ggml_tensor* ssm_alpha_weight = model_->get_ssm_alpha_weight(il);
    struct ggml_tensor* ssm_dt_bias = model_->get_ssm_dt_weight(il);
    struct ggml_tensor* ssm_a = model_->get_ssm_a_weight(il);
    struct ggml_tensor* ssm_conv1d_weight = model_->get_ssm_conv1d_weight(il);
    struct ggml_tensor* ssm_norm_weight = model_->get_ssm_norm_weight(il);
    struct ggml_tensor* ssm_out_weight = model_->get_ssm_out_weight(il);
    return build_deltanet_layer(
        ctx, gf, cur, dn_state,
        dn_idx, slot_idx, n_tokens,
        attn_qkv_weight,
        attn_gate_weight,
        ssm_beta_weight,
        ssm_alpha_weight,
        ssm_dt_bias,
        ssm_a,
        ssm_conv1d_weight,
        ssm_norm_weight,
        ssm_out_weight,
        static_cast<int>(n_embd),                                          // n_embd
        static_cast<int>(state_hp.head_v_dim * state_hp.num_v_heads),      // d_inner
        static_cast<int>(state_hp.head_k_dim),
        static_cast<int>(num_k_heads),
        static_cast<int>(state_hp.num_v_heads),
        static_cast<int>(state_hp.head_v_dim),
        static_cast<int>(state_hp.conv_channels),
        static_cast<int>(state_hp.conv_kernel),
        rms_norm_eps,
        il
    );
}

// All weight tensors are borrowed references; DeltaNetLayer does not own them.
// Unified build entry point. Returns output [n_embd, n_tokens/n_batch].
ggml_tensor* Qwen35moeForwardPass::build_all_deltanet_layer(
    ggml_context*      ctx,
    ggml_cgraph*       gf,
    ggml_tensor*       input,
    uint32_t           dn_idx,
    Phase              phase,
    const PrefillArgs& prefill_args,
    const DecodeArgs*  decode_args,
    DeltaNetState*     dn_state,
    const DeltaNetParams&     hp,
    int      il
)
{
    if (phase == Phase::Prefill) {
        return build_all_deltanet_layer_prefill(ctx, gf, input, dn_idx, prefill_args, dn_state, hp, il);
    } else {
        if (!decode_args) {
            throw std::runtime_error("DeltaNetLayer::build: decode_args must be non-null for Phase::Decode");
        }
        
        return build_all_deltanet_layer_decode(ctx, gf, input, dn_idx, *decode_args, dn_state, hp, il);
    }
}

ggml_tensor* Qwen35moeForwardPass::build_all_deltanet_layer_prefill(
    ggml_context*      ctx,
    ggml_cgraph*       gf,
    ggml_tensor*       input,
    uint32_t           dn_idx,
    const PrefillArgs& prefill_args,
    DeltaNetState*     dn_state,
    const DeltaNetParams&     hp,
    int      il
)
{
    // Derive n_embd from input shape
    const int n_embd = static_cast<int>(input->ne[0]);

    struct ggml_tensor* attn_qkv_weight = model_->get_attn_qkv_weight(il);
    struct ggml_tensor* attn_gate_weight = model_->get_attn_gate_weight(il);
    struct ggml_tensor* ssm_beta_weight = model_->get_ssm_beta_weight(il);
    struct ggml_tensor* ssm_alpha_weight = model_->get_ssm_alpha_weight(il);
    struct ggml_tensor* ssm_dt_bias = model_->get_ssm_dt_weight(il);
    struct ggml_tensor* ssm_a = model_->get_ssm_a_weight(il);
    struct ggml_tensor* ssm_conv1d_weight = model_->get_ssm_conv1d_weight(il);
    struct ggml_tensor* ssm_norm_weight = model_->get_ssm_norm_weight(il);
    struct ggml_tensor* ssm_out_weight = model_->get_ssm_out_weight(il);
    return build_deltanet_layer(
        ctx, gf, input, dn_state, dn_idx, prefill_args.slot_idx, prefill_args.n_tokens,
        attn_qkv_weight, attn_gate_weight, ssm_beta_weight, ssm_alpha_weight, ssm_dt_bias,
        ssm_a, ssm_conv1d_weight, ssm_norm_weight, ssm_out_weight,
        n_embd, hp.d_inner,
        hp.head_k_dim, hp.num_k_heads, hp.num_v_heads, hp.head_v_dim,
        hp.conv_channels, hp.conv_kernel, hp.rms_norm_eps,
        static_cast<int>(dn_idx)
    );
}

ggml_tensor* Qwen35moeForwardPass::build_all_deltanet_layer_decode(
    ggml_context*     ctx,
    ggml_cgraph*      gf,
    ggml_tensor*      input,
    uint32_t          dn_idx,
    const DecodeArgs& decode_args,
    DeltaNetState*     dn_state,
    const DeltaNetParams&     hp,
    int      il
)
{
    // Decode: batch of single tokens, one per slot.
    // We process each slot individually to keep state writes correct, then
    // concatenate the outputs. For n_batch == 1 this reduces to one prefill call.
    const int n_embd  = static_cast<int>(input->ne[0]);
    const int n_batch = static_cast<int>(decode_args.slots.size());

    if (n_batch == 0) {
        throw std::runtime_error("DeltaNetLayer::build_decode: slots must be non-empty");
    }

    if (n_batch == 1) {
        PrefillArgs pa;
        pa.n_tokens = 1;
        pa.slot_idx = decode_args.slots[0];
        return build_all_deltanet_layer_prefill(ctx, gf, input, dn_idx, pa, dn_state, hp, il);
    }

    // Multi-slot decode: slice the batch input per slot, process independently,
    // then concatenate. This is correct but not fused; Phase 4 can optimize.
    std::vector<ggml_tensor*> slot_outs;
    slot_outs.reserve(n_batch);

    for (int b = 0; b < n_batch; ++b) {
        // input[:, b] — one token for slot b
        ggml_tensor* token_in = ggml_view_2d(ctx, input,
            n_embd, 1,
            input->nb[1],
            static_cast<size_t>(b) * input->nb[1]
        );

        PrefillArgs pa;
        pa.n_tokens = 1;
        pa.slot_idx = decode_args.slots[b];
        ggml_tensor* out_b = build_all_deltanet_layer_prefill(ctx, gf, token_in, dn_idx, pa, dn_state, hp, il);
        slot_outs.push_back(out_b);
    }

    // Concatenate along token dimension
    ggml_tensor* combined = slot_outs[0];
    for (int b = 1; b < n_batch; ++b) {
        combined = ggml_concat(ctx, combined, slot_outs[b], 1);
    }
    
    return combined;
}

void Qwen35moeForwardPass::set_tensor_name(ggml_tensor* tensor, const char* name, int il) const {
    if (il != -1) {
        char new_name[128];
        snprintf(new_name, sizeof(new_name), "%s.%d", name, il);
        ggml_set_name(tensor, new_name);
    } else {
        ggml_set_name(tensor, name);
    }
}

void Qwen35moeForwardPass::build_output_head(ggml_cgraph* gf, ggml_tensor* cur) {
    struct ggml_tensor* output_norm = model_->get_output_norm_weight();
    cur = build_norm(gf, cur, output_norm, -1);
    set_tensor_name(cur, "final_norm");

    struct ggml_tensor* output = model_->get_output_weight();
    struct ggml_tensor* token_embedding = model_->get_token_embedding_weight();
    if (output != nullptr) {
        cur = ggml_mul_mat(ctx_, output, cur);
    } else {
        cur = ggml_mul_mat(ctx_, token_embedding, cur);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
}

bool Qwen35moeForwardPass::is_full_attention_layer(uint32_t layer_idx) const {
    // Pattern: layers (interval-1), 2*(interval-1)+1, ... i.e. layer_idx % interval == interval-1
    return (layer_idx % model_->meta_->qwen35moe.full_attention_interval) == (model_->meta_->qwen35moe.full_attention_interval - 1);
}

uint32_t Qwen35moeForwardPass::get_physical_cache_pos(uint32_t slot_idx) const {
    return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
}

// ── Cache management ─────────────────────────────────────────────────────
void Qwen35moeForwardPass::advance_cache(uint32_t n_tokens, uint32_t slot_idx) {
    if (kv_cache_) 
        kv_cache_->advance(n_tokens, slot_idx);
    // DeltaNet state is updated in-graph; no manual advance needed.
    snapkv_advance_seq_pos(slot_idx, n_tokens);
}

// Get the logical sequence position (0 = SnapKV not active for this slot).
uint32_t Qwen35moeForwardPass::snapkv_get_seq_pos(uint32_t slot_idx) const {
    return (slot_idx < snapkv_seq_pos_.size()) ? snapkv_seq_pos_[slot_idx] : 0;
}

// Advance the logical sequence position (call alongside advance_cache).
void Qwen35moeForwardPass::snapkv_advance_seq_pos(uint32_t slot_idx, uint32_t n_tokens) {
    if (slot_idx < snapkv_seq_pos_.size() && snapkv_seq_pos_[slot_idx] > 0)
        snapkv_seq_pos_[slot_idx] += n_tokens;
}

// Get output from GPU
std::vector<float> Qwen35moeForwardPass::get_output_logits(ggml_cgraph* gf) {
    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        throw std::runtime_error("logits tensor not found in graph");
    }
    
    size_t logits_size = ggml_nbytes(logits);
    std::vector<float> logits_result(logits_size / sizeof(float));
    ggml_backend_tensor_get(logits, logits_result.data(), 0, logits_size);
    
    return logits_result;
}
