#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <ggml.h>
#include <memory>
#include "model/model.h"
#include "pipeline/sampling.h"
#include "pipeline/tokenizer.h"
#include "graph/kv_cache_simple.h"
#include "graph/deltanet_state.h"

constexpr size_t FP_GRAPH_SIZE_METADATA = 128 * 1024 * 1024;
constexpr size_t FP_GRAPH_SIZE = 16384;

enum class Phase {
    Prefill,  // 批量提示处理：对所有输入标记进行完全因果关注
    Decode,   // 单步自回归：每个槽位一个令牌，关注键值缓存
};

// When --override-tensor splits MoE expert weights to CPU, GPU segments must
// not build mul_mat_id into the same graph (CUDA stages all experts on VRAM).
enum class LayerMoeMode {
    Full,
    PreMoeOnly,
};

// Args for Phase::Prefill — single slot, multiple tokens.
struct PrefillArgs {
    uint32_t n_tokens;
    uint32_t slot_idx;
};

// Args for Phase::Decode — one token per slot in the batch.
struct DecodeArgs {
    std::vector<uint32_t> slots;  // sequence slot per batch element
};

struct TopKSampleCandidates {
    std::vector<int32_t> token_ids;
    std::vector<float> logits;
};

class Qwen35moeForwardPass {
public:
    Qwen35moeForwardPass();
    ~Qwen35moeForwardPass();

    int init(const uint32_t context_len, const uint32_t max_batch_size, std::shared_ptr<Qwen35moeModel> model,
        uint32_t n_batch = 0, uint32_t n_ubatch = 0, bool enable_paged_kv = false, uint32_t paged_kv_block_size = 16);

    void reset_context();
    void reset_sequence(uint32_t slot_idx = 0);
    void set_flash_attention_enabled(bool enabled);

    // ── Graph building ───────────────────────────────────────────────────────
    ggml_cgraph* build_prefill_graph(
        const std::vector<int32_t>& tokens,
        int pos,
        uint32_t slot_idx = 0,
        uint32_t fixed_shared_n_kv = 0,
        bool dynamic_kv_write = false
    );
    ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions,
        uint32_t fixed_n_kv_len = 0
    );

    // ── Input setting ────────────────────────────────────────────────────────
    void set_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens, int pos);

    void set_batched_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots, const std::vector<int32_t>&  positions);

    std::vector<float> run_prefill(const std::vector<int32_t>& tokens, int pos, 
        uint32_t slot_idx, ggml_backend_sched_t scheduler);
    std::vector<float> run_decode_cached(int32_t token, int pos,
        uint32_t slot_idx, ggml_backend_sched_t scheduler);
    std::vector<std::vector<float>> run_decode_batch(const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots, const std::vector<int32_t>& positions,
        ggml_backend_sched_t scheduler);
    TopKSampleCandidates run_prefill_topk(const std::vector<int32_t>& tokens, int pos,
        uint32_t slot_idx, ggml_backend_sched_t scheduler);
    TopKSampleCandidates run_decode_cached_topk(int32_t token, int pos,
        uint32_t slot_idx, ggml_backend_sched_t scheduler);
    std::vector<TopKSampleCandidates> run_decode_batch_topk(const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots, const std::vector<int32_t>& positions,
        ggml_backend_sched_t scheduler);
    void configure_device_sampling(int top_k, float temperature);

    uint32_t get_cache_pos(uint32_t slot_idx) const;

    // ── MTP (Multi-Token Prediction / NextN) speculative decoding ─────────────
    // Enable the built-in nextn draft head. `mtp_active()` additionally requires
    // that the loaded GGUF actually carries an MTP block (model has_mtp()).
    void configure_mtp(bool enabled, int spec_draft_n_max);
    bool mtp_active() const { return mtp_enabled_ && mtp_supported_; }
    // Process the prompt (trunk + MTP KV seeding) and return the first generated
    // token (greedy). Fills `seed_h` with the trunk hidden that pairs with the
    // NEXT token for drafting, and returns the prompt length via `pos_out`.
    int32_t mtp_prefill(const std::vector<int32_t>& prompt_tokens, uint32_t slot,
                        std::vector<float>& seed_h, int& pos_out,
                        ggml_backend_sched_t scheduler);
    // One speculative step: draft up to spec_draft_n_max tokens from the built-in
    // head, verify them in a single batched trunk forward, accept the longest
    // matching greedy prefix plus one bonus token. Appends the confirmed
    // continuation tokens to `out` and updates (seed_tok, pos, seed_h) for the
    // next call. Returns the number of tokens appended.
    int mtp_step(int32_t& seed_tok, int& pos, std::vector<float>& seed_h,
                 uint32_t slot, std::vector<int32_t>& out,
                 ggml_backend_sched_t scheduler);
    void mtp_log_stats() const;

private:
    struct LayerSegment {
        uint32_t l0 = 0;
        uint32_t l1 = 0;
        ggml_backend_t backend = nullptr;
    };

    struct PrefillGraphSignature {
        uint32_t slot_idx = 0;
        uint32_t n_tokens = 0;
        uint32_t kv_capacity = 0;
        uint32_t context_len = 0;
        bool use_flash_attention = false;
        bool is_mixed_mode = false;
        uint64_t device_map_hash = 0;
        int sampling_top_k = 0;
        float sampling_temperature = 0.0f;
    };

    struct DecodeGraphSignature {
        uint32_t slot_idx = 0;
        uint32_t kv_capacity = 0;
        uint32_t context_len = 0;
        uint32_t n_batch_tokens = 0;
        uint32_t n_ubatch_tokens = 0;
        uint32_t n_decode_batch = 0;
        uint64_t slots_signature = 0;
        bool use_flash_attention = false;
        bool paged_fused_decode = false;
        // Mixed-mode guard: signature must match when AUTO places layers on
        // both GPU and CPU backends. Scheduler-first decode is preferred;
        // segmented decode remains the fallback.
        bool is_mixed_mode = false;
        // FNV-1a hash of the current layer→device assignment in AUTO_MODE.
        // Any change in layer placement forces a full graph recapture.
        uint64_t device_map_hash = 0;
    };

    struct SegmentedDecodeGraphCache {
        std::vector<uint8_t> ctx_buffer;
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_gallocr_t allocr = nullptr;
        ggml_backend_t backend = nullptr;
        uint32_t layer_begin = 0;
        uint32_t layer_end = 0;
        ggml_type hidden_type = GGML_TYPE_F32;
        bool uses_token_input = false;
        bool valid = false;
        ggml_tensor* tokens_tensor = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        ggml_tensor* hidden_in_tensor = nullptr;
        ggml_tensor* hidden_out_tensor = nullptr;
        ggml_tensor* kv_write_phys_tensor = nullptr;
        std::vector<ggml_tensor*> mask_tensors;
        std::vector<float> mask_f32;
        std::vector<ggml_fp16_t> mask_f16;
        int last_mask_pos = -1;
    };

    // Per-layer premoe + MoE graphs when --override-tensor splits expert weights.
    struct SplitMoeDecodeLayerCache {
        int layer_idx = -1;
        SegmentedDecodeGraphCache premoe;
        std::vector<uint8_t> moe_ctx_buffer;
        ggml_context* moe_ctx = nullptr;
        ggml_cgraph* moe_graph = nullptr;
        ggml_tensor* premoe_moe_norm_out = nullptr;
        ggml_tensor* premoe_ffn_res_out = nullptr;
        ggml_tensor* moe_norm_in = nullptr;
        ggml_tensor* moe_ffn_res_in = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_type moe_norm_type = GGML_TYPE_F32;
        ggml_type ffn_res_type = GGML_TYPE_F32;
        ggml_type hidden_type = GGML_TYPE_F32;
        ggml_backend_sched_t moe_scheduler = nullptr;
    };

    // Per-slot segmented decode cache bundle. Mixed-mode graphs bake slot_idx
    // into KV views, so each parallel slot needs its own cached segment graphs.
    struct SegmentedDecodeSlotCache {
        std::vector<SegmentedDecodeGraphCache> segment_caches;
        std::vector<SplitMoeDecodeLayerCache> split_layer_caches;
        bool use_split_decode = false;
        SegmentedDecodeGraphCache embed_cache;
        SegmentedDecodeGraphCache head_cache;
        DecodeGraphSignature signature{};
        bool signature_valid = false;
        int sampling_top_k = 0;
        float sampling_temperature = 0.0f;
    };

    // Create a new graph
    ggml_cgraph* new_graph();

    ggml_tensor* embedding(ggml_cgraph* gf, const std::vector<int32_t>& tokens);
    ggml_tensor* build_layer_range(
        ggml_cgraph* gf,
        ggml_tensor* inpL,
        ggml_tensor* inp_pos,
        uint32_t n_tok,
        uint32_t slot_idx,
        uint32_t layer_begin,
        uint32_t layer_end,
        uint32_t fixed_shared_n_kv = 0,
        ggml_tensor* kv_write_row = nullptr,
        LayerMoeMode moe_mode = LayerMoeMode::Full
    );
    ggml_cgraph* build_segment_graph_no_reset(
        uint32_t n_tok,
        int pos,
        uint32_t slot_idx,
        const LayerSegment& segment,
        bool use_token_input,
        ggml_type hidden_type,
        uint32_t fixed_shared_n_kv = 0,
        bool dynamic_kv_write = false,
        LayerMoeMode moe_mode = LayerMoeMode::Full
    );
    ggml_cgraph* build_prefill_segment_graph(
        uint32_t n_tok,
        int pos,
        uint32_t slot_idx,
        const LayerSegment& segment,
        bool use_token_input,
        ggml_type hidden_type
    );
    ggml_cgraph* build_decode_segment_graph(
        int32_t token,
        int pos,
        uint32_t slot_idx,
        const LayerSegment& segment,
        bool is_first_segment,
        bool is_last_segment,
        ggml_type hidden_type
    );
    ggml_cgraph* build_output_head_graph_from_hidden(ggml_type hidden_type);
    ggml_cgraph* build_decode_segment_graph_no_reset(
        int32_t token,
        int pos,
        uint32_t slot_idx,
        const LayerSegment& segment,
        bool is_first_segment,
        bool is_last_segment,
        ggml_type hidden_type,
        uint32_t fixed_shared_n_kv = 0,
        bool dynamic_kv_write = false
    );
    ggml_cgraph* build_output_head_graph_from_hidden_no_reset(ggml_type hidden_type);
    bool can_run_token_embedding_on_backend(ggml_backend_t backend) const;
    ggml_backend_t find_backend_for_tensor(const ggml_tensor* tensor) const;
    void maybe_log_segment_tensor(
        const char* scope,
        const LayerSegment* segment,
        ggml_tensor* tensor
    ) const;
    void validate_handoff_tensor(
        const char* where,
        ggml_tensor* tensor,
        ggml_type expected_type,
        size_t expected_bytes
    ) const;
    void copy_handoff_tensor(
        const char* where,
        ggml_tensor* src,
        ggml_tensor* dst,
        ggml_type expected_type
    ) const;
    // Async variant: CUDA<->CUDA uses event-chained copy; CPU<->CUDA uses pinned
    // staging (see copy_handoff_tensor_async).
    void copy_handoff_tensor_async(
        const char* where,
        ggml_backend_t src_backend,
        ggml_backend_t dst_backend,
        ggml_tensor* src,
        ggml_tensor* dst,
        ggml_type expected_type
    ) const;
    void ensure_handoff_pinned(size_t nbytes) const;
    void ensure_output_pinned(size_t nbytes) const;
    void feed_cached_head_hidden_bytes(
        const char* where,
        SegmentedDecodeGraphCache& head,
        const std::vector<uint8_t>& hidden_data,
        ggml_type hidden_type
    ) const;
    // CUDA host-pinning helpers. When QWEN35MOE_USE_CUDA is defined and a
    // CUDA backend is present, pinning lets cudaMemcpyAsync (used by ggml's
    // tensor_set_async / get_tensor_async) issue truly non-blocking H2D/D2H
    // transfers instead of falling back to a staged synchronous copy.
    void pin_host_region(void* ptr, size_t size) const;
    void unpin_host_region(void* ptr) const;
    void unpin_all_host_regions();
    void read_segment_hidden_out(
        const char* where,
        const LayerSegment* segment,
        ggml_cgraph* gf,
        std::vector<uint8_t>& hidden_data,
        ggml_type& hidden_type
    );
    void prepare_first_segment_hidden_from_token(
        const char* where,
        int32_t token,
        const LayerSegment& first_segment,
        std::vector<uint8_t>& hidden_data,
        ggml_type& hidden_type
    );
    void prepare_first_segment_hidden_from_tokens(
        const char* where,
        const std::vector<int32_t>& tokens,
        const LayerSegment& first_segment,
        std::vector<uint8_t>& hidden_data,
        ggml_type& hidden_type
    );
    void extract_last_token_hidden(
        const std::vector<uint8_t>& full_hidden,
        ggml_type hidden_type,
        int64_t n_embd,
        uint32_t n_tok,
        std::vector<uint8_t>& last_token_hidden
    ) const;
    void set_segment_inputs(
        ggml_cgraph* gf,
        const std::vector<int32_t>& tokens,
        int pos,
        ggml_type hidden_type,
        const std::vector<uint8_t>* hidden_data,
        const LayerSegment* segment,
        uint32_t layer_begin,
        uint32_t layer_end
    );
    std::vector<float> run_prefill_segmented_eager(
        const std::vector<int32_t>& tokens,
        int pos,
        uint32_t slot_idx
    );
    TopKSampleCandidates run_prefill_segmented_topk_eager(
        const std::vector<int32_t>& tokens,
        int pos,
        uint32_t slot_idx
    );
    std::vector<float> run_decode_segmented(int32_t token, int pos, uint32_t slot_idx);
    TopKSampleCandidates run_decode_segmented_topk(int32_t token, int pos, uint32_t slot_idx);
    std::vector<float> run_decode_segmented_eager(int32_t token, int pos, uint32_t slot_idx);
    TopKSampleCandidates run_decode_segmented_topk_eager(int32_t token, int pos, uint32_t slot_idx);
    std::vector<float> run_decode_segmented_cached(int32_t token, int pos, uint32_t slot_idx);
    TopKSampleCandidates run_decode_segmented_topk_cached(int32_t token, int pos, uint32_t slot_idx);
    void rebuild_layer_segments();
    bool segment_has_split_layers(const LayerSegment& segment) const;
    void run_segment_forward(
        ggml_cgraph* gf,
        const LayerSegment& segment,
        const char* where,
        const std::function<void()>& set_inputs,
        bool single_backend_only = false,
        const std::function<void()>& after_compute = {});
    uint32_t count_split_layers() const;
    ggml_cgraph* build_split_moe_ffn_graph_no_reset(
        ggml_context* ctx,
        int il,
        ggml_type hidden_type,
        ggml_tensor** moe_norm_in,
        ggml_tensor** ffn_res_in,
        ggml_tensor** hidden_out
    );
    void run_split_decode_body_eager(
        int32_t token,
        int pos,
        uint32_t slot_idx,
        std::vector<uint8_t>& hidden_data,
        ggml_type& hidden_type
    );
    void run_split_decode_body_cached(
        int32_t token,
        int pos,
        uint32_t slot_idx,
        SegmentedDecodeSlotCache& slot_cache,
        ggml_tensor*& hidden_out,
        ggml_backend_t& hidden_backend,
        ggml_type& hidden_type
    );
    void feed_cached_head_hidden_tensor(
        const char* where,
        SegmentedDecodeGraphCache& head,
        ggml_backend_t src_backend,
        ggml_tensor* hidden_src,
        ggml_type hidden_type
    ) const;
    std::vector<float> run_decode_split_layers_cached(
        int32_t token,
        int pos,
        uint32_t slot_idx,
        SegmentedDecodeSlotCache& slot_cache
    );

    // Inline MoE FFN for one physical layer, after the pre-FFN norm has been applied.
    // Returns the FFN output (before residual). il is the physical layer index.
    ggml_tensor* build_moe_layer(
        ggml_context* ctx,
        ggml_cgraph*  gf,
        ggml_tensor*  input,
        int il
    );
    ggml_tensor* build_moe_routed_experts(
        ggml_context* ctx,
        ggml_cgraph*  gf,
        ggml_tensor*  input,
        int il
    );
    ggml_tensor* build_moe_shared_combine(
        ggml_context* ctx,
        ggml_cgraph*  gf,
        ggml_tensor*  input,
        ggml_tensor*  routed_out,
        int il,
        ggml_tensor*  ffn_residual = nullptr
    );
    void read_named_layer_tensor(
        ggml_cgraph* gf,
        const char* base_name,
        int il,
        std::vector<uint8_t>& data,
        ggml_type& type
    ) const;
    void run_split_moe_layer_forward(
        int il,
        uint32_t n_tok,
        int pos,
        uint32_t slot_idx,
        bool first_uses_token_input,
        const std::vector<int32_t>* tokens,
        std::vector<uint8_t>& hidden_data,
        ggml_type& hidden_type
    );
    void run_prefill_segment_with_split_layers(
        const std::vector<int32_t>& tokens,
        int pos,
        uint32_t slot_idx,
        const LayerSegment& segment,
        bool first_uses_token_input,
        std::vector<uint8_t>& hidden_data,
        ggml_type& hidden_type
    );

    // Canonical RMS norm + weight scaling shared by all layer types.
    // Names the intermediate tensor "cur_rms_normed.il" (or "cur_rms_normed" when
    // il < 0) to match the ForwardPassBase::build_norm convention so that graph
    // debug output and tensor-name-based lookups in set_inputs remain stable.
    ggml_tensor* build_rms_norm(
        ggml_context* ctx,
        ggml_tensor*  cur,
        ggml_tensor*  weight,
        float         eps,
        int           il
    );

    ggml_tensor* build_ffn_swiglu(
        ggml_context* ctx,
        ggml_cgraph*  gf,
        ggml_tensor*  cur,
        ggml_tensor*  gate,
        ggml_tensor*  up,
        ggml_tensor*  down,
        int           il
    );

    ggml_tensor* build_norm(
        ggml_cgraph* gf,
        ggml_tensor* cur,
        ggml_tensor* mw,
        int il
    );

    // Core MHA kernel: Q@K^T → softmax(scale, mask) → @V.
    // Handles GQA, head permutation, stream splitting, and contiguous recombination.
    // Extracted from ForwardPassBase::build_attn_mha — identical logic.
    ggml_tensor* build_attn_mha(
        ggml_context* ctx,
        ggml_cgraph* gf,
        ggml_tensor* q,
        ggml_tensor* k,
        ggml_tensor* v,
        ggml_tensor* kq_mask,
        ggml_tensor* sinks,
        float        kq_scale,
        uint32_t     pos,
        int          il,
        bool         allow_flash_attn = true
    );

    // ── Gated attention variants (Qwen3.5, Qwen3.6) ─────────────────────────────
    // These models use a joint Q+Gate projection, Q/K RMS norms, partial RoPE, and
    // sigmoid gating after the attention output. They differ structurally from the
    // Qwen2/3 attention above, so they live as separate free functions.

    // Prefill / single-slot gated attention.
    // Takes raw normed input cur; performs Q/K/V projections, Q/K norms, partial
    // RoPE, KV cache write + full-history MHA, then sigmoid gating + output proj.
    ggml_tensor* build_gated_attention(
        ggml_context*    ctx,
        ggml_cgraph*     gf,
        simple_kv_cache* kv_cache,
        ggml_tensor*     cur,          // normed input [n_embd, n_tokens]
        ggml_tensor*     inp_pos,      // [n_tokens]
        int              kv_cache_layer,
        uint32_t         n_tokens,
        uint32_t         slot_idx,
        int              il,
        ggml_tensor*     w_q,          // attn_q_weight (joint Q+Gate)
        ggml_tensor*     w_q_norm,     // attn_q_norm_weight
        ggml_tensor*     w_k,          // attn_k_weight
        ggml_tensor*     w_k_norm,     // attn_k_norm_weight
        ggml_tensor*     w_v,          // attn_v_weight
        ggml_tensor*     w_out,        // attn_output_weight
        int              n_embd_head,
        int              n_head,
        int              n_head_kv,
        int              n_rot,
        float            freq_base,
        int              context_length,
        float            rms_norm_eps,
        ggml_tensor*     shared_kq_mask,
        ggml_tensor*     kv_write_row = nullptr,
        uint32_t           fixed_read_n_kv = 0
    );

    // Batched decode variant of build_gated_attention: same projections/norms/gating,
    // operates on a batch of slots with pre-built kq_mask and gather_indices.
    ggml_tensor* build_gated_batched_attention(
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
    );

    // Build the full GatedDeltaNet subgraph for one layer, single-slot prefill.
    // Performs: QKV proj → Z-gate proj → beta/alpha proj → conv1d → L2-norm →
    //   ggml_gated_delta_net (fused) → state write-back → gated RMSNorm → out-proj.
    // Returns output tensor [n_embd, n_tokens].
    ggml_tensor* build_deltanet_layer(
        ggml_context*   ctx,
        ggml_cgraph*    gf,
        ggml_tensor*    cur,          // normed input [n_embd, n_tokens]
        DeltaNetState*  dn_state,
        uint32_t        dn_idx,       // DeltaNet layer index within dn_state
        uint32_t        slot_idx,     // sequence slot
        uint32_t        n_tokens,
        ggml_tensor*    w_qkv,        // [n_embd, conv_channels] — joint QKV projection
        ggml_tensor*    w_gate,       // [n_embd, d_inner]       — output gate (Z)
        ggml_tensor*    w_beta,       // [n_embd, num_v_heads]   — beta gate (scalar per head)
        ggml_tensor*    w_a,          // [n_embd, num_v_heads]   — alpha for decay gate
        ggml_tensor*    w_dt_bias,    // [num_v_heads]           — dt bias
        ggml_tensor*    w_a_log,      // [num_v_heads]            — A_log scalar (1D)
        ggml_tensor*    w_conv,       // [conv_channels, 1, conv_kernel] — depthwise conv
        ggml_tensor*    w_norm,       // [d_inner]               — RMSNorm gamma
        ggml_tensor*    w_out,        // [d_inner, n_embd]       — output projection
        int             n_embd,
        int             d_inner,
        int             head_k_dim,
        int             num_k_heads,
        int             num_v_heads,
        int             head_v_dim,
        int             conv_channels,
        int             conv_kernel,
        float           rms_norm_eps,
        int             il              // physical layer index (for tensor naming)
    );

    // Single-token decode variant of build_deltanet_layer. Uses ggml_ssm_conv_step
    // to avoid concat/transpose of the conv sliding window on batch=1 decode.
    ggml_tensor* build_deltanet_layer_decode(
        ggml_context*   ctx,
        ggml_cgraph*    gf,
        ggml_tensor*    cur,
        DeltaNetState*  dn_state,
        uint32_t        dn_idx,
        uint32_t        slot_idx,
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
        int             il
    );

    ggml_tensor* build_dn_layer(
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
    );

    // Batched decode variant of build_deltanet_layer. Processes one token per
    // slot for `slots.size()` slots in a single graph (n_seq_tokens=1,
    // n_seqs=n_batch) and uses ggml_get_rows / ggml_set_rows against
    // dn_state's conv_/recurrent_ tensors so the per-slot state read/write is
    // done in a single fused gather/scatter pair instead of looping.
    //
    // The slot indices are pulled from a graph-scoped input tensor named
    // "dn_slot_idx" (shared across all DN layers in the graph). Callers must
    // populate it via ggml_backend_tensor_set with `slots` as int32 before
    // running the graph (handled inside set_batched_inputs and
    // set_cached_batched_decode_inputs).
    ggml_tensor* build_deltanet_layer_batched_decode(
        ggml_context*   ctx,
        ggml_cgraph*    gf,
        ggml_tensor*    cur,             // [n_embd, n_batch]
        DeltaNetState*  dn_state,
        uint32_t        dn_idx,
        const std::vector<uint32_t>& slots,
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
        int             il
    );

    ggml_tensor* build_all_deltanet_layer(
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
    );

    ggml_tensor* build_all_deltanet_layer_prefill(
        ggml_context*      ctx,
        ggml_cgraph*       gf,
        ggml_tensor*       input,
        uint32_t           dn_idx,
        const PrefillArgs& prefill_args,
        DeltaNetState*     dn_state,
        const DeltaNetParams&     hp,
        int      il
    );

    ggml_tensor* build_all_deltanet_layer_decode(
        ggml_context*     ctx,
        ggml_cgraph*      gf,
        ggml_tensor*      input,
        uint32_t          dn_idx,
        const DecodeArgs& decode_args,
        DeltaNetState*     dn_state,
        const DeltaNetParams&     hp,
        int      il
    );

    // ── MTP internal helpers (graph/mtp.cpp) ─────────────────────────────────
    void init_mtp();
    // Trunk forward over `tokens` that exposes per-column greedy argmax
    // ("mtp_argmax", I32[n_tok]) and per-column post-output_norm hidden
    // ("mtp_hidden", F32[n_embd, n_tok]); reuses build_layer_range so it
    // advances KV + DeltaNet state exactly like a prefill chunk.
    ggml_cgraph* build_mtp_verify_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx);
    // Single nextn draft-head block over `n_tok` tokens. Inputs: "tokens",
    // "inp_pos", "mtp_h_in" (F32[n_embd, n_tok]). Outputs (optional):
    // "mtp_argmax" (greedy token per column) and "mtp_out_h" (nextn hidden that
    // seeds the following draft step). Writes into mtp_kv_ at get_pos(slot).
    ggml_cgraph* build_mtp_block_graph(uint32_t n_tok, uint32_t slot_idx,
                                       bool need_argmax, bool need_hidden_out);
    void set_mtp_block_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens,
                              const std::vector<float>& hiddens, int pos, uint32_t slot_idx);
    void build_mtp_head(ggml_cgraph* gf, ggml_tensor* hidden_all,
                        bool need_argmax, bool need_hidden);
    // Run the nextn block over committed tokens to (re)write correct MTP KV.
    void run_mtp_process(const std::vector<int32_t>& tokens,
                         const std::vector<float>& hiddens_shifted,
                         int pos, uint32_t slot_idx, ggml_backend_sched_t scheduler);
    // Build (once per token-count) a persistent, reusable trunk graph and return it.
    ggml_cgraph* ensure_mtp_trunk_graph(uint32_t n_tok);
    // Advance trunk KV + DeltaNet state over `tokens` via a cached graph (used to
    // replay the committed prefix after a partial-acceptance verify).
    void run_mtp_advance(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx);
    // Snapshot path: verify graph that captures per-token DeltaNet state, plus the
    // commit graph that writes the accepted snapshot column into dn_state.
    void ensure_mtp_snap_verify_graph(uint32_t n_tok, uint32_t bucket);
    void run_mtp_verify_snapshot(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx,
                                 std::vector<int32_t>& argmax_out, std::vector<float>& hidden_out);
    void ensure_mtp_commit_graph();
    void run_mtp_commit(int accepted_col);
    // Run a verify/prefill trunk graph over `tokens`, returning per-column argmax
    // and (optionally) per-column hidden; advances trunk KV + DeltaNet state.
    void run_mtp_verify(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx,
                        ggml_backend_sched_t scheduler,
                        std::vector<int32_t>& argmax_out, std::vector<float>& hidden_out);

    void set_tensor_name(ggml_tensor* tensor, const char* name, int il = -1) const;
    bool is_full_attention_layer(uint32_t layer_idx) const;
    ggml_backend_t layer_backend(int il) const;
    bool layer_allows_flash_attn(int il) const;
    // CUDA-only DeltaNet decode fast path (ssm_conv_step + fused proj). CPU uses
    // the standard build_deltanet_layer path for correctness.
    bool deltanet_decode_fast_path_enabled() const;
    bool layer_allows_deltanet_decode_fast_path(int il) const;
    bool deltanet_decode_fused_proj_enabled(
        const ggml_tensor* w_qkv,
        const ggml_tensor* w_gate,
        const ggml_tensor* w_beta,
        const ggml_tensor* w_alpha
    ) const;
    bool attention_range_homogeneous(uint32_t layer_begin, uint32_t layer_end) const;
    // Build the output head: final norm → LM head matmul → "logits" tensor
    void build_output_head(ggml_cgraph* gf, ggml_tensor* cur);
    uint32_t get_physical_cache_pos(uint32_t slot_idx) const;

    void advance_cache(uint32_t n_tokens, uint32_t slot_idx);
    uint32_t prefill_token_bucket_capacity(uint32_t n_tokens) const;
    uint32_t prefill_graph_kv_capacity() const;
    uint32_t effective_prefill_batch_limit() const;
    uint32_t effective_prefill_ubatch_limit() const;
    bool can_use_cached_prefill(uint32_t n_tokens) const;
    void prepare_cached_prefill_graph(ggml_backend_sched_t scheduler, const PrefillGraphSignature& signature);
    bool is_cached_prefill_graph_compatible(const PrefillGraphSignature& signature) const;
    void ensure_cached_prefill_graph(ggml_backend_sched_t scheduler, const PrefillGraphSignature& signature);
    void clear_cached_prefill_graph();
    void collect_prefill_mask_tensors(ggml_cgraph* gf);
    void set_cached_prefill_inputs(const std::vector<int32_t>& tokens, int pos);
    void prepare_cached_decode_graph(ggml_backend_sched_t scheduler, uint32_t slot_idx, uint32_t kv_capacity);
    void prepare_cached_batched_decode_graph(
        ggml_backend_sched_t scheduler,
        const DecodeGraphSignature& signature,
        const std::vector<uint32_t>& slots
    );
    uint32_t decode_cache_bucket_capacity(uint32_t required_kv) const;
    bool is_cached_decode_graph_compatible(const DecodeGraphSignature& signature, uint32_t required_kv) const;
    bool is_cached_batched_decode_graph_compatible(const DecodeGraphSignature& signature, uint32_t required_kv) const;
    void ensure_cached_decode_graph(ggml_backend_sched_t scheduler, const DecodeGraphSignature& signature, uint32_t required_kv);
    void ensure_cached_batched_decode_graph(
        ggml_backend_sched_t scheduler,
        const DecodeGraphSignature& signature,
        const std::vector<uint32_t>& slots,
        uint32_t required_kv
    );
    void clear_cached_batched_decode_graph();
    void upload_decode_mask_tensor(ggml_tensor* kq_mask, const std::vector<float>& mask_f32);
    void fill_dynamic_decode_mask_1d(
        std::vector<float>& mask_f32,
        uint32_t graph_n_kv,
        uint32_t active_kv,
        bool incremental,
        int pos,
        int& last_mask_pos
    );
    void fill_dynamic_decode_mask_batched(
        std::vector<float>& mask_f32,
        uint32_t graph_n_kv,
        uint32_t n_batch,
        const std::vector<int32_t>& positions,
        bool incremental,
        std::vector<int32_t>& last_mask_positions
    );
    void fill_dynamic_prefill_mask(
        std::vector<float>& mask_f32,
        uint32_t graph_n_kv,
        uint32_t n_tok,
        int pos
    );
    bool run_prefill_chunk_scheduler(
        ggml_backend_sched_t scheduler,
        const std::vector<int32_t>& tokens,
        int pos,
        uint32_t slot_idx,
        bool use_topk,
        std::vector<float>* logits_out,
        TopKSampleCandidates* topk_out
    );
    void run_prefill_ubatch_eager_scheduler(
        ggml_backend_sched_t scheduler,
        const std::vector<int32_t>& tokens,
        int pos,
        uint32_t slot_idx,
        bool use_topk,
        std::vector<float>* logits_out,
        TopKSampleCandidates* topk_out
    );
    void run_prefill_microbatched_scheduler(
        ggml_backend_sched_t scheduler,
        const std::vector<int32_t>& tokens,
        int pos,
        uint32_t slot_idx,
        bool use_topk,
        std::vector<float>* final_logits,
        TopKSampleCandidates* final_topk
    );
    void run_prefill_microbatched_direct(
        ggml_gallocr_t allocr,
        ggml_backend_t backend,
        const std::vector<int32_t>& tokens,
        int pos,
        uint32_t slot_idx,
        bool use_topk,
        std::vector<float>* final_logits,
        TopKSampleCandidates* final_topk,
        bool allow_segmented = true
    );
    void maybe_log_decode_graph_stats();
    void set_cached_decode_inputs(ggml_cgraph* gf, int32_t token, int pos);
    void set_cached_batched_decode_inputs(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions
    );
    std::vector<std::vector<float>> run_decode_batch_cached(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions,
        ggml_backend_sched_t scheduler
    );
    std::vector<TopKSampleCandidates> run_decode_batch_topk_cached(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions,
        ggml_backend_sched_t scheduler
    );
    bool can_use_paged_fused_decode(const char** reason) const;
    bool paged_fused_decode_active() const;
    void maybe_log_paged_fused_fallback(const char* reason);
    void maybe_log_paged_fused_activation();
    void record_paged_fused_decode_timing(uint64_t delta_us);
    bool ensure_cached_decode_copy_ready(uint32_t slot_idx, uint32_t dst_pos);
    bool commit_cached_decode_step(uint32_t slot_idx, uint32_t dst_pos);
    bool is_segmented_decode_cache_compatible(
        const SegmentedDecodeSlotCache& slot_cache,
        const DecodeGraphSignature& signature,
        uint32_t required_kv
    ) const;
    void ensure_segmented_decode_cache(const DecodeGraphSignature& signature, uint32_t required_kv);
    void prepare_segmented_decode_cache(uint32_t slot_idx, uint32_t kv_capacity);
    bool ensure_segmented_decode_copy_ready(uint32_t slot_idx, uint32_t dst_pos) const;
    bool commit_segmented_decode_step(uint32_t slot_idx, uint32_t dst_pos);
    void clear_segmented_decode_cache();
    void clear_segmented_decode_slot_cache(SegmentedDecodeSlotCache& slot_cache);
    SegmentedDecodeSlotCache& segmented_decode_slot_cache(uint32_t slot_idx);
    const SegmentedDecodeSlotCache* find_segmented_decode_slot_cache(uint32_t slot_idx) const;
    void maybe_log_segmented_decode_cache_stats();
    void set_cached_segment_inputs(
        SegmentedDecodeGraphCache& cache,
        int32_t token,
        int pos,
        uint32_t slot_idx,
        ggml_type hidden_type,
        ggml_tensor* hidden_src,
        const LayerSegment* segment
    );
    uint32_t snapkv_get_seq_pos(uint32_t slot_idx) const;
    void snapkv_advance_seq_pos(uint32_t slot_idx, uint32_t n_tokens);
    std::vector<float> get_output_logits(ggml_cgraph* gf, ggml_backend_t compute_backend = nullptr);
    TopKSampleCandidates get_output_topk_candidates(
        ggml_cgraph* gf,
        uint32_t token_col,
        ggml_backend_t compute_backend = nullptr
    );

private:
    std::vector<uint8_t> ctx_buffer_;
    struct ggml_context* ctx_ = nullptr;
    std::shared_ptr<Qwen35moeModel> model_ = nullptr;

    // kv_layer_map_[il] = KV cache index (0‥9)  if attention layer, else -1.
    // dn_layer_map_[il] = DeltaNet index (0‥29) if DeltaNet layer,  else -1.
    std::vector<int32_t> kv_layer_map_;
    std::vector<int32_t> dn_layer_map_;

    std::unique_ptr<simple_kv_cache> kv_cache_;  // 10 attention layers
    std::unique_ptr<DeltaNetState> dn_state_;   // 30 DeltaNet layers

    // ── MTP speculative decoding state ────────────────────────────────────────
    // Dedicated single-layer KV cache for the nextn draft-head attention, plus a
    // device-resident shadow of the trunk DeltaNet state used to restore the
    // committed recurrent state after a partial-acceptance verify (see mtp.cpp).
    std::unique_ptr<simple_kv_cache> mtp_kv_;
    std::unique_ptr<DeltaNetState> dn_shadow_;
    // Persistent, reused trunk graphs keyed by token-count (each with its own
    // context + gallocr so they survive the reset_context() calls made by the
    // draft/process graphs). Built once per token-count and recomputed each step
    // with fresh inputs, avoiding the per-step rebuild+realloc+CUDA-graph
    // recapture. Index n_tok is used directly (arrays sized mtp_n_max_+2). Both
    // the verify (D+1 tokens) and the partial-accept replay (committed tokens)
    // paths reuse this cache.
    std::vector<ggml_context*> mtp_graph_ctx_;
    std::vector<std::vector<uint8_t>> mtp_graph_buf_;
    std::vector<ggml_gallocr_t> mtp_graph_allocr_;
    std::vector<ggml_cgraph*> mtp_graph_;
    uint32_t mtp_graph_bucket_ = 0;

    // ── MTP snapshot-based rollback (QWEN35MOE_MTP_SNAPSHOT) ───────────────────
    // Optional path that eliminates the partial-acceptance replay forward: the
    // verify graph captures per-token DeltaNet state (recurrent + conv) via the
    // gated_delta_net K>1 snapshot mechanism, then a tiny commit graph copies the
    // accepted column into dn_state. Off by default; on the stable path the
    // shadow+replay rollback is used.
    bool mtp_use_snapshots_ = false;
    // Set true only while building the snapshot verify graph so build_deltanet_layer
    // captures snapshots into mtp_rec_snap_/mtp_conv_snap_ instead of writing dn_state.
    bool mtp_snapshot_build_ = false;
    ggml_context* mtp_snap_ctx_ = nullptr;
    ggml_backend_buffer_t mtp_snap_buf_ = nullptr;
    std::vector<ggml_tensor*> mtp_rec_snap_;   // [n_dn][rec_slot_floats, K]
    std::vector<ggml_tensor*> mtp_conv_snap_;  // [n_dn][conv_slot_floats, K]
    uint32_t mtp_snap_k_ = 0;                  // K = mtp_n_max_+1
    // Dedicated snapshot-capturing verify graph (separate from the plain trunk cache).
    ggml_context* mtp_snap_verify_ctx_ = nullptr;
    std::vector<uint8_t> mtp_snap_verify_buf_;
    ggml_gallocr_t mtp_snap_verify_allocr_ = nullptr;
    ggml_cgraph* mtp_snap_verify_graph_ = nullptr;
    uint32_t mtp_snap_verify_n_tok_ = 0;
    uint32_t mtp_snap_verify_bucket_ = 0;
    // When set, the snapshot verify graph uses a KV bucket that grows with
    // position (256,512,...,ctx_len) instead of a fixed ctx_len, so early-
    // generation attention reads fewer KV rows. Rebuilds the verify graph on
    // each bucket step (a handful of times per long generation).
    bool mtp_grow_bucket_ = false;
    // Commit graph: copies snapshot column `mtp_commit_idx_` into dn_state.
    ggml_context* mtp_commit_ctx_ = nullptr;
    std::vector<uint8_t> mtp_commit_buf_;
    ggml_gallocr_t mtp_commit_allocr_ = nullptr;
    ggml_cgraph* mtp_commit_graph_ = nullptr;
    ggml_tensor* mtp_commit_idx_ = nullptr;

    bool mtp_enabled_ = false;    // requested via --mtp
    bool mtp_supported_ = false;  // model actually carries an MTP block
    int  mtp_n_max_ = 6;          // max draft tokens per step
    uint64_t mtp_step_count_ = 0;
    uint64_t mtp_draft_total_ = 0;
    uint64_t mtp_accept_total_ = 0;
    uint64_t mtp_emit_total_ = 0;
    // Per-phase wall-clock accounting (microseconds) so we can see exactly where
    // an MTP step spends its time instead of guessing.
    uint64_t mtp_draft_us_ = 0;
    uint64_t mtp_verify_us_ = 0;
    uint64_t mtp_process_us_ = 0;
    uint64_t mtp_replay_us_ = 0;
    uint64_t mtp_replay_count_ = 0;

    // Per-slot logical sequence position after SnapKV compaction.
    // 0 = SnapKV not active (use physical cache position).
    std::vector<uint32_t> snapkv_seq_pos_;

    uint32_t context_len_ = 0;
    uint32_t max_batch_size_ = 0;
    ggml_cgraph* cached_decode_graph_ = nullptr;
    bool cached_decode_graph_allocated_ = false;
    uint32_t cached_decode_slot_ = 0;
    uint32_t cached_decode_kv_capacity_ = 0;
    int cached_decode_last_mask_pos_ = -1;
    ggml_tensor* cached_decode_kv_write_phys_tensor_ = nullptr;
    std::vector<ggml_tensor*> cached_decode_mask_tensors_;
    ggml_tensor* cached_decode_tokens_tensor_ = nullptr;
    ggml_tensor* cached_decode_pos_tensor_ = nullptr;
    std::vector<float> cached_decode_mask_f32_;
    std::vector<ggml_fp16_t> cached_decode_mask_f16_;
    DecodeGraphSignature cached_decode_signature_{};
    bool cached_decode_signature_valid_ = false;
    ggml_cgraph* cached_prefill_graph_ = nullptr;
    bool cached_prefill_graph_allocated_ = false;
    ggml_tensor* cached_prefill_tokens_tensor_ = nullptr;
    ggml_tensor* cached_prefill_pos_tensor_ = nullptr;
    std::vector<ggml_tensor*> cached_prefill_mask_tensors_;
    PrefillGraphSignature cached_prefill_signature_{};
    bool cached_prefill_signature_valid_ = false;
    uint64_t prefill_graph_lookup_count_ = 0;
    uint64_t prefill_graph_hit_count_ = 0;
    uint64_t prefill_graph_miss_count_ = 0;
    uint64_t prefill_graph_recapture_count_ = 0;
    uint64_t prefill_graph_fallback_count_ = 0;
    uint64_t prefill_ubatch_chunk_count_ = 0;
    uint64_t prefill_ubatch_micro_step_count_ = 0;
    ggml_cgraph* cached_batched_decode_graph_ = nullptr;
    bool cached_batched_decode_graph_allocated_ = false;
    ggml_tensor* cached_batched_decode_tokens_tensor_ = nullptr;
    ggml_tensor* cached_batched_decode_pos_tensor_ = nullptr;
    ggml_tensor* cached_batched_decode_kq_mask_tensor_ = nullptr;
    ggml_tensor* cached_batched_decode_gather_indices_tensor_ = nullptr;
    // Slot-index input tensor used by batched DeltaNet decode (one I32 per
    // slot). Nullptr when the cached graph has no DeltaNet layers (e.g.
    // pure-attention models) or when n_batch == 1 (the single-slot decode
    // path falls back to per-slot views and does not allocate this).
    ggml_tensor* cached_batched_decode_dn_slot_idx_tensor_ = nullptr;
    DecodeGraphSignature cached_batched_decode_signature_{};
    bool cached_batched_decode_signature_valid_ = false;
    std::vector<float> cached_batched_decode_mask_f32_;
    std::vector<ggml_fp16_t> cached_batched_decode_mask_f16_;
    std::vector<int32_t> cached_batched_decode_last_mask_positions_;
    uint64_t decode_graph_lookup_count_ = 0;
    uint64_t decode_graph_hit_count_ = 0;
    uint64_t decode_graph_miss_count_ = 0;
    uint64_t decode_graph_recapture_count_ = 0;
    uint64_t decode_graph_fallback_count_ = 0;
    uint64_t decode_graph_scheduler_reset_count_ = 0;
    uint64_t decode_graph_last_logged_lookup_ = 0;
    uint64_t decode_graph_last_logged_recapture_ = 0;
    std::unordered_map<uint32_t, uint64_t> decode_graph_bucket_usage_;
    bool decode_graph_diag_enabled_ = false;
    uint32_t decode_graph_diag_interval_ = 256;
    std::vector<SegmentedDecodeSlotCache> segmented_decode_slot_caches_;
    bool segmented_decode_cache_enabled_ = true;
    bool segmented_decode_cache_fallback_warned_ = false;
    // Mixed-mode batched decode: in AUTO, default per-token segmented cache
    // (scales with --parallel slots). Opt-in scheduler batched via
    // QWEN35MOE_MIXED_BATCHED_SCHEDULER=1.
    bool mixed_batched_sequential_enabled_ = false;
    bool mixed_batched_warned_ = false;
    bool mixed_batched_eager_logged_ = false;
    // Mixed-mode prefill: default unified scheduler (fast when no split layers).
    // Opt-out: QWEN35MOE_MIXED_PREFILL_SEGMENTED=1.
    bool mixed_prefill_scheduler_enabled_ = true;
    bool mixed_prefill_segmented_warned_ = false;
    // Mixed-mode single-token decode: per-slot segmented cache (default).
    // Scheduler cached decode is opt-in via QWEN35MOE_MIXED_SCHEDULER_DECODE=1
    // because heterogeneous sched graphs pay CUDA-graph warmup and cross-backend
    // sync every step without matching segmented throughput.
    bool mixed_scheduler_decode_enabled_ = false;
    bool mixed_scheduler_decode_fallback_warned_ = false;
    uint64_t mixed_scheduler_decode_fallback_count_ = 0;
    // Track host buffers pinned via ggml_backend_cuda_register_host_buffer so
    // they can be unregistered on cache reset / destruction.
    mutable std::vector<std::pair<void*, size_t>> pinned_host_regions_;
    // Pinned staging for CPU<->GPU hidden handoffs (avoids ggml copy_async dual sync).
    mutable std::vector<uint8_t> handoff_pinned_;
    // Pinned staging for logits/top-k D2H on the compute stream (avoids cudaStreamPerThread resync).
    mutable std::vector<uint8_t> output_pinned_;
    uint64_t segmented_decode_lookup_count_ = 0;
    uint64_t segmented_decode_hit_count_ = 0;
    uint64_t segmented_decode_miss_count_ = 0;
    uint64_t segmented_decode_recapture_count_ = 0;
    int split_moe_last_sched_layer_idx_ = -1;
    uint64_t segmented_decode_fallback_count_ = 0;
    uint64_t segmented_decode_last_logged_lookup_ = 0;
    uint64_t segmented_decode_last_logged_recapture_ = 0;
    std::unordered_map<uint32_t, uint64_t> segmented_decode_bucket_usage_;
    uint32_t n_batch_tokens_ = 0;
    uint32_t n_ubatch_tokens_ = 0;
    bool use_flash_attention_ = false;
    bool paged_kv_enabled_ = false;
    bool paged_fused_decode_enabled_ = true;
    bool paged_fused_diag_enabled_ = false;
    bool paged_fused_fallback_warned_ = false;
    std::string paged_fused_last_fallback_reason_;
    uint64_t paged_fused_decode_attempt_count_ = 0;
    uint64_t paged_fused_decode_hit_count_ = 0;
    uint64_t paged_fused_decode_fallback_count_ = 0;
    uint64_t paged_fused_decode_compute_count_ = 0;
    uint64_t paged_fused_decode_compute_total_us_ = 0;
    int sampling_top_k_ = 0;
    float sampling_temperature_ = 0.0f;
    // Set via QWEN35MOE_DEV_CHECK=1 to enable lightweight device-consistency logging
    // on key layer boundaries (attention + ffn entry). Disabled by default so that
    // release performance is unaffected.
    bool dev_check_enabled_ = false;
    std::vector<LayerSegment> layer_segments_;
};
