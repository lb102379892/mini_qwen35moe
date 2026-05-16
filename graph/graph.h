#pragma once

#include <vector>
#include <unordered_map>
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

    int init(const uint32_t context_len, const uint32_t max_batch_size, std::shared_ptr<Qwen35moeModel> model, uint32_t n_batch = 0, uint32_t n_ubatch = 0);

    void reset_context();
    void reset_sequence(uint32_t slot_idx = 0);
    void set_flash_attention_enabled(bool enabled);

    // ── Graph building ───────────────────────────────────────────────────────
    ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0);
    ggml_cgraph* build_decoding_graph(const std::vector<int32_t>& tokens, const std::vector<uint32_t>& slots, const std::vector<int32_t>&  positions);

    // ── Input setting ────────────────────────────────────────────────────────
    void set_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens, int pos);

    void set_batched_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots, const std::vector<int32_t>&  positions);

    std::vector<float> run_prefill(const std::vector<int32_t>& tokens, int pos, 
        uint32_t slot_idx, ggml_backend_sched_t scheduler);
    std::vector<float> run_decode_cached(int32_t token, int pos,
        uint32_t slot_idx, ggml_backend_sched_t scheduler);
    TopKSampleCandidates run_prefill_topk(const std::vector<int32_t>& tokens, int pos,
        uint32_t slot_idx, ggml_backend_sched_t scheduler);
    TopKSampleCandidates run_decode_cached_topk(int32_t token, int pos,
        uint32_t slot_idx, ggml_backend_sched_t scheduler);
    void configure_device_sampling(int top_k, float temperature);

    uint32_t get_cache_pos(uint32_t slot_idx) const;

private:
    struct LayerSegment {
        uint32_t l0 = 0;
        uint32_t l1 = 0;
        ggml_backend_t backend = nullptr;
    };

    struct DecodeGraphSignature {
        uint32_t slot_idx = 0;
        uint32_t kv_capacity = 0;
        uint32_t context_len = 0;
        uint32_t n_batch_tokens = 0;
        uint32_t n_ubatch_tokens = 0;
        bool use_flash_attention = false;
        // Mixed-mode guard: when true the scheduler contains both GPU and CPU
        // backends. The cached-decode-graph path is disabled in this mode because
        // ggml's CUDA-graph capture/replay is unreliable across heterogeneous ops.
        bool is_mixed_mode = false;
        // FNV-1a hash of the current layer→device assignment in AUTO_MODE.
        // Any change in layer placement forces a full graph recapture.
        uint64_t device_map_hash = 0;
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
        uint32_t layer_end
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
    void set_segment_inputs(
        ggml_cgraph* gf,
        int32_t token,
        int pos,
        ggml_type hidden_type,
        const std::vector<uint8_t>* hidden_data,
        const LayerSegment* segment,
        uint32_t layer_begin,
        uint32_t layer_end
    );
    std::vector<float> run_decode_segmented(int32_t token, int pos, uint32_t slot_idx);
    TopKSampleCandidates run_decode_segmented_topk(int32_t token, int pos, uint32_t slot_idx);
    void rebuild_layer_segments();

    // Inline MoE FFN for one physical layer, after the pre-FFN norm has been applied.
    // Returns the FFN output (before residual). il is the physical layer index.
    ggml_tensor* build_moe_layer(
        ggml_context* ctx,
        ggml_cgraph*  gf,
        ggml_tensor*  input,
        int il
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
        int          il
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
        float            rms_norm_eps
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

    void set_tensor_name(ggml_tensor* tensor, const char* name, int il = -1) const;
    bool is_full_attention_layer(uint32_t layer_idx) const;
    // Build the output head: final norm → LM head matmul → "logits" tensor
    void build_output_head(ggml_cgraph* gf, ggml_tensor* cur);
    uint32_t get_physical_cache_pos(uint32_t slot_idx) const;

    void advance_cache(uint32_t n_tokens, uint32_t slot_idx);
    void prepare_cached_decode_graph(ggml_backend_sched_t scheduler, uint32_t slot_idx, uint32_t kv_capacity);
    uint32_t decode_cache_bucket_capacity(uint32_t required_kv) const;
    bool is_cached_decode_graph_compatible(const DecodeGraphSignature& signature, uint32_t required_kv) const;
    void ensure_cached_decode_graph(ggml_backend_sched_t scheduler, const DecodeGraphSignature& signature, uint32_t required_kv);
    void maybe_log_decode_graph_stats();
    void set_cached_decode_inputs(ggml_cgraph* gf, int32_t token, int pos);
    uint32_t snapkv_get_seq_pos(uint32_t slot_idx) const;
    void snapkv_advance_seq_pos(uint32_t slot_idx, uint32_t n_tokens);
    std::vector<float> get_output_logits(ggml_cgraph* gf);
    TopKSampleCandidates get_output_topk_candidates(ggml_cgraph* gf, uint32_t token_col);

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

    // Per-slot logical sequence position after SnapKV compaction.
    // 0 = SnapKV not active (use physical cache position).
    std::vector<uint32_t> snapkv_seq_pos_;

    uint32_t context_len_ = 0;
    uint32_t max_batch_size_ = 0;
    ggml_cgraph* cached_decode_graph_ = nullptr;
    bool cached_decode_graph_allocated_ = false;
    uint32_t cached_decode_slot_ = 0;
    uint32_t cached_decode_kv_capacity_ = 0;
    uint32_t cached_decode_scratch_pos_ = 0;
    int cached_decode_last_mask_pos_ = -1;
    std::vector<ggml_tensor*> cached_decode_mask_tensors_;
    ggml_tensor* cached_decode_tokens_tensor_ = nullptr;
    ggml_tensor* cached_decode_pos_tensor_ = nullptr;
    std::vector<float> cached_decode_mask_f32_;
    std::vector<ggml_fp16_t> cached_decode_mask_f16_;
    DecodeGraphSignature cached_decode_signature_{};
    bool cached_decode_signature_valid_ = false;
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
    uint32_t n_batch_tokens_ = 0;
    uint32_t n_ubatch_tokens_ = 0;
    bool use_flash_attention_ = false;
    int sampling_top_k_ = 0;
    float sampling_temperature_ = 0.0f;
    // Set via QWEN35MOE_DEV_CHECK=1 to enable lightweight device-consistency logging
    // on key layer boundaries (attention + ffn entry). Disabled by default so that
    // release performance is unaffected.
    bool dev_check_enabled_ = false;
    std::vector<LayerSegment> layer_segments_;
};
