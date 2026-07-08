/**
 * @file graph.cpp
 * @brief Qwen3.5-MoE 模型前向传播计算图构建实现
 * 
 * 该文件实现了 Qwen3.5-MoE 大语言模型的推理计算图构建，包括：
 * 1. Prefill 阶段：处理初始上下文输入，写入 KV 缓存
 * 2. Decode 阶段：逐 token 生成，支持多序列并行
 * 3. DeltaNet 层：Qwen3.5 特有的高效序列建模层
 * 4. MoE 层：混合专家前馈网络
 */
#include "graph/graph.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#ifdef QWEN35MOE_USE_CUDA
#include "ggml-cuda.h"
#endif

namespace {
constexpr const char* CUDA_BACKEND_NAME = "CUDA";

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
}

uint32_t env_u32_or_default(const char* name, uint32_t default_value) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || (end != nullptr && *end != '\0')) {
        return default_value;
    }
    if (parsed > static_cast<unsigned long long>(std::numeric_limits<uint32_t>::max())) {
        return default_value;
    }
    // Allow explicit 0 so diagnostics can be recapture only when interval=0.
    return static_cast<uint32_t>(parsed);
}

bool backend_name_contains_cuda(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    const char* name = ggml_backend_name(backend);
    return name != nullptr && std::strstr(name, CUDA_BACKEND_NAME) != nullptr;
}

struct CpuThreadScope {
    std::shared_ptr<Qwen35moeModel> model;

    CpuThreadScope(const std::shared_ptr<Qwen35moeModel>& model_in, bool for_batch)
        : model(model_in) {
        if (model) {
            model->apply_cpu_thread_pool(for_batch);
        }
    }
};

bool weight_tensor_on_cuda(const ggml_tensor* tensor) {
    if (!tensor || !tensor->buffer) {
        return false;
    }
    const char* buft_name = ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer));
    return buft_name != nullptr && std::strstr(buft_name, CUDA_BACKEND_NAME) != nullptr;
}

uint64_t hash_decode_slots(const std::vector<uint32_t>& slots) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;
    for (uint32_t slot : slots) {
        hash = (hash ^ static_cast<uint64_t>(slot)) * FNV_PRIME;
    }
    return hash;
}

// Decode buckets: fine 32..512 ladder while required_kv is small, then double
// from min_bucket (256) up to large_step (4096), then linear steps. Avoids a
// 1024-wide FA graph for short chats. See `decode_cache_bucket_capacity`.
constexpr uint32_t kDefaultDecodeMinBucket = 256;
constexpr uint32_t kDefaultDecodeLargeStep = 4096;

// Fine-grained powers of two used for the *prefill* cache, where the prompt
// length is highly variable and padding overhead matters more than recapture
// frequency. Decode uses the policy in `next_decode_kv_bucket` instead.
uint32_t fine_kv_bucket(uint32_t required_kv) {
    if (required_kv == 0) {
        return 0;
    }
    if (required_kv <= 32)   return 32;
    if (required_kv <= 64)   return 64;
    if (required_kv <= 128)  return 128;
    if (required_kv <= 256)  return 256;
    if (required_kv <= 512)  return 512;
    if (required_kv <= 1024) return 1024;
    if (required_kv <= 2048) return 2048;
    return ((required_kv + 2047) / 2048) * 2048;
}

uint32_t next_decode_kv_bucket(uint32_t required_kv, uint32_t min_bucket, uint32_t large_step) {
    if (required_kv == 0) {
        return 0;
    }
    // Clamp inputs to sane ranges so a misconfigured env knob never produces
    // a degenerate bucket sequence.
    if (min_bucket < 32) {
        min_bucket = 32;
    }
    if (large_step < 1024) {
        large_step = 1024;
    }
    if (large_step < min_bucket) {
        large_step = min_bucket;
    }
    if (required_kv <= min_bucket) {
        // Match prefill padding: e.g. prompt=15 → bucket 32, not min_bucket.
        return fine_kv_bucket(required_kv);
    }
    // Geometric growth (doubling) up to `large_step`.
    uint32_t bucket = min_bucket;
    while (bucket < required_kv && bucket < large_step) {
        bucket *= 2;
    }
    if (bucket >= required_kv) {
        return bucket;
    }
    // Past the geometric phase: ceil to a multiple of `large_step`. This is
    // coarser than the previous 2048-step policy and trades a bit of
    // padded-attention bandwidth for far fewer recaptures during long
    // generations.
    return ((required_kv + large_step - 1) / large_step) * large_step;
}

uint32_t legacy_decode_kv_bucket(uint32_t required_kv) {
    if (required_kv <= 256) {
        return 256;
    }
    if (required_kv <= 1024) {
        return 1024;
    }
    if (required_kv <= 2048) {
        return 2048;
    }
    return ((required_kv + 2047) / 2048) * 2048;
}

// ggml_gated_delta_net state layout (ggml >= 2025): (S_v*S_v*H, K, n_seqs), K=1 for inference.
int64_t gdn_rec_slot_floats(int head_v_dim, int num_v_heads) {
    return static_cast<int64_t>(head_v_dim) * head_v_dim * num_v_heads;
}

ggml_tensor* wrap_gdn_recurrent_state(
    ggml_context* ctx,
    ggml_tensor* rec_data,
    int64_t rec_slot_floats,
    int64_t n_seqs
) {
    return ggml_reshape_3d(ctx, rec_data, rec_slot_floats, 1, n_seqs);
}

size_t gdn_attn_scores_byte_offset(
    int head_v_dim, int num_v_heads, int n_seq_tokens, int n_seqs, ggml_type type
) {
    return static_cast<size_t>(head_v_dim) * num_v_heads * n_seq_tokens * n_seqs
        * ggml_type_size(type);
}
}

ggml_backend_t Qwen35moeForwardPass::layer_backend(int il) const {
    const auto& device_map = model_->get_layer_device_map();
    if (device_map.empty() || il < 0 || static_cast<size_t>(il) >= device_map.size()) {
        return model_->get_curr_backend();
    }
    return device_map[static_cast<size_t>(il)];
}

bool Qwen35moeForwardPass::layer_allows_flash_attn(int il) const {
    return use_flash_attention_ && backend_name_contains_cuda(layer_backend(il));
}

bool Qwen35moeForwardPass::deltanet_decode_fast_path_enabled() const {
    if (env_flag_enabled("QWEN35MOE_DN_DECODE_OFF")) {
        return false;
    }
    if (env_flag_enabled("QWEN35MOE_DN_DECODE_FORCE")) {
        return true;
    }
    return model_ && backend_name_contains_cuda(model_->get_curr_backend());
}

bool Qwen35moeForwardPass::layer_allows_deltanet_decode_fast_path(int il) const {
    if (!deltanet_decode_fast_path_enabled()) {
        return false;
    }
    // Fused DeltaNet decode kernels are CUDA-only; mixed AUTO must not route
    // CPU-resident layers through them just because a GPU backend exists.
    return backend_name_contains_cuda(layer_backend(il));
}

bool Qwen35moeForwardPass::deltanet_decode_fused_proj_enabled(
    const ggml_tensor* w_qkv,
    const ggml_tensor* w_gate,
    const ggml_tensor* w_beta,
    const ggml_tensor* w_alpha
) const {
    if (env_flag_enabled("QWEN35MOE_DN_FUSED_PROJ_OFF")) {
        return false;
    }
    const ggml_tensor* weights[] = {w_qkv, w_gate, w_beta, w_alpha};
    for (const ggml_tensor* w : weights) {
        if (w == nullptr) {
            return false;
        }
        // The fused CUDA projection pre-quantizes the activation once and reuses
        // it for all four weights. Q8_0 uses a different vector-dot contract from
        // the K-quants used by the original model, so mixed Q8_0/Q5_K UD weights
        // must stay on the standard mul_mat projection path.
        if (w->type == GGML_TYPE_Q8_0) {
            return false;
        }
    }
    return true;
}

bool Qwen35moeForwardPass::attention_range_homogeneous(uint32_t layer_begin, uint32_t layer_end) const {
    if (!model_->is_mixed_mode()) {
        return true;
    }
    bool saw_cuda = false;
    bool saw_non_cuda = false;
    for (uint32_t il = layer_begin; il < layer_end; ++il) {
        if (!is_full_attention_layer(il)) {
            continue;
        }
        if (backend_name_contains_cuda(layer_backend(static_cast<int>(il)))) {
            saw_cuda = true;
        } else {
            saw_non_cuda = true;
        }
        if (saw_cuda && saw_non_cuda) {
            return false;
        }
    }
    return true;
}

uint32_t Qwen35moeForwardPass::decode_cache_bucket_capacity(uint32_t required_kv) const {
    if (context_len_ == 0 || required_kv == 0) {
        return 0;
    }

    // Decode bucket policy
    // ---------------------
    // The decode cached-graph rebuild cost (sched_reset + alloc_graph) is
    // dominated by per-layer ggml backend setup, so reducing the *number* of
    // recaptures is the primary win. The bucket choice trades two costs:
    //   - smaller bucket  => less padded-attention bandwidth during decode,
    //                        but recapture every time required_kv crosses the
    //                        next bucket boundary;
    //   - larger bucket   => wider attention mask (FA still ignores masked
    //                        columns) but far fewer recaptures.
    //
    // Env knobs:
    //   QWEN35MOE_DECODE_KV_LOCK_CTX=1   – pin the bucket to context_len_, so
    //                                      decode is recapture-free after the
    //                                      first capture. Best for long
    //                                      generation; pays a constant mask
    //                                      width even for short decodes.
    //   QWEN35MOE_DECODE_KV_COARSE=1     – legacy 256/1024/2048-step buckets.
    //   QWEN35MOE_DECODE_KV_MIN_BUCKET=N – geometric anchor after the fine
    //                                      ladder (default 256). While
    //                                      required_kv <= N, fine_kv_bucket()
    //                                      applies (32..512). Set higher to
    //                                      reduce recaptures at the cost of
    //                                      wider attention graphs.
    //   QWEN35MOE_DECODE_KV_GROW_STEP=N  – linear bucket step past the
    //                                      geometric phase (default 4096).
    if (env_flag_enabled("QWEN35MOE_DECODE_KV_LOCK_CTX")) {
        return context_len_;
    }

    uint32_t bucket;
    if (env_flag_enabled("QWEN35MOE_DECODE_KV_COARSE")) {
        bucket = legacy_decode_kv_bucket(required_kv);
    } else {
        const uint32_t min_bucket = env_u32_or_default(
            "QWEN35MOE_DECODE_KV_MIN_BUCKET",
            kDefaultDecodeMinBucket
        );
        const uint32_t large_step = env_u32_or_default(
            "QWEN35MOE_DECODE_KV_GROW_STEP",
            kDefaultDecodeLargeStep
        );
        bucket = next_decode_kv_bucket(required_kv, min_bucket, large_step);
    }

    if (bucket > context_len_) {
        bucket = context_len_;
    }
    if (bucket < required_kv) {
        bucket = required_kv;
    }
    return bucket;
}

uint32_t Qwen35moeForwardPass::prefill_token_bucket_capacity(uint32_t n_tokens) const {
    if (n_tokens == 0) {
        return 0;
    }
    // Prefill keeps the fine-grained ladder. Prompt sizes are highly variable
    // and prefill is compute-bound, so padding the per-prompt graph up to a
    // A wide decode bucket floor would add real wall time. The
    // decode cache reuses these buckets too when seeding `prefill_graph_kv_*`,
    // and is pinned to context_len_ via `prefill_graph_kv_capacity()` below.
    return fine_kv_bucket(n_tokens);
}

uint32_t Qwen35moeForwardPass::prefill_graph_kv_capacity() const {
    return decode_cache_bucket_capacity(context_len_);
}

bool Qwen35moeForwardPass::can_use_cached_prefill(uint32_t n_tokens) const {
    if (n_tokens == 0) {
        return false;
    }
    // Segmented prefill builds per-backend segment graphs without the unified
    // scheduler cache; only the scheduler-owned prefill path can reuse graphs.
    if (model_->is_mixed_mode() && !mixed_prefill_scheduler_enabled_) {
        return false;
    }
    if (model_->has_any_split_device_layers() || model_->uses_tensor_overrides()) {
        return false;
    }
    return true;
}

void Qwen35moeForwardPass::collect_prefill_mask_tensors(ggml_cgraph* gf) {
    cached_prefill_mask_tensors_.clear();
    if (!gf) {
        return;
    }
    if (ggml_tensor* shared_kq_mask = ggml_graph_get_tensor(gf, "kq_mask_shared")) {
        cached_prefill_mask_tensors_.push_back(shared_kq_mask);
        return;
    }
    for (uint32_t il = 0; il < model_->trunk_layer_count(); ++il) {
        if (!is_full_attention_layer(il)) {
            continue;
        }
        char name[32];
        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
        if (ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, name)) {
            cached_prefill_mask_tensors_.push_back(kq_mask);
        }
    }
}

void Qwen35moeForwardPass::prepare_cached_prefill_graph(
    ggml_backend_sched_t scheduler,
    const PrefillGraphSignature& signature
) {
    if (scheduler == nullptr) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_prefill_graph: scheduler is null");
    }
    if (signature.n_tokens == 0 || signature.kv_capacity == 0) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_prefill_graph: invalid signature");
    }
    if (signature.slot_idx >= max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_prefill_graph: slot_idx out of range");
    }

    std::vector<int32_t> dummy_tokens(signature.n_tokens, 0);
    // Mask width must match active n_kv (= n_tokens at pos 0). Using kv_capacity
    // here pads the mask; ggml_flash_attn_ext requires a contiguous mask and
    // cannot take a non-contiguous prefix view of a larger tensor.
    cached_prefill_graph_ = build_prefill_graph(
        dummy_tokens,
        0,
        signature.slot_idx,
        signature.n_tokens
    );

    cached_prefill_tokens_tensor_ = ggml_graph_get_tensor(cached_prefill_graph_, "tokens");
    cached_prefill_pos_tensor_ = ggml_graph_get_tensor(cached_prefill_graph_, "inp_pos");
    if (!cached_prefill_tokens_tensor_ || !cached_prefill_pos_tensor_) {
        throw std::runtime_error(
            "Qwen35moeForwardPass::prepare_cached_prefill_graph: missing cached prefill input tensors"
        );
    }
    collect_prefill_mask_tensors(cached_prefill_graph_);

    ggml_backend_sched_reset(scheduler);
    decode_graph_scheduler_reset_count_++;
    if (!ggml_backend_sched_alloc_graph(scheduler, cached_prefill_graph_)) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_prefill_graph: failed to allocate graph");
    }

    cached_prefill_graph_allocated_ = true;
    cached_prefill_signature_ = signature;
    cached_prefill_signature_valid_ = true;
    prefill_graph_recapture_count_++;
}

bool Qwen35moeForwardPass::is_cached_prefill_graph_compatible(const PrefillGraphSignature& signature) const {
    return cached_prefill_graph_ != nullptr &&
           cached_prefill_graph_allocated_ &&
           cached_prefill_signature_valid_ &&
           cached_prefill_signature_.slot_idx == signature.slot_idx &&
           // Exact token count match: the graph tensor shapes are built for
           // signature.n_tokens columns. Reusing a larger cached graph for a
           // shorter prompt would require padding + trimmed KV writes; keep
           // exact match here and let ensure_cached_prefill_graph rebuild on
           // a new length (still amortized across repeated prompts).
           cached_prefill_signature_.n_tokens == signature.n_tokens &&
           cached_prefill_signature_.kv_capacity == signature.kv_capacity &&
           cached_prefill_signature_.context_len == signature.context_len &&
           cached_prefill_signature_.use_flash_attention == signature.use_flash_attention &&
           cached_prefill_signature_.is_mixed_mode == signature.is_mixed_mode &&
           cached_prefill_signature_.device_map_hash == signature.device_map_hash &&
           cached_prefill_signature_.sampling_top_k == signature.sampling_top_k &&
           std::fabs(cached_prefill_signature_.sampling_temperature - signature.sampling_temperature) < 1e-6f;
}

void Qwen35moeForwardPass::ensure_cached_prefill_graph(
    ggml_backend_sched_t scheduler,
    const PrefillGraphSignature& signature
) {
    prefill_graph_lookup_count_++;
    if (is_cached_prefill_graph_compatible(signature)) {
        prefill_graph_hit_count_++;
        return;
    }

    prefill_graph_miss_count_++;
    try {
        prepare_cached_prefill_graph(scheduler, signature);
    } catch (const std::exception& ex) {
        std::fprintf(
            stderr,
            "[PERF][prefill-graph] recapture_failed n_tokens=%u err=%s\n",
            signature.n_tokens,
            ex.what()
        );
        prefill_graph_fallback_count_++;
        clear_cached_prefill_graph();
        throw;
    }
}

void Qwen35moeForwardPass::set_cached_prefill_inputs(const std::vector<int32_t>& tokens, int pos) {
    if (!cached_prefill_graph_) {
        throw std::runtime_error("Qwen35moeForwardPass::set_cached_prefill_inputs: graph is null");
    }
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    if (n_tok != cached_prefill_signature_.n_tokens) {
        throw std::runtime_error("Qwen35moeForwardPass::set_cached_prefill_inputs: token count mismatch");
    }

    ggml_backend_tensor_set(cached_prefill_tokens_tensor_, tokens.data(), 0, n_tok * sizeof(int32_t));

    std::vector<int32_t> pos_data(n_tok);
    for (uint32_t i = 0; i < n_tok; ++i) {
        pos_data[i] = pos + static_cast<int>(i);
    }
    ggml_backend_tensor_set(cached_prefill_pos_tensor_, pos_data.data(), 0, n_tok * sizeof(int32_t));

    if (cached_prefill_mask_tensors_.empty()) {
        return;
    }

    const uint32_t graph_n_kv = static_cast<uint32_t>(cached_prefill_mask_tensors_.front()->ne[0]);
    std::vector<float> mask_f32;
    fill_dynamic_prefill_mask(mask_f32, graph_n_kv, n_tok, pos);
    for (ggml_tensor* kq_mask : cached_prefill_mask_tensors_) {
        upload_decode_mask_tensor(kq_mask, mask_f32);
    }
}

bool Qwen35moeForwardPass::run_prefill_chunk_scheduler(
    ggml_backend_sched_t scheduler,
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx,
    bool use_topk,
    std::vector<float>* logits_out,
    TopKSampleCandidates* topk_out
) {
    if (!scheduler || !can_use_cached_prefill(static_cast<uint32_t>(tokens.size()))) {
        return false;
    }
    if (use_topk && sampling_top_k_ <= 0) {
        return false;
    }
    // Cached graph is captured at KV position 0 with mask width == chunk length.
    if (pos != 0) {
        return false;
    }

    try {
        PrefillGraphSignature signature;
        signature.slot_idx = slot_idx;
        signature.n_tokens = static_cast<uint32_t>(tokens.size());
        signature.kv_capacity = prefill_graph_kv_capacity();
        signature.context_len = context_len_;
        signature.use_flash_attention = use_flash_attention_;
        signature.is_mixed_mode = model_->is_mixed_mode();
        signature.device_map_hash = model_->compute_device_map_hash();
        signature.sampling_top_k = sampling_top_k_;
        signature.sampling_temperature = sampling_temperature_;

        ensure_cached_prefill_graph(scheduler, signature);
        set_cached_prefill_inputs(tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, cached_prefill_graph_);
        advance_cache(static_cast<uint32_t>(tokens.size()), slot_idx);

        if (use_topk) {
            if (!topk_out) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_chunk_scheduler: topk_out is null");
            }
            *topk_out = get_output_topk_candidates(cached_prefill_graph_, 0);
        } else {
            if (!logits_out) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_chunk_scheduler: logits_out is null");
            }
            *logits_out = get_output_logits(cached_prefill_graph_);
        }
        return true;
    } catch (const std::exception& ex) {
        prefill_graph_fallback_count_++;
        if (decode_graph_diag_enabled_) {
            std::fprintf(
                stderr,
                "[PERF][prefill-graph] cached path failed n_tokens=%zu pos=%d: %s\n",
                tokens.size(),
                pos,
                ex.what()
            );
        }
        clear_cached_prefill_graph();
        return false;
    }
}

uint32_t Qwen35moeForwardPass::effective_prefill_batch_limit() const {
    uint32_t limit = n_batch_tokens_;
    if (limit == 0) {
        limit = context_len_;
    }
    return std::max(1u, std::min(limit, context_len_));
}

uint32_t Qwen35moeForwardPass::effective_prefill_ubatch_limit() const {
    uint32_t limit = n_ubatch_tokens_;
    if (limit == 0) {
        limit = effective_prefill_batch_limit();
    }
    limit = std::min(limit, effective_prefill_batch_limit());
    return std::max(1u, std::min(limit, context_len_));
}

void Qwen35moeForwardPass::run_prefill_ubatch_eager_scheduler(
    ggml_backend_sched_t scheduler,
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx,
    bool use_topk,
    std::vector<float>* logits_out,
    TopKSampleCandidates* topk_out
) {
    ggml_backend_sched_reset(scheduler);
    decode_graph_scheduler_reset_count_++;
    ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
    if (!ggml_backend_sched_alloc_graph(scheduler, gf)) {
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_ubatch_eager_scheduler: failed to allocate graph");
    }
    set_inputs(gf, tokens, pos);
    ggml_backend_sched_graph_compute(scheduler, gf);
    advance_cache(static_cast<uint32_t>(tokens.size()), slot_idx);

    if (use_topk) {
        if (!topk_out) {
            throw std::runtime_error("Qwen35moeForwardPass::run_prefill_ubatch_eager_scheduler: topk_out is null");
        }
        *topk_out = get_output_topk_candidates(gf, 0);
    } else {
        if (!logits_out) {
            throw std::runtime_error("Qwen35moeForwardPass::run_prefill_ubatch_eager_scheduler: logits_out is null");
        }
        *logits_out = get_output_logits(gf);
    }
}

void Qwen35moeForwardPass::run_prefill_microbatched_scheduler(
    ggml_backend_sched_t scheduler,
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx,
    bool use_topk,
    std::vector<float>* final_logits,
    TopKSampleCandidates* final_topk
) {
    if (tokens.empty()) {
        return;
    }

    const bool use_segmented_prefill =
        model_->is_mixed_mode() &&
        !mixed_prefill_scheduler_enabled_ &&
        !layer_segments_.empty();

    if (use_segmented_prefill) {
        if (!mixed_prefill_segmented_warned_) {
            std::fprintf(
                stderr,
                "[PERF][prefill-segmented] auto/mixed mode: prefill uses per-segment scheduler execution "
                "(same pipeline as decode). Set QWEN35MOE_MIXED_PREFILL_SCHEDULER=1 only when "
                "weights are not split across backends.\n"
            );
            mixed_prefill_segmented_warned_ = true;
        }
        if (use_topk) {
            if (!final_topk) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_microbatched_scheduler: final_topk is null");
            }
            *final_topk = run_prefill_segmented_topk_eager(tokens, pos, slot_idx);
        } else {
            if (!final_logits) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_microbatched_scheduler: final_logits is null");
            }
            *final_logits = run_prefill_segmented_eager(tokens, pos, slot_idx);
        }
        return;
    }

    const uint32_t ubatch = effective_prefill_ubatch_limit();
    if (tokens.size() <= ubatch) {
        if (!run_prefill_chunk_scheduler(scheduler, tokens, pos, slot_idx, use_topk, final_logits, final_topk)) {
            run_prefill_ubatch_eager_scheduler(scheduler, tokens, pos, slot_idx, use_topk, final_logits, final_topk);
        }
        return;
    }

    prefill_ubatch_chunk_count_++;
    const uint32_t micro_count =
        static_cast<uint32_t>((tokens.size() + ubatch - 1) / ubatch);
    prefill_ubatch_micro_step_count_ += micro_count;

    for (size_t start = 0; start < tokens.size(); start += ubatch) {
        const size_t micro_size = std::min(static_cast<size_t>(ubatch), tokens.size() - start);
        std::vector<int32_t> micro_tokens(tokens.begin() + start, tokens.begin() + start + micro_size);
        const int micro_pos = pos + static_cast<int>(start);

        if (run_prefill_chunk_scheduler(
                scheduler, micro_tokens, micro_pos, slot_idx, use_topk, final_logits, final_topk)) {
            continue;
        }
        run_prefill_ubatch_eager_scheduler(
            scheduler, micro_tokens, micro_pos, slot_idx, use_topk, final_logits, final_topk);
    }

    if (decode_graph_diag_enabled_ && micro_count > 1) {
        std::fprintf(
            stderr,
            "[PERF][prefill-ubatch] tokens=%zu ubatch=%u micro_steps=%u pos=%d\n",
            tokens.size(),
            ubatch,
            micro_count,
            pos
        );
    }
}

void Qwen35moeForwardPass::run_prefill_microbatched_direct(
    ggml_gallocr_t allocr,
    ggml_backend_t backend,
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx,
    bool use_topk,
    std::vector<float>* final_logits,
    TopKSampleCandidates* final_topk,
    bool allow_segmented
) {
    if (tokens.empty()) {
        return;
    }

    const bool use_segmented_prefill =
        allow_segmented &&
        model_->is_mixed_mode() &&
        !mixed_prefill_scheduler_enabled_ &&
        !layer_segments_.empty();

    if (use_segmented_prefill) {
        if (use_topk) {
            if (!final_topk) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_microbatched_direct: final_topk is null");
            }
            *final_topk = run_prefill_segmented_topk_eager(tokens, pos, slot_idx);
        } else {
            if (!final_logits) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_microbatched_direct: final_logits is null");
            }
            *final_logits = run_prefill_segmented_eager(tokens, pos, slot_idx);
        }
        return;
    }

    const uint32_t ubatch = effective_prefill_ubatch_limit();
    if (tokens.size() <= ubatch) {
        ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
        ggml_gallocr_alloc_graph(allocr, gf);
        set_inputs(gf, tokens, pos);
        ggml_backend_graph_compute(backend, gf);
        advance_cache(static_cast<uint32_t>(tokens.size()), slot_idx);
        if (use_topk) {
            if (!final_topk) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_microbatched_direct: final_topk is null");
            }
            *final_topk = get_output_topk_candidates(gf, 0);
        } else {
            if (!final_logits) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_microbatched_direct: final_logits is null");
            }
            *final_logits = get_output_logits(gf);
        }
        return;
    }

    prefill_ubatch_chunk_count_++;
    const uint32_t micro_count =
        static_cast<uint32_t>((tokens.size() + ubatch - 1) / ubatch);
    prefill_ubatch_micro_step_count_ += micro_count;

    for (size_t start = 0; start < tokens.size(); start += ubatch) {
        const size_t micro_size = std::min(static_cast<size_t>(ubatch), tokens.size() - start);
        std::vector<int32_t> micro_tokens(tokens.begin() + start, tokens.begin() + start + micro_size);
        const int micro_pos = pos + static_cast<int>(start);

        ggml_cgraph* gf = build_prefill_graph(micro_tokens, micro_pos, slot_idx);
        ggml_gallocr_alloc_graph(allocr, gf);
        set_inputs(gf, micro_tokens, micro_pos);
        ggml_backend_graph_compute(backend, gf);
        advance_cache(static_cast<uint32_t>(micro_size), slot_idx);

        if (use_topk) {
            if (!final_topk) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_microbatched_direct: final_topk is null");
            }
            *final_topk = get_output_topk_candidates(gf, 0);
        } else {
            if (!final_logits) {
                throw std::runtime_error("Qwen35moeForwardPass::run_prefill_microbatched_direct: final_logits is null");
            }
            *final_logits = get_output_logits(gf);
        }
    }

    if (decode_graph_diag_enabled_ && micro_count > 1) {
        std::fprintf(
            stderr,
            "[PERF][prefill-ubatch] tokens=%zu ubatch=%u micro_steps=%u pos=%d (direct)\n",
            tokens.size(),
            ubatch,
            micro_count,
            pos
        );
    }
}

void set_mask_data(ggml_tensor* tensor, const std::vector<float>& mask) {
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> mask_f16(mask.size());
        ggml_fp32_to_fp16_row(mask.data(), mask_f16.data(), static_cast<int64_t>(mask.size()));
        ggml_backend_tensor_set(tensor, mask_f16.data(), 0, mask_f16.size() * sizeof(ggml_fp16_t));
        return;
    }
    ggml_backend_tensor_set(tensor, mask.data(), 0, mask.size() * sizeof(float));
}

void Qwen35moeForwardPass::upload_decode_mask_tensor(
    ggml_tensor* kq_mask,
    const std::vector<float>& mask_f32
) {
    if (!kq_mask) {
        return;
    }
    set_mask_data(kq_mask, mask_f32);
}

void Qwen35moeForwardPass::fill_dynamic_decode_mask_1d(
    std::vector<float>& mask_f32,
    uint32_t graph_n_kv,
    uint32_t active_kv,
    bool incremental,
    int pos,
    int& last_mask_pos
) {
    if (graph_n_kv == 0) {
        return;
    }
    if (active_kv > graph_n_kv) {
        active_kv = graph_n_kv;
    }

    const auto block_padding_tail = [&]() {
        for (uint32_t j = active_kv; j < graph_n_kv; ++j) {
            mask_f32[j] = -INFINITY;
        }
    };

    const bool can_incremental =
        incremental &&
        use_flash_attention_ &&
        last_mask_pos >= 0 &&
        pos == last_mask_pos + 1 &&
        mask_f32.size() == graph_n_kv;

    if (can_incremental) {
        if (static_cast<uint32_t>(pos) < graph_n_kv) {
            mask_f32[static_cast<uint32_t>(pos)] = 0.0f;
        }
        block_padding_tail();
        last_mask_pos = pos;
        return;
    }

    mask_f32.assign(graph_n_kv, -INFINITY);
    for (uint32_t j = 0; j < active_kv; ++j) {
        mask_f32[j] = 0.0f;
    }
    block_padding_tail();
    last_mask_pos = pos;
}

void Qwen35moeForwardPass::fill_dynamic_prefill_mask(
    std::vector<float>& mask_f32,
    uint32_t graph_n_kv,
    uint32_t n_tok,
    int pos
) {
    if (graph_n_kv == 0 || n_tok == 0) {
        return;
    }
    mask_f32.assign(static_cast<size_t>(graph_n_kv) * n_tok, -INFINITY);
    for (uint32_t t = 0; t < n_tok; ++t) {
        const uint32_t q_pos = static_cast<uint32_t>(pos) + t;
        const uint32_t active_kv = std::min(graph_n_kv, q_pos + 1);
        float* row = mask_f32.data() + static_cast<size_t>(t) * graph_n_kv;
        for (uint32_t j = 0; j < active_kv; ++j) {
            row[j] = 0.0f;
        }
    }
}

void Qwen35moeForwardPass::fill_dynamic_decode_mask_batched(
    std::vector<float>& mask_f32,
    uint32_t graph_n_kv,
    uint32_t n_batch,
    const std::vector<int32_t>& positions,
    bool incremental,
    std::vector<int32_t>& last_mask_positions
) {
    if (graph_n_kv == 0 || n_batch == 0) {
        return;
    }
    if (last_mask_positions.size() < n_batch) {
        last_mask_positions.assign(n_batch, -1);
    }

    const size_t mask_elems = static_cast<size_t>(graph_n_kv) * n_batch;

    bool all_incremental = incremental && use_flash_attention_;
    if (all_incremental) {
        for (uint32_t b = 0; b < n_batch; ++b) {
            const int pos = positions[b];
            if (pos < 0 ||
                last_mask_positions[b] < 0 ||
                pos != last_mask_positions[b] + 1) {
                all_incremental = false;
                break;
            }
        }
    }

    if (!all_incremental || mask_f32.size() != mask_elems) {
        mask_f32.assign(mask_elems, -INFINITY);
        all_incremental = false;
    }

    if (all_incremental) {
        for (uint32_t b = 0; b < n_batch; ++b) {
            const int pos = positions[b];
            const uint32_t active_kv = static_cast<uint32_t>(pos) + 1;
            float* row = mask_f32.data() + static_cast<size_t>(b) * graph_n_kv;
            if (static_cast<uint32_t>(pos) < graph_n_kv) {
                row[pos] = 0.0f;
            }
            for (uint32_t j = active_kv; j < graph_n_kv; ++j) {
                row[j] = -INFINITY;
            }
            last_mask_positions[b] = pos;
        }
        return;
    }

    for (uint32_t b = 0; b < n_batch; ++b) {
        const int pos = positions[b];
        if (pos < 0) {
            last_mask_positions[b] = -1;
            continue;
        }
        const uint32_t active_kv = std::min(graph_n_kv, static_cast<uint32_t>(pos) + 1);
        float* row = mask_f32.data() + static_cast<size_t>(b) * graph_n_kv;
        for (uint32_t j = 0; j < active_kv; ++j) {
            row[j] = 0.0f;
        }
        for (uint32_t j = active_kv; j < graph_n_kv; ++j) {
            row[j] = -INFINITY;
        }
        last_mask_positions[b] = pos;
    }
}

bool Qwen35moeForwardPass::can_run_token_embedding_on_backend(ggml_backend_t backend) const {
    if (!backend) {
        return false;
    }
    ggml_tensor* token_embedding = model_->get_token_embedding_weight();
    if (!token_embedding) {
        throw std::runtime_error("Qwen35moeForwardPass::can_run_token_embedding_on_backend: token embedding weight missing");
    }
    if (!token_embedding->buffer) {
        return true;
    }
    const ggml_backend_buffer_type_t token_buft = ggml_backend_buffer_get_type(token_embedding->buffer);
    const ggml_backend_buffer_type_t backend_buft = ggml_backend_get_default_buffer_type(backend);
    return token_buft == backend_buft;
}

ggml_backend_t Qwen35moeForwardPass::find_backend_for_tensor(const ggml_tensor* tensor) const {
    ggml_backend_t fallback = model_->get_curr_backend();
    if (!tensor || !tensor->buffer) {
        return fallback;
    }

    const ggml_backend_buffer_type_t tensor_buft = ggml_backend_buffer_get_type(tensor->buffer);
    if (fallback && ggml_backend_get_default_buffer_type(fallback) == tensor_buft) {
        return fallback;
    }
    for (const auto& segment : layer_segments_) {
        if (segment.backend && ggml_backend_get_default_buffer_type(segment.backend) == tensor_buft) {
            return segment.backend;
        }
    }
    return fallback;
}

void Qwen35moeForwardPass::maybe_log_segment_tensor(
    const char* scope,
    const LayerSegment* segment,
    ggml_tensor* tensor
) const {
    if (!dev_check_enabled_ || !tensor) {
        return;
    }
    const char* backend_name = "n/a";
    if (segment && segment->backend) {
        backend_name = ggml_backend_name(segment->backend);
    } else if (tensor->buffer) {
        backend_name = ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer));
    }
    std::fprintf(
        stderr,
        "[dev-check] segmented %s backend=%s tensor=%s type=%s ne=[%lld,%lld,%lld,%lld] bytes=%zu\n",
        scope,
        backend_name,
        tensor->name,
        ggml_type_name(tensor->type),
        static_cast<long long>(tensor->ne[0]),
        static_cast<long long>(tensor->ne[1]),
        static_cast<long long>(tensor->ne[2]),
        static_cast<long long>(tensor->ne[3]),
        static_cast<size_t>(ggml_nbytes(tensor))
    );
}

void Qwen35moeForwardPass::validate_handoff_tensor(
    const char* where,
    ggml_tensor* tensor,
    ggml_type expected_type,
    size_t expected_bytes
) const {
    if (!tensor) {
        throw std::runtime_error(std::string(where) + ": missing tensor");
    }
    const int64_t expected_ne0 = static_cast<int64_t>(model_->meta_->qwen35moe.embedding_length);
    const size_t row_bytes = ggml_row_size(expected_type, expected_ne0);
    if (row_bytes == 0 || expected_bytes == 0 || expected_bytes % row_bytes != 0) {
        throw std::runtime_error(
            std::string(where) +
            ": hidden byte size is not an integral embedding row (bytes=" +
            std::to_string(expected_bytes) +
            ", row_bytes=" + std::to_string(row_bytes) + ")");
    }
    const int64_t expected_ne1 = static_cast<int64_t>(expected_bytes / row_bytes);
    if (tensor->ne[0] != expected_ne0 || tensor->ne[1] != expected_ne1) {
        throw std::runtime_error(
            std::string(where) +
            ": hidden shape mismatch (expected [" + std::to_string(expected_ne0) + "," +
            std::to_string(expected_ne1) + "], got [" +
            std::to_string(tensor->ne[0]) + "," + std::to_string(tensor->ne[1]) + "])");
    }
    const size_t actual_bytes = static_cast<size_t>(ggml_nbytes(tensor));
    if (tensor->type != expected_type || actual_bytes != expected_bytes) {
        throw std::runtime_error(
            std::string(where) +
            ": hidden handoff mismatch (expected type=" + ggml_type_name(expected_type) +
            ", bytes=" + std::to_string(expected_bytes) +
            "; got type=" + ggml_type_name(tensor->type) +
            ", bytes=" + std::to_string(actual_bytes) + ")");
    }
}

void Qwen35moeForwardPass::copy_handoff_tensor(
    const char* where,
    ggml_tensor* src,
    ggml_tensor* dst,
    ggml_type expected_type
) const {
    if (!src || !dst) {
        throw std::runtime_error(std::string(where) + ": missing hidden handoff tensor");
    }
    const size_t hidden_bytes = static_cast<size_t>(ggml_nbytes(src));
    if (hidden_bytes == 0) {
        throw std::runtime_error(std::string(where) + ": hidden handoff has zero bytes");
    }
    validate_handoff_tensor(where, src, expected_type, hidden_bytes);
    validate_handoff_tensor(where, dst, expected_type, hidden_bytes);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] != dst->ne[i] || src->nb[i] != dst->nb[i]) {
            throw std::runtime_error(std::string(where) + ": hidden handoff layout mismatch");
        }
    }
    maybe_log_segment_tensor("segment_hidden_copy_src", nullptr, src);
    maybe_log_segment_tensor("segment_hidden_copy_dst", nullptr, dst);
    ggml_backend_tensor_copy(src, dst);
}

void Qwen35moeForwardPass::ensure_handoff_pinned(size_t nbytes) const {
#ifdef QWEN35MOE_USE_CUDA
    if (handoff_pinned_.size() >= nbytes) {
        return;
    }
    if (!handoff_pinned_.empty()) {
        unpin_host_region(handoff_pinned_.data());
    }
    handoff_pinned_.resize(nbytes);
    pin_host_region(handoff_pinned_.data(), handoff_pinned_.size());
#else
    (void)nbytes;
#endif
}

void Qwen35moeForwardPass::ensure_output_pinned(size_t nbytes) const {
#ifdef QWEN35MOE_USE_CUDA
    if (output_pinned_.size() >= nbytes) {
        return;
    }
    if (!output_pinned_.empty()) {
        unpin_host_region(output_pinned_.data());
    }
    output_pinned_.resize(nbytes);
    pin_host_region(output_pinned_.data(), output_pinned_.size());
#else
    (void)nbytes;
#endif
}

void Qwen35moeForwardPass::copy_handoff_tensor_async(
    const char* where,
    ggml_backend_t src_backend,
    ggml_backend_t dst_backend,
    ggml_tensor* src,
    ggml_tensor* dst,
    ggml_type expected_type
) const {
    if (!src || !dst) {
        throw std::runtime_error(std::string(where) + ": missing hidden handoff tensor");
    }
    const size_t hidden_bytes = static_cast<size_t>(ggml_nbytes(src));
    if (hidden_bytes == 0) {
        throw std::runtime_error(std::string(where) + ": hidden handoff has zero bytes");
    }
    validate_handoff_tensor(where, src, expected_type, hidden_bytes);
    validate_handoff_tensor(where, dst, expected_type, hidden_bytes);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] != dst->ne[i] || src->nb[i] != dst->nb[i]) {
            throw std::runtime_error(std::string(where) + ": hidden handoff layout mismatch");
        }
    }
    maybe_log_segment_tensor("segment_hidden_copy_src", nullptr, src);
    maybe_log_segment_tensor("segment_hidden_copy_dst", nullptr, dst);

#ifdef QWEN35MOE_USE_CUDA
    const bool src_on_cuda = src->buffer && !ggml_backend_buffer_is_host(src->buffer);
    const bool dst_on_cuda = dst->buffer && !ggml_backend_buffer_is_host(dst->buffer);

    // Async segment graphs: sync before host reads or cross-backend copies.
    // Same-device GPU→GPU relies on explicit producer sync at call sites.
    if (src_backend && (!src_on_cuda || src_on_cuda != dst_on_cuda)) {
        ggml_backend_synchronize(src_backend);
    }
#else
    if (src_backend) {
        ggml_backend_synchronize(src_backend);
    }
#endif

#ifdef QWEN35MOE_USE_CUDA
    if (src_on_cuda && !dst_on_cuda && src_backend && ggml_backend_is_cuda(src_backend)) {
        // GPU -> CPU: async D2H into pinned staging, then one stream sync.
        ensure_handoff_pinned(hidden_bytes);
        ggml_backend_tensor_get_async(
            src_backend, src, handoff_pinned_.data(), 0, hidden_bytes);
        ggml_backend_synchronize(src_backend);
        ggml_backend_tensor_set(dst, handoff_pinned_.data(), 0, hidden_bytes);
        return;
    }

    if (!src_on_cuda && dst_on_cuda && dst_backend && ggml_backend_is_cuda(dst_backend)) {
        // CPU -> GPU: H2D queued on the CUDA stream (no dual backend flush).
        ensure_handoff_pinned(hidden_bytes);
        std::memcpy(handoff_pinned_.data(), src->data, hidden_bytes);
        ggml_backend_tensor_set_async(
            dst_backend, dst, handoff_pinned_.data(), 0, hidden_bytes);
        return;
    }

    if (src_on_cuda && dst_on_cuda && src_backend && dst_backend &&
        ggml_backend_is_cuda(src_backend) && ggml_backend_is_cuda(dst_backend)) {
        ggml_backend_tensor_copy_async(src_backend, dst_backend, src, dst);
        return;
    }
#endif

    if (dst_backend) {
        ggml_backend_tensor_copy_async(src_backend, dst_backend, src, dst);
        return;
    }
    ggml_backend_tensor_copy(src, dst);
}

void Qwen35moeForwardPass::pin_host_region(void* ptr, size_t size) const {
#ifdef QWEN35MOE_USE_CUDA
    if (!ptr || size == 0) {
        return;
    }
    if (!model_ || !model_->backend_gpu_) {
        return;
    }
    if (!ggml_backend_is_cuda(model_->backend_gpu_)) {
        return;
    }
    for (const auto& region : pinned_host_regions_) {
        if (region.first == ptr) {
            return;
        }
    }
    if (ggml_backend_cuda_register_host_buffer(ptr, size)) {
        pinned_host_regions_.emplace_back(ptr, size);
    }
#else
    (void)ptr;
    (void)size;
#endif
}

void Qwen35moeForwardPass::unpin_host_region(void* ptr) const {
#ifdef QWEN35MOE_USE_CUDA
    for (auto it = pinned_host_regions_.begin(); it != pinned_host_regions_.end(); ++it) {
        if (it->first == ptr) {
            ggml_backend_cuda_unregister_host_buffer(ptr);
            pinned_host_regions_.erase(it);
            return;
        }
    }
#else
    (void)ptr;
#endif
}

void Qwen35moeForwardPass::unpin_all_host_regions() {
#ifdef QWEN35MOE_USE_CUDA
    for (auto& region : pinned_host_regions_) {
        ggml_backend_cuda_unregister_host_buffer(region.first);
    }
#endif
    pinned_host_regions_.clear();
    handoff_pinned_.clear();
    output_pinned_.clear();
}

void Qwen35moeForwardPass::read_segment_hidden_out(
    const char* where,
    const LayerSegment* segment,
    ggml_cgraph* gf,
    std::vector<uint8_t>& hidden_data,
    ggml_type& hidden_type
) {
    ggml_tensor* hidden_out = ggml_graph_get_tensor(gf, "segment_hidden_out");
    if (!hidden_out) {
        throw std::runtime_error(std::string(where) + ": missing segment_hidden_out");
    }
    maybe_log_segment_tensor("segment_hidden_out", segment, hidden_out);
    hidden_type = hidden_out->type;
    const size_t hidden_bytes = static_cast<size_t>(ggml_nbytes(hidden_out));
    if (hidden_bytes == 0) {
        throw std::runtime_error(std::string(where) + ": segment_hidden_out has zero bytes");
    }
    hidden_data.resize(hidden_bytes);
    ggml_backend_tensor_get(hidden_out, hidden_data.data(), 0, hidden_data.size());
}

void Qwen35moeForwardPass::prepare_first_segment_hidden_from_token(
    const char* where,
    int32_t token,
    const LayerSegment& first_segment,
    std::vector<uint8_t>& hidden_data,
    ggml_type& hidden_type
) {
    ggml_tensor* token_embedding = model_->get_token_embedding_weight();
    if (!token_embedding) {
        throw std::runtime_error(std::string(where) + ": token embedding weight missing");
    }

    reset_context();
    ggml_cgraph* gf = new_graph();
    ggml_tensor* tok_t = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, 1);
    ggml_set_input(tok_t);
    set_tensor_name(tok_t, "tokens");
    ggml_build_forward_expand(gf, tok_t);

    ggml_tensor* hidden_out = ggml_get_rows(ctx_, token_embedding, tok_t);
    set_tensor_name(hidden_out, "segment_hidden_out");
    ggml_build_forward_expand(gf, hidden_out);

    ggml_backend_t embed_backend = find_backend_for_tensor(token_embedding);
    if (!embed_backend) {
        throw std::runtime_error(std::string(where) + ": no backend available for token embedding");
    }

    ggml_gallocr_t embed_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(embed_backend));
    if (!ggml_gallocr_alloc_graph(embed_alloc, gf)) {
        ggml_gallocr_free(embed_alloc);
        throw std::runtime_error(std::string(where) + ": token embedding graph alloc failed");
    }

    maybe_log_segment_tensor("tokens", &first_segment, tok_t);
    ggml_backend_tensor_set(tok_t, &token, 0, sizeof(token));
    ggml_backend_graph_compute(embed_backend, gf);
    read_segment_hidden_out(where, &first_segment, gf, hidden_data, hidden_type);
    ggml_gallocr_free(embed_alloc);

    if (dev_check_enabled_) {
        std::fprintf(
            stderr,
            "[dev-check] segmented first-segment embedding bridge token_backend=%s segment_backend=%s\n",
            ggml_backend_name(embed_backend),
            first_segment.backend ? ggml_backend_name(first_segment.backend) : "n/a"
        );
    }
}

void Qwen35moeForwardPass::prepare_first_segment_hidden_from_tokens(
    const char* where,
    const std::vector<int32_t>& tokens,
    const LayerSegment& first_segment,
    std::vector<uint8_t>& hidden_data,
    ggml_type& hidden_type
) {
    if (tokens.empty()) {
        throw std::runtime_error(std::string(where) + ": empty token input for embedding bridge");
    }
    ggml_tensor* token_embedding = model_->get_token_embedding_weight();
    if (!token_embedding) {
        throw std::runtime_error(std::string(where) + ": token embedding weight missing");
    }

    reset_context();
    ggml_cgraph* gf = new_graph();
    ggml_tensor* hidden_out = embedding(gf, tokens);
    set_tensor_name(hidden_out, "segment_hidden_out");
    ggml_build_forward_expand(gf, hidden_out);

    ggml_backend_t embed_backend = find_backend_for_tensor(token_embedding);
    if (!embed_backend) {
        throw std::runtime_error(std::string(where) + ": no backend available for token embedding");
    }

    ggml_gallocr_t embed_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(embed_backend));
    if (!ggml_gallocr_alloc_graph(embed_alloc, gf)) {
        ggml_gallocr_free(embed_alloc);
        throw std::runtime_error(std::string(where) + ": token embedding graph alloc failed");
    }

    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) {
        ggml_gallocr_free(embed_alloc);
        throw std::runtime_error(std::string(where) + ": tokens tensor missing from embedding graph");
    }
    maybe_log_segment_tensor("tokens", &first_segment, tok_t);
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, tokens.size() * sizeof(int32_t));
    ggml_backend_graph_compute(embed_backend, gf);
    read_segment_hidden_out(where, &first_segment, gf, hidden_data, hidden_type);
    ggml_gallocr_free(embed_alloc);

    if (dev_check_enabled_) {
        std::fprintf(
            stderr,
            "[dev-check] segmented prefill embedding bridge n_tokens=%zu token_backend=%s segment_backend=%s\n",
            tokens.size(),
            ggml_backend_name(embed_backend),
            first_segment.backend ? ggml_backend_name(first_segment.backend) : "n/a"
        );
    }
}

void Qwen35moeForwardPass::extract_last_token_hidden(
    const std::vector<uint8_t>& full_hidden,
    ggml_type hidden_type,
    int64_t n_embd,
    uint32_t n_tok,
    std::vector<uint8_t>& last_token_hidden
) const {
    if (n_tok == 0) {
        throw std::runtime_error("Qwen35moeForwardPass::extract_last_token_hidden: n_tok is zero");
    }
    const size_t row_bytes = ggml_row_size(hidden_type, n_embd);
    const size_t expected_bytes = row_bytes * static_cast<size_t>(n_tok);
    if (full_hidden.size() != expected_bytes) {
        throw std::runtime_error("Qwen35moeForwardPass::extract_last_token_hidden: hidden byte size mismatch");
    }
    last_token_hidden.resize(row_bytes);
    const size_t offset = static_cast<size_t>(n_tok - 1) * row_bytes;
    std::memcpy(last_token_hidden.data(), full_hidden.data() + offset, row_bytes);
}

/**
 * @brief 构造函数：初始化 Qwen35moeForwardPass 对象
 */
Qwen35moeForwardPass::Qwen35moeForwardPass() {
}

/**
 * @brief 析构函数：释放 ggml 上下文资源
 */
Qwen35moeForwardPass::~Qwen35moeForwardPass() {
    clear_segmented_decode_cache();
    if (ctx_) {
        ggml_free(ctx_);
    }
}

/**
 * @brief 初始化前向传播器
 * 
 * @param context_len 上下文长度
 * @param max_batch_size 最大批处理大小
 * @param model 模型对象指针
 * @return 0 表示成功
 * 
 * 主要初始化工作：
 * 1. 初始化 ggml 上下文
 * 2. 构建层映射表（区分 Full Attention 层和 DeltaNet 层）
 * 3. 创建 KV 缓存（用于 Full Attention 层）
 * 4. 创建 DeltaNet 状态（用于 DeltaNet 层）
 */
int Qwen35moeForwardPass::init(const uint32_t context_len, const uint32_t max_batch_size, std::shared_ptr<Qwen35moeModel> model,
    uint32_t n_batch, uint32_t n_ubatch, bool enable_paged_kv, uint32_t paged_kv_block_size) {
    model_ = model;
    use_flash_attention_ = env_flag_enabled("QWEN35MOE_FLASH_ATTN");
    decode_graph_diag_enabled_ = env_flag_enabled("QWEN35MOE_DECODE_GRAPH_DIAG");
    decode_graph_diag_interval_ = env_u32_or_default("QWEN35MOE_DECODE_GRAPH_DIAG_INTERVAL", 256);
    segmented_decode_cache_enabled_ = !env_flag_enabled("QWEN35MOE_SEGMENT_CACHE_OFF");
    // Mixed-mode batched decode defaults to per-token segmented cache (fast in
    // AUTO with --parallel). Opt-in scheduler batched:
    // QWEN35MOE_MIXED_BATCHED_SCHEDULER=1. Legacy opt-in sequential flag still
    // honored when set explicitly.
    mixed_batched_sequential_enabled_ = env_flag_enabled("QWEN35MOE_MIXED_BATCHED_SEQUENTIAL");
    if (!mixed_batched_sequential_enabled_) {
        if (const char* legacy = std::getenv("QWEN35MOE_MIXED_BATCHED_EAGER");
            legacy && legacy[0] == '0') {
            mixed_batched_sequential_enabled_ = true;
        }
    }
    if (model_->is_mixed_mode()) {
        if (!env_flag_enabled("QWEN35MOE_MIXED_BATCHED_SCHEDULER")) {
            mixed_batched_sequential_enabled_ = true;
        } else if (env_flag_enabled("QWEN35MOE_MIXED_BATCHED_SEQUENTIAL")) {
            mixed_batched_sequential_enabled_ = true;
        } else {
            mixed_batched_sequential_enabled_ = false;
        }
    }
    // Default: unified scheduler prefill in AUTO mode (fast when weights are not
    // split across backends). Opt-out: QWEN35MOE_MIXED_PREFILL_SEGMENTED=1.
    mixed_prefill_scheduler_enabled_ = !env_flag_enabled("QWEN35MOE_MIXED_PREFILL_SEGMENTED");
    if (env_flag_enabled("QWEN35MOE_MIXED_PREFILL_SCHEDULER")) {
        mixed_prefill_scheduler_enabled_ = true;
    }
    if (const char* legacy = std::getenv("QWEN35MOE_MIXED_PREFILL_SCHEDULER");
        legacy && legacy[0] == '0') {
        mixed_prefill_scheduler_enabled_ = false;
    }
    if (model_->is_mixed_mode() &&
        (model_->has_any_split_device_layers() || model_->uses_tensor_overrides())) {
        if (mixed_prefill_scheduler_enabled_) {
            std::fprintf(stderr,
                "[override-tensor] split-layer weights detected: forcing segmented prefill "
                "(unified scheduler would OOM).\n");
        }
        mixed_prefill_scheduler_enabled_ = false;
        if (segmented_decode_cache_enabled_ && model_->has_any_split_device_layers()) {
            std::fprintf(stderr,
                "[override-tensor] split-layer decode: async premoe→MoE cross-device pipeline "
                "with per-layer graph cache (env: QWEN35MOE_SEGMENT_CACHE_OFF to disable)\n");
        }
    }
    mixed_scheduler_decode_enabled_ = env_flag_enabled("QWEN35MOE_MIXED_SCHEDULER_DECODE");
    if (env_flag_enabled("QWEN35MOE_MIXED_SCHEDULER_DECODE_OFF")) {
        mixed_scheduler_decode_enabled_ = false;
    }
    if (model_->is_mixed_mode() && model_->has_any_split_device_layers()) {
        if (mixed_scheduler_decode_enabled_) {
            std::fprintf(stderr,
                "[override-tensor] split-layer weights detected: ignoring QWEN35MOE_MIXED_SCHEDULER_DECODE "
                "(unified scheduler decode would OOM staging routed experts). "
                "Using split-layer premoe→MoE pipeline instead.\n");
        }
        mixed_scheduler_decode_enabled_ = false;
    }
    dev_check_enabled_ = env_flag_enabled("QWEN35MOE_DEV_CHECK");
    paged_kv_enabled_ = enable_paged_kv || env_flag_enabled("QWEN35MOE_PAGED_KV");
    paged_fused_decode_enabled_ = !env_flag_enabled("QWEN35MOE_PAGED_FUSED_ATTN_OFF");
    paged_fused_diag_enabled_ = env_flag_enabled("QWEN35MOE_PAGED_FUSED_DIAG") || decode_graph_diag_enabled_ || dev_check_enabled_;
    const uint32_t paged_block_size = std::max<uint32_t>(1u, env_u32_or_default("QWEN35MOE_PAGED_BLOCK_SIZE", paged_kv_block_size));

    // ── Mixed-mode 下的 paged KV 处理 ─────────────────────────────────────────
    //
    // 当前 paged KV 与 fused flash-attention decode 是一对组合优化：
    //   - simple_kv_cache 走 paged 布局：所有 K/V 读写都得过 block 间接寻址。
    //   - attention decode 走 ggml_flash_attn_ext fused kernel：一次性吃掉
    //     QK^T、softmax、@V 全流程，把 paged 间接寻址开销分摊掉。
    //
    // 但 fused FA decode kernel 仅在 CUDA backend 上实现；mixed (AUTO_MODE)
    // 下部分 attention 层落在 CPU backend，那部分必然走非 fused fallback。
    // can_use_paged_fused_decode() 里已经显式拒绝 mixed（见下方逻辑），
    // 这意味着在 mixed 下我们【保留了 paged 间接寻址成本，却拿不到 fused
    // FA 的收益】，几乎注定比关掉 --paged-kv 还慢。
    //
    // 因此默认行为：mixed + --paged-kv → 自动禁用 paged_kv_enabled_，
    // 并打印一条说明。逃生口 QWEN35MOE_PAGED_KV_FORCE_MIXED=1 允许强行
    // 启用（仅用于做 A/B 性能对比，正常生产请关掉）。
    //
    // 当真正想把 paged + fused FA 推广到 mixed，需要：
    //   (1) 让 ggml-cpu 也实现 ggml_flash_attn_ext（或在 CPU 层走非 paged
    //       fallback 而 GPU 层走 fused）；
    //   (2) kv_cache_simple paged 布局的 gather/scatter 支持跨 backend 复制
    //       （目前 paged block 是设备本地的）。
    // 这是较大的工作量，本次只做 "明确禁用 + 诊断信息" 的兜底。
    const bool paged_kv_requested = paged_kv_enabled_;
    const bool paged_kv_force_mixed = env_flag_enabled("QWEN35MOE_PAGED_KV_FORCE_MIXED");
    if (paged_kv_requested && model_->is_mixed_mode()) {
        if (paged_kv_force_mixed) {
            std::fprintf(stderr,
                "[paged-kv] WARNING: force-enabled under AUTO_MODE (mixed GPU/CPU) via "
                "QWEN35MOE_PAGED_KV_FORCE_MIXED=1. Fused flash-attention decode is NOT "
                "supported in mixed mode and will fall back to the standard attention "
                "path; KV storage will still pay the paged-block indirection cost, "
                "which is typically SLOWER than running without --paged-kv. Use only "
                "for A/B benchmarking.\n");
        } else {
            std::fprintf(stderr,
                "[paged-kv] disabled automatically: model is running in AUTO_MODE "
                "(mixed GPU/CPU). The fused flash-attention decode path requires all "
                "attention layers on the CUDA backend; under mixed mode that kernel "
                "always falls back, so paged KV layout would add block-indirection "
                "cost without delivering the fused-attention speedup. To override for "
                "A/B testing, set QWEN35MOE_PAGED_KV_FORCE_MIXED=1.\n");
            paged_kv_enabled_ = false;
        }
    }
    auto& m = model_->meta_->qwen35moe;

    context_len_ = context_len;
    max_batch_size_ = max_batch_size;
    // Single-slot decode without CUDA fused FA only pays paged block indirection
    // (kv_write_phys, gather indices, prefix growth) with no fused-attn upside.
    // Keep paged KV for parallel>1 (shared block pool). Override: QWEN35MOE_PAGED_KV_FORCE=1.
    const bool paged_kv_force = env_flag_enabled("QWEN35MOE_PAGED_KV_FORCE");
    if (paged_kv_enabled_ && !paged_kv_force && max_batch_size_ <= 1) {
        const char* fused_reason = nullptr;
        if (!can_use_paged_fused_decode(&fused_reason)) {
            std::fprintf(stderr,
                "[paged-kv] disabled automatically: single-sequence decode without fused "
                "flash-attention (%s). Paged layout adds per-token block addressing without "
                "the fused FA kernel that amortizes it. Use dense KV (omit --paged-kv), or "
                "enable --flash-attn on CUDA, or set parallel>1 for multi-slot block pooling. "
                "Set QWEN35MOE_PAGED_KV_FORCE=1 to override for A/B testing.\n",
                fused_reason ? fused_reason : "unknown");
            paged_kv_enabled_ = false;
        }
    }
    cached_batched_decode_last_mask_positions_.assign(max_batch_size_, -1);
    n_batch_tokens_ = n_batch > 0 ? std::min(n_batch, context_len_) : context_len_;
    n_ubatch_tokens_ = n_ubatch > 0 ? std::min(n_ubatch, n_batch_tokens_) : n_batch_tokens_;

    // One-shot info line about the active decode bucket policy. Helps users
    // correlate `[PERF][decode-graph] recapture=...` numbers with the policy
    // actually in effect for the run, and surfaces the env knobs they can
    // turn for further tuning.
    {
        const bool lock_ctx = env_flag_enabled("QWEN35MOE_DECODE_KV_LOCK_CTX");
        const bool coarse = env_flag_enabled("QWEN35MOE_DECODE_KV_COARSE");
        const uint32_t min_bucket = env_u32_or_default(
            "QWEN35MOE_DECODE_KV_MIN_BUCKET", kDefaultDecodeMinBucket);
        const uint32_t large_step = env_u32_or_default(
            "QWEN35MOE_DECODE_KV_GROW_STEP", kDefaultDecodeLargeStep);
        const char* policy = lock_ctx ? "lock-ctx"
                            : coarse  ? "coarse"
                                      : "geometric";
        std::fprintf(stderr,
            "[decode-graph] bucket policy=%s ctx_len=%u min_bucket=%u grow_step=%u "
            "(env: QWEN35MOE_DECODE_KV_LOCK_CTX, QWEN35MOE_DECODE_KV_COARSE, "
            "QWEN35MOE_DECODE_KV_MIN_BUCKET, QWEN35MOE_DECODE_KV_GROW_STEP)\n",
            policy, context_len_, min_bucket, large_step);
    }

    // 预分配持久化缓冲区用于存储计算图元数据
    ctx_buffer_.resize(FP_GRAPH_SIZE_METADATA);

    struct ggml_init_params params = {
        .mem_size   = ctx_buffer_.size(),
        .mem_buffer = ctx_buffer_.data(),
        .no_alloc   = true,  // 使用预分配缓冲区，不允许自动分配
    };
    ctx_ = ggml_init(params);

    // 初始化层映射表：将物理层索引映射到 KV/DeltaNet 层索引
    const uint32_t trunk_layers = model_->trunk_layer_count();
    kv_layer_map_.assign(trunk_layers, -1);  // Full Attention 层映射
    dn_layer_map_.assign(trunk_layers, -1);  // DeltaNet 层映射
    int kv_idx = 0, dn_idx = 0;
    for (uint32_t il = 0; il < trunk_layers; ++il) {
        if (is_full_attention_layer(il))
            kv_layer_map_[il] = kv_idx++;  // 标记为 Full Attention 层
        else
            dn_layer_map_[il] = dn_idx++;  // 标记为 DeltaNet 层
    }

    // 创建 KV 缓存：用于存储注意力机制的 Key/Value 张量
    // Qwen3.5-MoE 有 10 个 Full Attention 层
    ggml_backend_t cache_backend = model_->get_curr_backend();
    std::vector<ggml_backend_t> kv_layer_backends(static_cast<size_t>(kv_idx), cache_backend);
    for (uint32_t il = 0; il < trunk_layers; ++il) {
        const int32_t mapped = kv_layer_map_[il];
        if (mapped >= 0) {
            kv_layer_backends[static_cast<size_t>(mapped)] = model_->get_layer_backend(il);
        }
    }
    if (model_->is_mixed_mode()) {
        std::fprintf(stderr,
            "[dev-mode=auto] mixed GPU/CPU detected (device_map_hash=0x%llx): "
            "prefill=%s decode=%s batched_decode=%s segmented_decode_cache=%s "
            "(env: QWEN35MOE_MIXED_PREFILL_SEGMENTED, QWEN35MOE_MIXED_SCHEDULER_DECODE, "
            "QWEN35MOE_MIXED_BATCHED_SCHEDULER, QWEN35MOE_SEGMENT_CACHE_OFF)\n",
            static_cast<unsigned long long>(model_->compute_device_map_hash()),
            mixed_prefill_scheduler_enabled_ ? "scheduler" : "segmented",
            mixed_scheduler_decode_enabled_ ? "scheduler-opt-in" : "segmented",
            mixed_batched_sequential_enabled_ ? "per-token-segmented" : "scheduler-batched",
            segmented_decode_cache_enabled_ ? "per-slot" : "disabled");
    }

    const int n_kv_layers = kv_idx;  // Full Attention 层数量
    const uint32_t n_embd_k = static_cast<uint32_t>(m.head_count_kv * m.key_length);  // K 维度
    const uint32_t n_embd_v = static_cast<uint32_t>(m.head_count_kv * m.value_length); // V 维度
    kv_cache_ = std::make_unique<simple_kv_cache>(
        static_cast<uint32_t>(n_kv_layers),  // KV 层数
        context_len_,                         // 上下文长度
        max_batch_size_,                      // 最大批大小
        n_embd_k, n_embd_v,                   // K/V 维度
        GGML_TYPE_F16, GGML_TYPE_F16,         // 数据类型
        cache_backend,                        // 默认后端设备
        kv_layer_backends,                    // 每层设备映射
        PagedKVConfig{paged_kv_enabled_, paged_block_size, dev_check_enabled_}
    );
    if (paged_kv_enabled_) {
        std::fprintf(stderr,
            "[paged-kv] enabled block_size=%u total_blocks=%u\n",
            paged_block_size,
            kv_cache_->paged_total_blocks());
    }

    rebuild_layer_segments();
    if (model_->is_mixed_mode()) {
        const uint32_t interval = model_->meta_->qwen35moe.full_attention_interval;
        bool any_split_in_block = false;
        for (size_t i = 0; i < layer_segments_.size(); ++i) {
            const auto& seg = layer_segments_[i];
            const bool l0_aligned = (interval == 0) || (seg.l0 % interval == 0);
            const bool l1_aligned = (interval == 0) ||
                                    (seg.l1 % interval == 0) ||
                                    seg.l1 == trunk_layers;
            std::fprintf(stderr,
                "[dev-mode=auto] segment[%zu]=[%u,%u) backend=%s%s\n",
                i,
                seg.l0,
                seg.l1,
                ggml_backend_name(seg.backend),
                (interval > 0 && (!l0_aligned || !l1_aligned))
                    ? " (boundary mid-attention-block)" : "");
            if (interval > 0 && (!l0_aligned || !l1_aligned)) {
                any_split_in_block = true;
            }
        }
        std::fprintf(stderr,
            "[dev-mode=auto] segments=%zu attention_interval=%u%s\n",
            layer_segments_.size(),
            interval,
            any_split_in_block
                ? " — boundary not block-aligned; consider unset QWEN35MOE_AUTO_BOUNDARY_DISABLE or adjust --gpu-layer / --n-gpu-layers"
                : "");
    }

    // 创建 DeltaNet 状态：用于存储 DeltaNet 层的循环状态
    // Qwen3.5-MoE 有 30 个 DeltaNet 层
    const int n_dn_layers = dn_idx;
    std::vector<ggml_backend_t> dn_layer_backends(static_cast<size_t>(n_dn_layers), cache_backend);
    for (uint32_t il = 0; il < trunk_layers; ++il) {
        const int32_t mapped = dn_layer_map_[il];
        if (mapped >= 0) {
            dn_layer_backends[static_cast<size_t>(mapped)] = model_->get_layer_backend(il);
        }
    }
    const uint32_t d_inner       = m.inner_size;     // 内部维度 = 4096
    const uint32_t num_v_heads   = m.time_step_rank; // V 头数量 = 32
    const uint32_t num_k_heads   = m.group_count;    // K 头数量 = 16
    const uint32_t head_v_dim    = d_inner / num_v_heads; // 每个 V 头维度 = 128
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.state_size; // 卷积通道数 = 8192

    DeltaNetStateParams dn_state_hp {
        static_cast<uint32_t>(n_dn_layers),
        max_batch_size_,
        head_v_dim,
        m.state_size,  // head_k_dim = 128
        num_v_heads,
        conv_channels,
        m.conv_kernel, // 卷积核大小 = 4
        cache_backend,
        dn_layer_backends
    };
    dn_state_ = std::make_unique<DeltaNetState>(dn_state_hp);

    // Allocate MTP draft-head resources if --mtp was already requested before
    // init(); otherwise configure_mtp() will do it lazily.
    init_mtp();

    return 0;
}

void Qwen35moeForwardPass::set_flash_attention_enabled(bool enabled) {
    use_flash_attention_ = enabled;
    clear_segmented_decode_cache();
    clear_cached_batched_decode_graph();
    clear_cached_prefill_graph();
}

void Qwen35moeForwardPass::configure_device_sampling(int top_k, float temperature) {
    sampling_top_k_ = top_k;
    sampling_temperature_ = temperature;
    clear_segmented_decode_cache();
    clear_cached_batched_decode_graph();
    clear_cached_prefill_graph();
}

/**
 * @brief 重置 ggml 上下文
 * 
 * 在每次构建新的计算图前调用，确保上下文干净
 */
void Qwen35moeForwardPass::reset_context() {
    if (ctx_) {
        ggml_free(ctx_);
    }
    cached_decode_graph_ = nullptr;
    cached_decode_graph_allocated_ = false;
    cached_decode_kv_capacity_ = 0;
    cached_decode_kv_write_phys_tensor_ = nullptr;
    cached_decode_last_mask_pos_ = -1;
    cached_decode_mask_tensors_.clear();
    cached_decode_tokens_tensor_ = nullptr;
    cached_decode_pos_tensor_ = nullptr;
    cached_decode_mask_f32_.clear();
    cached_decode_mask_f16_.clear();
    cached_decode_signature_valid_ = false;
    clear_cached_batched_decode_graph();
    clear_cached_prefill_graph();
    struct ggml_init_params params = {
        .mem_size   = ctx_buffer_.size(),
        .mem_buffer = ctx_buffer_.data(),
        .no_alloc   = true, 
    };
    ctx_ = ggml_init(params);
}

void Qwen35moeForwardPass::clear_cached_prefill_graph() {
    cached_prefill_graph_ = nullptr;
    cached_prefill_graph_allocated_ = false;
    cached_prefill_tokens_tensor_ = nullptr;
    cached_prefill_pos_tensor_ = nullptr;
    cached_prefill_mask_tensors_.clear();
    cached_prefill_signature_valid_ = false;
}

void Qwen35moeForwardPass::clear_cached_batched_decode_graph() {
    cached_batched_decode_graph_ = nullptr;
    cached_batched_decode_graph_allocated_ = false;
    cached_batched_decode_tokens_tensor_ = nullptr;
    cached_batched_decode_pos_tensor_ = nullptr;
    cached_batched_decode_kq_mask_tensor_ = nullptr;
    cached_batched_decode_gather_indices_tensor_ = nullptr;
    cached_batched_decode_dn_slot_idx_tensor_ = nullptr;
    cached_batched_decode_signature_valid_ = false;
    cached_batched_decode_mask_f32_.clear();
    cached_batched_decode_mask_f16_.clear();
    cached_batched_decode_last_mask_positions_.assign(max_batch_size_, -1);
}

void Qwen35moeForwardPass::reset_sequence(uint32_t slot_idx) {
    if (slot_idx >= max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::reset_sequence: slot_idx out of range");
    }
    if (kv_cache_) {
        kv_cache_->clear_slot(slot_idx);
    }
    if (dn_state_) {
        dn_state_->clear_slot(slot_idx);
    }
    if (mtp_kv_) {
        mtp_kv_->clear_slot(slot_idx);
    }
    if (dn_shadow_) {
        dn_shadow_->clear_slot(slot_idx);
    }
    if (slot_idx < snapkv_seq_pos_.size()) {
        snapkv_seq_pos_[slot_idx] = 0;
    }
}

void Qwen35moeForwardPass::prepare_cached_decode_graph(ggml_backend_sched_t scheduler, uint32_t slot_idx, uint32_t kv_capacity) {
    if (scheduler == nullptr) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: scheduler is null");
    }
    if (context_len_ == 0) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: context length is zero");
    }
    if (slot_idx >= max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: slot_idx out of range");
    }
    if (kv_capacity == 0) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: kv_capacity cannot be zero");
    }
    if (kv_capacity > context_len_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: kv_capacity exceeds context_len_");
    }

    DecodeGraphSignature signature;
    signature.slot_idx = slot_idx;
    signature.kv_capacity = kv_capacity;
    signature.context_len = context_len_;
    signature.n_batch_tokens = n_batch_tokens_;
    signature.n_ubatch_tokens = n_ubatch_tokens_;
    signature.use_flash_attention = use_flash_attention_;
    signature.paged_fused_decode = paged_fused_decode_active();
    signature.is_mixed_mode = model_->is_mixed_mode();
    signature.device_map_hash = model_->compute_device_map_hash();

    std::vector<int32_t> dummy_tokens(1, 0);
    cached_decode_graph_ = build_prefill_graph(dummy_tokens, 0, slot_idx, kv_capacity, true);
    cached_decode_slot_ = slot_idx;
    cached_decode_kv_capacity_ = kv_capacity;
    cached_decode_last_mask_pos_ = -1;
    cached_decode_kv_write_phys_tensor_ = ggml_graph_get_tensor(cached_decode_graph_, "kv_write_phys");
    cached_decode_tokens_tensor_ = ggml_graph_get_tensor(cached_decode_graph_, "tokens");
    cached_decode_pos_tensor_ = ggml_graph_get_tensor(cached_decode_graph_, "inp_pos");
    if (!cached_decode_tokens_tensor_ || !cached_decode_pos_tensor_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: missing cached decode input tensors");
    }
    cached_decode_mask_tensors_.clear();
    if (ggml_tensor* shared_kq_mask = ggml_graph_get_tensor(cached_decode_graph_, "kq_mask_shared")) {
        cached_decode_mask_tensors_.push_back(shared_kq_mask);
    } else {
        for (uint32_t il = 0; il < model_->trunk_layer_count(); ++il) {
            if (!is_full_attention_layer(il)) {
                continue;
            }
            char name[32];
            std::snprintf(name, sizeof(name), "kq_mask.%u", il);
            ggml_tensor* kq_mask = ggml_graph_get_tensor(cached_decode_graph_, name);
            if (kq_mask) {
                cached_decode_mask_tensors_.push_back(kq_mask);
            }
        }
    }

    ggml_backend_sched_reset(scheduler);
    decode_graph_scheduler_reset_count_++;
    if (!ggml_backend_sched_alloc_graph(scheduler, cached_decode_graph_)) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: failed to allocate graph");
    }
    cached_decode_graph_allocated_ = true;
    cached_decode_signature_ = signature;
    cached_decode_signature_valid_ = true;
    cached_decode_mask_f32_.clear();
    cached_decode_mask_f16_.clear();
    cached_decode_last_mask_pos_ = -1;
    decode_graph_bucket_usage_[kv_capacity]++;
    decode_graph_recapture_count_++;
}

bool Qwen35moeForwardPass::is_cached_decode_graph_compatible(const DecodeGraphSignature& signature, uint32_t required_kv) const {
    return cached_decode_graph_ != nullptr &&
           cached_decode_graph_allocated_ &&
           cached_decode_signature_valid_ &&
           cached_decode_signature_.slot_idx == signature.slot_idx &&
           cached_decode_signature_.context_len == signature.context_len &&
           cached_decode_signature_.n_batch_tokens == signature.n_batch_tokens &&
           cached_decode_signature_.n_ubatch_tokens == signature.n_ubatch_tokens &&
           cached_decode_signature_.use_flash_attention == signature.use_flash_attention &&
           cached_decode_signature_.paged_fused_decode == signature.paged_fused_decode &&
           cached_decode_signature_.is_mixed_mode == signature.is_mixed_mode &&
           cached_decode_signature_.device_map_hash == signature.device_map_hash &&
           required_kv <= cached_decode_signature_.kv_capacity;
}

void Qwen35moeForwardPass::ensure_cached_decode_graph(ggml_backend_sched_t scheduler, const DecodeGraphSignature& signature, uint32_t required_kv) {
    decode_graph_lookup_count_++;
    if (is_cached_decode_graph_compatible(signature, required_kv)) {
        decode_graph_hit_count_++;
        return;
    }

    decode_graph_miss_count_++;
    try {
        prepare_cached_decode_graph(scheduler, signature.slot_idx, signature.kv_capacity);
    } catch (const std::exception& ex) {
        std::fprintf(
            stderr,
            "[PERF][decode-graph] recapture_failed requested_bucket=%u err=%s\n",
            signature.kv_capacity,
            ex.what()
        );
        // Pick a smaller, *bucket-aligned* fallback capacity so the next
        // decode step does not immediately miss again on +1 token. Halve the
        // requested bucket then re-bucketize; if that still equals the
        // original capacity (no smaller bucket exists), rethrow.
        const uint32_t halved = std::max<uint32_t>(required_kv, signature.kv_capacity / 2);
        const uint32_t fallback_capacity = decode_cache_bucket_capacity(halved);
        if (fallback_capacity == 0 || fallback_capacity >= signature.kv_capacity) {
            throw;
        }
        decode_graph_fallback_count_++;
        prepare_cached_decode_graph(scheduler, signature.slot_idx, fallback_capacity);
    }
}

bool Qwen35moeForwardPass::is_cached_batched_decode_graph_compatible(
    const DecodeGraphSignature& signature,
    uint32_t required_kv
) const {
    return cached_batched_decode_graph_ != nullptr &&
           cached_batched_decode_graph_allocated_ &&
           cached_batched_decode_signature_valid_ &&
           cached_batched_decode_signature_.n_decode_batch == signature.n_decode_batch &&
           cached_batched_decode_signature_.slots_signature == signature.slots_signature &&
           cached_batched_decode_signature_.context_len == signature.context_len &&
           cached_batched_decode_signature_.n_batch_tokens == signature.n_batch_tokens &&
           cached_batched_decode_signature_.n_ubatch_tokens == signature.n_ubatch_tokens &&
           cached_batched_decode_signature_.use_flash_attention == signature.use_flash_attention &&
           cached_batched_decode_signature_.paged_fused_decode == signature.paged_fused_decode &&
           cached_batched_decode_signature_.device_map_hash == signature.device_map_hash &&
           required_kv <= cached_batched_decode_signature_.kv_capacity;
}

void Qwen35moeForwardPass::prepare_cached_batched_decode_graph(
    ggml_backend_sched_t scheduler,
    const DecodeGraphSignature& signature,
    const std::vector<uint32_t>& slots
) {
    if (scheduler == nullptr) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_batched_decode_graph: scheduler is null");
    }
    const uint32_t n_batch = signature.n_decode_batch;
    if (n_batch == 0 || n_batch > max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_batched_decode_graph: invalid batch size");
    }
    if (slots.size() != n_batch) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_batched_decode_graph: slots size mismatch");
    }
    const uint32_t kv_capacity = signature.kv_capacity;
    if (kv_capacity == 0 || kv_capacity > context_len_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_batched_decode_graph: invalid kv capacity");
    }

    std::vector<int32_t> dummy_tokens(n_batch, 0);
    std::vector<int32_t> dummy_positions(n_batch, static_cast<int32_t>(kv_capacity - 1));
    cached_batched_decode_graph_ = build_decoding_graph(
        dummy_tokens,
        slots,
        dummy_positions,
        kv_capacity
    );

    cached_batched_decode_tokens_tensor_ = ggml_graph_get_tensor(cached_batched_decode_graph_, "tokens");
    cached_batched_decode_pos_tensor_ = ggml_graph_get_tensor(cached_batched_decode_graph_, "inp_pos");
    cached_batched_decode_kq_mask_tensor_ = ggml_graph_get_tensor(cached_batched_decode_graph_, "kq_mask_b");
    cached_batched_decode_gather_indices_tensor_ = ggml_graph_get_tensor(cached_batched_decode_graph_, "gather_indices");
    // Optional: only present when graph has DeltaNet layers AND batched-decode
    // path built the shared slot-index input. n_batch == 1 takes the
    // per-slot view fast path (no shared input tensor).
    cached_batched_decode_dn_slot_idx_tensor_ = ggml_graph_get_tensor(cached_batched_decode_graph_, "dn_slot_idx");
    if (!cached_batched_decode_tokens_tensor_ || !cached_batched_decode_pos_tensor_) {
        throw std::runtime_error(
            "Qwen35moeForwardPass::prepare_cached_batched_decode_graph: missing cached batched decode input tensors"
        );
    }

    ggml_backend_sched_reset(scheduler);
    decode_graph_scheduler_reset_count_++;
    if (!ggml_backend_sched_alloc_graph(scheduler, cached_batched_decode_graph_)) {
        throw std::runtime_error(
            "Qwen35moeForwardPass::prepare_cached_batched_decode_graph: failed to allocate graph"
        );
    }

    cached_batched_decode_graph_allocated_ = true;
    cached_batched_decode_signature_ = signature;
    cached_batched_decode_signature_valid_ = true;
    cached_batched_decode_mask_f32_.clear();
    cached_batched_decode_mask_f16_.clear();
    cached_batched_decode_last_mask_positions_.assign(max_batch_size_, -1);
    decode_graph_bucket_usage_[kv_capacity]++;
    decode_graph_recapture_count_++;
}

void Qwen35moeForwardPass::ensure_cached_batched_decode_graph(
    ggml_backend_sched_t scheduler,
    const DecodeGraphSignature& signature,
    const std::vector<uint32_t>& slots,
    uint32_t required_kv
) {
    decode_graph_lookup_count_++;
    if (is_cached_batched_decode_graph_compatible(signature, required_kv)) {
        decode_graph_hit_count_++;
        return;
    }

    decode_graph_miss_count_++;
    try {
        prepare_cached_batched_decode_graph(scheduler, signature, slots);
    } catch (const std::exception& ex) {
        std::fprintf(
            stderr,
            "[PERF][decode-graph-batched] recapture_failed batch=%u bucket=%u err=%s\n",
            signature.n_decode_batch,
            signature.kv_capacity,
            ex.what()
        );
        // Same bucket-aligned shrink-and-retry policy as the single-stream
        // decode cache; see ensure_cached_decode_graph for the rationale.
        const uint32_t halved = std::max<uint32_t>(required_kv, signature.kv_capacity / 2);
        const uint32_t fallback_capacity = decode_cache_bucket_capacity(halved);
        if (fallback_capacity == 0 || fallback_capacity >= signature.kv_capacity) {
            throw;
        }
        decode_graph_fallback_count_++;
        DecodeGraphSignature fallback = signature;
        fallback.kv_capacity = fallback_capacity;
        prepare_cached_batched_decode_graph(scheduler, fallback, slots);
    }
}

void Qwen35moeForwardPass::set_cached_batched_decode_inputs(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions
) {
    if (!cached_batched_decode_graph_) {
        throw std::runtime_error("Qwen35moeForwardPass::set_cached_batched_decode_inputs: graph is null");
    }
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());

    if (cached_batched_decode_tokens_tensor_) {
        ggml_backend_tensor_set(
            cached_batched_decode_tokens_tensor_,
            tokens.data(),
            0,
            n_batch * sizeof(int32_t)
        );
    }
    if (cached_batched_decode_pos_tensor_) {
        ggml_backend_tensor_set(
            cached_batched_decode_pos_tensor_,
            positions.data(),
            0,
            n_batch * sizeof(int32_t)
        );
    }

    ggml_tensor* kq_mask = cached_batched_decode_kq_mask_tensor_;
    if (kq_mask) {
        const uint32_t graph_n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        fill_dynamic_decode_mask_batched(
            cached_batched_decode_mask_f32_,
            graph_n_kv,
            n_batch,
            positions,
            true,
            cached_batched_decode_last_mask_positions_
        );
        upload_decode_mask_tensor(kq_mask, cached_batched_decode_mask_f32_);
    }

    ggml_tensor* gi = cached_batched_decode_gather_indices_tensor_;
    if (gi) {
        const uint32_t n_kv = static_cast<uint32_t>(gi->ne[0]) / n_batch;
        if (paged_kv_enabled_ && kv_cache_) {
            for (size_t b = 0; b < slots.size(); ++b) {
                const int32_t p = positions[b];
                if (p < 0) {
                    continue;
                }
                kv_cache_->ensure_contiguous_kv_prefix(
                    slots[b],
                    static_cast<uint32_t>(p) + 1
                );
            }
        }
        std::vector<int32_t> idx;
        if (kv_cache_) {
            kv_cache_->fill_gather_indices(slots, n_kv, idx);
        } else {
            idx.resize(n_batch * n_kv);
            for (uint32_t b = 0; b < n_batch; ++b) {
                const uint32_t slot = slots[b];
                for (uint32_t j = 0; j < n_kv; ++j) {
                    idx[b * n_kv + j] = static_cast<int32_t>(slot * context_len_ + j);
                }
            }
        }
        ggml_backend_tensor_set(gi, idx.data(), 0, idx.size() * sizeof(int32_t));
    }

    // DeltaNet batched-decode slot indices. Present only when the cached
    // graph has DN layers and this batch uses the batched (n_batch > 1)
    // path. The same int32 vector is shared by all DN layers in the graph.
    if (cached_batched_decode_dn_slot_idx_tensor_) {
        std::vector<int32_t> slot_idx_i32(slots.begin(), slots.end());
        ggml_backend_tensor_set(
            cached_batched_decode_dn_slot_idx_tensor_,
            slot_idx_i32.data(),
            0,
            slot_idx_i32.size() * sizeof(int32_t)
        );
    }
}

void Qwen35moeForwardPass::maybe_log_decode_graph_stats() {
    if (!decode_graph_diag_enabled_) {
        return;
    }
    if (decode_graph_lookup_count_ == 0) {
        return;
    }

    bool interval_reached = false;
    if (decode_graph_diag_interval_ > 0) {
        interval_reached = (decode_graph_lookup_count_ % decode_graph_diag_interval_) == 0;
    }
    const bool has_new_recapture = decode_graph_recapture_count_ > decode_graph_last_logged_recapture_;
    if (!interval_reached && !has_new_recapture) {
        return;
    }

    uint32_t hottest_bucket = 0;
    uint64_t hottest_bucket_count = 0;
    for (const auto& item : decode_graph_bucket_usage_) {
        if (item.second > hottest_bucket_count) {
            hottest_bucket_count = item.second;
            hottest_bucket = item.first;
        }
    }

    std::fprintf(
        stderr,
        "[PERF][decode-graph] lookups=%llu hit=%llu miss=%llu recapture=%llu fallback=%llu sched_reset=%llu active_bucket=%u hot_bucket=%u hot_bucket_uses=%llu\n",
        static_cast<unsigned long long>(decode_graph_lookup_count_),
        static_cast<unsigned long long>(decode_graph_hit_count_),
        static_cast<unsigned long long>(decode_graph_miss_count_),
        static_cast<unsigned long long>(decode_graph_recapture_count_),
        static_cast<unsigned long long>(decode_graph_fallback_count_),
        static_cast<unsigned long long>(decode_graph_scheduler_reset_count_),
        cached_decode_signature_valid_ ? cached_decode_signature_.kv_capacity : 0,
        hottest_bucket,
        static_cast<unsigned long long>(hottest_bucket_count)
    );
    std::fprintf(
        stderr,
        "[PERF][paged-fused] attempt=%llu hit=%llu fallback=%llu avg_decode_us=%llu\n",
        static_cast<unsigned long long>(paged_fused_decode_attempt_count_),
        static_cast<unsigned long long>(paged_fused_decode_hit_count_),
        static_cast<unsigned long long>(paged_fused_decode_fallback_count_),
        static_cast<unsigned long long>(
            paged_fused_decode_compute_count_ == 0
                ? 0
                : (paged_fused_decode_compute_total_us_ / paged_fused_decode_compute_count_))
    );
    decode_graph_last_logged_lookup_ = decode_graph_lookup_count_;
    decode_graph_last_logged_recapture_ = decode_graph_recapture_count_;

    if (prefill_graph_lookup_count_ > 0) {
        std::fprintf(
            stderr,
            "[PERF][prefill-graph] lookups=%llu hit=%llu miss=%llu recapture=%llu fallback=%llu active_n_tokens=%u\n",
            static_cast<unsigned long long>(prefill_graph_lookup_count_),
            static_cast<unsigned long long>(prefill_graph_hit_count_),
            static_cast<unsigned long long>(prefill_graph_miss_count_),
            static_cast<unsigned long long>(prefill_graph_recapture_count_),
            static_cast<unsigned long long>(prefill_graph_fallback_count_),
            cached_prefill_signature_valid_ ? cached_prefill_signature_.n_tokens : 0
        );
    }
    if (prefill_ubatch_micro_step_count_ > 0) {
        std::fprintf(
            stderr,
            "[PERF][prefill-ubatch] chunks=%llu micro_steps=%llu n_batch=%u n_ubatch=%u\n",
            static_cast<unsigned long long>(prefill_ubatch_chunk_count_),
            static_cast<unsigned long long>(prefill_ubatch_micro_step_count_),
            effective_prefill_batch_limit(),
            effective_prefill_ubatch_limit()
        );
    }
}

/**
 * @brief 设置缓存解码图的输入数据（图复用模式）
 * 
 * 该函数用于解码阶段（Decode），在预构建的计算图上更新输入数据，实现图复用。
 * 相比于每次解码都重建图，这种方式可以显著减少 CPU 开销，提高 GPU 利用率。
 * 
 * @param gf 预构建的解码计算图（由 build_cached_decode_graph 创建）
 * @param token 当前要解码的单个 token ID
 * @param pos 当前 token 在序列中的位置（从 0 开始）
 * 
 * @throws std::runtime_error 如果图中缺少必要的输入张量
 * 
 * 实现步骤：
 * 1. 更新 token 输入张量
 * 2. 更新位置输入张量
 * 3. 动态更新注意力掩码（确保因果性）
 */
void Qwen35moeForwardPass::set_cached_decode_inputs(ggml_cgraph* gf, int32_t token, int pos) {
    (void)gf;
    // ==================== 步骤1：更新 Token 输入 ====================
    // 从计算图中获取名为 "tokens" 的输入张量
    ggml_tensor* tok_t = cached_decode_tokens_tensor_;
    if (!tok_t) {
        throw std::runtime_error("qwen35moe: 'tokens' tensor missing from cached decode graph");
    }
    // 将当前 token ID 写入张量（解码阶段每次只处理一个 token）
    ggml_backend_tensor_set(tok_t, &token, 0, sizeof(int32_t));

    // ==================== 步骤2：更新位置输入 ====================
    // 从计算图中获取名为 "inp_pos" 的位置张量
    ggml_tensor* pos_t = cached_decode_pos_tensor_;
    if (!pos_t) {
        throw std::runtime_error("qwen35moe: 'inp_pos' tensor missing from cached decode graph");
    }
    // 将当前位置写入张量
    ggml_backend_tensor_set(pos_t, &pos, 0, sizeof(int32_t));

    // ==================== 步骤3：KV 写入行（paged 布局为物理行索引）====================
    const uint32_t active_kv = static_cast<uint32_t>(pos) + 1;
    if (paged_kv_enabled_ && kv_cache_) {
        kv_cache_->ensure_contiguous_kv_prefix(cached_decode_slot_, active_kv);
        if (cached_decode_kv_write_phys_tensor_) {
            const uint32_t logical_pos = static_cast<uint32_t>(pos);
            if (!kv_cache_->ensure_materialized_logical_pos(cached_decode_slot_, logical_pos)) {
                throw std::runtime_error("qwen35moe: failed to materialize paged KV slot for decode write");
            }
            const uint32_t phys =
                kv_cache_->physical_row_for_contiguous_write(cached_decode_slot_, logical_pos);
            ggml_backend_tensor_set(cached_decode_kv_write_phys_tensor_, &phys, 0, sizeof(int32_t));
        }
    }

    // ==================== 步骤4：动态更新注意力掩码 ====================
    if (cached_decode_mask_tensors_.empty()) {
        throw std::runtime_error("qwen35moe: cached decode masks missing from cached decode graph");
    }

    uint32_t prepared_n_kv = 0;
    for (ggml_tensor* kq_mask : cached_decode_mask_tensors_) {
        const uint32_t graph_n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        if (prepared_n_kv != graph_n_kv) {
            fill_dynamic_decode_mask_1d(
                cached_decode_mask_f32_,
                graph_n_kv,
                active_kv,
                true,
                pos,
                cached_decode_last_mask_pos_
            );
            prepared_n_kv = graph_n_kv;
        }
        upload_decode_mask_tensor(kq_mask, cached_decode_mask_f32_);
    }
}

bool Qwen35moeForwardPass::paged_fused_decode_active() const {
    return can_use_paged_fused_decode(nullptr);
}

bool Qwen35moeForwardPass::can_use_paged_fused_decode(const char** reason) const {
    if (!paged_kv_enabled_) {
        if (reason) *reason = "paged-kv-disabled";
        return false;
    }
    if (!paged_fused_decode_enabled_) {
        if (reason) *reason = "paged-fused-disabled-by-env";
        return false;
    }
    if (!kv_cache_ || !kv_cache_->paged_enabled()) {
        if (reason) *reason = "paged-cache-unavailable";
        return false;
    }
    if (!use_flash_attention_) {
        if (reason) *reason = "flash-attn-disabled";
        return false;
    }
    if (!backend_name_contains_cuda(model_->get_curr_backend())) {
        if (reason) *reason = "cuda-backend-unavailable";
        return false;
    }
    if (model_->is_mixed_mode()) {
        // 第二层兜底：init() 默认会在 mixed + paged-kv 时直接把
        // paged_kv_enabled_ 关掉，因此正常情况下这条分支根本走不到。
        // 只有 QWEN35MOE_PAGED_KV_FORCE_MIXED=1 强行启用 paged-kv 后才
        // 会走到这里，此时 fused FA 必须显式 fallback，否则 mixed 下
        // 部分 attention 层位于 CPU backend，fused FA kernel 不可用。
        if (reason) *reason = "mixed-mode-unsupported";
        return false;
    }
    if (reason) *reason = nullptr;
    return true;
}

void Qwen35moeForwardPass::maybe_log_paged_fused_fallback(const char* reason) {
    if (!reason) {
        return;
    }
    paged_fused_decode_fallback_count_++;
    if (!paged_fused_diag_enabled_) {
        return;
    }
    const bool first = !paged_fused_fallback_warned_;
    const bool changed = paged_fused_last_fallback_reason_ != reason;
    if (first || changed) {
        std::fprintf(stderr,
            "[paged-fused] decode fallback reason=%s (falling back to standard decode path)\n",
            reason);
        paged_fused_last_fallback_reason_ = reason;
        paged_fused_fallback_warned_ = true;
    }
}

void Qwen35moeForwardPass::maybe_log_paged_fused_activation() {
    if (!paged_fused_diag_enabled_ || paged_fused_decode_hit_count_ != 1) {
        return;
    }
    std::fprintf(stderr,
        "[paged-fused] decode fused path active backend=%s flash_attn=1 paged_kv=1\n",
        ggml_backend_name(model_->get_curr_backend()));
}

void Qwen35moeForwardPass::record_paged_fused_decode_timing(uint64_t delta_us) {
    paged_fused_decode_compute_count_++;
    paged_fused_decode_compute_total_us_ += delta_us;
}

bool Qwen35moeForwardPass::ensure_cached_decode_copy_ready(uint32_t slot_idx, uint32_t dst_pos) {
    if (!paged_kv_enabled_ || !kv_cache_) {
        return true;
    }
    if (!kv_cache_->ensure_materialized_logical_pos(slot_idx, dst_pos)) {
        std::fprintf(stderr,
            "[paged-kv] cached decode materialize fallback: slot=%u pos=%u\n",
            slot_idx,
            dst_pos);
        return false;
    }
    return true;
}

bool Qwen35moeForwardPass::commit_cached_decode_step(uint32_t slot_idx, uint32_t dst_pos) {
    (void)dst_pos;
    advance_cache(1, slot_idx);
    maybe_log_decode_graph_stats();
    return true;
}

void Qwen35moeForwardPass::clear_segmented_decode_slot_cache(SegmentedDecodeSlotCache& slot_cache) {
    auto clear_one = [this](SegmentedDecodeGraphCache& cache) {
        unpin_host_region(cache.mask_f32.data());
        unpin_host_region(cache.mask_f16.data());
        if (cache.allocr) {
            ggml_gallocr_free(cache.allocr);
            cache.allocr = nullptr;
        }
        if (cache.ctx) {
            ggml_free(cache.ctx);
            cache.ctx = nullptr;
        }
        cache = SegmentedDecodeGraphCache{};
    };

    for (auto& cache : slot_cache.segment_caches) {
        clear_one(cache);
    }
    slot_cache.segment_caches.clear();
    for (auto& lc : slot_cache.split_layer_caches) {
        clear_one(lc.premoe);
        if (lc.moe_scheduler) {
            ggml_backend_sched_free(lc.moe_scheduler);
            lc.moe_scheduler = nullptr;
        }
        if (lc.moe_ctx) {
            ggml_free(lc.moe_ctx);
            lc.moe_ctx = nullptr;
        }
        lc = SplitMoeDecodeLayerCache{};
    }
    slot_cache.split_layer_caches.clear();
    slot_cache.use_split_decode = false;
    clear_one(slot_cache.embed_cache);
    clear_one(slot_cache.head_cache);
    slot_cache.signature_valid = false;
}

Qwen35moeForwardPass::SegmentedDecodeSlotCache& Qwen35moeForwardPass::segmented_decode_slot_cache(uint32_t slot_idx) {
    if (slot_idx >= max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::segmented_decode_slot_cache: slot_idx out of range");
    }
    if (segmented_decode_slot_caches_.size() <= slot_idx) {
        segmented_decode_slot_caches_.resize(slot_idx + 1);
    }
    return segmented_decode_slot_caches_[slot_idx];
}

const Qwen35moeForwardPass::SegmentedDecodeSlotCache* Qwen35moeForwardPass::find_segmented_decode_slot_cache(
    uint32_t slot_idx
) const {
    if (slot_idx >= segmented_decode_slot_caches_.size()) {
        return nullptr;
    }
    return &segmented_decode_slot_caches_[slot_idx];
}

void Qwen35moeForwardPass::clear_segmented_decode_cache() {
    for (auto& slot_cache : segmented_decode_slot_caches_) {
        clear_segmented_decode_slot_cache(slot_cache);
    }
    segmented_decode_slot_caches_.clear();
    unpin_all_host_regions();
}

bool Qwen35moeForwardPass::is_segmented_decode_cache_compatible(
    const SegmentedDecodeSlotCache& slot_cache,
    const DecodeGraphSignature& signature,
    uint32_t required_kv
) const {
    if (!slot_cache.signature_valid || !slot_cache.head_cache.valid) {
        return false;
    }
    if (slot_cache.use_split_decode) {
        if (slot_cache.split_layer_caches.size() != count_split_layers()) {
            return false;
        }
        if (slot_cache.segment_caches.size() != layer_segments_.size()) {
            return false;
        }
    } else if (slot_cache.segment_caches.size() != layer_segments_.size()) {
        return false;
    }
    return slot_cache.signature.context_len == signature.context_len &&
           slot_cache.signature.n_batch_tokens == signature.n_batch_tokens &&
           slot_cache.signature.n_ubatch_tokens == signature.n_ubatch_tokens &&
           slot_cache.signature.use_flash_attention == signature.use_flash_attention &&
           slot_cache.signature.paged_fused_decode == signature.paged_fused_decode &&
           slot_cache.signature.is_mixed_mode == signature.is_mixed_mode &&
           slot_cache.signature.device_map_hash == signature.device_map_hash &&
           slot_cache.sampling_top_k == sampling_top_k_ &&
           slot_cache.sampling_temperature == sampling_temperature_ &&
           required_kv <= slot_cache.signature.kv_capacity;
}

void Qwen35moeForwardPass::ensure_segmented_decode_cache(
    const DecodeGraphSignature& signature,
    uint32_t required_kv
) {
    segmented_decode_lookup_count_++;
    SegmentedDecodeSlotCache& slot_cache = segmented_decode_slot_cache(signature.slot_idx);
    if (is_segmented_decode_cache_compatible(slot_cache, signature, required_kv)) {
        segmented_decode_hit_count_++;
        return;
    }

    segmented_decode_miss_count_++;
    try {
        prepare_segmented_decode_cache(signature.slot_idx, signature.kv_capacity);
    } catch (const std::exception& ex) {
        std::fprintf(
            stderr,
            "[PERF][segmented-decode-cache] recapture_failed requested_bucket=%u err=%s\n",
            signature.kv_capacity,
            ex.what()
        );
        const uint32_t fallback_capacity = std::min(context_len_, required_kv);
        if (fallback_capacity == signature.kv_capacity) {
            throw;
        }
        segmented_decode_fallback_count_++;
        prepare_segmented_decode_cache(signature.slot_idx, fallback_capacity);
    }
}

void Qwen35moeForwardPass::prepare_segmented_decode_cache(uint32_t slot_idx, uint32_t kv_capacity) {
    if (layer_segments_.empty()) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_segmented_decode_cache: no layer segments");
    }
    if (kv_capacity == 0 || kv_capacity > context_len_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_segmented_decode_cache: invalid kv capacity");
    }
    if (slot_idx >= max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_segmented_decode_cache: slot_idx out of range");
    }

    clear_segmented_decode_slot_cache(segmented_decode_slot_cache(slot_idx));

    SegmentedDecodeSlotCache& slot_cache = segmented_decode_slot_cache(slot_idx);

    const auto init_cache_context = [](SegmentedDecodeGraphCache& cache) {
        cache.ctx_buffer.assign(FP_GRAPH_SIZE_METADATA, 0);
        struct ggml_init_params params = {
            .mem_size   = cache.ctx_buffer.size(),
            .mem_buffer = cache.ctx_buffer.data(),
            .no_alloc   = true,
        };
        cache.ctx = ggml_init(params);
        if (!cache.ctx) {
            throw std::runtime_error("segmented decode cache: failed to init ggml context");
        }
    };

    const auto allocate_cache_graph = [](SegmentedDecodeGraphCache& cache, const char* where) {
        if (!cache.backend) {
            throw std::runtime_error(std::string(where) + ": backend is null");
        }
        cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cache.backend));
        if (!cache.allocr || !ggml_gallocr_alloc_graph(cache.allocr, cache.graph)) {
            throw std::runtime_error(std::string(where) + ": graph allocation failed");
        }
        cache.valid = true;
    };

    try {
        ggml_type next_hidden_type = GGML_TYPE_F32;
        const bool first_uses_token_input = can_run_token_embedding_on_backend(layer_segments_.front().backend);

        if (!first_uses_token_input) {
            ggml_tensor* token_embedding = model_->get_token_embedding_weight();
            if (!token_embedding) {
                throw std::runtime_error("segmented decode cache: token embedding weight missing");
            }
            SegmentedDecodeGraphCache& embed = slot_cache.embed_cache;
            embed.backend = find_backend_for_tensor(token_embedding);
            init_cache_context(embed);

            ggml_context* saved_ctx = ctx_;
            ctx_ = embed.ctx;
            try {
                embed.graph = new_graph();
                embed.tokens_tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, 1);
                ggml_set_input(embed.tokens_tensor);
                set_tensor_name(embed.tokens_tensor, "tokens");
                ggml_build_forward_expand(embed.graph, embed.tokens_tensor);

                embed.hidden_out_tensor = ggml_get_rows(ctx_, token_embedding, embed.tokens_tensor);
                set_tensor_name(embed.hidden_out_tensor, "segment_hidden_out");
                ggml_build_forward_expand(embed.graph, embed.hidden_out_tensor);
            } catch (...) {
                ctx_ = saved_ctx;
                throw;
            }
            ctx_ = saved_ctx;

            allocate_cache_graph(embed, "segmented decode embedding cache");
            next_hidden_type = embed.hidden_out_tensor->type;
        }

        slot_cache.use_split_decode = model_->has_any_split_device_layers();
        split_moe_last_sched_layer_idx_ = -1;

        if (slot_cache.use_split_decode) {
            const bool first_uses_token_input =
                can_run_token_embedding_on_backend(layer_segments_.front().backend);

            slot_cache.segment_caches.resize(layer_segments_.size());
            slot_cache.split_layer_caches.reserve(count_split_layers());

            for (size_t seg_i = 0; seg_i < layer_segments_.size(); ++seg_i) {
                const bool first = seg_i == 0;
                const LayerSegment& segment = layer_segments_[seg_i];

                if (segment_has_split_layers(segment)) {
                    for (uint32_t il = segment.l0; il < segment.l1; ++il) {
                        if (!model_->layer_has_split_devices(static_cast<int>(il))) {
                            continue;
                        }

                        SplitMoeDecodeLayerCache lc;
                        lc.layer_idx = static_cast<int>(il);
                        const LayerSegment layer_seg{ il, il + 1, layer_backend(static_cast<int>(il)) };
                        const bool use_token_input =
                            il == 0 &&
                            first_uses_token_input &&
                            !slot_cache.embed_cache.valid;

                        SegmentedDecodeGraphCache& premoe = lc.premoe;
                        premoe.backend = layer_seg.backend;
                        premoe.layer_begin = layer_seg.l0;
                        premoe.layer_end = layer_seg.l1;
                        premoe.hidden_type = next_hidden_type;
                        premoe.uses_token_input = use_token_input;
                        init_cache_context(premoe);

                        ggml_context* saved_ctx = ctx_;
                        ctx_ = premoe.ctx;
                        try {
                            premoe.graph = build_segment_graph_no_reset(
                                1,
                                0,
                                slot_idx,
                                layer_seg,
                                use_token_input,
                                next_hidden_type,
                                kv_capacity,
                                true,
                                LayerMoeMode::PreMoeOnly
                            );
                        } catch (...) {
                            ctx_ = saved_ctx;
                            throw;
                        }
                        ctx_ = saved_ctx;

                        premoe.tokens_tensor = ggml_graph_get_tensor(premoe.graph, "tokens");
                        premoe.pos_tensor = ggml_graph_get_tensor(premoe.graph, "inp_pos");
                        premoe.kv_write_phys_tensor = ggml_graph_get_tensor(premoe.graph, "kv_write_phys");
                        premoe.hidden_in_tensor = ggml_graph_get_tensor(premoe.graph, "segment_hidden_in");
                        if (!premoe.pos_tensor) {
                            throw std::runtime_error("split decode cache: missing premoe inp_pos tensor");
                        }
                        if (ggml_tensor* shared_kq_mask = ggml_graph_get_tensor(premoe.graph, "kq_mask_shared")) {
                            premoe.mask_tensors.push_back(shared_kq_mask);
                        } else {
                            char name[32];
                            std::snprintf(name, sizeof(name), "kq_mask.%u", il);
                            if (ggml_tensor* kq_mask = ggml_graph_get_tensor(premoe.graph, name)) {
                                premoe.mask_tensors.push_back(kq_mask);
                            }
                        }
                        allocate_cache_graph(premoe, "split decode premoe cache");

                        char tensor_name[128];
                        std::snprintf(tensor_name, sizeof(tensor_name), "moe_norm_in.%d", lc.layer_idx);
                        lc.premoe_moe_norm_out = ggml_graph_get_tensor(premoe.graph, tensor_name);
                        std::snprintf(tensor_name, sizeof(tensor_name), "ffn_residual_in.%d", lc.layer_idx);
                        lc.premoe_ffn_res_out = ggml_graph_get_tensor(premoe.graph, tensor_name);
                        if (!lc.premoe_moe_norm_out || !lc.premoe_ffn_res_out) {
                            throw std::runtime_error(
                                "split decode cache: missing premoe MoE handoff tensors");
                        }
                        lc.moe_norm_type = lc.premoe_moe_norm_out->type;
                        lc.ffn_res_type = lc.premoe_ffn_res_out->type;

                        lc.moe_ctx_buffer.assign(FP_GRAPH_SIZE_METADATA, 0);
                        struct ggml_init_params moe_params = {
                            .mem_size   = lc.moe_ctx_buffer.size(),
                            .mem_buffer = lc.moe_ctx_buffer.data(),
                            .no_alloc   = true,
                        };
                        lc.moe_ctx = ggml_init(moe_params);
                        if (!lc.moe_ctx) {
                            throw std::runtime_error("split decode cache: failed to init MoE ggml context");
                        }

                        lc.hidden_type = lc.moe_norm_type;
                        saved_ctx = ctx_;
                        ctx_ = lc.moe_ctx;
                        try {
                            lc.moe_graph = build_split_moe_ffn_graph_no_reset(
                                lc.moe_ctx,
                                lc.layer_idx,
                                lc.hidden_type,
                                &lc.moe_norm_in,
                                &lc.moe_ffn_res_in,
                                &lc.hidden_out
                            );
                        } catch (...) {
                            ctx_ = saved_ctx;
                            throw;
                        }
                        ctx_ = saved_ctx;

                        if (!lc.moe_norm_in || !lc.moe_ffn_res_in || !lc.hidden_out) {
                            throw std::runtime_error("split decode cache: missing MoE graph I/O tensors");
                        }

                        lc.moe_scheduler = model_->create_scheduler();
                        if (!lc.moe_scheduler ||
                            !ggml_backend_sched_alloc_graph(lc.moe_scheduler, lc.moe_graph)) {
                            throw std::runtime_error(
                                "split decode cache: per-layer MoE scheduler alloc failed");
                        }

                        slot_cache.split_layer_caches.push_back(std::move(lc));
                        next_hidden_type = lc.hidden_out->type;
                    }
                    continue;
                }

                const bool segment_uses_token_input = first && first_uses_token_input;
                SegmentedDecodeGraphCache& cache = slot_cache.segment_caches[seg_i];
                cache.backend = segment.backend;
                cache.layer_begin = segment.l0;
                cache.layer_end = segment.l1;
                cache.hidden_type = next_hidden_type;
                cache.uses_token_input = segment_uses_token_input;
                init_cache_context(cache);

                ggml_context* saved_ctx = ctx_;
                ctx_ = cache.ctx;
                try {
                    cache.graph = build_decode_segment_graph_no_reset(
                        0,
                        0,
                        slot_idx,
                        segment,
                        segment_uses_token_input,
                        false,
                        next_hidden_type,
                        kv_capacity,
                        true
                    );
                } catch (...) {
                    ctx_ = saved_ctx;
                    throw;
                }
                ctx_ = saved_ctx;

                cache.tokens_tensor = ggml_graph_get_tensor(cache.graph, "tokens");
                cache.pos_tensor = ggml_graph_get_tensor(cache.graph, "inp_pos");
                cache.kv_write_phys_tensor = ggml_graph_get_tensor(cache.graph, "kv_write_phys");
                cache.hidden_in_tensor = ggml_graph_get_tensor(cache.graph, "segment_hidden_in");
                cache.hidden_out_tensor = ggml_graph_get_tensor(cache.graph, "segment_hidden_out");
                if (!cache.pos_tensor || !cache.hidden_out_tensor) {
                    throw std::runtime_error(
                        "split decode cache: missing non-split segment I/O tensors");
                }
                if (ggml_tensor* shared_kq_mask = ggml_graph_get_tensor(cache.graph, "kq_mask_shared")) {
                    cache.mask_tensors.push_back(shared_kq_mask);
                } else {
                    for (uint32_t il = segment.l0; il < segment.l1; ++il) {
                        if (!is_full_attention_layer(il)) {
                            continue;
                        }
                        char name[32];
                        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
                        if (ggml_tensor* kq_mask = ggml_graph_get_tensor(cache.graph, name)) {
                            cache.mask_tensors.push_back(kq_mask);
                        }
                    }
                }

                allocate_cache_graph(cache, "split decode non-split segment cache");
                next_hidden_type = cache.hidden_out_tensor->type;
            }
        } else {
        slot_cache.segment_caches.resize(layer_segments_.size());
        for (size_t i = 0; i < layer_segments_.size(); ++i) {
            const bool first = i == 0;
            const LayerSegment& segment = layer_segments_[i];
            const bool segment_uses_token_input = first && first_uses_token_input;
            SegmentedDecodeGraphCache& cache = slot_cache.segment_caches[i];
            cache.backend = segment.backend;
            cache.layer_begin = segment.l0;
            cache.layer_end = segment.l1;
            cache.hidden_type = next_hidden_type;
            cache.uses_token_input = segment_uses_token_input;
            init_cache_context(cache);

            ggml_context* saved_ctx = ctx_;
            ctx_ = cache.ctx;
            try {
                cache.graph = build_decode_segment_graph_no_reset(
                    0,
                    0,
                    slot_idx,
                    segment,
                    segment_uses_token_input,
                    false,
                    next_hidden_type,
                    kv_capacity,
                    true
                );
            } catch (...) {
                ctx_ = saved_ctx;
                throw;
            }
            ctx_ = saved_ctx;

            cache.tokens_tensor = ggml_graph_get_tensor(cache.graph, "tokens");
            cache.pos_tensor = ggml_graph_get_tensor(cache.graph, "inp_pos");
            cache.kv_write_phys_tensor = ggml_graph_get_tensor(cache.graph, "kv_write_phys");
            cache.hidden_in_tensor = ggml_graph_get_tensor(cache.graph, "segment_hidden_in");
            cache.hidden_out_tensor = ggml_graph_get_tensor(cache.graph, "segment_hidden_out");
            if (!cache.pos_tensor || !cache.hidden_out_tensor) {
                throw std::runtime_error("segmented decode cache: missing segment input/output tensors");
            }
            if (ggml_tensor* shared_kq_mask = ggml_graph_get_tensor(cache.graph, "kq_mask_shared")) {
                cache.mask_tensors.push_back(shared_kq_mask);
            } else {
                for (uint32_t il = segment.l0; il < segment.l1; ++il) {
                    if (!is_full_attention_layer(il)) {
                        continue;
                    }
                    char name[32];
                    std::snprintf(name, sizeof(name), "kq_mask.%u", il);
                    if (ggml_tensor* kq_mask = ggml_graph_get_tensor(cache.graph, name)) {
                        cache.mask_tensors.push_back(kq_mask);
                    }
                }
            }

            allocate_cache_graph(cache, "segmented decode layer cache");
            next_hidden_type = cache.hidden_out_tensor->type;
        }
        }

        SegmentedDecodeGraphCache& head = slot_cache.head_cache;
        head.backend = model_->get_curr_backend();
        head.hidden_type = next_hidden_type;
        init_cache_context(head);

        ggml_context* saved_ctx = ctx_;
        ctx_ = head.ctx;
        try {
            head.graph = build_output_head_graph_from_hidden_no_reset(next_hidden_type);
        } catch (...) {
            ctx_ = saved_ctx;
            throw;
        }
        ctx_ = saved_ctx;

        head.hidden_in_tensor = ggml_graph_get_tensor(head.graph, "segment_hidden_in");
        if (!head.hidden_in_tensor) {
            throw std::runtime_error("segmented decode cache: missing head hidden input tensor");
        }
        allocate_cache_graph(head, "segmented decode head cache");

        slot_cache.signature.slot_idx = slot_idx;
        slot_cache.signature.kv_capacity = kv_capacity;
        slot_cache.signature.context_len = context_len_;
        slot_cache.signature.n_batch_tokens = n_batch_tokens_;
        slot_cache.signature.n_ubatch_tokens = n_ubatch_tokens_;
        slot_cache.signature.use_flash_attention = use_flash_attention_;
        slot_cache.signature.paged_fused_decode = paged_fused_decode_active();
        slot_cache.signature.is_mixed_mode = model_->is_mixed_mode();
        slot_cache.signature.device_map_hash = model_->compute_device_map_hash();
        slot_cache.signature_valid = true;
        slot_cache.sampling_top_k = sampling_top_k_;
        slot_cache.sampling_temperature = sampling_temperature_;
        segmented_decode_bucket_usage_[kv_capacity]++;
        segmented_decode_recapture_count_++;
    } catch (...) {
        clear_segmented_decode_slot_cache(slot_cache);
        throw;
    }
}

bool Qwen35moeForwardPass::ensure_segmented_decode_copy_ready(uint32_t slot_idx, uint32_t dst_pos) const {
    if (!paged_kv_enabled_ || !kv_cache_) {
        return true;
    }
    if (!kv_cache_->ensure_materialized_logical_pos(slot_idx, dst_pos)) {
        std::fprintf(stderr,
            "[paged-kv] segmented decode materialize fallback: slot=%u pos=%u\n",
            slot_idx,
            dst_pos);
        return false;
    }
    return true;
}

bool Qwen35moeForwardPass::commit_segmented_decode_step(uint32_t slot_idx, uint32_t dst_pos) {
    (void)dst_pos;
    advance_cache(1, slot_idx);
    maybe_log_segmented_decode_cache_stats();
    return true;
}

void Qwen35moeForwardPass::maybe_log_segmented_decode_cache_stats() {
    if (!decode_graph_diag_enabled_ || segmented_decode_lookup_count_ == 0) {
        return;
    }

    bool interval_reached = false;
    if (decode_graph_diag_interval_ > 0) {
        interval_reached = (segmented_decode_lookup_count_ % decode_graph_diag_interval_) == 0;
    }
    const bool has_new_recapture = segmented_decode_recapture_count_ > segmented_decode_last_logged_recapture_;
    if (!interval_reached && !has_new_recapture) {
        return;
    }

    uint32_t hottest_bucket = 0;
    uint64_t hottest_bucket_count = 0;
    for (const auto& item : segmented_decode_bucket_usage_) {
        if (item.second > hottest_bucket_count) {
            hottest_bucket_count = item.second;
            hottest_bucket = item.first;
        }
    }

    uint32_t active_bucket = 0;
    for (const auto& slot_cache : segmented_decode_slot_caches_) {
        if (slot_cache.signature_valid && slot_cache.signature.kv_capacity > active_bucket) {
            active_bucket = slot_cache.signature.kv_capacity;
        }
    }

    std::fprintf(
        stderr,
        "[PERF][segmented-decode-cache] lookups=%llu hit=%llu miss=%llu recapture=%llu fallback=%llu active_bucket=%u segments=%zu hot_bucket=%u hot_bucket_uses=%llu\n",
        static_cast<unsigned long long>(segmented_decode_lookup_count_),
        static_cast<unsigned long long>(segmented_decode_hit_count_),
        static_cast<unsigned long long>(segmented_decode_miss_count_),
        static_cast<unsigned long long>(segmented_decode_recapture_count_),
        static_cast<unsigned long long>(segmented_decode_fallback_count_),
        active_bucket,
        layer_segments_.size(),
        hottest_bucket,
        static_cast<unsigned long long>(hottest_bucket_count)
    );
    segmented_decode_last_logged_lookup_ = segmented_decode_lookup_count_;
    segmented_decode_last_logged_recapture_ = segmented_decode_recapture_count_;
}

void Qwen35moeForwardPass::set_cached_segment_inputs(
    SegmentedDecodeGraphCache& cache,
    int32_t token,
    int pos,
    uint32_t slot_idx,
    ggml_type hidden_type,
    ggml_tensor* hidden_src,
    const LayerSegment* segment
) {
    if (cache.tokens_tensor) {
        if (cache.tokens_tensor->type != GGML_TYPE_I32 ||
            static_cast<size_t>(ggml_nbytes(cache.tokens_tensor)) != sizeof(int32_t)) {
            throw std::runtime_error("Qwen35moeForwardPass::set_cached_segment_inputs: tokens tensor mismatch");
        }
        maybe_log_segment_tensor("tokens", segment, cache.tokens_tensor);
        // Tokens are tiny (4B); use sync set so the i32 lives in pageable memory
        // doesn't cause an extra staging copy. The next compute is on the same
        // backend, so ordering is preserved.
        ggml_backend_tensor_set(cache.tokens_tensor, &token, 0, sizeof(token));
    }

    if (cache.pos_tensor) {
        if (cache.pos_tensor->type != GGML_TYPE_I32 ||
            static_cast<size_t>(ggml_nbytes(cache.pos_tensor)) != sizeof(int32_t)) {
            throw std::runtime_error("Qwen35moeForwardPass::set_cached_segment_inputs: inp_pos tensor mismatch");
        }
        ggml_backend_tensor_set_async(cache.backend, cache.pos_tensor, &pos, 0, sizeof(pos));
    }

    if (paged_kv_enabled_ && kv_cache_) {
        const uint32_t logical_pos = static_cast<uint32_t>(pos);
        kv_cache_->ensure_contiguous_kv_prefix(slot_idx, logical_pos + 1);
        if (cache.kv_write_phys_tensor) {
            if (!kv_cache_->ensure_materialized_logical_pos(slot_idx, logical_pos)) {
                throw std::runtime_error(
                    "Qwen35moeForwardPass::set_cached_segment_inputs: failed to materialize paged KV for decode write"
                );
            }
            const uint32_t phys = kv_cache_->physical_row_for_contiguous_write(slot_idx, logical_pos);
            ggml_backend_tensor_set(cache.kv_write_phys_tensor, &phys, 0, sizeof(int32_t));
        }
    }

    if (cache.hidden_in_tensor) {
        if (hidden_src) {
            ggml_backend_t src_backend = find_backend_for_tensor(hidden_src);
            copy_handoff_tensor_async(
                "Qwen35moeForwardPass::set_cached_segment_inputs",
                src_backend,
                cache.backend,
                hidden_src,
                cache.hidden_in_tensor,
                hidden_type
            );
        }
    } else if (!cache.tokens_tensor && hidden_src == nullptr) {
        throw std::runtime_error("Qwen35moeForwardPass::set_cached_segment_inputs: missing graph input");
    }

    uint32_t prepared_n_kv = 0;
    bool f16_ready = false;
    const bool can_incremental = cache.last_mask_pos >= 0 && pos == cache.last_mask_pos + 1;
    for (ggml_tensor* kq_mask : cache.mask_tensors) {
        const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        if (prepared_n_kv != n_kv) {
            const bool need_full_refresh =
                !can_incremental ||
                cache.mask_f32.size() != n_kv ||
                cache.last_mask_pos < 0;
            if (need_full_refresh) {
                // If the vector capacity changes its data() pointer, drop any
                // stale pinning before reallocating.
                if (cache.mask_f32.capacity() < n_kv) {
                    unpin_host_region(cache.mask_f32.data());
                }
                cache.mask_f32.assign(n_kv, -INFINITY);
                const uint32_t active_kv = std::min(n_kv, static_cast<uint32_t>(pos) + 1);
                for (uint32_t j = 0; j < active_kv; ++j) {
                    cache.mask_f32[j] = 0.0f;
                }
                // Pin the (now stably-sized) host-side mask buffer so subsequent
                // tensor_set_async to the GPU can use cudaMemcpyAsync without
                // an internal staging copy. No-op when CUDA is not built or
                // the backend isn't CUDA.
                pin_host_region(cache.mask_f32.data(), cache.mask_f32.size() * sizeof(float));
            } else if (pos > 0) {
                const uint32_t visible_prev_pos = static_cast<uint32_t>(pos - 1);
                if (visible_prev_pos < n_kv) {
                    cache.mask_f32[visible_prev_pos] = 0.0f;
                }
            }
            prepared_n_kv = n_kv;
            f16_ready = false;
        }

        if (kq_mask->type == GGML_TYPE_F16) {
            if (!f16_ready) {
                if (cache.mask_f16.capacity() < cache.mask_f32.size()) {
                    unpin_host_region(cache.mask_f16.data());
                }
                cache.mask_f16.resize(cache.mask_f32.size());
                ggml_fp32_to_fp16_row(
                    cache.mask_f32.data(),
                    cache.mask_f16.data(),
                    static_cast<int64_t>(cache.mask_f32.size())
                );
                pin_host_region(cache.mask_f16.data(), cache.mask_f16.size() * sizeof(ggml_fp16_t));
                f16_ready = true;
            }
            ggml_backend_tensor_set_async(
                cache.backend,
                kq_mask,
                cache.mask_f16.data(),
                0,
                cache.mask_f16.size() * sizeof(ggml_fp16_t)
            );
        } else {
            ggml_backend_tensor_set_async(
                cache.backend,
                kq_mask,
                cache.mask_f32.data(),
                0,
                cache.mask_f32.size() * sizeof(float)
            );
        }
    }
    if (!cache.mask_tensors.empty()) {
        cache.last_mask_pos = pos;
    }
}

/**
 * @brief 构建 Prefill 阶段计算图
 * 
 * Prefill 阶段用于处理初始上下文输入，将所有 token 的 KV 值写入缓存
 * 
 * @param tokens 输入的 token 序列
 * @param pos 起始位置（通常为 0）
 * @param slot_idx 槽位索引（默认为 0）
 * @return 构建好的计算图
 * 
 * 处理流程：
 * 1. Token Embedding 查找
 * 2. 创建位置编码张量
 * 3. 遍历所有 Transformer 层：
 *    - Pre-attention RMSNorm
 *    - Full Attention 或 DeltaNet 层
 *    - 残差连接
 *    - Pre-FFN RMSNorm
 *    - MoE FFN 层
 *    - 残差连接
 * 4. 最终归一化 + LM Head
 */
ggml_cgraph* Qwen35moeForwardPass::build_prefill_graph(
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx,
    uint32_t fixed_shared_n_kv,
    bool dynamic_kv_write
) {
    reset_context();

    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    if (n_tok == 0) {
        throw std::runtime_error(
            "Qwen35moeForwardPass::build_prefill_graph: empty token input"
        );
    }
    if (static_cast<uint32_t>(pos) + n_tok > context_len_) {
        throw std::runtime_error(
            "Qwen35moeForwardPass::build_prefill_graph: input exceeds context_len_"
        );
    }

    ggml_cgraph* gf = new_graph();

    auto& m = model_->meta_->qwen35moe;

    // 1. Token Embedding：将 token ID 转换为词向量
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(inpL, "inpL");

    // 2. 位置张量（所有注意力层共享）
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp_pos);  // 标记为输入张量
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    ggml_tensor* kv_write_row = nullptr;
    if (dynamic_kv_write && fixed_shared_n_kv > 0) {
        if (paged_kv_enabled_) {
            kv_write_row = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
            ggml_set_input(kv_write_row);
            set_tensor_name(kv_write_row, "kv_write_phys");
            ggml_build_forward_expand(gf, kv_write_row);
        } else {
            kv_write_row = inp_pos;
        }
    }

    // 3. Transformer 层循环
    inpL = build_layer_range(
        gf, inpL, inp_pos, n_tok, slot_idx, 0, model_->trunk_layer_count(), fixed_shared_n_kv, kv_write_row);

    // 4. 最终归一化 + LM Head。Prefill 只需要最后一个 token 的 logits
    // 来采样首个 decode token，避免对整段 prompt 做 vocab 投影。
    ggml_tensor* last_token = inpL;
    if (n_tok > 1) {
        last_token = ggml_view_2d(
            ctx_,
            inpL,
            inpL->ne[0],
            1,
            inpL->nb[1],
            static_cast<size_t>(n_tok - 1) * inpL->nb[1]
        );
        set_tensor_name(last_token, "prefill_last_hidden");
    }
    build_output_head(gf, last_token);

    return gf;
}

ggml_tensor* Qwen35moeForwardPass::build_layer_range(
    ggml_cgraph* gf,
    ggml_tensor* inpL,
    ggml_tensor* inp_pos,
    uint32_t n_tok,
    uint32_t slot_idx,
    uint32_t layer_begin,
    uint32_t layer_end,
    uint32_t fixed_shared_n_kv,
    ggml_tensor* kv_write_row,
    LayerMoeMode moe_mode
) {
    auto& m = model_->meta_->qwen35moe;
    const uint32_t d_inner = m.inner_size;
    const uint32_t num_v_heads = m.time_step_rank;
    const uint32_t num_k_heads = m.group_count;
    const uint32_t head_v_dim = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.state_size;
    DeltaNetStateParams dn_hp {
        0, 0,
        head_v_dim,
        m.state_size,
        num_v_heads,
        conv_channels,
        m.conv_kernel,
        nullptr
    };

    ggml_tensor* shared_kq_mask = nullptr;
    uint32_t shared_n_kv = fixed_shared_n_kv;
    if (shared_n_kv == 0) {
        shared_n_kv = kv_cache_ ? kv_cache_->get_pos(slot_idx) + n_tok : n_tok;
    }
    auto get_shared_kq_mask = [&]() -> ggml_tensor* {
        if (!shared_kq_mask) {
            if (!attention_range_homogeneous(layer_begin, layer_end)) {
                return nullptr;
            }
            const bool shared_fa = [&]() {
                for (uint32_t il = layer_begin; il < layer_end; ++il) {
                    if (is_full_attention_layer(il) && !layer_allows_flash_attn(static_cast<int>(il))) {
                        return false;
                    }
                }
                return use_flash_attention_;
            }();
            shared_kq_mask = ggml_new_tensor_2d(
                ctx_,
                shared_fa ? GGML_TYPE_F16 : GGML_TYPE_F32,
                shared_n_kv,
                n_tok
            );
            set_tensor_name(shared_kq_mask, "kq_mask_shared");
            ggml_build_forward_expand(gf, shared_kq_mask);
        }
        return shared_kq_mask;
    };

    for (uint32_t il = layer_begin; il < layer_end; ++il) {
        ggml_tensor* inpSA = inpL;
        ggml_tensor* attn_norm_weight = model_->get_attn_norm_weight(il);
        if (dev_check_enabled_ && attn_norm_weight) {
            const char* norm_dev = attn_norm_weight->buffer
                ? ggml_backend_buft_name(ggml_backend_buffer_get_type(attn_norm_weight->buffer))
                : "unallocated";
            std::fprintf(stderr, "[dev-check] prefill layer=%u attn_norm_weight.backend=%s\n", il, norm_dev);
        }

        ggml_tensor* cur = build_norm(gf, inpL, attn_norm_weight, il);
        if (is_full_attention_layer(il)) {
            int kv_idx = kv_layer_map_[il];
            ggml_tensor* attn_q_weight = model_->get_attn_q_weight(il);
            ggml_tensor* attn_q_norm_weight = model_->get_attn_q_norm_weight(il);
            ggml_tensor* attn_k_weight = model_->get_attn_k_weight(il);
            ggml_tensor* attn_k_norm_weight = model_->get_attn_k_norm_weight(il);
            ggml_tensor* attn_v_weight = model_->get_attn_v_weight(il);
            ggml_tensor* attn_output_weight = model_->get_attn_output_weight(il);

            if (dev_check_enabled_ && attn_q_weight && attn_k_weight) {
                const char* q_dev = attn_q_weight->buffer
                    ? ggml_backend_buft_name(ggml_backend_buffer_get_type(attn_q_weight->buffer))
                    : "unallocated";
                const char* k_dev = attn_k_weight->buffer
                    ? ggml_backend_buft_name(ggml_backend_buffer_get_type(attn_k_weight->buffer))
                    : "unallocated";
                int kc_layer = kv_layer_map_[il];
                const char* kc_dev = (kc_layer >= 0 && kv_cache_)
                    ? (kv_cache_->get_k_cache_tensor(kc_layer) &&
                       kv_cache_->get_k_cache_tensor(kc_layer)->buffer
                       ? ggml_backend_buft_name(ggml_backend_buffer_get_type(
                           kv_cache_->get_k_cache_tensor(kc_layer)->buffer))
                       : "unallocated")
                    : "n/a";
                std::fprintf(stderr,
                    "[dev-check] prefill layer=%u attn_q.backend=%s attn_k.backend=%s kv_cache.backend=%s\n",
                    il, q_dev, k_dev, kc_dev);
            }

            cur = build_gated_attention(
                ctx_, gf, kv_cache_.get(), cur, inp_pos,
                kv_idx, n_tok, slot_idx, il, attn_q_weight,
                attn_q_norm_weight, attn_k_weight, attn_k_norm_weight, attn_v_weight, attn_output_weight,
                m.key_length, m.head_count, m.head_count_kv, m.dimension_count, m.freq_base,
                static_cast<int>(context_len_), m.layer_norm_rms_epsilon, get_shared_kq_mask(),
                kv_write_row, fixed_shared_n_kv
            );
        } else {
            uint32_t dn_idx = static_cast<uint32_t>(dn_layer_map_[il]);
            cur = build_dn_layer(ctx_, gf, cur, dn_state_.get(), dn_hp, num_k_heads,
                m.embedding_length, dn_idx, n_tok, slot_idx, m.layer_norm_rms_epsilon, il
            );
        }

        cur = ggml_add(ctx_, cur, inpSA);

        ggml_tensor* ffn_inp = cur;
        ggml_tensor* post_attention_norm = model_->get_post_attention_norm_weight(il);
        if (dev_check_enabled_) {
            ggml_tensor* ffn_gate_inp_w = model_->get_ffn_gate_inp_weight(il);
            if (ffn_gate_inp_w) {
                const char* ffn_dev = ffn_gate_inp_w->buffer
                    ? ggml_backend_buft_name(ggml_backend_buffer_get_type(ffn_gate_inp_w->buffer))
                    : "unallocated";
                std::fprintf(stderr, "[dev-check] prefill layer=%u ffn_gate_inp.backend=%s\n", il, ffn_dev);
            }
        }

        cur = build_norm(gf, cur, post_attention_norm, il);
        if (moe_mode == LayerMoeMode::PreMoeOnly &&
            model_->layer_has_split_devices(static_cast<int>(il))) {
            const int64_t n_embd = cur->ne[0];
            const int64_t n_tok_ll = cur->ne[1];
            ggml_tensor* moe_norm_copy = ggml_cpy(
                ctx_,
                cur,
                ggml_new_tensor_2d(ctx_, cur->type, n_embd, n_tok_ll)
            );
            set_tensor_name(moe_norm_copy, "moe_norm_in", il);
            ggml_tensor* ffn_res_copy = ggml_cpy(
                ctx_,
                ffn_inp,
                ggml_new_tensor_2d(ctx_, ffn_inp->type, n_embd, n_tok_ll)
            );
            set_tensor_name(ffn_res_copy, "ffn_residual_in", il);
            ggml_build_forward_expand(gf, moe_norm_copy);
            ggml_build_forward_expand(gf, ffn_res_copy);
            inpL = moe_norm_copy;
            continue;
        }
        cur = build_moe_layer(ctx_, gf, cur, il);
        cur = ggml_add(ctx_, cur, ffn_inp);
        set_tensor_name(cur, "layer_out", il);
        inpL = cur;
    }
    return inpL;
}

void Qwen35moeForwardPass::rebuild_layer_segments() {
    layer_segments_.clear();
    const auto& device_map = model_->get_layer_device_map();
    if (device_map.empty()) {
        return;
    }
    uint32_t i = 0;
    while (i < device_map.size()) {
        uint32_t j = i + 1;
        while (j < device_map.size() && device_map[j] == device_map[i]) {
            ++j;
        }
        layer_segments_.push_back(LayerSegment{i, j, device_map[i]});
        i = j;
    }
}

/**
 * @brief 构建 Decode 阶段计算图（支持多序列并行）
 * 
 * Decode 阶段用于逐 token 生成，支持多序列并行处理
 * 每个序列每次只生成一个 token，但多个序列可以并行处理
 * 
 * @param tokens 每个序列当前的 token（batch 大小）
 * @param slots 每个 token 对应的槽位索引
 * @param positions 每个序列的当前位置
 * @return 构建好的计算图
 * 
 * 处理流程：
 * 1. Token Embedding
 * 2. 创建位置编码张量（每个 batch 元素一个）
 * 3. 创建 KV gather mask 和 gather indices（用于从稀疏缓存中收集 KV）
 * 4. 遍历 Transformer 层（与 Prefill 类似，但使用批处理版本）
 * 5. 最终归一化 + LM Head
 */
ggml_cgraph* Qwen35moeForwardPass::build_decoding_graph(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions,
    uint32_t fixed_n_kv_len
) {
    reset_context();

    ggml_cgraph* gf = new_graph();

    auto& m = model_->meta_->qwen35moe;
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());  // 批大小 = 序列数量
    if (n_batch == 0 || slots.size() != n_batch || positions.size() != n_batch) {
        throw std::runtime_error("Qwen35moeForwardPass::build_decoding_graph: input size mismatch");
    }

    // 从元数据派生 DeltaNet 参数
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

    // 1. Token Embedding
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(inpL, "inpL");

    // 2. 位置张量（每个 batch 元素一个位置）
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_batch);
    ggml_set_input(inp_pos);
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. KV gather mask — 所有注意力层共享
    uint32_t n_kv_len = fixed_n_kv_len;
    if (n_kv_len == 0) {
        uint32_t max_physical = 0;
        for (uint32_t s : slots) {
            uint32_t phys = get_physical_cache_pos(s);
            if (phys > max_physical) {
                max_physical = phys;
            }
        }
        uint32_t max_pos = 0;
        for (int32_t p : positions) {
            if (p >= 0 && static_cast<uint32_t>(p) > max_pos) {
                max_pos = static_cast<uint32_t>(p);
            }
        }
        n_kv_len = std::max(max_physical, max_pos) + 1;
    }
    
    // 创建 KV mask: [n_kv_len, 1, 1, n_batch]
    const bool batched_shared_fa = use_flash_attention_ && !model_->is_mixed_mode();
    ggml_tensor* kq_mask = ggml_new_tensor_4d(
        ctx_,
        batched_shared_fa ? GGML_TYPE_F16 : GGML_TYPE_F32,
        n_kv_len,
        1,
        1,
        n_batch
    );
    ggml_set_input(kq_mask);
    set_tensor_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    // 创建 gather indices: [n_batch * n_kv_len]
    // 用于从稀疏 KV 缓存中收集正确的 KV 对
    ggml_tensor* gather_indices = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, static_cast<int64_t>(n_batch * n_kv_len));
    ggml_set_input(gather_indices);
    set_tensor_name(gather_indices, "gather_indices");

    // 4. Transformer 层循环
    for (uint32_t il = 0; il < model_->trunk_layer_count(); ++il) {
        ggml_tensor* inpSA = inpL;

        // Pre-attention RMSNorm
        struct ggml_tensor* attn_norm_weight = model_->get_attn_norm_weight(il);
        ggml_tensor* cur = build_norm(gf, inpL, attn_norm_weight, il);

        if (is_full_attention_layer(il)) {
            // Full Attention 层：使用批处理版本
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
                static_cast<int>(context_len_),
                m.layer_norm_rms_epsilon
            );
        } else {
            // DeltaNet 层：解码阶段，每个槽位一个 token
            uint32_t dn_idx = static_cast<uint32_t>(dn_layer_map_[il]);
            DecodeArgs da{slots};
            PrefillArgs pa_unused{1, 0};

            cur = build_all_deltanet_layer(ctx_, gf, cur, dn_idx, Phase::Decode, pa_unused, &da, dn_state_.get(), dn_hp, il);
        }

        // 残差连接 1
        cur = ggml_add(ctx_, cur, inpSA);

        // Pre-FFN RMSNorm + MoE
        ggml_tensor* ffn_inp = cur;
        struct ggml_tensor* post_attention_norm = model_->get_post_attention_norm_weight(il);
        cur = build_norm(gf, cur, post_attention_norm, il);
        cur = build_moe_layer(ctx_, gf, cur, il);

        // 残差连接 2
        cur = ggml_add(ctx_, cur, ffn_inp);
        inpL = cur;
    }

    // 最终归一化 + LM Head
    build_output_head(gf, inpL);
    return gf;
}

/**
 * @brief 设置 Prefill 阶段的输入数据
 * 
 * @param gf 计算图
 * @param tokens 输入 token 序列
 * @param pos 起始位置
 * 
 * 设置内容：
 * 1. Token ID 张量
 * 2. 位置 ID 张量
 * 3. 因果注意力掩码（Causal mask）
 */
void Qwen35moeForwardPass::set_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens, int pos) {
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());

    // 1. 设置 Token 张量
    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) 
        throw std::runtime_error("qwen36: 'tokens' tensor missing from graph");
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, n_tok * sizeof(int32_t));

    // 2. 设置位置 ID 张量
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) 
        throw std::runtime_error("qwen36: 'inp_pos' tensor missing from graph");

    std::vector<int32_t> pos_data(n_tok);
    for (uint32_t i = 0; i < n_tok; ++i) 
        pos_data[i] = pos + static_cast<int>(i);  // 位置从 pos 开始递增
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, n_tok * sizeof(int32_t));

    // 3. 设置因果注意力掩码（仅用于 Full Attention 层）
    const auto apply_prefill_masks = [&](ggml_tensor* kq_mask) {
        if (!kq_mask) {
            return;
        }
        const uint32_t graph_n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        std::vector<float> mask;
        fill_dynamic_prefill_mask(mask, graph_n_kv, n_tok, pos);
        upload_decode_mask_tensor(kq_mask, mask);
    };

    if (ggml_tensor* shared_kq_mask = ggml_graph_get_tensor(gf, "kq_mask_shared")) {
        apply_prefill_masks(shared_kq_mask);
        return;
    }

    for (uint32_t il = 0; il < model_->trunk_layer_count(); ++il) {
        if (!is_full_attention_layer(il)) {
            continue;
        }
        char name[32];
        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
        apply_prefill_masks(ggml_graph_get_tensor(gf, name));
    }
}

/**
 * @brief 设置 Decode 阶段的批处理输入数据
 * 
 * @param gf 计算图
 * @param tokens 每个序列当前的 token
 * @param slots 槽位索引数组
 * @param positions 每个序列的当前位置
 * 
 * 设置内容：
 * 1. Token ID 张量
 * 2. 位置 ID 张量（每个序列一个）
 * 3. 共享 KV 掩码
 * 4. Gather indices（用于从稀疏缓存收集 KV）
 */
void Qwen35moeForwardPass::set_batched_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots, const std::vector<int32_t>&  positions) {
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());

    // 1. 设置 Token 张量
    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) throw std::runtime_error("qwen36: 'tokens' tensor missing from graph");
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, n_batch * sizeof(int32_t));

    // 2. 设置位置 ID 张量（每个 batch 槽位一个位置）
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) throw std::runtime_error("qwen36: 'inp_pos' tensor missing from graph");
    ggml_backend_tensor_set(pos_t, positions.data(), 0, n_batch * sizeof(int32_t));

    // 3. 动态 KV 掩码 [graph_n_kv, 1, 1, n_batch]：仅开放 causal 前缀，桶内尾部保持 -inf
    ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, "kq_mask_b");
    if (kq_mask) {
        const uint32_t graph_n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        std::vector<float> mask;
        std::vector<int32_t> last_positions;
        fill_dynamic_decode_mask_batched(
            mask,
            graph_n_kv,
            n_batch,
            positions,
            false,
            last_positions
        );
        upload_decode_mask_tensor(kq_mask, mask);
    }

    // 4. 设置 Gather indices [n_batch * n_kv_len]
    // 用于从稀疏 KV 缓存中收集正确的 KV 对
    ggml_tensor* gi = ggml_graph_get_tensor(gf, "gather_indices");
    if (gi) {
        const uint32_t n_kv   = static_cast<uint32_t>(gi->ne[0]) / n_batch;
        std::vector<int32_t> idx;
        if (kv_cache_) {
            kv_cache_->fill_gather_indices(slots, n_kv, idx);
        } else {
            idx.resize(n_batch * n_kv);
            for (uint32_t b = 0; b < n_batch; ++b) {
                const uint32_t slot = slots[b];
                for (uint32_t j = 0; j < n_kv; ++j) {
                    idx[b * n_kv + j] = static_cast<int32_t>(slot * context_len_ + j);
                }
            }
        }
        ggml_backend_tensor_set(gi, idx.data(), 0, idx.size() * sizeof(int32_t));
    }

    // 5. 设置 DeltaNet batched-decode slot 索引 [n_batch]
    // 仅当 graph 含 DN 层、且走 batched decode 路径（n_batch > 1）时存在；
    // 所有 DN 层共享同一个 input 张量，供 ggml_get_rows / ggml_set_rows
    // 按 slot 拉取 / 写回 conv 与 recurrent state。
    ggml_tensor* dn_slot_idx = ggml_graph_get_tensor(gf, "dn_slot_idx");
    if (dn_slot_idx) {
        std::vector<int32_t> slot_idx_i32(slots.begin(), slots.end());
        ggml_backend_tensor_set(
            dn_slot_idx,
            slot_idx_i32.data(),
            0,
            slot_idx_i32.size() * sizeof(int32_t)
        );
    }
}

ggml_cgraph* Qwen35moeForwardPass::build_segment_graph_no_reset(
    uint32_t n_tok,
    int pos,
    uint32_t slot_idx,
    const LayerSegment& segment,
    bool use_token_input,
    ggml_type hidden_type,
    uint32_t fixed_shared_n_kv,
    bool dynamic_kv_write,
    LayerMoeMode moe_mode
) {
    (void)pos;
    if (n_tok == 0) {
        throw std::runtime_error("Qwen35moeForwardPass::build_segment_graph_no_reset: n_tok is zero");
    }
    ggml_cgraph* gf = new_graph();

    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp_pos);
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    ggml_tensor* kv_write_row = nullptr;
    if (dynamic_kv_write && fixed_shared_n_kv > 0) {
        if (paged_kv_enabled_) {
            kv_write_row = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
            ggml_set_input(kv_write_row);
            set_tensor_name(kv_write_row, "kv_write_phys");
            ggml_build_forward_expand(gf, kv_write_row);
        } else {
            kv_write_row = inp_pos;
        }
    }

    ggml_tensor* inpL = nullptr;
    if (use_token_input) {
        if (n_tok == 1) {
            std::vector<int32_t> token_vec = { 0 };
            inpL = embedding(gf, token_vec);
        } else {
            inpL = embedding(gf, std::vector<int32_t>(n_tok, 0));
        }
        set_tensor_name(inpL, "inpL");
    } else {
        const int64_t n_embd = static_cast<int64_t>(model_->meta_->qwen35moe.embedding_length);
        inpL = ggml_new_tensor_2d(ctx_, hidden_type, n_embd, static_cast<int64_t>(n_tok));
        ggml_set_input(inpL);
        set_tensor_name(inpL, "segment_hidden_in");
        ggml_build_forward_expand(gf, inpL);
    }

    inpL = build_layer_range(
        gf,
        inpL,
        inp_pos,
        n_tok,
        slot_idx,
        segment.l0,
        segment.l1,
        fixed_shared_n_kv,
        kv_write_row,
        moe_mode
    );
    if (moe_mode != LayerMoeMode::PreMoeOnly) {
        set_tensor_name(inpL, "segment_hidden_out");
    }
    ggml_build_forward_expand(gf, inpL);

    return gf;
}

ggml_cgraph* Qwen35moeForwardPass::build_prefill_segment_graph(
    uint32_t n_tok,
    int pos,
    uint32_t slot_idx,
    const LayerSegment& segment,
    bool use_token_input,
    ggml_type hidden_type
) {
    reset_context();
    return build_segment_graph_no_reset(
        n_tok,
        pos,
        slot_idx,
        segment,
        use_token_input,
        hidden_type,
        0,
        false
    );
}

ggml_cgraph* Qwen35moeForwardPass::build_decode_segment_graph(
    int32_t token,
    int pos,
    uint32_t slot_idx,
    const LayerSegment& segment,
    bool is_first_segment,
    bool is_last_segment,
    ggml_type hidden_type
) {
    (void)token;
    reset_context();
    return build_decode_segment_graph_no_reset(
        token,
        pos,
        slot_idx,
        segment,
        is_first_segment,
        is_last_segment,
        hidden_type
    );
}

ggml_cgraph* Qwen35moeForwardPass::build_decode_segment_graph_no_reset(
    int32_t token,
    int pos,
    uint32_t slot_idx,
    const LayerSegment& segment,
    bool is_first_segment,
    bool is_last_segment,
    ggml_type hidden_type,
    uint32_t fixed_shared_n_kv,
    bool dynamic_kv_write
) {
    (void)token;
    (void)is_last_segment;
    return build_segment_graph_no_reset(
        1,
        pos,
        slot_idx,
        segment,
        is_first_segment,
        hidden_type,
        fixed_shared_n_kv,
        dynamic_kv_write
    );
}

ggml_cgraph* Qwen35moeForwardPass::build_output_head_graph_from_hidden(ggml_type hidden_type) {
    reset_context();
    return build_output_head_graph_from_hidden_no_reset(hidden_type);
}

ggml_cgraph* Qwen35moeForwardPass::build_output_head_graph_from_hidden_no_reset(ggml_type hidden_type) {
    ggml_cgraph* gf = new_graph();
    const int64_t n_embd = static_cast<int64_t>(model_->meta_->qwen35moe.embedding_length);
    ggml_tensor* inpL = ggml_new_tensor_2d(ctx_, hidden_type, n_embd, 1);
    ggml_set_input(inpL);
    set_tensor_name(inpL, "segment_hidden_in");
    ggml_build_forward_expand(gf, inpL);
    build_output_head(gf, inpL);
    return gf;
}

void Qwen35moeForwardPass::set_segment_inputs(
    ggml_cgraph* gf,
    const std::vector<int32_t>& tokens,
    int pos,
    ggml_type hidden_type,
    const std::vector<uint8_t>* hidden_data,
    const LayerSegment* segment,
    uint32_t layer_begin,
    uint32_t layer_end
) {
    uint32_t n_tok = 1;
    if (ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos")) {
        n_tok = static_cast<uint32_t>(pos_t->ne[0]);
    }

    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (hidden_data == nullptr && !tok_t) {
        throw std::runtime_error("Qwen35moeForwardPass::set_segment_inputs: missing both hidden_data and tokens tensor for first segment");
    }
    if (tok_t) {
        if (tok_t->type != GGML_TYPE_I32 ||
            static_cast<size_t>(ggml_nbytes(tok_t)) != static_cast<size_t>(n_tok) * sizeof(int32_t)) {
            throw std::runtime_error("Qwen35moeForwardPass::set_segment_inputs: tokens tensor mismatch");
        }
        if (tokens.size() != n_tok) {
            throw std::runtime_error("Qwen35moeForwardPass::set_segment_inputs: token count mismatch");
        }
        maybe_log_segment_tensor("tokens", segment, tok_t);
        ggml_backend_tensor_set(tok_t, tokens.data(), 0, static_cast<size_t>(n_tok) * sizeof(int32_t));
    }
    if (ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos")) {
        if (pos_t->type != GGML_TYPE_I32 ||
            static_cast<size_t>(ggml_nbytes(pos_t)) != static_cast<size_t>(n_tok) * sizeof(int32_t)) {
            throw std::runtime_error("Qwen35moeForwardPass::set_segment_inputs: inp_pos tensor mismatch");
        }
        std::vector<int32_t> pos_data(n_tok);
        for (uint32_t i = 0; i < n_tok; ++i) {
            pos_data[i] = pos + static_cast<int>(i);
        }
        ggml_backend_tensor_set(pos_t, pos_data.data(), 0, static_cast<size_t>(n_tok) * sizeof(int32_t));
    }
    if (ggml_tensor* hidden_t = ggml_graph_get_tensor(gf, "segment_hidden_in")) {
        if (hidden_data == nullptr) {
            throw std::runtime_error("Qwen35moeForwardPass::set_segment_inputs: hidden input missing");
        }
        validate_handoff_tensor("Qwen35moeForwardPass::set_segment_inputs", hidden_t, hidden_type, hidden_data->size());
        maybe_log_segment_tensor("segment_hidden_in", segment, hidden_t);
        ggml_backend_tensor_set(hidden_t, hidden_data->data(), 0, hidden_data->size());
    }

    if (ggml_tensor* shared_kq_mask = ggml_graph_get_tensor(gf, "kq_mask_shared")) {
        const uint32_t n_kv = static_cast<uint32_t>(shared_kq_mask->ne[0]);
        const uint32_t mask_n_tok = static_cast<uint32_t>(shared_kq_mask->ne[1]);
        if (mask_n_tok != n_tok) {
            throw std::runtime_error("Qwen35moeForwardPass::set_segment_inputs: shared mask token dimension mismatch");
        }
        std::vector<float> mask;
        if (n_tok > 1) {
            fill_dynamic_prefill_mask(mask, n_kv, n_tok, pos);
        } else {
            mask.resize(n_kv);
            for (uint32_t j = 0; j < n_kv; ++j) {
                mask[j] = (j <= static_cast<uint32_t>(pos)) ? 0.0f : -INFINITY;
            }
        }
        set_mask_data(shared_kq_mask, mask);
        return;
    }

    for (uint32_t il = layer_begin; il < layer_end; ++il) {
        if (!is_full_attention_layer(il)) {
            continue;
        }
        char name[32];
        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, name);
        if (!kq_mask) {
            continue;
        }
        const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        if (n_tok > 1) {
            const uint32_t mask_n_tok = static_cast<uint32_t>(kq_mask->ne[1]);
            std::vector<float> mask;
            fill_dynamic_prefill_mask(mask, n_kv, mask_n_tok, pos);
            set_mask_data(kq_mask, mask);
            continue;
        }
        std::vector<float> mask(n_kv, -INFINITY);
        for (uint32_t j = 0; j <= static_cast<uint32_t>(pos) && j < n_kv; ++j) {
            mask[j] = 0.0f;
        }
        set_mask_data(kq_mask, mask);
    }
}

bool Qwen35moeForwardPass::segment_has_split_layers(const LayerSegment& segment) const {
    for (uint32_t il = segment.l0; il < segment.l1; ++il) {
        if (model_->layer_has_split_devices(static_cast<int>(il))) {
            return true;
        }
    }
    return false;
}

uint32_t Qwen35moeForwardPass::count_split_layers() const {
    uint32_t count = 0;
    for (const LayerSegment& segment : layer_segments_) {
        for (uint32_t il = segment.l0; il < segment.l1; ++il) {
            if (model_->layer_has_split_devices(static_cast<int>(il))) {
                ++count;
            }
        }
    }
    return count;
}

ggml_cgraph* Qwen35moeForwardPass::build_split_moe_ffn_graph_no_reset(
    ggml_context* ctx,
    int il,
    ggml_type hidden_type,
    ggml_tensor** moe_norm_in,
    ggml_tensor** ffn_res_in,
    ggml_tensor** hidden_out
) {
    const int64_t n_embd = static_cast<int64_t>(model_->meta_->qwen35moe.embedding_length);
    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* norm_in = ggml_new_tensor_2d(ctx, hidden_type, n_embd, 1);
    ggml_set_input(norm_in);
    set_tensor_name(norm_in, "moe_norm_in", il);
    ggml_build_forward_expand(gf, norm_in);

    ggml_tensor* ffn_in = ggml_new_tensor_2d(ctx, hidden_type, n_embd, 1);
    ggml_set_input(ffn_in);
    set_tensor_name(ffn_in, "ffn_residual_in", il);
    ggml_build_forward_expand(gf, ffn_in);

    ggml_context* saved_ctx = ctx_;
    ctx_ = ctx;
    ggml_tensor* routed = build_moe_routed_experts(ctx, gf, norm_in, il);
    ggml_build_forward_expand(gf, routed);
    ggml_tensor* out = build_moe_shared_combine(ctx, gf, norm_in, routed, il, ffn_in);
    set_tensor_name(out, "segment_hidden_out");
    ggml_build_forward_expand(gf, out);
    ctx_ = saved_ctx;

    if (moe_norm_in) {
        *moe_norm_in = norm_in;
    }
    if (ffn_res_in) {
        *ffn_res_in = ffn_in;
    }
    if (hidden_out) {
        *hidden_out = out;
    }
    return gf;
}

void Qwen35moeForwardPass::run_segment_forward(
    ggml_cgraph* gf,
    const LayerSegment& segment,
    const char* where,
    const std::function<void()>& set_inputs,
    bool single_backend_only,
    const std::function<void()>& after_compute
) {
    // Mixed-mode segments reference weights/KV/DeltaNet state on multiple backends.
    // Single-backend gallocr + graph_compute cannot route those ops; use the model
    // scheduler (same as unified prefill) on the segment subgraph instead.
    const bool use_scheduler =
        !single_backend_only &&
        (model_->is_mixed_mode() || segment_has_split_layers(segment));
    if (use_scheduler) {
        ggml_backend_sched_t scheduler = model_->get_scheduler();
        if (!scheduler) {
            throw std::runtime_error(
                std::string(where) + ": mixed segment requires scheduler backend");
        }
        ggml_backend_sched_reset(scheduler);
        decode_graph_scheduler_reset_count_++;
        if (!ggml_backend_sched_alloc_graph(scheduler, gf)) {
            throw std::runtime_error(
                std::string(where) + ": segment scheduler alloc failed");
        }
        set_inputs();
        ggml_backend_sched_graph_compute(scheduler, gf);
        if (after_compute) {
            after_compute();
        }
        return;
    }

    const ggml_backend_buffer_type_t seg_buft =
        ggml_backend_get_default_buffer_type(segment.backend);
    ggml_gallocr_t seg_alloc = ggml_gallocr_new(seg_buft);
    if (!ggml_gallocr_alloc_graph(seg_alloc, gf)) {
        ggml_gallocr_free(seg_alloc);
        throw std::runtime_error(std::string(where) + ": segment alloc failed");
    }
    set_inputs();
    ggml_backend_graph_compute(segment.backend, gf);
    if (after_compute) {
        after_compute();
    }
    ggml_gallocr_free(seg_alloc);
}

void Qwen35moeForwardPass::read_named_layer_tensor(
    ggml_cgraph* gf,
    const char* base_name,
    int il,
    std::vector<uint8_t>& data,
    ggml_type& type
) const {
    char name[128];
    std::snprintf(name, sizeof(name), "%s.%d", base_name, il);
    ggml_tensor* tensor = ggml_graph_get_tensor(gf, name);
    if (!tensor) {
        throw std::runtime_error(
            std::string("Qwen35moeForwardPass::read_named_layer_tensor: missing ") + name);
    }
    type = tensor->type;
    const size_t nbytes = static_cast<size_t>(ggml_nbytes(tensor));
    if (nbytes == 0) {
        throw std::runtime_error(
            std::string("Qwen35moeForwardPass::read_named_layer_tensor: zero bytes for ") + name);
    }
    data.resize(nbytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, nbytes);
}

void Qwen35moeForwardPass::run_split_moe_layer_forward(
    int il,
    uint32_t n_tok,
    int pos,
    uint32_t slot_idx,
    bool first_uses_token_input,
    const std::vector<int32_t>* tokens,
    std::vector<uint8_t>& hidden_data,
    ggml_type& hidden_type
) {
    const LayerSegment layer_seg{
        static_cast<uint32_t>(il),
        static_cast<uint32_t>(il + 1),
        layer_backend(il)
    };

    // Phase 1: GPU attention + pre-FFN norm (no mul_mat_id in graph).
    reset_context();
    ggml_cgraph* gf_pre = build_segment_graph_no_reset(
        n_tok,
        pos,
        slot_idx,
        layer_seg,
        first_uses_token_input,
        hidden_type,
        0,
        false,
        LayerMoeMode::PreMoeOnly
    );

    std::vector<uint8_t> moe_norm_data;
    std::vector<uint8_t> ffn_residual_data;
    ggml_type moe_norm_type = hidden_type;

    run_segment_forward(
        gf_pre,
        layer_seg,
        "Qwen35moeForwardPass::split_moe premoe",
        [&]() {
            if (first_uses_token_input) {
                if (!tokens || tokens->size() != n_tok) {
                    throw std::runtime_error("Qwen35moeForwardPass::run_split_moe_layer_forward: token mismatch");
                }
                set_segment_inputs(
                    gf_pre,
                    *tokens,
                    pos,
                    hidden_type,
                    nullptr,
                    &layer_seg,
                    layer_seg.l0,
                    layer_seg.l1
                );
            } else {
                set_segment_inputs(
                    gf_pre,
                    tokens ? *tokens : std::vector<int32_t>(n_tok, 0),
                    pos,
                    hidden_type,
                    &hidden_data,
                    &layer_seg,
                    layer_seg.l0,
                    layer_seg.l1
                );
            }
        },
        true,
        [&]() {
            read_named_layer_tensor(gf_pre, "moe_norm_in", il, moe_norm_data, moe_norm_type);
            read_named_layer_tensor(gf_pre, "ffn_residual_in", il, ffn_residual_data, hidden_type);
        }
    );

    // Phase 2+3: routed experts (CPU) + shared/residual (GPU) in one isolated MoE graph.
    reset_context();
    const int64_t n_embd = static_cast<int64_t>(model_->meta_->qwen35moe.embedding_length);
    ggml_cgraph* gf_moe = new_graph();

    ggml_tensor* moe_norm_in = ggml_new_tensor_2d(ctx_, moe_norm_type, n_embd, static_cast<int64_t>(n_tok));
    ggml_set_input(moe_norm_in);
    set_tensor_name(moe_norm_in, "moe_norm_in", il);
    ggml_build_forward_expand(gf_moe, moe_norm_in);

    ggml_tensor* ffn_residual_in = ggml_new_tensor_2d(ctx_, hidden_type, n_embd, static_cast<int64_t>(n_tok));
    ggml_set_input(ffn_residual_in);
    set_tensor_name(ffn_residual_in, "ffn_residual_in", il);
    ggml_build_forward_expand(gf_moe, ffn_residual_in);

    ggml_tensor* routed = build_moe_routed_experts(ctx_, gf_moe, moe_norm_in, il);
    ggml_build_forward_expand(gf_moe, routed);

    ggml_tensor* layer_out = build_moe_shared_combine(
        ctx_, gf_moe, moe_norm_in, routed, il, ffn_residual_in);
    set_tensor_name(layer_out, "segment_hidden_out");
    ggml_build_forward_expand(gf_moe, layer_out);

    run_segment_forward(gf_moe, layer_seg, "Qwen35moeForwardPass::split_moe ffns", [&]() {
        ggml_backend_tensor_set(moe_norm_in, moe_norm_data.data(), 0, moe_norm_data.size());
        ggml_backend_tensor_set(ffn_residual_in, ffn_residual_data.data(), 0, ffn_residual_data.size());
    });

    read_segment_hidden_out(
        "Qwen35moeForwardPass::run_split_moe_layer_forward",
        &layer_seg,
        gf_moe,
        hidden_data,
        hidden_type
    );
}

void Qwen35moeForwardPass::run_prefill_segment_with_split_layers(
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx,
    const LayerSegment& segment,
    bool first_uses_token_input,
    std::vector<uint8_t>& hidden_data,
    ggml_type& hidden_type
) {
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    bool use_token_input = first_uses_token_input;
    for (uint32_t il = segment.l0; il < segment.l1; ++il) {
        if (model_->layer_has_split_devices(static_cast<int>(il))) {
            run_split_moe_layer_forward(
                static_cast<int>(il),
                n_tok,
                pos,
                slot_idx,
                use_token_input,
                &tokens,
                hidden_data,
                hidden_type
            );
            use_token_input = false;
            continue;
        }

        const LayerSegment layer_seg{ il, il + 1, segment.backend };
        reset_context();
        ggml_cgraph* gf = build_segment_graph_no_reset(
            n_tok,
            pos,
            slot_idx,
            layer_seg,
            use_token_input,
            hidden_type,
            0,
            false,
            LayerMoeMode::Full
        );
        run_segment_forward(gf, layer_seg, "Qwen35moeForwardPass::prefill split segment layer", [&]() {
            set_segment_inputs(
                gf,
                tokens,
                pos,
                hidden_type,
                use_token_input ? nullptr : &hidden_data,
                &layer_seg,
                layer_seg.l0,
                layer_seg.l1
            );
        });
        read_segment_hidden_out(
            "Qwen35moeForwardPass::run_prefill_segment_with_split_layers",
            &layer_seg,
            gf,
            hidden_data,
            hidden_type
        );
        use_token_input = false;
    }
}

std::vector<float> Qwen35moeForwardPass::run_prefill_segmented_eager(
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx
) {
    if (tokens.empty()) {
        return {};
    }
    if (layer_segments_.empty()) {
        auto backend = model_->get_curr_backend();
        auto buf_type = ggml_backend_get_default_buffer_type(backend);
        ggml_gallocr_t allocr_prefill = ggml_gallocr_new(buf_type);
        std::vector<float> result;
        run_prefill_microbatched_direct(
            allocr_prefill, backend, tokens, pos, slot_idx, false, &result, nullptr, false);
        if (allocr_prefill) {
            ggml_gallocr_free(allocr_prefill);
        }
        return result;
    }

    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    const int64_t n_embd = static_cast<int64_t>(model_->meta_->qwen35moe.embedding_length);

    std::vector<uint8_t> hidden_data;
    ggml_type hidden_type = GGML_TYPE_F32;
    for (size_t i = 0; i < layer_segments_.size(); ++i) {
        const bool first = i == 0;
        const LayerSegment& segment = layer_segments_[i];
        const bool first_uses_token_input = first && can_run_token_embedding_on_backend(segment.backend);
        if (first && !first_uses_token_input) {
            prepare_first_segment_hidden_from_tokens(
                "Qwen35moeForwardPass::run_prefill_segmented",
                tokens,
                segment,
                hidden_data,
                hidden_type
            );
        }

        if (segment_has_split_layers(segment)) {
            run_prefill_segment_with_split_layers(
                tokens,
                pos,
                slot_idx,
                segment,
                first_uses_token_input,
                hidden_data,
                hidden_type
            );
            continue;
        }

        ggml_cgraph* gf = build_prefill_segment_graph(
            n_tok,
            pos,
            slot_idx,
            segment,
            first_uses_token_input,
            hidden_type
        );

        run_segment_forward(gf, segment, "Qwen35moeForwardPass::run_prefill_segmented", [&]() {
            set_segment_inputs(
                gf,
                tokens,
                pos,
                hidden_type,
                first_uses_token_input ? nullptr : &hidden_data,
                &segment,
                segment.l0,
                segment.l1
            );
        });
        read_segment_hidden_out("Qwen35moeForwardPass::run_prefill_segmented", &segment, gf, hidden_data, hidden_type);
    }

    std::vector<uint8_t> head_hidden = hidden_data;
    if (n_tok > 1) {
        extract_last_token_hidden(hidden_data, hidden_type, n_embd, n_tok, head_hidden);
    }

    ggml_cgraph* gf_head = build_output_head_graph_from_hidden(hidden_type);
    ggml_backend_t head_backend = model_->get_curr_backend();
    ggml_gallocr_t head_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(head_backend));
    if (!ggml_gallocr_alloc_graph(head_alloc, gf_head)) {
        ggml_gallocr_free(head_alloc);
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_segmented: head alloc failed");
    }
    if (ggml_tensor* hidden_in = ggml_graph_get_tensor(gf_head, "segment_hidden_in")) {
        if (head_hidden.empty()) {
            ggml_gallocr_free(head_alloc);
            throw std::runtime_error("Qwen35moeForwardPass::run_prefill_segmented: head hidden mismatch");
        }
        validate_handoff_tensor(
            "Qwen35moeForwardPass::run_prefill_segmented: head",
            hidden_in,
            hidden_type,
            head_hidden.size()
        );
        maybe_log_segment_tensor("head_segment_hidden_in", nullptr, hidden_in);
        ggml_backend_tensor_set(hidden_in, head_hidden.data(), 0, head_hidden.size());
    }
    ggml_backend_graph_compute(head_backend, gf_head);
    ggml_tensor* head_logits = ggml_graph_get_tensor(gf_head, "logits");
    if (!head_logits) {
        ggml_gallocr_free(head_alloc);
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_segmented: logits tensor missing in head graph");
    }
    maybe_log_segment_tensor("head_logits", nullptr, head_logits);
    std::vector<float> logits = get_output_logits(gf_head);
    ggml_gallocr_free(head_alloc);
    advance_cache(n_tok, slot_idx);
    maybe_log_decode_graph_stats();
    return logits;
}

TopKSampleCandidates Qwen35moeForwardPass::run_prefill_segmented_topk_eager(
    const std::vector<int32_t>& tokens,
    int pos,
    uint32_t slot_idx
) {
    if (sampling_top_k_ <= 0) {
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_segmented_topk: device sampling not configured");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_segmented_topk: empty token input");
    }
    if (layer_segments_.empty()) {
        auto backend = model_->get_curr_backend();
        auto buf_type = ggml_backend_get_default_buffer_type(backend);
        ggml_gallocr_t allocr_prefill = ggml_gallocr_new(buf_type);
        TopKSampleCandidates result;
        run_prefill_microbatched_direct(
            allocr_prefill, backend, tokens, pos, slot_idx, true, nullptr, &result, false);
        if (allocr_prefill) {
            ggml_gallocr_free(allocr_prefill);
        }
        return result;
    }

    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    const int64_t n_embd = static_cast<int64_t>(model_->meta_->qwen35moe.embedding_length);

    std::vector<uint8_t> hidden_data;
    ggml_type hidden_type = GGML_TYPE_F32;
    for (size_t i = 0; i < layer_segments_.size(); ++i) {
        const bool first = i == 0;
        const LayerSegment& segment = layer_segments_[i];
        const bool first_uses_token_input = first && can_run_token_embedding_on_backend(segment.backend);
        if (first && !first_uses_token_input) {
            prepare_first_segment_hidden_from_tokens(
                "Qwen35moeForwardPass::run_prefill_segmented_topk",
                tokens,
                segment,
                hidden_data,
                hidden_type
            );
        }

        if (segment_has_split_layers(segment)) {
            run_prefill_segment_with_split_layers(
                tokens,
                pos,
                slot_idx,
                segment,
                first_uses_token_input,
                hidden_data,
                hidden_type
            );
            continue;
        }

        ggml_cgraph* gf = build_prefill_segment_graph(
            n_tok,
            pos,
            slot_idx,
            segment,
            first_uses_token_input,
            hidden_type
        );

        run_segment_forward(gf, segment, "Qwen35moeForwardPass::run_prefill_segmented_topk", [&]() {
            set_segment_inputs(
                gf,
                tokens,
                pos,
                hidden_type,
                first_uses_token_input ? nullptr : &hidden_data,
                &segment,
                segment.l0,
                segment.l1
            );
        });
        read_segment_hidden_out("Qwen35moeForwardPass::run_prefill_segmented_topk", &segment, gf, hidden_data, hidden_type);
    }

    std::vector<uint8_t> head_hidden = hidden_data;
    if (n_tok > 1) {
        extract_last_token_hidden(hidden_data, hidden_type, n_embd, n_tok, head_hidden);
    }

    ggml_cgraph* gf_head = build_output_head_graph_from_hidden(hidden_type);
    ggml_backend_t head_backend = model_->get_curr_backend();
    ggml_gallocr_t head_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(head_backend));
    if (!ggml_gallocr_alloc_graph(head_alloc, gf_head)) {
        ggml_gallocr_free(head_alloc);
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_segmented_topk: head alloc failed");
    }
    if (ggml_tensor* hidden_in = ggml_graph_get_tensor(gf_head, "segment_hidden_in")) {
        if (head_hidden.empty()) {
            ggml_gallocr_free(head_alloc);
            throw std::runtime_error("Qwen35moeForwardPass::run_prefill_segmented_topk: head hidden mismatch");
        }
        validate_handoff_tensor(
            "Qwen35moeForwardPass::run_prefill_segmented_topk: head",
            hidden_in,
            hidden_type,
            head_hidden.size()
        );
        maybe_log_segment_tensor("head_segment_hidden_in", nullptr, hidden_in);
        ggml_backend_tensor_set(hidden_in, head_hidden.data(), 0, head_hidden.size());
    }
    ggml_backend_graph_compute(head_backend, gf_head);
    ggml_tensor* head_logits = ggml_graph_get_tensor(gf_head, "logits");
    if (!head_logits) {
        ggml_gallocr_free(head_alloc);
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_segmented_topk: logits tensor missing in head graph");
    }
    maybe_log_segment_tensor("head_logits", nullptr, head_logits);
    TopKSampleCandidates result = get_output_topk_candidates(gf_head, 0);
    ggml_gallocr_free(head_alloc);
    advance_cache(n_tok, slot_idx);
    maybe_log_decode_graph_stats();
    return result;
}

std::vector<float> Qwen35moeForwardPass::run_decode_segmented(int32_t token, int pos, uint32_t slot_idx) {
    if (!segmented_decode_cache_enabled_) {
        return run_decode_segmented_eager(token, pos, slot_idx);
    }
    try {
        return run_decode_segmented_cached(token, pos, slot_idx);
    } catch (const std::exception& ex) {
        segmented_decode_fallback_count_++;
        if (!segmented_decode_cache_fallback_warned_) {
            std::fprintf(stderr,
                "[PERF][segmented-decode-cache] disabled after cache failure: %s\n",
                ex.what());
            segmented_decode_cache_fallback_warned_ = true;
        }
        clear_segmented_decode_cache();
        segmented_decode_cache_enabled_ = false;
        return run_decode_segmented_eager(token, pos, slot_idx);
    }
}

std::vector<float> Qwen35moeForwardPass::run_decode_segmented_eager(int32_t token, int pos, uint32_t slot_idx) {
    if (layer_segments_.empty()) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill(token_vec, pos, slot_idx, nullptr);
    }

    std::vector<uint8_t> hidden_data;
    ggml_type hidden_type = GGML_TYPE_F32;
    for (size_t i = 0; i < layer_segments_.size(); ++i) {
        const bool first = i == 0;
        const LayerSegment& segment = layer_segments_[i];
        const bool first_uses_token_input = first && can_run_token_embedding_on_backend(segment.backend);
        if (first && !first_uses_token_input) {
            prepare_first_segment_hidden_from_token(
                "Qwen35moeForwardPass::run_decode_segmented",
                token,
                segment,
                hidden_data,
                hidden_type
            );
        }

        if (segment_has_split_layers(segment)) {
            run_prefill_segment_with_split_layers(
                std::vector<int32_t>{ token },
                pos,
                slot_idx,
                segment,
                first_uses_token_input,
                hidden_data,
                hidden_type
            );
            continue;
        }

        ggml_cgraph* gf = build_decode_segment_graph(token, pos, slot_idx, segment, first_uses_token_input, false, hidden_type);

        run_segment_forward(gf, segment, "Qwen35moeForwardPass::run_decode_segmented", [&]() {
            set_segment_inputs(
                gf,
                std::vector<int32_t>{ token },
                pos,
                hidden_type,
                first_uses_token_input ? nullptr : &hidden_data,
                &segment,
                segment.l0,
                segment.l1
            );
        });
        read_segment_hidden_out("Qwen35moeForwardPass::run_decode_segmented", &segment, gf, hidden_data, hidden_type);
    }

    ggml_cgraph* gf_head = build_output_head_graph_from_hidden(hidden_type);
    ggml_backend_t head_backend = model_->get_curr_backend();
    ggml_gallocr_t head_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(head_backend));
    if (!ggml_gallocr_alloc_graph(head_alloc, gf_head)) {
        ggml_gallocr_free(head_alloc);
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented: head alloc failed");
    }
    if (ggml_tensor* hidden_in = ggml_graph_get_tensor(gf_head, "segment_hidden_in")) {
        if (hidden_data.empty()) {
            ggml_gallocr_free(head_alloc);
            throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented: head hidden mismatch");
        }
        validate_handoff_tensor("Qwen35moeForwardPass::run_decode_segmented: head", hidden_in, hidden_type, hidden_data.size());
        maybe_log_segment_tensor("head_segment_hidden_in", nullptr, hidden_in);
        ggml_backend_tensor_set(hidden_in, hidden_data.data(), 0, hidden_data.size());
    }
    const size_t head_reserve = ggml_gallocr_get_buffer_size(head_alloc, 0);
    //std::fprintf(stderr, "[dev-mode=auto] segmented reserve output_head backend=%s bytes=%zu\n", ggml_backend_name(head_backend), head_reserve);
    ggml_backend_graph_compute(head_backend, gf_head);
    ggml_tensor* head_logits = ggml_graph_get_tensor(gf_head, "logits");
    if (!head_logits) {
        ggml_gallocr_free(head_alloc);
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented: logits tensor missing in head graph");
    }
    maybe_log_segment_tensor("head_logits", nullptr, head_logits);
    std::vector<float> logits = get_output_logits(gf_head);
    ggml_gallocr_free(head_alloc);
    advance_cache(1, slot_idx);
    return logits;
}

std::vector<float> Qwen35moeForwardPass::run_decode_segmented_cached(int32_t token, int pos, uint32_t slot_idx) {
    if (layer_segments_.empty()) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill(token_vec, pos, slot_idx, nullptr);
    }

    const uint32_t required_kv = static_cast<uint32_t>(pos) + 1;
    DecodeGraphSignature requested_signature;
    requested_signature.slot_idx = slot_idx;
    requested_signature.context_len = context_len_;
    requested_signature.n_batch_tokens = n_batch_tokens_;
    requested_signature.n_ubatch_tokens = n_ubatch_tokens_;
    requested_signature.use_flash_attention = use_flash_attention_;
    requested_signature.paged_fused_decode = paged_fused_decode_active();
    requested_signature.is_mixed_mode = model_->is_mixed_mode();
    requested_signature.device_map_hash = model_->compute_device_map_hash();
    requested_signature.kv_capacity = decode_cache_bucket_capacity(required_kv);

    ensure_segmented_decode_cache(requested_signature, required_kv);
    if (!ensure_segmented_decode_copy_ready(slot_idx, static_cast<uint32_t>(pos))) {
        segmented_decode_fallback_count_++;
        return run_decode_segmented_eager(token, pos, slot_idx);
    }

    SegmentedDecodeSlotCache& slot_cache = segmented_decode_slot_cache(slot_idx);

    if (slot_cache.use_split_decode) {
        return run_decode_split_layers_cached(token, pos, slot_idx, slot_cache);
    }

    ggml_type hidden_type = GGML_TYPE_F32;
    ggml_tensor* hidden_tensor = nullptr;

    // Track the backend that produced each `hidden_tensor` so we can issue an
    // async cross-backend copy for the next consumer. When backends are the
    // same and CUDA, ggml_backend_tensor_copy_async chains via stream events
    // (see ggml_backend_cuda_cpy_tensor_async); for cross-device transfers it
    // falls back to a sync copy with backend-level synchronize. Either way is
    // correct; the win is when consecutive segments live on the same CUDA
    // device, where this avoids the implicit per-stage synchronize we'd
    // otherwise pay inside ggml_backend_graph_compute.
    ggml_backend_t hidden_backend = nullptr;

    if (slot_cache.embed_cache.valid) {
        set_cached_segment_inputs(
            slot_cache.embed_cache,
            token,
            pos,
            slot_idx,
            hidden_type,
            nullptr,
            &layer_segments_.front()
        );
        ggml_backend_graph_compute_async(
            slot_cache.embed_cache.backend,
            slot_cache.embed_cache.graph
        );
        hidden_tensor = slot_cache.embed_cache.hidden_out_tensor;
        if (!hidden_tensor) {
            throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_cached: missing embedding hidden output");
        }
        hidden_type = hidden_tensor->type;
        hidden_backend = slot_cache.embed_cache.backend;
        maybe_log_segment_tensor("segment_hidden_out", &layer_segments_.front(), hidden_tensor);
    }

    for (size_t i = 0; i < slot_cache.segment_caches.size(); ++i) {
        const LayerSegment& segment = layer_segments_[i];
        SegmentedDecodeGraphCache& cache = slot_cache.segment_caches[i];
        set_cached_segment_inputs(
            cache,
            token,
            pos,
            slot_idx,
            hidden_type,
            cache.hidden_in_tensor ? hidden_tensor : nullptr,
            &segment
        );
        ggml_backend_graph_compute_async(cache.backend, cache.graph);
        hidden_tensor = cache.hidden_out_tensor;
        if (!hidden_tensor) {
            throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_cached: missing segment hidden output");
        }
        hidden_type = hidden_tensor->type;
        hidden_backend = cache.backend;
        maybe_log_segment_tensor("segment_hidden_out", &segment, hidden_tensor);
    }

    SegmentedDecodeGraphCache& head = slot_cache.head_cache;
    if (!hidden_tensor) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_cached: head hidden mismatch");
    }
    if (hidden_backend) {
        ggml_backend_synchronize(hidden_backend);
    }
    copy_handoff_tensor_async(
        "Qwen35moeForwardPass::run_decode_segmented_cached: head",
        hidden_backend,
        head.backend,
        hidden_tensor,
        head.hidden_in_tensor,
        hidden_type
    );

    ggml_backend_graph_compute_async(head.backend, head.graph);
    std::vector<float> logits = get_output_logits(head.graph, head.backend);
    if (!commit_segmented_decode_step(slot_idx, static_cast<uint32_t>(pos))) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_cached: failed to commit decode step");
    }
    return logits;
}

void Qwen35moeForwardPass::run_split_decode_body_eager(
    int32_t token,
    int pos,
    uint32_t slot_idx,
    std::vector<uint8_t>& hidden_data,
    ggml_type& hidden_type
) {
    hidden_type = GGML_TYPE_F32;
    hidden_data.clear();
    for (size_t i = 0; i < layer_segments_.size(); ++i) {
        const bool first = i == 0;
        const LayerSegment& segment = layer_segments_[i];
        const bool first_uses_token_input = first && can_run_token_embedding_on_backend(segment.backend);
        if (first && !first_uses_token_input) {
            prepare_first_segment_hidden_from_token(
                "Qwen35moeForwardPass::run_split_decode_body_eager",
                token,
                segment,
                hidden_data,
                hidden_type
            );
        }

        if (segment_has_split_layers(segment)) {
            run_prefill_segment_with_split_layers(
                std::vector<int32_t>{ token },
                pos,
                slot_idx,
                segment,
                first_uses_token_input,
                hidden_data,
                hidden_type
            );
            continue;
        }

        ggml_cgraph* gf = build_decode_segment_graph(token, pos, slot_idx, segment, first_uses_token_input, false, hidden_type);
        run_segment_forward(gf, segment, "Qwen35moeForwardPass::run_split_decode_body_eager", [&]() {
            set_segment_inputs(
                gf,
                std::vector<int32_t>{ token },
                pos,
                hidden_type,
                first_uses_token_input ? nullptr : &hidden_data,
                &segment,
                segment.l0,
                segment.l1
            );
        });
        read_segment_hidden_out(
            "Qwen35moeForwardPass::run_split_decode_body_eager",
            &segment,
            gf,
            hidden_data,
            hidden_type
        );
    }
}

void Qwen35moeForwardPass::run_split_decode_body_cached(
    int32_t token,
    int pos,
    uint32_t slot_idx,
    SegmentedDecodeSlotCache& slot_cache,
    ggml_tensor*& hidden_out,
    ggml_backend_t& hidden_backend,
    ggml_type& hidden_type
) {
    hidden_out = nullptr;
    hidden_backend = nullptr;
    hidden_type = GGML_TYPE_F32;

    ggml_tensor* hidden_tensor = nullptr;

    if (slot_cache.embed_cache.valid) {
        set_cached_segment_inputs(
            slot_cache.embed_cache,
            token,
            pos,
            slot_idx,
            hidden_type,
            nullptr,
            &layer_segments_.front()
        );
        ggml_backend_graph_compute_async(
            slot_cache.embed_cache.backend,
            slot_cache.embed_cache.graph
        );
        hidden_tensor = slot_cache.embed_cache.hidden_out_tensor;
        if (!hidden_tensor) {
            throw std::runtime_error(
                "Qwen35moeForwardPass::run_split_decode_body_cached: missing embedding hidden output");
        }
        hidden_type = hidden_tensor->type;
        hidden_backend = slot_cache.embed_cache.backend;
    }

    const auto find_split_layer_cache = [&](int layer_idx) -> SplitMoeDecodeLayerCache* {
        for (SplitMoeDecodeLayerCache& lc : slot_cache.split_layer_caches) {
            if (lc.layer_idx == layer_idx) {
                return &lc;
            }
        }
        return nullptr;
    };

    const auto run_cached_split_layer = [&](SplitMoeDecodeLayerCache& lc, const LayerSegment& layer_seg) {
        if (!lc.premoe_moe_norm_out || !lc.premoe_ffn_res_out) {
            throw std::runtime_error(
                "Qwen35moeForwardPass::run_split_decode_body_cached: missing premoe handoff tensors");
        }

        set_cached_segment_inputs(
            lc.premoe,
            token,
            pos,
            slot_idx,
            hidden_type,
            hidden_tensor,
            &layer_seg
        );

        ggml_backend_graph_compute_async(lc.premoe.backend, lc.premoe.graph);
        ggml_backend_synchronize(lc.premoe.backend);

        ggml_backend_t moe_norm_backend = find_backend_for_tensor(lc.moe_norm_in);
        ggml_backend_t ffn_res_backend = find_backend_for_tensor(lc.moe_ffn_res_in);
        copy_handoff_tensor_async(
            "Qwen35moeForwardPass::run_split_decode_body_cached: premoe→moe norm",
            lc.premoe.backend,
            moe_norm_backend,
            lc.premoe_moe_norm_out,
            lc.moe_norm_in,
            lc.moe_norm_type
        );
        copy_handoff_tensor_async(
            "Qwen35moeForwardPass::run_split_decode_body_cached: premoe→moe residual",
            lc.premoe.backend,
            ffn_res_backend,
            lc.premoe_ffn_res_out,
            lc.moe_ffn_res_in,
            lc.ffn_res_type
        );

        if (!lc.moe_scheduler) {
            throw std::runtime_error(
                "Qwen35moeForwardPass::run_split_decode_body_cached: missing MoE scheduler");
        }
        ggml_backend_sched_graph_compute_async(lc.moe_scheduler, lc.moe_graph);

        if (!lc.hidden_out) {
            throw std::runtime_error(
                "Qwen35moeForwardPass::run_split_decode_body_cached: missing MoE hidden output");
        }
        hidden_tensor = lc.hidden_out;
        hidden_type = lc.hidden_out->type;
        hidden_backend = find_backend_for_tensor(lc.hidden_out);
        split_moe_last_sched_layer_idx_ = lc.layer_idx;
    };

    std::vector<uint8_t> bootstrap_hidden;
    bool have_bootstrap_hidden = false;

    for (size_t i = 0; i < layer_segments_.size(); ++i) {
        const bool first = i == 0;
        const LayerSegment& segment = layer_segments_[i];
        const bool first_uses_token_input = first && can_run_token_embedding_on_backend(segment.backend);
        if (first && !first_uses_token_input && !slot_cache.embed_cache.valid) {
            prepare_first_segment_hidden_from_token(
                "Qwen35moeForwardPass::run_split_decode_body_cached",
                token,
                segment,
                bootstrap_hidden,
                hidden_type
            );
            if (bootstrap_hidden.empty()) {
                throw std::runtime_error(
                    "Qwen35moeForwardPass::run_split_decode_body_cached: missing bootstrap hidden");
            }
            have_bootstrap_hidden = true;
        }

        if (segment_has_split_layers(segment)) {
            for (uint32_t il = segment.l0; il < segment.l1; ++il) {
                if (!model_->layer_has_split_devices(static_cast<int>(il))) {
                    continue;
                }
                SplitMoeDecodeLayerCache* lc = find_split_layer_cache(static_cast<int>(il));
                if (!lc) {
                    throw std::runtime_error(
                        "Qwen35moeForwardPass::run_split_decode_body_cached: missing split layer cache");
                }
                const LayerSegment layer_seg{ il, il + 1, segment.backend };
                run_cached_split_layer(*lc, layer_seg);
            }
            continue;
        }

        SegmentedDecodeGraphCache& cache = slot_cache.segment_caches[i];
        if (!cache.valid) {
            throw std::runtime_error(
                "Qwen35moeForwardPass::run_split_decode_body_cached: missing non-split segment cache");
        }
        set_cached_segment_inputs(
            cache,
            token,
            pos,
            slot_idx,
            hidden_type,
            hidden_tensor,
            &segment
        );
        if (have_bootstrap_hidden) {
            if (!cache.hidden_in_tensor) {
                throw std::runtime_error(
                    "Qwen35moeForwardPass::run_split_decode_body_cached: missing bootstrap hidden_in");
            }
            validate_handoff_tensor(
                "Qwen35moeForwardPass::run_split_decode_body_cached: bootstrap hidden_in",
                cache.hidden_in_tensor,
                hidden_type,
                bootstrap_hidden.size()
            );
            ggml_backend_tensor_set(
                cache.hidden_in_tensor,
                bootstrap_hidden.data(),
                0,
                bootstrap_hidden.size()
            );
            have_bootstrap_hidden = false;
        }
        ggml_backend_graph_compute_async(cache.backend, cache.graph);
        hidden_tensor = cache.hidden_out_tensor;
        if (!hidden_tensor) {
            throw std::runtime_error(
                "Qwen35moeForwardPass::run_split_decode_body_cached: missing segment hidden output");
        }
        hidden_type = hidden_tensor->type;
        hidden_backend = cache.backend;
    }

    if (!hidden_tensor) {
        throw std::runtime_error(
            "Qwen35moeForwardPass::run_split_decode_body_cached: missing final hidden tensor");
    }
    hidden_out = hidden_tensor;
}

void Qwen35moeForwardPass::feed_cached_head_hidden_bytes(
    const char* where,
    SegmentedDecodeGraphCache& head,
    const std::vector<uint8_t>& hidden_data,
    ggml_type hidden_type
) const {
    if (!head.hidden_in_tensor) {
        throw std::runtime_error(std::string(where) + ": missing head hidden input");
    }
    if (hidden_data.empty()) {
        throw std::runtime_error(std::string(where) + ": head hidden mismatch");
    }
    if (hidden_type != head.hidden_type) {
        throw std::runtime_error(
            std::string(where) + ": head hidden type mismatch (runtime=" +
            std::string(ggml_type_name(hidden_type)) + " cached=" +
            std::string(ggml_type_name(head.hidden_type)) + ")");
    }
    validate_handoff_tensor(where, head.hidden_in_tensor, hidden_type, hidden_data.size());
    maybe_log_segment_tensor("head_segment_hidden_in", nullptr, head.hidden_in_tensor);
    ggml_backend_tensor_set(head.hidden_in_tensor, hidden_data.data(), 0, hidden_data.size());
    if (head.backend) {
        ggml_backend_synchronize(head.backend);
    }
}

void Qwen35moeForwardPass::feed_cached_head_hidden_tensor(
    const char* where,
    SegmentedDecodeGraphCache& head,
    ggml_backend_t src_backend,
    ggml_tensor* hidden_src,
    ggml_type hidden_type
) const {
    if (!head.hidden_in_tensor) {
        throw std::runtime_error(std::string(where) + ": missing head hidden input");
    }
    if (!hidden_src) {
        throw std::runtime_error(std::string(where) + ": head hidden mismatch");
    }
    copy_handoff_tensor_async(
        where,
        src_backend,
        head.backend,
        hidden_src,
        head.hidden_in_tensor,
        hidden_type
    );
}

std::vector<float> Qwen35moeForwardPass::run_decode_split_layers_cached(
    int32_t token,
    int pos,
    uint32_t slot_idx,
    SegmentedDecodeSlotCache& slot_cache
) {
    ggml_tensor* hidden_tensor = nullptr;
    ggml_backend_t hidden_backend = nullptr;
    ggml_type hidden_type = GGML_TYPE_F32;
    run_split_decode_body_cached(
        token,
        pos,
        slot_idx,
        slot_cache,
        hidden_tensor,
        hidden_backend,
        hidden_type
    );

    SegmentedDecodeGraphCache& head = slot_cache.head_cache;
    if (hidden_backend) {
        ggml_backend_synchronize(hidden_backend);
    }
    feed_cached_head_hidden_tensor(
        "Qwen35moeForwardPass::run_decode_split_layers_cached: head",
        head,
        hidden_backend,
        hidden_tensor,
        hidden_type
    );
    ggml_backend_graph_compute_async(head.backend, head.graph);
    std::vector<float> logits = get_output_logits(head.graph, head.backend);
    if (!commit_segmented_decode_step(slot_idx, static_cast<uint32_t>(pos))) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_split_layers_cached: commit failed");
    }
    return logits;
}

TopKSampleCandidates Qwen35moeForwardPass::run_decode_segmented_topk(int32_t token, int pos, uint32_t slot_idx) {
    if (!segmented_decode_cache_enabled_) {
        return run_decode_segmented_topk_eager(token, pos, slot_idx);
    }
    try {
        return run_decode_segmented_topk_cached(token, pos, slot_idx);
    } catch (const std::exception& ex) {
        segmented_decode_fallback_count_++;
        if (!segmented_decode_cache_fallback_warned_) {
            std::fprintf(stderr,
                "[PERF][segmented-decode-cache] disabled after cache failure: %s\n",
                ex.what());
            segmented_decode_cache_fallback_warned_ = true;
        }
        clear_segmented_decode_cache();
        segmented_decode_cache_enabled_ = false;
        return run_decode_segmented_topk_eager(token, pos, slot_idx);
    }
}

TopKSampleCandidates Qwen35moeForwardPass::run_decode_segmented_topk_eager(int32_t token, int pos, uint32_t slot_idx) {
    if (sampling_top_k_ <= 0) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk: device sampling not configured");
    }
    if (layer_segments_.empty()) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill_topk(token_vec, pos, slot_idx, nullptr);
    }

    std::vector<uint8_t> hidden_data;
    ggml_type hidden_type = GGML_TYPE_F32;
    for (size_t i = 0; i < layer_segments_.size(); ++i) {
        const bool first = i == 0;
        const LayerSegment& segment = layer_segments_[i];
        const bool first_uses_token_input = first && can_run_token_embedding_on_backend(segment.backend);
        if (first && !first_uses_token_input) {
            prepare_first_segment_hidden_from_token(
                "Qwen35moeForwardPass::run_decode_segmented_topk",
                token,
                segment,
                hidden_data,
                hidden_type
            );
        }

        if (segment_has_split_layers(segment)) {
            run_prefill_segment_with_split_layers(
                std::vector<int32_t>{ token },
                pos,
                slot_idx,
                segment,
                first_uses_token_input,
                hidden_data,
                hidden_type
            );
            continue;
        }

        ggml_cgraph* gf = build_decode_segment_graph(token, pos, slot_idx, segment, first_uses_token_input, false, hidden_type);

        run_segment_forward(gf, segment, "Qwen35moeForwardPass::run_decode_segmented_topk", [&]() {
            set_segment_inputs(
                gf,
                std::vector<int32_t>{ token },
                pos,
                hidden_type,
                first_uses_token_input ? nullptr : &hidden_data,
                &segment,
                segment.l0,
                segment.l1
            );
        });
        read_segment_hidden_out("Qwen35moeForwardPass::run_decode_segmented_topk", &segment, gf, hidden_data, hidden_type);
    }

    ggml_cgraph* gf_head = build_output_head_graph_from_hidden(hidden_type);
    ggml_backend_t head_backend = model_->get_curr_backend();
    ggml_gallocr_t head_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(head_backend));
    if (!ggml_gallocr_alloc_graph(head_alloc, gf_head)) {
        ggml_gallocr_free(head_alloc);
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk: head alloc failed");
    }
    if (ggml_tensor* hidden_in = ggml_graph_get_tensor(gf_head, "segment_hidden_in")) {
        if (hidden_data.empty()) {
            ggml_gallocr_free(head_alloc);
            throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk: head hidden mismatch");
        }
        validate_handoff_tensor("Qwen35moeForwardPass::run_decode_segmented_topk: head", hidden_in, hidden_type, hidden_data.size());
        maybe_log_segment_tensor("head_segment_hidden_in", nullptr, hidden_in);
        ggml_backend_tensor_set(hidden_in, hidden_data.data(), 0, hidden_data.size());
    }
    const size_t head_reserve = ggml_gallocr_get_buffer_size(head_alloc, 0);
    //std::fprintf(stderr, "[dev-mode=auto] segmented reserve output_head backend=%s bytes=%zu\n", ggml_backend_name(head_backend), head_reserve);
    ggml_backend_graph_compute(head_backend, gf_head);
    ggml_tensor* logits_scaled = ggml_graph_get_tensor(gf_head, "logits_scaled");
    ggml_tensor* topk_idx = ggml_graph_get_tensor(gf_head, "sample_topk_idx");
    if (!topk_idx) {
        ggml_gallocr_free(head_alloc);
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk: missing sample_topk_idx in head graph");
    }
    if (logits_scaled) {
        maybe_log_segment_tensor("head_logits_scaled", nullptr, logits_scaled);
    }
    maybe_log_segment_tensor("head_topk_idx", nullptr, topk_idx);
    TopKSampleCandidates result = get_output_topk_candidates(gf_head, 0);
    ggml_gallocr_free(head_alloc);
    advance_cache(1, slot_idx);
    return result;
}

TopKSampleCandidates Qwen35moeForwardPass::run_decode_segmented_topk_cached(int32_t token, int pos, uint32_t slot_idx) {
    if (sampling_top_k_ <= 0) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk_cached: device sampling not configured");
    }
    if (layer_segments_.empty()) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill_topk(token_vec, pos, slot_idx, nullptr);
    }

    const uint32_t required_kv = static_cast<uint32_t>(pos) + 1;
    DecodeGraphSignature requested_signature;
    requested_signature.slot_idx = slot_idx;
    requested_signature.context_len = context_len_;
    requested_signature.n_batch_tokens = n_batch_tokens_;
    requested_signature.n_ubatch_tokens = n_ubatch_tokens_;
    requested_signature.use_flash_attention = use_flash_attention_;
    requested_signature.paged_fused_decode = paged_fused_decode_active();
    requested_signature.is_mixed_mode = model_->is_mixed_mode();
    requested_signature.device_map_hash = model_->compute_device_map_hash();
    requested_signature.kv_capacity = decode_cache_bucket_capacity(required_kv);

    ensure_segmented_decode_cache(requested_signature, required_kv);
    if (!ensure_segmented_decode_copy_ready(slot_idx, static_cast<uint32_t>(pos))) {
        segmented_decode_fallback_count_++;
        return run_decode_segmented_topk_eager(token, pos, slot_idx);
    }

    SegmentedDecodeSlotCache& slot_cache = segmented_decode_slot_cache(slot_idx);

    if (slot_cache.use_split_decode) {
        ggml_tensor* hidden_tensor = nullptr;
        ggml_backend_t hidden_backend = nullptr;
        ggml_type hidden_type = GGML_TYPE_F32;
        run_split_decode_body_cached(
            token,
            pos,
            slot_idx,
            slot_cache,
            hidden_tensor,
            hidden_backend,
            hidden_type
        );

        SegmentedDecodeGraphCache& head = slot_cache.head_cache;
        if (hidden_backend) {
            ggml_backend_synchronize(hidden_backend);
        }
        feed_cached_head_hidden_tensor(
            "Qwen35moeForwardPass::run_decode_segmented_topk_cached: head",
            head,
            hidden_backend,
            hidden_tensor,
            hidden_type
        );
        ggml_backend_graph_compute_async(head.backend, head.graph);
        TopKSampleCandidates result = get_output_topk_candidates(head.graph, 0, head.backend);
        if (!commit_segmented_decode_step(slot_idx, static_cast<uint32_t>(pos))) {
            throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk_cached: failed to commit decode step");
        }
        return result;
    }

    ggml_type hidden_type = GGML_TYPE_F32;
    ggml_tensor* hidden_tensor = nullptr;
    ggml_backend_t hidden_backend = nullptr;

    if (slot_cache.embed_cache.valid) {
        set_cached_segment_inputs(
            slot_cache.embed_cache,
            token,
            pos,
            slot_idx,
            hidden_type,
            nullptr,
            &layer_segments_.front()
        );
        ggml_backend_graph_compute_async(
            slot_cache.embed_cache.backend,
            slot_cache.embed_cache.graph
        );
        hidden_tensor = slot_cache.embed_cache.hidden_out_tensor;
        if (!hidden_tensor) {
            throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk_cached: missing embedding hidden output");
        }
        hidden_type = hidden_tensor->type;
        hidden_backend = slot_cache.embed_cache.backend;
        maybe_log_segment_tensor("segment_hidden_out", &layer_segments_.front(), hidden_tensor);
    }

    for (size_t i = 0; i < slot_cache.segment_caches.size(); ++i) {
        const LayerSegment& segment = layer_segments_[i];
        SegmentedDecodeGraphCache& cache = slot_cache.segment_caches[i];
        set_cached_segment_inputs(
            cache,
            token,
            pos,
            slot_idx,
            hidden_type,
            cache.hidden_in_tensor ? hidden_tensor : nullptr,
            &segment
        );
        ggml_backend_graph_compute_async(cache.backend, cache.graph);
        hidden_tensor = cache.hidden_out_tensor;
        if (!hidden_tensor) {
            throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk_cached: missing segment hidden output");
        }
        hidden_type = hidden_tensor->type;
        hidden_backend = cache.backend;
        maybe_log_segment_tensor("segment_hidden_out", &segment, hidden_tensor);
    }

    SegmentedDecodeGraphCache& head = slot_cache.head_cache;
    if (!hidden_tensor) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk_cached: head hidden mismatch");
    }
    if (hidden_backend) {
        ggml_backend_synchronize(hidden_backend);
    }
    copy_handoff_tensor_async(
        "Qwen35moeForwardPass::run_decode_segmented_topk_cached: head",
        hidden_backend,
        head.backend,
        hidden_tensor,
        head.hidden_in_tensor,
        hidden_type
    );

    ggml_backend_graph_compute_async(head.backend, head.graph);
    TopKSampleCandidates result = get_output_topk_candidates(head.graph, 0, head.backend);
    if (!commit_segmented_decode_step(slot_idx, static_cast<uint32_t>(pos))) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_segmented_topk_cached: failed to commit decode step");
    }
    return result;
}

/**
 * @brief 执行 Prefill 阶段推理
 * 
 * @param tokens 输入 token 序列
 * @param pos 起始位置
 * @param slot_idx 槽位索引
 * @param scheduler 后端调度器
 * @return 输出 logits
 * 
 * 执行流程：
 * 1. 重置调度器
 * 2. 构建计算图
 * 3. 分配计算图内存
 * 4. 设置输入数据
 * 5. 执行计算
 * 6. 更新缓存位置
 * 7. 返回输出 logits
 */
std::vector<float> Qwen35moeForwardPass::run_prefill(const std::vector<int32_t>& tokens, int pos, 
    uint32_t slot_idx, ggml_backend_sched_t scheduler) {
    if (tokens.empty()) {
        return {};
    }

    CpuThreadScope thread_scope(model_, true);

    const uint32_t batch_limit = effective_prefill_batch_limit();
    if (scheduler != nullptr) {
        std::vector<float> last_logits;
        for (size_t start = 0; start < tokens.size(); start += batch_limit) {
            const size_t chunk_size = std::min(static_cast<size_t>(batch_limit), tokens.size() - start);
            std::vector<int32_t> chunk_tokens(tokens.begin() + start, tokens.begin() + start + chunk_size);
            const int chunk_pos = pos + static_cast<int>(start);
            run_prefill_microbatched_scheduler(
                scheduler, chunk_tokens, chunk_pos, slot_idx, false, &last_logits, nullptr);
        }
        maybe_log_decode_graph_stats();
        return last_logits;
    }

    auto backend = model_->get_curr_backend();
    auto buf_type = ggml_backend_get_default_buffer_type(backend);
    ggml_gallocr_t allocr_prefill = ggml_gallocr_new(buf_type);
    std::vector<float> result;
    for (size_t start = 0; start < tokens.size(); start += batch_limit) {
        const size_t chunk_size = std::min(static_cast<size_t>(batch_limit), tokens.size() - start);
        std::vector<int32_t> chunk_tokens(tokens.begin() + start, tokens.begin() + start + chunk_size);
        const int chunk_pos = pos + static_cast<int>(start);
        run_prefill_microbatched_direct(
            allocr_prefill, backend, chunk_tokens, chunk_pos, slot_idx, false, &result, nullptr);
    }
    if (allocr_prefill) {
        ggml_gallocr_free(allocr_prefill);
    }
    return result;
}

TopKSampleCandidates Qwen35moeForwardPass::run_prefill_topk(const std::vector<int32_t>& tokens, int pos,
    uint32_t slot_idx, ggml_backend_sched_t scheduler) {
    if (sampling_top_k_ <= 0) {
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_topk: device sampling not configured");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_topk: empty token input");
    }

    CpuThreadScope thread_scope(model_, true);

    const uint32_t batch_limit = effective_prefill_batch_limit();
    if (scheduler != nullptr) {
        TopKSampleCandidates result;
        for (size_t start = 0; start < tokens.size(); start += batch_limit) {
            const size_t chunk_size = std::min(static_cast<size_t>(batch_limit), tokens.size() - start);
            std::vector<int32_t> chunk_tokens(tokens.begin() + start, tokens.begin() + start + chunk_size);
            const int chunk_pos = pos + static_cast<int>(start);
            run_prefill_microbatched_scheduler(
                scheduler, chunk_tokens, chunk_pos, slot_idx, true, nullptr, &result);
        }
        maybe_log_decode_graph_stats();
        return result;
    }

    auto backend = model_->get_curr_backend();
    auto buf_type = ggml_backend_get_default_buffer_type(backend);
    ggml_gallocr_t allocr_prefill = ggml_gallocr_new(buf_type);
    TopKSampleCandidates result;
    for (size_t start = 0; start < tokens.size(); start += batch_limit) {
        const size_t chunk_size = std::min(static_cast<size_t>(batch_limit), tokens.size() - start);
        std::vector<int32_t> chunk_tokens(tokens.begin() + start, tokens.begin() + start + chunk_size);
        const int chunk_pos = pos + static_cast<int>(start);
        run_prefill_microbatched_direct(
            allocr_prefill, backend, chunk_tokens, chunk_pos, slot_idx, true, nullptr, &result);
    }
    if (allocr_prefill) {
        ggml_gallocr_free(allocr_prefill);
    }
    return result;
}

std::vector<float> Qwen35moeForwardPass::run_decode_cached(int32_t token, int pos,
    uint32_t slot_idx, ggml_backend_sched_t scheduler) {
    if (scheduler == nullptr) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill(token_vec, pos, slot_idx, scheduler);
    }
    if (static_cast<uint32_t>(pos) >= context_len_) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_cached: position exceeds context_len_");
    }

    CpuThreadScope thread_scope(model_, false);

    if (paged_kv_enabled_) {
        paged_fused_decode_attempt_count_++;
        const char* fallback_reason = nullptr;
        const bool use_fused_decode = can_use_paged_fused_decode(&fallback_reason);
        if (!use_fused_decode) {
            maybe_log_paged_fused_fallback(fallback_reason);
        } else {
            paged_fused_decode_hit_count_++;
            maybe_log_paged_fused_activation();
        }
    }

    // Mixed GPU/CPU: per-slot segmented decode cache (default). Scheduler cached
    // decode is opt-in because it triggers CUDA-graph warmup and is slower than
    // the async CPU→GPU segment pipeline in AUTO mode.
    if (model_->is_mixed_mode()) {
        if (mixed_scheduler_decode_enabled_ && !model_->has_any_split_device_layers()) {
            try {
                const uint32_t required_kv = static_cast<uint32_t>(pos) + 1;
                DecodeGraphSignature requested_signature;
                requested_signature.slot_idx = slot_idx;
                requested_signature.context_len = context_len_;
                requested_signature.n_batch_tokens = n_batch_tokens_;
                requested_signature.n_ubatch_tokens = n_ubatch_tokens_;
                requested_signature.use_flash_attention = use_flash_attention_;
                requested_signature.paged_fused_decode = paged_fused_decode_active();
                requested_signature.is_mixed_mode = true;
                requested_signature.device_map_hash = model_->compute_device_map_hash();
                requested_signature.kv_capacity = decode_cache_bucket_capacity(required_kv);

                ensure_cached_decode_graph(scheduler, requested_signature, required_kv);
                if (!ensure_cached_decode_copy_ready(slot_idx, static_cast<uint32_t>(pos))) {
                    throw std::runtime_error("mixed scheduler decode: paged-kv copy guard failed");
                }

                set_cached_decode_inputs(cached_decode_graph_, token, pos);
                ggml_backend_sched_graph_compute(scheduler, cached_decode_graph_);

                if (!commit_cached_decode_step(slot_idx, static_cast<uint32_t>(pos))) {
                    throw std::runtime_error("mixed scheduler decode: failed to commit KV scratch");
                }
                return get_output_logits(cached_decode_graph_);
            } catch (const std::exception& ex) {
                mixed_scheduler_decode_fallback_count_++;
                if (!mixed_scheduler_decode_fallback_warned_) {
                    std::fprintf(
                        stderr,
                        "[mixed-mode] scheduler decode fallback to segmented cache: %s\n",
                        ex.what()
                    );
                    mixed_scheduler_decode_fallback_warned_ = true;
                }
            }
        }
        return run_decode_segmented(token, pos, slot_idx);
    }

    const uint32_t required_kv = static_cast<uint32_t>(pos) + 1;
    DecodeGraphSignature requested_signature;
    requested_signature.slot_idx = slot_idx;
    requested_signature.context_len = context_len_;
    requested_signature.n_batch_tokens = n_batch_tokens_;
    requested_signature.n_ubatch_tokens = n_ubatch_tokens_;
    requested_signature.use_flash_attention = use_flash_attention_;
    requested_signature.paged_fused_decode = paged_fused_decode_active();
    requested_signature.is_mixed_mode = model_->is_mixed_mode();
    requested_signature.device_map_hash = model_->compute_device_map_hash();
    requested_signature.kv_capacity = decode_cache_bucket_capacity(required_kv);

    ensure_cached_decode_graph(scheduler, requested_signature, required_kv);
    if (!ensure_cached_decode_copy_ready(slot_idx, static_cast<uint32_t>(pos))) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill(token_vec, pos, slot_idx, scheduler);
    }

    set_cached_decode_inputs(cached_decode_graph_, token, pos);
    const auto decode_start_time = std::chrono::steady_clock::now();
    ggml_backend_sched_graph_compute(scheduler, cached_decode_graph_);
    if (paged_fused_decode_active()) {
        const auto decode_end = std::chrono::steady_clock::now();
        const auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start_time).count();
        record_paged_fused_decode_timing(static_cast<uint64_t>(dt_us));
    }

    if (!commit_cached_decode_step(slot_idx, static_cast<uint32_t>(pos))) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill(token_vec, pos, slot_idx, scheduler);
    }
    return get_output_logits(cached_decode_graph_);
}

TopKSampleCandidates Qwen35moeForwardPass::run_decode_cached_topk(int32_t token, int pos,
    uint32_t slot_idx, ggml_backend_sched_t scheduler) {
    if (sampling_top_k_ <= 0) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_cached_topk: device sampling not configured");
    }
    if (scheduler == nullptr) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill_topk(token_vec, pos, slot_idx, scheduler);
    }
    if (static_cast<uint32_t>(pos) >= context_len_) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_cached_topk: position exceeds context_len_");
    }

    CpuThreadScope thread_scope(model_, false);

    if (paged_kv_enabled_) {
        paged_fused_decode_attempt_count_++;
        const char* fallback_reason = nullptr;
        const bool use_fused_decode = can_use_paged_fused_decode(&fallback_reason);
        if (!use_fused_decode) {
            maybe_log_paged_fused_fallback(fallback_reason);
        } else {
            paged_fused_decode_hit_count_++;
            maybe_log_paged_fused_activation();
        }
    }

    // Mixed GPU/CPU: per-slot segmented decode cache (default).
    if (model_->is_mixed_mode()) {
        if (mixed_scheduler_decode_enabled_ && !model_->has_any_split_device_layers()) {
            try {
                const uint32_t required_kv = static_cast<uint32_t>(pos) + 1;
                DecodeGraphSignature requested_signature;
                requested_signature.slot_idx = slot_idx;
                requested_signature.context_len = context_len_;
                requested_signature.n_batch_tokens = n_batch_tokens_;
                requested_signature.n_ubatch_tokens = n_ubatch_tokens_;
                requested_signature.use_flash_attention = use_flash_attention_;
                requested_signature.paged_fused_decode = paged_fused_decode_active();
                requested_signature.is_mixed_mode = true;
                requested_signature.device_map_hash = model_->compute_device_map_hash();
                requested_signature.kv_capacity = decode_cache_bucket_capacity(required_kv);

                ensure_cached_decode_graph(scheduler, requested_signature, required_kv);
                if (!ensure_cached_decode_copy_ready(slot_idx, static_cast<uint32_t>(pos))) {
                    throw std::runtime_error("mixed scheduler decode topk: paged-kv copy guard failed");
                }

                set_cached_decode_inputs(cached_decode_graph_, token, pos);
                ggml_backend_sched_graph_compute(scheduler, cached_decode_graph_);

                if (!commit_cached_decode_step(slot_idx, static_cast<uint32_t>(pos))) {
                    throw std::runtime_error("mixed scheduler decode topk: failed to commit KV scratch");
                }
                return get_output_topk_candidates(cached_decode_graph_, 0);
            } catch (const std::exception& ex) {
                mixed_scheduler_decode_fallback_count_++;
                if (!mixed_scheduler_decode_fallback_warned_) {
                    std::fprintf(
                        stderr,
                        "[mixed-mode] scheduler decode_topk fallback to segmented cache: %s\n",
                        ex.what()
                    );
                    mixed_scheduler_decode_fallback_warned_ = true;
                }
            }
        }
        return run_decode_segmented_topk(token, pos, slot_idx);
    }

    const uint32_t required_kv = static_cast<uint32_t>(pos) + 1;
    DecodeGraphSignature requested_signature;
    requested_signature.slot_idx = slot_idx;
    requested_signature.context_len = context_len_;
    requested_signature.n_batch_tokens = n_batch_tokens_;
    requested_signature.n_ubatch_tokens = n_ubatch_tokens_;
    requested_signature.use_flash_attention = use_flash_attention_;
    requested_signature.paged_fused_decode = paged_fused_decode_active();
    requested_signature.is_mixed_mode = model_->is_mixed_mode();
    requested_signature.device_map_hash = model_->compute_device_map_hash();
    requested_signature.kv_capacity = decode_cache_bucket_capacity(required_kv);

    ensure_cached_decode_graph(scheduler, requested_signature, required_kv);
    if (!ensure_cached_decode_copy_ready(slot_idx, static_cast<uint32_t>(pos))) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill_topk(token_vec, pos, slot_idx, scheduler);
    }

    set_cached_decode_inputs(cached_decode_graph_, token, pos);
    const auto decode_start_time = std::chrono::steady_clock::now();
    ggml_backend_sched_graph_compute(scheduler, cached_decode_graph_);
    if (paged_fused_decode_active()) {
        const auto decode_end = std::chrono::steady_clock::now();
        const auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start_time).count();
        record_paged_fused_decode_timing(static_cast<uint64_t>(dt_us));
    }

    if (!commit_cached_decode_step(slot_idx, static_cast<uint32_t>(pos))) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill_topk(token_vec, pos, slot_idx, scheduler);
    }
    return get_output_topk_candidates(cached_decode_graph_, 0);
}

std::vector<std::vector<float>> Qwen35moeForwardPass::run_decode_batch_cached(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions,
    ggml_backend_sched_t scheduler
) {
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());
    uint32_t max_pos = 0;
    for (int32_t pos : positions) {
        if (pos >= 0 && static_cast<uint32_t>(pos) > max_pos) {
            max_pos = static_cast<uint32_t>(pos);
        }
    }
    const uint32_t required_kv = max_pos + 1;
    DecodeGraphSignature signature;
    signature.kv_capacity = decode_cache_bucket_capacity(required_kv);
    signature.context_len = context_len_;
    signature.n_batch_tokens = n_batch_tokens_;
    signature.n_ubatch_tokens = n_ubatch_tokens_;
    signature.n_decode_batch = n_batch;
    signature.slots_signature = hash_decode_slots(slots);
    signature.use_flash_attention = use_flash_attention_;
    signature.paged_fused_decode = paged_fused_decode_active();
    signature.device_map_hash = model_->compute_device_map_hash();

    ensure_cached_batched_decode_graph(scheduler, signature, slots, required_kv);
    set_cached_batched_decode_inputs(tokens, slots, positions);
    ggml_backend_sched_graph_compute(scheduler, cached_batched_decode_graph_);

    ggml_tensor* logits = ggml_graph_get_tensor(cached_batched_decode_graph_, "logits");
    if (!logits) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch_cached: logits tensor missing");
    }
    if (logits->ne[1] < static_cast<int64_t>(n_batch)) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch_cached: logits batch shape mismatch");
    }
    const uint32_t vocab_size = static_cast<uint32_t>(logits->ne[0]);
    std::vector<std::vector<float>> result(n_batch, std::vector<float>(vocab_size));
    for (uint32_t b = 0; b < n_batch; ++b) {
        const size_t offset = static_cast<size_t>(b) * logits->nb[1];
        ggml_backend_tensor_get(logits, result[b].data(), offset, vocab_size * sizeof(float));
    }
    for (uint32_t slot : slots) {
        advance_cache(1, slot);
    }
    maybe_log_decode_graph_stats();
    return result;
}

std::vector<std::vector<float>> Qwen35moeForwardPass::run_decode_batch(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions,
    ggml_backend_sched_t scheduler
) {
    if (tokens.empty()) {
        return {};
    }
    if (tokens.size() != slots.size() || tokens.size() != positions.size()) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch: input size mismatch");
    }
    if (tokens.size() == 1) {
        return {run_decode_cached(tokens[0], positions[0], slots[0], scheduler)};
    }
    if (scheduler == nullptr) {
        std::vector<std::vector<float>> result;
        result.reserve(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            result.push_back(run_decode_cached(tokens[i], positions[i], slots[i], scheduler));
        }
        return result;
    }
    if (tokens.size() > max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch: batch exceeds max_batch_size_");
    }
    for (int pos : positions) {
        if (pos < 0 || static_cast<uint32_t>(pos) >= context_len_) {
            throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch: position exceeds context_len_");
        }
    }

    CpuThreadScope thread_scope(model_, true);

    if (!model_->is_mixed_mode()) {
        try {
            return run_decode_batch_cached(tokens, slots, positions, scheduler);
        } catch (const std::exception& ex) {
            decode_graph_fallback_count_++;
            std::fprintf(
                stderr,
                "[PERF][decode-graph-batched] fallback to eager path: %s\n",
                ex.what()
            );
            clear_cached_batched_decode_graph();
        }
    }

    if (model_->is_mixed_mode() && mixed_batched_sequential_enabled_) {
        // Default in AUTO: per-token segmented cache (one cached graph set per slot).
        if (!mixed_batched_warned_) {
            std::fprintf(
                stderr,
                "[mixed-mode] batched decode (batch=%zu) using per-token segmented cache "
                "(default in AUTO). Set QWEN35MOE_MIXED_BATCHED_SCHEDULER=1 for scheduler batched.\n",
                tokens.size()
            );
            mixed_batched_warned_ = true;
        }
        std::vector<std::vector<float>> result;
        result.reserve(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            result.push_back(run_decode_cached(tokens[i], positions[i], slots[i], scheduler));
        }
        return result;
    } else if (model_->is_mixed_mode() && !mixed_batched_eager_logged_) {
        std::fprintf(
            stderr,
            "[mixed-mode] batched decode (batch=%zu) using scheduler batched eager path "
            "(QWEN35MOE_MIXED_BATCHED_SCHEDULER=1). Default AUTO uses per-token segmented.\n",
            tokens.size()
        );
        mixed_batched_eager_logged_ = true;
    }

    ggml_backend_sched_reset(scheduler);
    decode_graph_scheduler_reset_count_++;
    ggml_cgraph* gf = build_decoding_graph(tokens, slots, positions);
    if (!ggml_backend_sched_alloc_graph(scheduler, gf)) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch: failed to allocate graph");
    }
    set_batched_inputs(gf, tokens, slots, positions);
    ggml_backend_sched_graph_compute(scheduler, gf);

    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch: logits tensor missing");
    }
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());
    if (logits->ne[1] < static_cast<int64_t>(n_batch)) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch: logits batch shape mismatch");
    }
    const uint32_t vocab_size = static_cast<uint32_t>(logits->ne[0]);
    std::vector<std::vector<float>> result(n_batch, std::vector<float>(vocab_size));
    for (uint32_t b = 0; b < n_batch; ++b) {
        const size_t offset = static_cast<size_t>(b) * logits->nb[1];
        ggml_backend_tensor_get(logits, result[b].data(), offset, vocab_size * sizeof(float));
    }
    for (uint32_t slot : slots) {
        advance_cache(1, slot);
    }
    return result;
}

std::vector<TopKSampleCandidates> Qwen35moeForwardPass::run_decode_batch_topk_cached(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions,
    ggml_backend_sched_t scheduler
) {
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());
    uint32_t max_pos = 0;
    for (int32_t pos : positions) {
        if (pos >= 0 && static_cast<uint32_t>(pos) > max_pos) {
            max_pos = static_cast<uint32_t>(pos);
        }
    }
    const uint32_t required_kv = max_pos + 1;
    DecodeGraphSignature signature;
    signature.kv_capacity = decode_cache_bucket_capacity(required_kv);
    signature.context_len = context_len_;
    signature.n_batch_tokens = n_batch_tokens_;
    signature.n_ubatch_tokens = n_ubatch_tokens_;
    signature.n_decode_batch = n_batch;
    signature.slots_signature = hash_decode_slots(slots);
    signature.use_flash_attention = use_flash_attention_;
    signature.paged_fused_decode = paged_fused_decode_active();
    signature.device_map_hash = model_->compute_device_map_hash();

    ensure_cached_batched_decode_graph(scheduler, signature, slots, required_kv);
    set_cached_batched_decode_inputs(tokens, slots, positions);
    ggml_backend_sched_graph_compute(scheduler, cached_batched_decode_graph_);

    std::vector<TopKSampleCandidates> result;
    result.reserve(tokens.size());
    for (uint32_t b = 0; b < n_batch; ++b) {
        result.push_back(get_output_topk_candidates(cached_batched_decode_graph_, b));
    }
    for (uint32_t slot : slots) {
        advance_cache(1, slot);
    }
    maybe_log_decode_graph_stats();
    return result;
}

std::vector<TopKSampleCandidates> Qwen35moeForwardPass::run_decode_batch_topk(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions,
    ggml_backend_sched_t scheduler
) {
    if (sampling_top_k_ <= 0) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch_topk: device sampling not configured");
    }
    if (tokens.empty()) {
        return {};
    }
    if (tokens.size() != slots.size() || tokens.size() != positions.size()) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch_topk: input size mismatch");
    }
    if (tokens.size() == 1) {
        return {run_decode_cached_topk(tokens[0], positions[0], slots[0], scheduler)};
    }
    if (scheduler == nullptr) {
        std::vector<TopKSampleCandidates> result;
        result.reserve(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            result.push_back(run_decode_cached_topk(tokens[i], positions[i], slots[i], scheduler));
        }
        return result;
    }
    if (tokens.size() > max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch_topk: batch exceeds max_batch_size_");
    }
    for (int pos : positions) {
        if (pos < 0 || static_cast<uint32_t>(pos) >= context_len_) {
            throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch_topk: position exceeds context_len_");
        }
    }

    CpuThreadScope thread_scope(model_, true);

    if (!model_->is_mixed_mode()) {
        try {
            return run_decode_batch_topk_cached(tokens, slots, positions, scheduler);
        } catch (const std::exception& ex) {
            decode_graph_fallback_count_++;
            std::fprintf(
                stderr,
                "[PERF][decode-graph-batched] topk fallback to eager path: %s\n",
                ex.what()
            );
            clear_cached_batched_decode_graph();
        }
    }

    if (model_->is_mixed_mode() && mixed_batched_sequential_enabled_) {
        if (!mixed_batched_warned_) {
            std::fprintf(
                stderr,
                "[mixed-mode] batched decode_topk (batch=%zu) using per-token segmented cache "
                "(default in AUTO). Set QWEN35MOE_MIXED_BATCHED_SCHEDULER=1 for scheduler batched.\n",
                tokens.size()
            );
            mixed_batched_warned_ = true;
        }
        std::vector<TopKSampleCandidates> result;
        result.reserve(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            result.push_back(run_decode_cached_topk(tokens[i], positions[i], slots[i], scheduler));
        }
        return result;
    } else if (model_->is_mixed_mode() && !mixed_batched_eager_logged_) {
        std::fprintf(
            stderr,
            "[mixed-mode] batched decode_topk (batch=%zu) using scheduler batched eager path "
            "(QWEN35MOE_MIXED_BATCHED_SCHEDULER=1). Default AUTO uses per-token segmented.\n",
            tokens.size()
        );
        mixed_batched_eager_logged_ = true;
    }

    ggml_backend_sched_reset(scheduler);
    decode_graph_scheduler_reset_count_++;
    ggml_cgraph* gf = build_decoding_graph(tokens, slots, positions);
    if (!ggml_backend_sched_alloc_graph(scheduler, gf)) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_batch_topk: failed to allocate graph");
    }
    set_batched_inputs(gf, tokens, slots, positions);
    ggml_backend_sched_graph_compute(scheduler, gf);

    std::vector<TopKSampleCandidates> result;
    result.reserve(tokens.size());
    for (uint32_t b = 0; b < static_cast<uint32_t>(tokens.size()); ++b) {
        result.push_back(get_output_topk_candidates(gf, b));
    }
    for (uint32_t slot : slots) {
        advance_cache(1, slot);
    }
    return result;
}

uint32_t Qwen35moeForwardPass::get_cache_pos(uint32_t slot_idx) const {
    uint32_t seq = snapkv_get_seq_pos(slot_idx);
    if (seq > 0) return seq;
    return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
}

ggml_cgraph* Qwen35moeForwardPass::new_graph() {
    return ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);
}

/**
 * @brief Token Embedding 查找
 * 
 * @param gf 计算图
 * @param tokens 输入 token ID 序列
 * @return 嵌入向量张量 [n_embd, n_tokens]
 * 
 * 实现步骤：
 * 1. 创建 token ID 张量
 * 2. 使用 ggml_get_rows 进行嵌入查找（相当于 embedding lookup）
 */
ggml_tensor* Qwen35moeForwardPass::embedding(ggml_cgraph* gf, const std::vector<int32_t>& tokens) {
    const size_t n_tokens = tokens.size();

    // 1. 创建 1D token ID 张量
    struct ggml_tensor* tokens_tensor = ggml_new_tensor_1d(
        ctx_,
        GGML_TYPE_I32,
        n_tokens
    );
    
    ggml_set_input(tokens_tensor);  // 标记为输入张量
    set_tensor_name(tokens_tensor, "tokens");
    ggml_build_forward_expand(gf, tokens_tensor);

    // 2. 使用 ggml_get_rows 进行嵌入查找
    // 从嵌入矩阵中提取对应 token 的嵌入向量
    struct ggml_tensor* token_embedding = model_->get_token_embedding_weight();
    ggml_tensor * cur = ggml_get_rows(
        ctx_,
        token_embedding,
        tokens_tensor
    );

    set_tensor_name(cur, "embed_lookup");
    return cur;
}

/**
 * @brief 构建 MoE（混合专家）FFN 层
 * 
 * 在 Pre-FFN 归一化之后应用，返回 FFN 输出（残差连接之前）
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param input 输入张量 [n_embd, n_tokens]
 * @param il 物理层索引
 * @return MoE 层输出 [n_embd, n_tokens]
 * 
 * MoE 层结构：
 * 1. 路由 (Routing)：为每个 token 选择 top-k 专家
 * 2. Expert Dispatch：将 token 分配到选中的专家
 * 3. Weighted Sum：加权汇总专家输出
 * 4. Shared Expert：共享专家（可选）
 */
ggml_tensor* Qwen35moeForwardPass::build_moe_routed_experts(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  input,
    int il
)
{
    struct ggml_tensor* ffn_gate_inp = model_->get_ffn_gate_inp_weight(il);
    struct ggml_tensor* ffn_gate_exps = model_->get_ffn_gate_exps_weight(il);
    struct ggml_tensor* ffn_up_exps = model_->get_ffn_up_exps_weight(il);
    struct ggml_tensor* ffn_down_exps = model_->get_ffn_down_exps_weight(il);

    auto& m = model_->meta_->qwen35moe;
    const int64_t n_embd   = input->ne[0];
    const int64_t n_tokens = input->ne[1];
    const int     n_exp    = m.expert_count;
    const int     top_k    = m.expert_used_count;
    const int64_t ffn_dim  = m.expert_feed_forward_length;
    const bool layer_on_cuda = backend_name_contains_cuda(layer_backend(il));
    const bool experts_on_cuda = weight_tensor_on_cuda(ffn_gate_exps);
    const bool use_cuda_moe_fusion = layer_on_cuda && experts_on_cuda;

    ggml_tensor* logits = ggml_mul_mat(ctx, ffn_gate_inp, input);
    set_tensor_name(logits, "moe_logits", il);

    // Reshape before ARGSORT so ggml_cuda topk_moe fusion can match:
    // RESHAPE -> ARGSORT -> VIEW -> GET_ROWS -> RESHAPE -> SOFT_MAX -> RESHAPE.
    ggml_tensor* logits_3d = ggml_reshape_3d(ctx, logits, 1, n_exp, n_tokens);
    set_tensor_name(logits_3d, "moe_logits_3d", il);

    ggml_tensor* expert_idx = nullptr;
    if (use_cuda_moe_fusion) {
        ggml_tensor* sorted = ggml_argsort(ctx, logits, GGML_SORT_ORDER_DESC);
        set_tensor_name(sorted, "moe_sorted", il);
        expert_idx = ggml_view_4d(
            ctx,
            sorted,
            top_k,
            sorted->ne[1],
            sorted->ne[2],
            sorted->ne[3],
            sorted->nb[1],
            sorted->nb[2],
            sorted->nb[3],
            0);
    } else {
        expert_idx = ggml_top_k(ctx, logits, top_k);
    }
    set_tensor_name(expert_idx, "moe_idx", il);

    ggml_tensor* expert_logits = ggml_get_rows(ctx, logits_3d, expert_idx);
    expert_logits = ggml_reshape_2d(ctx, expert_logits, top_k, n_tokens);
    set_tensor_name(expert_logits, "moe_sel_logits", il);

    ggml_tensor* expert_weights = ggml_soft_max(ctx, expert_logits);
    expert_weights = ggml_reshape_2d(ctx, expert_weights, top_k, n_tokens);
    set_tensor_name(expert_weights, "moe_weights", il);

    const int64_t n_slots = static_cast<int64_t>(top_k) * n_tokens;
    const bool use_flat_dispatch = n_tokens > 1;
    const bool use_cuda_routed_reduce = use_cuda_moe_fusion;

    ggml_tensor* input_3d = nullptr;
    ggml_tensor* ids_for_mm = expert_idx;

    if (use_flat_dispatch) {
        ggml_tensor* input_1t = ggml_reshape_3d(ctx, input, n_embd, 1, n_tokens);
        input_3d = ggml_repeat_4d(ctx, input_1t, n_embd, top_k, n_tokens, 1);
        input_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, input_3d), n_embd, n_slots, 1);
        ids_for_mm = ggml_reshape_2d(ctx, ggml_cont(ctx, expert_idx), n_slots, 1);
        set_tensor_name(input_3d, "moe_input_slots", il);
        set_tensor_name(ids_for_mm, "moe_idx_flat", il);
    } else {
        input_3d = ggml_reshape_3d(ctx, input, n_embd, 1, n_tokens);
    }

    const auto reshape_moe_mm_out = [&](ggml_tensor* out, int64_t out_dim) -> ggml_tensor* {
        if (!use_flat_dispatch) {
            return out;
        }
        return ggml_reshape_3d(ctx, ggml_cont(ctx, out), out_dim, top_k, n_tokens);
    };

    ggml_tensor* exp_gate_out = reshape_moe_mm_out(
        ggml_mul_mat_id(ctx, ffn_gate_exps, input_3d, ids_for_mm),
        ffn_dim
    );
    set_tensor_name(exp_gate_out, "moe_exp_gate", il);

    ggml_tensor* exp_up_out = reshape_moe_mm_out(
        ggml_mul_mat_id(ctx, ffn_up_exps, input_3d, ids_for_mm),
        ffn_dim
    );
    set_tensor_name(exp_up_out, "moe_exp_up", il);

    ggml_tensor* exp_act = ggml_swiglu_split(ctx, exp_gate_out, exp_up_out);
    set_tensor_name(exp_act, "moe_exp_act", il);

    ggml_tensor* exp_act_mm = use_flat_dispatch
        ? ggml_reshape_3d(ctx, ggml_cont(ctx, exp_act), ffn_dim, n_slots, 1)
        : exp_act;
    ggml_tensor* exp_down_out = reshape_moe_mm_out(
        ggml_mul_mat_id(ctx, ffn_down_exps, exp_act_mm, ids_for_mm),
        n_embd
    );
    set_tensor_name(exp_down_out, "moe_exp_down", il);

    ggml_tensor* down_perm = ggml_permute(ctx, exp_down_out, 1, 0, 2, 3);
    set_tensor_name(down_perm, "moe_exp_down_perm", il);

    ggml_tensor* w_3d = ggml_reshape_3d(ctx, expert_weights, top_k, 1, n_tokens);

    ggml_tensor* routed_3d = nullptr;
    if (!use_cuda_routed_reduce) {
        ggml_tensor* down_cont = ggml_cont(ctx, down_perm);
        set_tensor_name(down_cont, "moe_exp_down_cont", il);
        routed_3d = ggml_mul_mat(ctx, down_cont, w_3d);
        set_tensor_name(routed_3d, "moe_routed_mm", il);
    } else {
        ggml_tensor* weighted = ggml_mul(ctx, down_perm, w_3d);
        set_tensor_name(weighted, "moe_weighted", il);
        routed_3d = ggml_sum_rows(ctx, weighted);
        set_tensor_name(routed_3d, "moe_routed_sum", il);
    }

    ggml_tensor* routed_out = ggml_reshape_2d(ctx, routed_3d, n_embd, n_tokens);
    set_tensor_name(routed_out, "moe_routed_out", il);
    return routed_out;
}

ggml_tensor* Qwen35moeForwardPass::build_moe_shared_combine(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  input,
    ggml_tensor*  routed_out,
    int il,
    ggml_tensor*  ffn_residual
)
{
    struct ggml_tensor* ffn_gate_shexp = model_->get_ffn_gate_shexp_weight(il);
    struct ggml_tensor* ffn_up_shexp = model_->get_ffn_up_shexp_weight(il);
    struct ggml_tensor* ffn_down_shexp = model_->get_ffn_down_shexp_weight(il);
    struct ggml_tensor* ffn_gate_inp_shexp = model_->get_ffn_gate_inp_shexp_weight(il);

    ggml_tensor* sh_gate_out = ggml_mul_mat(ctx, ffn_gate_shexp, input);
    ggml_tensor* sh_up_out   = ggml_mul_mat(ctx, ffn_up_shexp,   input);
    ggml_tensor* sh_act      = ggml_swiglu_split(ctx, sh_gate_out, sh_up_out);
    ggml_tensor* sh_down_out = ggml_mul_mat(ctx, ffn_down_shexp, sh_act);
    set_tensor_name(sh_down_out, "moe_shared_out", il);

    ggml_tensor* sh_gate_logit = ggml_mul_mat(ctx, ffn_gate_inp_shexp, input);
    ggml_tensor* sh_gate       = ggml_sigmoid(ctx, sh_gate_logit);
    ggml_tensor* sh_contribution = ggml_mul(ctx, sh_down_out, sh_gate);
    set_tensor_name(sh_contribution, "moe_shared_contrib", il);

    ggml_tensor* combined = ggml_add(ctx, routed_out, sh_contribution);
    set_tensor_name(combined, "moe_combined", il);
    if (ffn_residual != nullptr) {
        combined = ggml_add(ctx, combined, ffn_residual);
    }
    return combined;
}

ggml_tensor* Qwen35moeForwardPass::build_moe_layer(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  input,
    int il
)
{
    ggml_tensor* routed = build_moe_routed_experts(ctx, gf, input, il);
    return build_moe_shared_combine(ctx, gf, input, routed, il, nullptr);
}

/**
 * @brief 构建 RMS 归一化层
 * 
 * RMSNorm 公式: output = weight * (input / sqrt(mean(input^2) + eps))
 * 
 * @param ctx ggml 上下文
 * @param cur 输入张量
 * @param weight 缩放权重
 * @param eps epsilon（防止除零）
 * @param il 层索引
 * @return 归一化后的张量
 */
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
    cur = ggml_mul(ctx, cur, weight);  // 乘以可学习的缩放权重
    return cur;
}

/**
 * @brief 构建 SwiGLU FFN 层
 * 
 * SwiGLU 公式: down @ (silu(gate @ x) * (up @ x))
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图（未使用，保持 API 一致性）
 * @param cur 输入张量
 * @param gate gate 权重矩阵
 * @param up up 权重矩阵
 * @param down down 权重矩阵
 * @param il 层索引
 * @return FFN 输出
 */
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

    // Up 投影
    ggml_tensor* tmp = ggml_mul_mat(ctx, up, cur);
    snprintf(name, sizeof(name), "ffn_up.%d", il);
    set_tensor_name(tmp, name);

    // Gate 投影
    cur = ggml_mul_mat(ctx, gate, cur);
    snprintf(name, sizeof(name), "ffn_gate.%d", il);
    set_tensor_name(cur, name);

    // SwiGLU 激活: silu(gate) * up
    cur = ggml_swiglu_split(ctx, cur, tmp);
    snprintf(name, sizeof(name), "ffn_swiglu.%d", il);
    set_tensor_name(cur, name);

    // Down 投影
    cur = ggml_mul_mat(ctx, down, cur);
    return cur;
}

/**
 * @brief 构建归一化层（封装函数）
 * 
 * 调用 build_rms_norm，使用模型配置的 epsilon
 * 
 * @param gf 计算图
 * @param cur 输入张量
 * @param mw 权重张量
 * @param il 层索引
 * @return 归一化后的张量
 */
ggml_tensor* Qwen35moeForwardPass::build_norm(
    ggml_cgraph* gf,
    ggml_tensor* cur,
    ggml_tensor* mw,
    int il
)
{
    return build_rms_norm(ctx_, cur, mw, model_->meta_->qwen35moe.layer_norm_rms_epsilon, il);
}

/**
 * @brief 构建多头注意力 (MHA) 层
 * 
 * 标准多头注意力计算：Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图（未直接使用）
 * @param q Query 张量
 * @param k Key 张量
 * @param v Value 张量
 * @param kq_mask 注意力掩码
 * @param sinks sink token（未使用）
 * @param kq_scale 缩放因子（1/sqrt(d_k)）
 * @param pos 位置（未使用）
 * @param il 层索引
 * @return 注意力输出
 */
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
    int           il,
    bool          allow_flash_attn
)
{
    (void)gf; (void)pos; // 保持 API 对称性，未直接使用

    const bool v_trans = v->nb[1] > v->nb[2];
    (void)v_trans;

    const auto n_stream = k->ne[3];

    // 重塑 Q 张量
    q = ggml_reshape_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream);
    set_tensor_name(q, "q_reshaped", il);

    // 调整张量维度顺序以适配注意力计算
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    set_tensor_name(q, "q_permuted", il);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    set_tensor_name(k, "k_permuted", il);

    if (allow_flash_attn && use_flash_attention_ && kq_mask && kq_mask->type == GGML_TYPE_F16) {
        ggml_tensor* v_fa = ggml_permute(ctx, v, 0, 2, 1, 3);
        set_tensor_name(v_fa, "v_flash", il);

        ggml_tensor* kqv = ggml_flash_attn_ext(ctx, q, k, v_fa, kq_mask, kq_scale, 0.0f, 0.0f);
        set_tensor_name(kqv, "kqv_flash", il);
        if (sinks) {
            ggml_flash_attn_ext_add_sinks(kqv, sinks);
        }

        ggml_tensor* cur = ggml_cont_2d(ctx, kqv, kqv->ne[0] * kqv->ne[1], kqv->ne[2] * kqv->ne[3]);
        set_tensor_name(cur, "attn_flash_recombined", il);
        return cur;
    }

    v = ggml_permute(ctx, v, 1, 2, 0, 3);
    set_tensor_name(v, "v_permuted", il);
    v = ggml_cont(ctx, v);  // 确保连续内存
    set_tensor_name(v, "v_cont", il);

    ggml_tensor* cur;
    {
        // Q * K^T
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
        set_tensor_name(kq, "kq", il);

        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);  // 使用 F32 精度

        // softmax(Q*K^T / scale + mask)
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0);
        set_tensor_name(kq, "kq_soft", il);

        ggml_soft_max_add_sinks(kq, sinks);  // 添加 sink token

        // attention * V
        ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);
        set_tensor_name(kqv, "kqv", il);

        // 调整输出维度顺序
        cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        set_tensor_name(cur, "kqv_permuted", il);

        // 合并为 2D 张量
        cur = ggml_cont_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        set_tensor_name(cur, "attn_recombined", il);
    }

    return cur;
}

/**
 * @brief 构建 Qwen3.5/Qwen3.6 的门控注意力层
 * 
 * Gated Attention 特点：
 * 1. 联合 Q+Gate 投影
 * 2. Q/K 独立 RMS 归一化
 * 3. Partial RoPE 位置编码
 * 4. 输出端 Sigmoid 门控
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param kv_cache KV 缓存
 * @param cur 输入张量
 * @param inp_pos 位置张量
 * @param kv_cache_layer KV 缓存层索引
 * @param n_tokens token 数量
 * @param slot_idx 槽位索引
 * @param il 层索引
 * @param w_q Q 投影权重
 * @param w_q_norm Q 归一化权重
 * @param w_k K 投影权重
 * @param w_k_norm K 归一化权重
 * @param w_v V 投影权重
 * @param w_out 输出投影权重
 * @param n_embd_head 每头维度
 * @param n_head Q 头数
 * @param n_head_kv KV 头数
 * @param n_rot RoPE 旋转维度
 * @param freq_base RoPE 频率基数
 * @param context_length 上下文长度
 * @param rms_norm_eps RMS 归一化 epsilon
 * @return 注意力输出
 */
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
    float            rms_norm_eps,
    ggml_tensor*     shared_kq_mask,
    ggml_tensor*     kv_write_row,
    uint32_t           fixed_read_n_kv
)
{
    // A. 联合 Q+Gate 投影
    // 输出: [(n_embd_head*2)*n_head, n_tokens]
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx, w_q, cur);
    set_tensor_name(Qcur_full, "Qcur_full", il);

    // B. 通过跨步视图提取 Q
    // Q 和 Gate 交替存储，Q 在偶数位置
    ggml_tensor* Qcur = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    set_tensor_name(Qcur, "Qcur", il);

    // Q RMS 归一化
    Qcur = build_rms_norm(ctx, Qcur, w_q_norm, rms_norm_eps, il);
    set_tensor_name(Qcur, "Qcur_normed", il);

    // C. K 和 V 投影
    ggml_tensor* Kcur = ggml_mul_mat(ctx, w_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx, w_v, cur);

    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    // K RMS 归一化
    Kcur = build_rms_norm(ctx, Kcur, w_k_norm, rms_norm_eps, il);
    set_tensor_name(Kcur, "Kcur_normed", il);

    // D. 提取 Gate
    // Gate 在奇数位置（偏移 n_embd_head）
    ggml_tensor* gate = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);
    set_tensor_name(gate, "gate", il);

    // E. Partial RoPE（部分旋转位置编码）
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // F. KV 缓存写入 + 完整历史读取
    const float    kq_scale  = 1.0f / sqrtf(float(n_embd_head));
    const uint32_t cache_pos = kv_cache->get_pos(slot_idx);
    const uint32_t n_kv      = fixed_read_n_kv > 0 ? fixed_read_n_kv : (cache_pos + n_tokens);

    // 写入 KV 缓存（decode 缓存图用 kv_write_row 在运行时指定行，避免 scratch+copy）
    if (kv_write_row) {
        ggml_build_forward_expand(gf, kv_cache->cpy_k(ctx, Kcur, kv_cache_layer, slot_idx, kv_write_row));
        ggml_build_forward_expand(gf, kv_cache->cpy_v(ctx, Vcur, kv_cache_layer, slot_idx, kv_write_row));
    } else {
        ggml_build_forward_expand(gf, kv_cache->cpy_k(ctx, Kcur, kv_cache_layer, slot_idx));
        ggml_build_forward_expand(gf, kv_cache->cpy_v(ctx, Vcur, kv_cache_layer, slot_idx));
    }

    // 读取完整的 KV 历史（缓存图捕获时放宽 paged 前缀检查，运行时再 materialize）
    const bool relax_paged_prefix = kv_write_row != nullptr;
    ggml_tensor* k_full = kv_cache->get_k(ctx, kv_cache_layer, n_kv, slot_idx, relax_paged_prefix);
    ggml_tensor* v_full = kv_cache->get_v(ctx, kv_cache_layer, n_kv, slot_idx, relax_paged_prefix);

    ggml_tensor* k_view = ggml_view_3d(ctx, k_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * k_full->nb[0], k_full->nb[1], 0);
    ggml_tensor* v_view = ggml_view_3d(ctx, v_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * v_full->nb[0], v_full->nb[1], 0);

    ggml_tensor* kq_mask = shared_kq_mask;
    const bool layer_fa = layer_allows_flash_attn(il);
    if (kq_mask) {
        if (kq_mask->ne[0] != static_cast<int64_t>(n_kv) ||
            kq_mask->ne[1] != static_cast<int64_t>(n_tokens)) {
            throw std::runtime_error("Qwen35moeForwardPass::build_gated_attention: shared mask shape mismatch");
        }
    } else {
        kq_mask = ggml_new_tensor_2d(
            ctx,
            layer_fa ? GGML_TYPE_F16 : GGML_TYPE_F32,
            n_kv,
            n_tokens
        );
        set_tensor_name(kq_mask, "kq_mask", il);
        ggml_build_forward_expand(gf, kq_mask);
    }

    // 计算多头注意力
    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, cache_pos, il, layer_fa);

    // G. Sigmoid 门控
    cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));
    set_tensor_name(cur, "attn_gated", il);

    // H. 输出投影
    cur = ggml_mul_mat(ctx, w_out, cur);
    set_tensor_name(cur, "attn_output", il);

    return cur;
}

/**
 * @brief 构建批处理版本的门控注意力层（Decode 阶段）
 * 
 * 与 build_gated_attention 类似，但支持多序列并行处理
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param kv_cache KV 缓存
 * @param cur 输入张量
 * @param inp_pos 位置张量
 * @param kq_mask 注意力掩码
 * @param gather_indices KV gather 索引
 * @param kv_cache_layer KV 缓存层索引
 * @param slots 槽位索引数组
 * @param positions 位置数组
 * @param il 层索引
 * @param w_q Q 投影权重
 * @param w_q_norm Q 归一化权重
 * @param w_k K 投影权重
 * @param w_k_norm K 归一化权重
 * @param w_v V 投影权重
 * @param w_out 输出投影权重
 * @param n_embd_head 每头维度
 * @param n_head Q 头数
 * @param n_head_kv KV 头数
 * @param n_rot RoPE 旋转维度
 * @param freq_base RoPE 频率基数
 * @param context_length 上下文长度
 * @param rms_norm_eps RMS 归一化 epsilon
 * @return 注意力输出
 */
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

    // A. 联合 Q+Gate 投影 → [(n_embd_head*2)*n_head, n_batch]
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx, w_q, cur);

    // B. 通过跨步视图提取 Q → [n_embd_head, n_head, n_batch]
    ggml_tensor* Qcur = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_batch,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0
    );

    Qcur = build_rms_norm(ctx, Qcur, w_q_norm, rms_norm_eps, il);

    // C. K 和 V 投影
    ggml_tensor* Kcur = ggml_mul_mat(ctx, w_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx, w_v, cur);

    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_batch);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_batch);

    Kcur = build_rms_norm(ctx, Kcur, w_k_norm, rms_norm_eps, il);

    // D. 提取 Gate → [n_embd_head*n_head, n_batch]
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

    // F. 逐槽位写入 KV 缓存
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

    // G. Gather 所有槽位的 KV（活跃长度 = max(pos)+1，图内 padding 由动态 mask 屏蔽）
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p >= 0 && static_cast<uint32_t>(p) > max_pos) {
            max_pos = static_cast<uint32_t>(p);
        }
    }
    uint32_t n_kv_len = max_pos + 1;
    if (kq_mask) {
        const uint32_t graph_n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        if (n_kv_len > graph_n_kv) {
            n_kv_len = graph_n_kv;
        }
    }
    ggml_tensor* k_gathered = kv_cache->gather_k(ctx, gf, kv_cache_layer, gather_indices, n_batch, n_kv_len);
    ggml_tensor* v_gathered = kv_cache->gather_v(ctx, gf, kv_cache_layer, gather_indices, n_batch, n_kv_len);

    ggml_tensor* k_view = ggml_view_4d(ctx, k_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * k_gathered->nb[0],
        k_gathered->nb[1],
        k_gathered->nb[2], 0
    );
    ggml_tensor* v_view = ggml_view_4d(ctx, v_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * v_gathered->nb[0],
        v_gathered->nb[1],
        v_gathered->nb[2], 0
    );

    // H. 注意力计算
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
    const bool layer_fa = layer_allows_flash_attn(il) &&
        kq_mask &&
        kq_mask->type == GGML_TYPE_F16;
    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, 0, il, layer_fa);

    // I. Sigmoid 门控
    cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));

    // J. 输出投影
    cur = ggml_mul_mat(ctx, w_out, cur);

    return cur;
}

/**
 * @brief 构建 DeltaNet 层（核心实现）
 * 
 * DeltaNet 是 Qwen3.5 引入的高效序列建模层，替代传统 Transformer 中的多头注意力机制。
 * 它结合了以下核心组件：
 * 
 * 1. **因果卷积 (Causal Conv1d)**：捕捉局部序列依赖，类似于注意力的窗口机制
 * 2. **门控 Delta 规则 (Gated Delta Rule)**：一种高效的状态更新机制
 * 3. **循环状态 (Recurrent State)**：维护长期依赖的记忆
 * 
 * DeltaNet 的优势：
 * - 时间复杂度 O(n)，远低于标准注意力的 O(n²)
 * - 内存效率更高，无需存储完整的 KV 矩阵
 * - 适合长序列建模
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param cur 输入张量 [n_embd, n_tokens]
 * @param dn_state DeltaNet 状态（包含卷积状态和循环状态）
 * @param dn_idx DeltaNet 层索引（在所有 DeltaNet 层中的位置）
 * @param slot_idx 槽位索引（用于多序列并行）
 * @param n_tokens 当前批次的 token 数量
 * @param w_qkv QKV 混合投影权重 [n_embd, conv_channels]
 * @param w_gate 输出门控权重 [n_embd, d_inner]
 * @param w_beta beta 门控权重 [n_embd, num_v_heads]
 * @param w_a alpha/decay 权重 [n_embd, num_v_heads]
 * @param w_dt_bias dt bias 权重 [num_v_heads]
 * @param w_a_log A_log 权重 [num_v_heads]（用于状态衰减）
 * @param w_conv 卷积核权重
 * @param w_norm RMS 归一化权重 [head_v_dim]
 * @param w_out 输出投影权重 [n_embd, d_inner]
 * @param n_embd 嵌入维度
 * @param d_inner 内部维度 (head_v_dim * num_v_heads)
 * @param head_k_dim K 头维度
 * @param num_k_heads K 头数量
 * @param num_v_heads V 头数量
 * @param head_v_dim V 头维度
 * @param conv_channels 卷积通道数
 * @param conv_kernel 卷积核大小
 * @param rms_norm_eps RMS 归一化 epsilon
 * @param il 物理层索引（用于命名）
 * @return DeltaNet 输出 [n_embd, n_tokens]
 */
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
    // 转换为 int64_t 类型以避免溢出
    const int64_t n_seq_tokens = static_cast<int64_t>(n_tokens);
    const int64_t n_seqs = 1;  // Prefill 阶段处理单个序列

    // =========================================================================
    // 阶段 1: 输入投影 (Input Projections)
    // =========================================================================
    // QKV 混合投影: 将输入嵌入映射到卷积通道空间
    // 输入: cur [n_embd, n_tokens]
    // 权重: w_qkv [n_embd, conv_channels]
    // 输出: qkv_mixed [conv_channels, n_tokens]
    ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, w_qkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    set_tensor_name(qkv_mixed, "dn_qkv", il);

    // Z 门控投影: 用于最后的门控归一化，控制信息流动
    // 输入: cur [n_embd, n_tokens]
    // 权重: w_gate [n_embd, d_inner]
    // 输出: z [d_inner, n_tokens]
    ggml_tensor* z = ggml_mul_mat(ctx, w_gate, cur);
    set_tensor_name(z, "dn_z", il);

    // =========================================================================
    // 阶段 2: Beta 和 Decay Gate 投影
    // =========================================================================
    // Beta 门控: 控制新信息的接受程度（类似注意力中的权重）
    // 经过 sigmoid 激活后取值范围为 (0, 1)
    // 输入: cur [n_embd, n_tokens]
    // 权重: w_beta [n_embd, num_v_heads]
    // 输出: beta [1, num_v_heads, n_tokens, 1]
    ggml_tensor* beta = ggml_mul_mat(ctx, w_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);
    set_tensor_name(beta, "dn_beta", il);

    // Decay Gate: 控制循环状态的衰减速率，决定历史信息的遗忘程度
    // 公式: decay = softplus(alpha @ x + dt_bias) * A_log
    // softplus 确保输出非负，A_log 是可学习的对数衰减因子
    // 输入: cur [n_embd, n_tokens]
    // 权重: w_a [n_embd, num_v_heads]
    // 输出: decay_gate [1, num_v_heads, n_tokens, 1]
    ggml_tensor* alpha = ggml_mul_mat(ctx, w_a, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* alpha_biased = ggml_add(ctx, alpha, w_dt_bias);  // 添加时间步偏置
    ggml_tensor* alpha_sp     = ggml_softplus(ctx, alpha_biased);  // softplus 激活
    ggml_tensor* decay_gate   = ggml_mul(ctx, alpha_sp, w_a_log);  // 乘以对数衰减因子
    decay_gate = ggml_reshape_4d(ctx, decay_gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    set_tensor_name(decay_gate, "dn_decay", il);

    // =========================================================================
    // 阶段 3: 因果卷积 (Causal Conv1d)
    // =========================================================================
    // 获取该层的卷积状态张量（存储所有槽位的卷积状态）
    ggml_tensor* conv_all = dn_state->conv_tensor(dn_idx);
    const int64_t conv_state_elems = static_cast<int64_t>(conv_kernel - 1) * conv_channels;

    // 提取当前槽位的滑动窗口状态
    // 卷积状态存储了前 (conv_kernel - 1) 个 token 的信息，用于构建因果卷积窗口
    ggml_tensor* conv_states = ggml_view_1d(ctx, conv_all, conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    conv_states = ggml_reshape_3d(ctx, conv_states, conv_kernel - 1, conv_channels, n_seqs);
    set_tensor_name(conv_states, "dn_conv_st", il);

    // 沿时间轴拼接历史状态和新的 QKV token
    // conv_input: [conv_kernel, conv_channels, n_seqs]
    ggml_tensor* qkv_t = ggml_transpose(ctx, qkv_mixed);
    ggml_tensor* conv_input = ggml_concat(ctx, conv_states, qkv_t, 0);
    set_tensor_name(conv_input, "dn_conv_in", il);

    // 更新卷积状态：保留最后 (conv_kernel - 1) 个 token
    // 这是实现因果性的关键：只保留最近的历史，防止信息泄露
    const bool mtp_snap = mtp_snapshot_build_ &&
        static_cast<size_t>(dn_idx) < mtp_rec_snap_.size() &&
        mtp_rec_snap_[dn_idx] != nullptr && mtp_conv_snap_[dn_idx] != nullptr;
    if (mtp_snap) {
        // MTP verify: capture the conv window AFTER every token (not just the
        // final one) into successive columns of this layer's snapshot scratch.
        // Window after token t = conv_input rows [t+1 .. t+kernel-1] over all
        // channels — same shape/strides as `last_conv` below (which is exactly the
        // t == n_tok-1 case), just shifted by t rows. A per-token loop avoids an
        // overlapping-stride 3D view (which ggml's contiguous-size bounds check
        // rejects).
        for (int64_t t = 0; t < n_seq_tokens; ++t) {
            ggml_tensor* win_t = ggml_view_3d(ctx, conv_input,
                conv_kernel - 1, conv_channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                static_cast<size_t>(t + 1) * conv_input->nb[0]);
            ggml_tensor* col_t = ggml_view_1d(ctx, mtp_conv_snap_[dn_idx],
                conv_state_elems, static_cast<size_t>(t) * mtp_conv_snap_[dn_idx]->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, win_t, col_t));
        }
    } else {
        ggml_tensor* last_conv = ggml_view_3d(ctx, conv_input,
            conv_kernel - 1, conv_channels, n_seqs,
            conv_input->nb[1], conv_input->nb[2],
            static_cast<size_t>(conv_input->ne[0] - (conv_kernel - 1)) * ggml_element_size(conv_input)
        );
        ggml_tensor* conv_dst = ggml_view_1d(ctx, conv_all, conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, last_conv, conv_dst));
    }

    // 执行深度卷积 + SiLU 激活
    // 深度卷积：每个通道独立卷积，参数共享，计算高效
    ggml_tensor* conv_out = ggml_ssm_conv(ctx, conv_input, w_conv);
    conv_out = ggml_silu(ctx, conv_out);
    set_tensor_name(conv_out, "dn_conv_out", il);

    // =========================================================================
    // 阶段 4: 拆分卷积输出为 Q, K, V
    // =========================================================================
    // QKV 在卷积输出中是连续存储的，需要通过视图拆分
    // 布局: [Q | K | V] = [head_k_dim*num_k_heads | head_k_dim*num_k_heads | head_v_dim*num_v_heads]
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

    // =========================================================================
    // 阶段 5: Q 和 K 的 L2 归一化
    // =========================================================================
    // L2 归一化使 Q 和 K 的范数为 1，有助于稳定训练和推理
    // 避免因范数差异导致的梯度不稳定
    q_conv = ggml_l2_norm(ctx, q_conv, rms_norm_eps);
    k_conv = ggml_l2_norm(ctx, k_conv, rms_norm_eps);

    // GQA 风格的头部匹配：如果 K 头数不等于 V 头数，重复 K/Q 头
    // 例如: num_k_heads=16, num_v_heads=32，则每个 K 头对应 2 个 V 头
    // 这样可以减少计算量同时保持表达能力
    if (num_k_heads != num_v_heads) {
        q_conv = ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // =========================================================================
    // 阶段 6: 门控 Delta Net 循环（核心操作）
    // =========================================================================
    if (head_k_dim != head_v_dim) {
        throw std::runtime_error(
            "build_deltanet_layer: ggml_gated_delta_net requires head_k_dim == head_v_dim");
    }
    ggml_tensor* rec_all = dn_state->recurrent_tensor(dn_idx);
    const int64_t rec_slot_floats = gdn_rec_slot_floats(head_v_dim, num_v_heads);

    ggml_tensor* S = ggml_view_1d(ctx, rec_all, rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    if (mtp_snap) {
        // MTP verify snapshot mode: request K = n_tok per-token state snapshots by
        // widening the input state's slot dimension to n_tok (plane 0 = current
        // state; padded planes are ignored by the kernel, which reads slot 0).
        S = ggml_reshape_3d(ctx, S, rec_slot_floats, 1, 1);
        S = ggml_pad(ctx, S, 0, static_cast<int>(n_seq_tokens) - 1, 0, 0);
    } else {
        S = wrap_gdn_recurrent_state(ctx, S, rec_slot_floats, n_seqs);
    }
    set_tensor_name(S, "dn_state_in", il);

    // 调用融合的门控 Delta Net 操作
    // Delta 规则: S_{t+1} = (1 - decay) * S_t + beta * outer(K, V)
    // 输出打包了两部分内容：
    //   1. per-token output: [head_v_dim, num_v_heads, n_tokens, n_seqs]
    //   2. final state:      [head_v_dim, head_v_dim, num_v_heads, n_seqs] (S_v×S_v per head)
    ggml_tensor* result = ggml_gated_delta_net(ctx, q_conv, k_conv, v_conv, decay_gate, beta, S);
    set_tensor_name(result, "dn_gdn_result", il);

    // 提取 per-token 输出（结果的第一部分）
    ggml_tensor* output = ggml_view_4d(ctx, result,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * num_v_heads),
        ggml_row_size(result->type, head_v_dim * num_v_heads * n_seq_tokens),
        0
    );
    set_tensor_name(output, "dn_delta_out", il);

    // 提取并写回新的循环状态
    const size_t rec_snap_off =
        gdn_attn_scores_byte_offset(head_v_dim, num_v_heads, n_seq_tokens, n_seqs, result->type);
    if (mtp_snap) {
        // K = n_tok snapshots are contiguous at rec_snap_off; column t = state
        // after token t. Copy them all into this layer's snapshot scratch instead
        // of committing to dn_state (the committed column is selected post-verify).
        ggml_tensor* snaps = ggml_view_2d(ctx, result,
            rec_slot_floats, n_seq_tokens,
            ggml_row_size(result->type, rec_slot_floats),
            rec_snap_off);
        ggml_tensor* rec_snap_dst = ggml_view_2d(ctx, mtp_rec_snap_[dn_idx],
            rec_slot_floats, n_seq_tokens, mtp_rec_snap_[dn_idx]->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, snaps, rec_snap_dst));
    } else {
        // K=1 snapshot: the single final state.
        ggml_tensor* new_state = ggml_view_1d(ctx, result, rec_slot_floats, rec_snap_off);
        ggml_tensor* rec_dst = ggml_view_1d(ctx, rec_all, rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, rec_dst));
    }

    // =========================================================================
    // 阶段 7: 门控 RMSNorm
    // =========================================================================
    // 对输出进行归一化，并应用门控机制
    // 类似于 Transformer 中的 pre-attention norm，但增加了门控来控制信息流
    // w_norm 是 [head_v_dim]，通过广播应用到所有 head 和 token
    ggml_tensor* z_4d   = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* normed = ggml_rms_norm(ctx, output, rms_norm_eps);
    normed = ggml_mul(ctx, normed, w_norm);
    // gated = silu(z_4d) * normed
    //   - 老路径: silu + mul，两个独立内核 + 一个中间张量
    //   - 新路径: ggml_swiglu_split(z_4d, normed) 把 silu+mul 合到
    //     GGML_OP_GLU 一个内核里（CPU/CUDA 后端都已实现），少一次
    //     launch / 一次中间分配。形状/类型/行连续约束天然满足。
    ggml_tensor* gated  = ggml_swiglu_split(ctx, z_4d, normed);
    set_tensor_name(gated, "dn_gated", il);

    // =========================================================================
    // 阶段 8: 输出投影
    // =========================================================================
    // 将内部维度投影回嵌入维度
    // gated: [head_v_dim, num_v_heads, n_tokens, n_seqs]
    // flat: [head_v_dim * num_v_heads, n_tokens, n_seqs]
    // out_proj: [n_embd, n_tokens]
    ggml_tensor* flat = ggml_reshape_3d(ctx, gated, static_cast<int64_t>(head_v_dim) * num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* out_proj = ggml_mul_mat(ctx, w_out, flat);
    set_tensor_name(out_proj, "dn_output", il);
    out_proj = ggml_reshape_2d(ctx, out_proj, n_embd, n_seq_tokens * n_seqs);

    return out_proj;
}

ggml_tensor* Qwen35moeForwardPass::build_deltanet_layer_decode(
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
    int             il)
{
    (void)d_inner;
    const int64_t n_seq_tokens = 1;
    const int64_t n_seqs = 1;

    const int64_t n_qkv   = w_qkv->ne[1];
    const int64_t n_gate  = w_gate->ne[1];
    const int64_t n_beta  = w_beta->ne[1];
    const int64_t n_alpha = w_a->ne[1];

    ggml_tensor* qkv_mixed = nullptr;
    ggml_tensor* z         = nullptr;
    ggml_tensor* beta      = nullptr;
    ggml_tensor* alpha     = nullptr;

    if (deltanet_decode_fused_proj_enabled(w_qkv, w_gate, w_beta, w_a)) {
        ggml_tensor* proj = ggml_deltanet_decode_proj(ctx, cur, w_qkv, w_gate, w_beta, w_a);
        set_tensor_name(proj, "dn_proj", il);

        qkv_mixed = ggml_view_1d(ctx, proj, n_qkv, 0);
        qkv_mixed = ggml_reshape_2d(ctx, qkv_mixed, n_qkv, 1);
        set_tensor_name(qkv_mixed, "dn_qkv", il);

        z = ggml_view_1d(ctx, proj, n_gate, static_cast<size_t>(n_qkv) * ggml_element_size(proj));
        z = ggml_reshape_2d(ctx, z, n_gate, 1);
        set_tensor_name(z, "dn_z", il);

        alpha = ggml_view_1d(
            ctx, proj, n_alpha,
            static_cast<size_t>(n_qkv + n_gate + n_beta) * ggml_element_size(proj));
        alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);

        beta = ggml_view_1d(
            ctx, proj, n_beta,
            static_cast<size_t>(n_qkv + n_gate) * ggml_element_size(proj));
        beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
        beta = ggml_sigmoid(ctx, beta);
        set_tensor_name(beta, "dn_beta", il);
    } else {
        qkv_mixed = ggml_mul_mat(ctx, w_qkv, cur);
        set_tensor_name(qkv_mixed, "dn_qkv", il);

        z = ggml_mul_mat(ctx, w_gate, cur);
        set_tensor_name(z, "dn_z", il);

        beta = ggml_mul_mat(ctx, w_beta, cur);
        beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
        beta = ggml_sigmoid(ctx, beta);
        set_tensor_name(beta, "dn_beta", il);

        alpha = ggml_mul_mat(ctx, w_a, cur);
        alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor* alpha_biased = ggml_add(ctx, alpha, w_dt_bias);
    ggml_tensor* alpha_sp     = ggml_softplus(ctx, alpha_biased);
    ggml_tensor* decay_gate   = ggml_mul(ctx, alpha_sp, w_a_log);
    decay_gate = ggml_reshape_4d(ctx, decay_gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    set_tensor_name(decay_gate, "dn_decay", il);

    ggml_tensor* conv_all = dn_state->conv_tensor(dn_idx);
    const int64_t conv_state_elems = static_cast<int64_t>(conv_kernel - 1) * conv_channels;

    ggml_tensor* conv_states = ggml_view_1d(ctx, conv_all, conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    conv_states = ggml_reshape_2d(ctx, conv_states, conv_kernel - 1, conv_channels);
    set_tensor_name(conv_states, "dn_conv_st", il);

    ggml_tensor* step_result = ggml_ssm_conv_step(ctx, conv_states, qkv_mixed, w_conv);
    set_tensor_name(step_result, "dn_conv_step", il);

    ggml_tensor* new_conv_state = ggml_view_1d(
        ctx, step_result, conv_state_elems, static_cast<size_t>(conv_channels) * ggml_element_size(step_result));
    ggml_tensor* conv_dst = ggml_view_1d(ctx, conv_all, conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_conv_state, conv_dst));

    ggml_tensor* conv_out = ggml_view_1d(ctx, step_result, conv_channels, 0);
    conv_out = ggml_reshape_3d(ctx, conv_out, conv_channels, n_seq_tokens, n_seqs);
    conv_out = ggml_silu(ctx, conv_out);
    set_tensor_name(conv_out, "dn_conv_out", il);

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

    q_conv = ggml_l2_norm(ctx, q_conv, rms_norm_eps);
    k_conv = ggml_l2_norm(ctx, k_conv, rms_norm_eps);

    if (num_k_heads != num_v_heads) {
        q_conv = ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    if (head_k_dim != head_v_dim) {
        throw std::runtime_error(
            "build_deltanet_layer_decode: ggml_gated_delta_net requires head_k_dim == head_v_dim");
    }

    ggml_tensor* rec_all = dn_state->recurrent_tensor(dn_idx);
    const int64_t rec_slot_floats = gdn_rec_slot_floats(head_v_dim, num_v_heads);

    ggml_tensor* S = ggml_view_1d(ctx, rec_all, rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    S = wrap_gdn_recurrent_state(ctx, S, rec_slot_floats, n_seqs);
    set_tensor_name(S, "dn_state_in", il);

    ggml_tensor* result = ggml_gated_delta_net(ctx, q_conv, k_conv, v_conv, decay_gate, beta, S);
    set_tensor_name(result, "dn_gdn_result", il);

    ggml_tensor* output = ggml_view_4d(ctx, result,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * num_v_heads),
        ggml_row_size(result->type, head_v_dim * num_v_heads * n_seq_tokens),
        0
    );
    set_tensor_name(output, "dn_delta_out", il);

    ggml_tensor* new_state = ggml_view_1d(
        ctx,
        result,
        rec_slot_floats,
        gdn_attn_scores_byte_offset(head_v_dim, num_v_heads, n_seq_tokens, n_seqs, result->type)
    );

    ggml_tensor* rec_dst = ggml_view_1d(ctx, rec_all, rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, rec_dst));

    ggml_tensor* z_4d   = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* normed = ggml_rms_norm(ctx, output, rms_norm_eps);
    normed = ggml_mul(ctx, normed, w_norm);
    ggml_tensor* gated  = ggml_swiglu_split(ctx, z_4d, normed);
    set_tensor_name(gated, "dn_gated", il);

    ggml_tensor* flat = ggml_reshape_3d(ctx, gated, static_cast<int64_t>(head_v_dim) * num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* out_proj = ggml_mul_mat(ctx, w_out, flat);
    set_tensor_name(out_proj, "dn_output", il);
    out_proj = ggml_reshape_2d(ctx, out_proj, n_embd, n_seq_tokens * n_seqs);

    return out_proj;
}

/**
 * @brief 批化 DeltaNet 解码层构建（多 slot 并行）
 *
 * 与 build_deltanet_layer 数学等价，但把每步 decode 的 N 个 slot 视为
 * N 个长度为 1 的序列（n_seq_tokens=1, n_seqs=N），并用一对
 * ggml_get_rows / ggml_set_rows 完成 conv-state 与 recurrent-state 的
 * 按 slot 收集 / 散布（gather/scatter），从而避免老路径里
 * 逐 slot 切片 → 调用 prefill → 拼接输出 带来的 N 倍小算子开销。
 *
 * slot 索引来自 graph 范围共享的输入张量 "dn_slot_idx"（int32, [n_seqs]），
 * 由 set_batched_inputs / set_cached_batched_decode_inputs 在执行前写入。
 *
 * 数学路径与 build_deltanet_layer 完全一致：QKV 投影 → Z gate → beta/decay
 * → 因果 conv1d → L2 norm → ggml_gated_delta_net（融合） → 状态写回
 * → gated RMSNorm → 输出投影。区别仅在状态读写从单 slot view 变为
 * 跨 slot 的 gather/scatter，并在 n_seqs 维度自然批化。
 *
 * @return DeltaNet 输出 [n_embd, n_batch]
 */
ggml_tensor* Qwen35moeForwardPass::build_deltanet_layer_batched_decode(
    ggml_context*   ctx,
    ggml_cgraph*    gf,
    ggml_tensor*    cur,
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
    int             il)
{
    (void)d_inner;
    const int64_t n_seq_tokens = 1;
    const int64_t n_seqs = static_cast<int64_t>(slots.size());
    if (n_seqs <= 0) {
        throw std::runtime_error("build_deltanet_layer_batched_decode: slots must be non-empty");
    }

    // 阶段 1: 输入投影（QKV mix + Z gate） — 与 prefill 等价，
    // 只是 n_seq_tokens 与 n_seqs 角色互换。
    ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, w_qkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    set_tensor_name(qkv_mixed, "dn_qkv", il);

    ggml_tensor* z = ggml_mul_mat(ctx, w_gate, cur);
    set_tensor_name(z, "dn_z", il);

    // 阶段 2: Beta 与 Decay gate
    ggml_tensor* beta = ggml_mul_mat(ctx, w_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);
    set_tensor_name(beta, "dn_beta", il);

    ggml_tensor* alpha = ggml_mul_mat(ctx, w_a, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* alpha_biased = ggml_add(ctx, alpha, w_dt_bias);
    ggml_tensor* alpha_sp     = ggml_softplus(ctx, alpha_biased);
    ggml_tensor* decay_gate   = ggml_mul(ctx, alpha_sp, w_a_log);
    decay_gate = ggml_reshape_4d(ctx, decay_gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    set_tensor_name(decay_gate, "dn_decay", il);

    // 阶段 3a: slot 索引输入张量（graph 范围共享，多 DN 层只生成一次）
    ggml_tensor* slots_idx = ggml_graph_get_tensor(gf, "dn_slot_idx");
    if (slots_idx == nullptr) {
        slots_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
        ggml_set_input(slots_idx);
        set_tensor_name(slots_idx, "dn_slot_idx");
        ggml_build_forward_expand(gf, slots_idx);
    } else if (slots_idx->ne[0] != n_seqs) {
        // 防御性检查：同一 graph 内所有 DN 层共享同一个 slot 索引输入，
        // 因此长度必须一致。若不一致说明上层调用违反了批一致性约定。
        throw std::runtime_error(
            "build_deltanet_layer_batched_decode: shared dn_slot_idx tensor size mismatch"
        );
    }

    // 阶段 3b: 因果卷积 — 按 slot 收集历史状态
    ggml_tensor* conv_all = dn_state->conv_tensor(dn_idx);
    const int64_t conv_state_elems = static_cast<int64_t>(conv_kernel - 1) * conv_channels;

    // 一次性 gather N 个 slot 的 conv state：
    // conv_all [conv_state_elems, n_slots_total]
    //   -> [conv_state_elems, n_seqs]，再 reshape 到 [conv_kernel-1, conv_channels, n_seqs]
    ggml_tensor* conv_states = ggml_get_rows(ctx, conv_all, slots_idx);
    conv_states = ggml_reshape_3d(ctx, conv_states, conv_kernel - 1, conv_channels, n_seqs);
    set_tensor_name(conv_states, "dn_conv_st", il);

    // 拼接历史 + 新 QKV token => [conv_kernel, conv_channels, n_seqs]
    ggml_tensor* qkv_t     = ggml_transpose(ctx, qkv_mixed);
    ggml_tensor* conv_input = ggml_concat(ctx, conv_states, qkv_t, 0);
    set_tensor_name(conv_input, "dn_conv_in", il);

    // 取最后 (conv_kernel-1) 个 token 作为新的 conv state（每个 slot 独立）
    ggml_tensor* last_conv = ggml_view_3d(
        ctx, conv_input,
        conv_kernel - 1, conv_channels, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        static_cast<size_t>(conv_input->ne[0] - (conv_kernel - 1)) * ggml_element_size(conv_input)
    );
    // 这个 view 的行步长继承自 conv_input（=conv_kernel*sizeof），
    // 与 (conv_kernel-1)*sizeof 不连续，set_rows 要求源行连续 ⇒ cont 一下。
    ggml_tensor* last_conv_flat = ggml_cont_2d(ctx, last_conv, conv_state_elems, n_seqs);
    // 一次性 scatter 回 N 个 slot：等价于 N 个 ggml_cpy 但只产生一个算子。
    ggml_tensor* conv_write = ggml_set_rows(ctx, conv_all, last_conv_flat, slots_idx);
    ggml_build_forward_expand(gf, conv_write);

    // 深度卷积 + SiLU
    ggml_tensor* conv_out = ggml_ssm_conv(ctx, conv_input, w_conv);
    conv_out = ggml_silu(ctx, conv_out);
    set_tensor_name(conv_out, "dn_conv_out", il);

    // 阶段 4: 拆 Q/K/V（与 prefill 完全一致，只是 ne[2]/ne[3] 角色不同）
    const int64_t qkv_dim = static_cast<int64_t>(head_k_dim) * num_k_heads * 2
                          + static_cast<int64_t>(head_v_dim) * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

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

    // 阶段 5: L2 norm + 必要时做 GQA 头复制
    q_conv = ggml_l2_norm(ctx, q_conv, rms_norm_eps);
    k_conv = ggml_l2_norm(ctx, k_conv, rms_norm_eps);
    if (num_k_heads != num_v_heads) {
        q_conv = ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // 阶段 6: gated DeltaNet（一次性处理 N 个 slot）
    if (head_k_dim != head_v_dim) {
        throw std::runtime_error(
            "build_deltanet_layer_batched_decode: ggml_gated_delta_net requires head_k_dim == head_v_dim");
    }
    ggml_tensor* rec_all = dn_state->recurrent_tensor(dn_idx);
    const int64_t rec_slot_floats = gdn_rec_slot_floats(head_v_dim, num_v_heads);

    ggml_tensor* S_gathered = ggml_get_rows(ctx, rec_all, slots_idx);
    ggml_tensor* S = wrap_gdn_recurrent_state(ctx, S_gathered, rec_slot_floats, n_seqs);
    set_tensor_name(S, "dn_state_in", il);

    ggml_tensor* result = ggml_gated_delta_net(ctx, q_conv, k_conv, v_conv, decay_gate, beta, S);
    set_tensor_name(result, "dn_gdn_result", il);

    // 提取 per-token 输出（result 的前半段）
    ggml_tensor* output = ggml_view_4d(ctx, result,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * num_v_heads),
        ggml_row_size(result->type, head_v_dim * num_v_heads * n_seq_tokens),
        0
    );
    set_tensor_name(output, "dn_delta_out", il);

    // 提取新状态（result 的后半段）并按 slot 一次性 scatter 回 rec_all
    ggml_tensor* new_state = ggml_view_2d(
        ctx,
        result,
        rec_slot_floats,
        n_seqs,
        result->nb[0],
        gdn_attn_scores_byte_offset(head_v_dim, num_v_heads, n_seq_tokens, n_seqs, result->type)
    );

    ggml_tensor* new_state_flat = ggml_cont_2d(ctx, new_state, rec_slot_floats, n_seqs);
    ggml_tensor* rec_write = ggml_set_rows(ctx, rec_all, new_state_flat, slots_idx);
    ggml_build_forward_expand(gf, rec_write);

    // 阶段 7: gated RMSNorm — 与 prefill 对应路径同款 swiglu_split 融合
    ggml_tensor* z_4d   = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* normed = ggml_rms_norm(ctx, output, rms_norm_eps);
    normed = ggml_mul(ctx, normed, w_norm);
    ggml_tensor* gated  = ggml_swiglu_split(ctx, z_4d, normed);
    set_tensor_name(gated, "dn_gated", il);

    // 阶段 8: 输出投影 → [n_embd, n_batch]
    ggml_tensor* flat = ggml_reshape_3d(ctx, gated, static_cast<int64_t>(head_v_dim) * num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* out_proj = ggml_mul_mat(ctx, w_out, flat);
    set_tensor_name(out_proj, "dn_output", il);
    out_proj = ggml_reshape_2d(ctx, out_proj, n_embd, n_seq_tokens * n_seqs);

    return out_proj;
}

/**
 * @brief 构建 DeltaNet 层（封装函数）
 * 
 * 为物理层 il 构建 DeltaNet 子图，封装了权重获取逻辑
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param cur 输入张量
 * @param dn_state DeltaNet 状态
 * @param state_hp DeltaNet 状态参数
 * @param num_k_heads K 头数量
 * @param n_embd 嵌入维度
 * @param dn_idx DeltaNet 层索引
 * @param n_tokens token 数量
 * @param slot_idx 槽位索引
 * @param rms_norm_eps RMS 归一化 epsilon
 * @param il 物理层索引
 * @return DeltaNet 输出
 */
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
    // 获取 DeltaNet 层的各项权重
    struct ggml_tensor* attn_qkv_weight = model_->get_attn_qkv_weight(il);
    struct ggml_tensor* attn_gate_weight = model_->get_attn_gate_weight(il);
    struct ggml_tensor* ssm_beta_weight = model_->get_ssm_beta_weight(il);
    struct ggml_tensor* ssm_alpha_weight = model_->get_ssm_alpha_weight(il);
    struct ggml_tensor* ssm_dt_bias = model_->get_ssm_dt_weight(il);
    struct ggml_tensor* ssm_a = model_->get_ssm_a_weight(il);
    struct ggml_tensor* ssm_conv1d_weight = model_->get_ssm_conv1d_weight(il);
    struct ggml_tensor* ssm_norm_weight = model_->get_ssm_norm_weight(il);
    struct ggml_tensor* ssm_out_weight = model_->get_ssm_out_weight(il);

    if (n_tokens == 1 && layer_allows_deltanet_decode_fast_path(il)) {
        const bool fused_proj = deltanet_decode_fused_proj_enabled(
            attn_qkv_weight, attn_gate_weight, ssm_beta_weight, ssm_alpha_weight);
        if (env_flag_enabled("QWEN35MOE_DN_DECODE_VERIFY") && dn_idx == 0) {
            std::fprintf(stderr,
                "[VERIFY] DeltaNet decode path active (n_tokens=1): fused_proj=%d ssm_conv_step=1 layer=%d\n",
                fused_proj ? 1 : 0,
                il);
            std::fflush(stderr);
        }
        return build_deltanet_layer_decode(
            ctx, gf, cur, dn_state, dn_idx, slot_idx,
            attn_qkv_weight, attn_gate_weight, ssm_beta_weight, ssm_alpha_weight,
            ssm_dt_bias, ssm_a, ssm_conv1d_weight, ssm_norm_weight, ssm_out_weight,
            static_cast<int>(n_embd),
            static_cast<int>(state_hp.head_v_dim * state_hp.num_v_heads),
            static_cast<int>(state_hp.head_k_dim),
            static_cast<int>(num_k_heads),
            static_cast<int>(state_hp.num_v_heads),
            static_cast<int>(state_hp.head_v_dim),
            static_cast<int>(state_hp.conv_channels),
            static_cast<int>(state_hp.conv_kernel),
            rms_norm_eps,
            il);
    }

    // 调用核心 DeltaNet 构建函数
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

/**
 * @brief DeltaNet 层统一构建入口
 * 
 * 根据阶段（Prefill/Decode）调用不同的构建函数
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param input 输入张量
 * @param dn_idx DeltaNet 层索引
 * @param phase 阶段（Prefill/Decode）
 * @param prefill_args Prefill 参数
 * @param decode_args Decode 参数
 * @param dn_state DeltaNet 状态
 * @param hp DeltaNet 参数
 * @param il 物理层索引
 * @return DeltaNet 输出 [n_embd, n_tokens/n_batch]
 */
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

/**
 * @brief Prefill 阶段的 DeltaNet 构建
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param input 输入张量
 * @param dn_idx DeltaNet 层索引
 * @param prefill_args Prefill 参数
 * @param dn_state DeltaNet 状态
 * @param hp DeltaNet 参数
 * @param il 物理层索引
 * @return DeltaNet 输出
 */
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
    // 从输入形状推导 n_embd
    const int n_embd = static_cast<int>(input->ne[0]);

    // 获取权重并调用核心构建函数
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

/**
 * @brief Decode 阶段的 DeltaNet 构建（支持多序列并行）
 * 
 * Decode 阶段每个槽位处理一个 token，但支持多槽位并行
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param input 输入张量
 * @param dn_idx DeltaNet 层索引
 * @param decode_args Decode 参数
 * @param dn_state DeltaNet 状态
 * @param hp DeltaNet 参数
 * @param il 物理层索引
 * @return DeltaNet 输出
 */
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
    // Decode: 每个 slot 一个 token。
    //   - n_batch == 1: 走 decode 专用路径（ssm_conv_step，避免 conv concat）
    //   - n_batch  > 1: 走批化 decode 路径
    //       build_deltanet_layer_batched_decode：把每个 slot 视为一条
    //       长度为 1 的序列（n_seqs=n_batch），用一对
    //       ggml_get_rows/ggml_set_rows 完成 conv & recurrent state 的
    //       gather/scatter，所有 matmul / conv / gated_delta_net 在
    //       n_seqs 维度上自然批化。
    const int n_batch = static_cast<int>(decode_args.slots.size());

    if (n_batch == 0) {
        throw std::runtime_error("DeltaNetLayer::build_decode: slots must be non-empty");
    }

    const int n_embd = static_cast<int>(input->ne[0]);

    struct ggml_tensor* attn_qkv_weight     = model_->get_attn_qkv_weight(il);
    struct ggml_tensor* attn_gate_weight    = model_->get_attn_gate_weight(il);
    struct ggml_tensor* ssm_beta_weight     = model_->get_ssm_beta_weight(il);
    struct ggml_tensor* ssm_alpha_weight    = model_->get_ssm_alpha_weight(il);
    struct ggml_tensor* ssm_dt_bias         = model_->get_ssm_dt_weight(il);
    struct ggml_tensor* ssm_a               = model_->get_ssm_a_weight(il);
    struct ggml_tensor* ssm_conv1d_weight   = model_->get_ssm_conv1d_weight(il);
    struct ggml_tensor* ssm_norm_weight     = model_->get_ssm_norm_weight(il);
    struct ggml_tensor* ssm_out_weight      = model_->get_ssm_out_weight(il);

    if (n_batch == 1 && layer_allows_deltanet_decode_fast_path(il)) {
        return build_deltanet_layer_decode(
            ctx, gf, input, dn_state, dn_idx, decode_args.slots[0],
            attn_qkv_weight, attn_gate_weight, ssm_beta_weight, ssm_alpha_weight,
            ssm_dt_bias, ssm_a, ssm_conv1d_weight, ssm_norm_weight, ssm_out_weight,
            n_embd, hp.d_inner,
            hp.head_k_dim, hp.num_k_heads, hp.num_v_heads, hp.head_v_dim,
            hp.conv_channels, hp.conv_kernel, hp.rms_norm_eps,
            il
        );
    }

    if (n_batch == 1) {
        return build_deltanet_layer(
            ctx, gf, input, dn_state,
            dn_idx, decode_args.slots[0], 1,
            attn_qkv_weight, attn_gate_weight, ssm_beta_weight, ssm_alpha_weight,
            ssm_dt_bias, ssm_a, ssm_conv1d_weight, ssm_norm_weight, ssm_out_weight,
            n_embd, hp.d_inner,
            hp.head_k_dim, hp.num_k_heads, hp.num_v_heads, hp.head_v_dim,
            hp.conv_channels, hp.conv_kernel, hp.rms_norm_eps,
            static_cast<int>(dn_idx)
        );
    }

    // 多 slot 批化路径
    return build_deltanet_layer_batched_decode(
        ctx, gf, input, dn_state, dn_idx, decode_args.slots,
        attn_qkv_weight, attn_gate_weight, ssm_beta_weight, ssm_alpha_weight,
        ssm_dt_bias, ssm_a, ssm_conv1d_weight, ssm_norm_weight, ssm_out_weight,
        n_embd, hp.d_inner,
        hp.head_k_dim, hp.num_k_heads, hp.num_v_heads, hp.head_v_dim,
        hp.conv_channels, hp.conv_kernel, hp.rms_norm_eps,
        static_cast<int>(dn_idx)
    );
}

/**
 * @brief 设置张量名称（带层索引）
 * 
 * @param tensor 张量指针
 * @param name 基础名称
 * @param il 层索引（-1 表示不带索引）
 */
void Qwen35moeForwardPass::set_tensor_name(ggml_tensor* tensor, const char* name, int il) const {
    if (il != -1) {
        char new_name[128];
        snprintf(new_name, sizeof(new_name), "%s.%d", name, il);
        ggml_set_name(tensor, new_name);
    } else {
        ggml_set_name(tensor, name);
    }
}

/**
 * @brief 构建输出头（LM Head）
 * 
 * 包含最终归一化和线性投影到 vocab 空间
 * 
 * @param gf 计算图
 * @param cur 输入张量（Transformer 最后一层输出）
 */
void Qwen35moeForwardPass::build_output_head(ggml_cgraph* gf, ggml_tensor* cur) {
    // 最终 RMS 归一化
    struct ggml_tensor* output_norm = model_->get_output_norm_weight();
    cur = build_norm(gf, cur, output_norm, -1);
    set_tensor_name(cur, "final_norm");

    // LM Head 投影
    // 如果有独立的输出权重则使用，否则复用嵌入矩阵（权重共享）
    struct ggml_tensor* output = model_->get_output_weight();
    struct ggml_tensor* token_embedding = model_->get_token_embedding_weight();
    if (output != nullptr) {
        cur = ggml_mul_mat(ctx_, output, cur);
    } else {
        cur = ggml_mul_mat(ctx_, token_embedding, cur);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    if (sampling_top_k_ > 0) {
        const int64_t n_vocab = cur->ne[0];
        const int64_t n_tokens = cur->ne[1];
        int64_t k = sampling_top_k_;
        if (k > n_vocab) {
            k = n_vocab;
        }
        if (k <= 0) {
            return;
        }

        // 快速路径：只需要 argmax 的情形
        //   1) k == 1（用户显式 top_k=1）
        //   2) sampling_temperature_ <= 0（greedy；sample_from_topk_candidates
        //      在 temperature_<=0 时直接返回 token_ids[0]，不读 logits）
        //
        // 收益：
        // - argmax 不受单调正缩放影响，跳过 ggml_scale 节省一次全词表 element-wise
        // - 用 ggml_argmax 替代 ggml_top_k：CUDA 后端 top_k 在 CUB < 3.2 时会
        //   退化成整词表 argsort（vocab≈248K，开销巨大），argmax 是单次 reduce
        // - 跳过 ggml_get_rows 拉 topk logits
        const bool argmax_only = (k == 1) || (sampling_temperature_ <= 0.0f);
        if (argmax_only) {
            ggml_tensor* argmax_idx = ggml_argmax(ctx_, cur);
            // argmax_idx: I32, shape [n_tokens]
            // reshape 为 [1, n_tokens]，让 get_output_topk_candidates 用同一份
            // (token_col * nb[1]) 偏移逻辑读取
            ggml_tensor* topk_indices = ggml_reshape_2d(ctx_, argmax_idx, 1, n_tokens);
            set_tensor_name(topk_indices, "sample_topk_idx");
            ggml_build_forward_expand(gf, topk_indices);
            return;
        }

        // 通用 top_k 路径
        ggml_tensor* sampling_logits = cur;
        if (sampling_temperature_ > 0.0f) {
            const float inv_temp = 1.0f / sampling_temperature_;
            if (!std::isfinite(inv_temp)) {
                throw std::runtime_error("invalid sampling temperature for device-side scaling");
            }
            sampling_logits = ggml_scale(ctx_, cur, inv_temp);
            set_tensor_name(sampling_logits, "logits_scaled");
            ggml_build_forward_expand(gf, sampling_logits);
        }

        ggml_tensor* topk_indices = ggml_top_k(ctx_, sampling_logits, static_cast<int32_t>(k));
        set_tensor_name(topk_indices, "sample_topk_idx");
        ggml_build_forward_expand(gf, topk_indices);

        if (sampling_temperature_ > 0.0f) {
            ggml_tensor* logits_3d = ggml_reshape_3d(ctx_, sampling_logits, 1, n_vocab, n_tokens);
            ggml_tensor* topk_logits = ggml_get_rows(ctx_, logits_3d, topk_indices);
            topk_logits = ggml_reshape_2d(ctx_, topk_logits, k, n_tokens);
            set_tensor_name(topk_logits, "sample_topk_logits");
            ggml_build_forward_expand(gf, topk_logits);
        }
    }
}

/**
 * @brief 判断某层是否为 Full Attention 层
 * 
 * Qwen3.5-MoE 使用混合架构：大部分层使用 DeltaNet，每隔一定间隔使用 Full Attention
 * 模式：layers (interval-1), 2*(interval-1)+1, ... 即 layer_idx % interval == interval-1
 * 
 * @param layer_idx 物理层索引
 * @return true 表示是 Full Attention 层，false 表示是 DeltaNet 层
 */
bool Qwen35moeForwardPass::is_full_attention_layer(uint32_t layer_idx) const {
    return (layer_idx % model_->meta_->qwen35moe.full_attention_interval) == (model_->meta_->qwen35moe.full_attention_interval - 1);
}

/**
 * @brief 获取槽位的物理缓存位置
 * 
 * @param slot_idx 槽位索引
 * @return 物理缓存位置
 */
uint32_t Qwen35moeForwardPass::get_physical_cache_pos(uint32_t slot_idx) const {
    return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
}

// =========================================================================
// 缓存管理相关函数
// =========================================================================

/**
 * @brief 推进缓存位置（KV 缓存和 SnapKV）
 * 
 * 在推理完成后调用，将缓存指针向前移动 n_tokens 个位置
 * 
 * @param n_tokens 处理的 token 数量
 * @param slot_idx 槽位索引
 */
void Qwen35moeForwardPass::advance_cache(uint32_t n_tokens, uint32_t slot_idx) {
    // 推进 KV 缓存位置
    if (kv_cache_) 
        kv_cache_->advance(n_tokens, slot_idx);
    // DeltaNet 状态在计算图中自动更新，无需手动推进
    // 推进 SnapKV 的逻辑序列位置
    snapkv_advance_seq_pos(slot_idx, n_tokens);
}

/**
 * @brief 获取 SnapKV 的逻辑序列位置
 * 
 * SnapKV 是一个优化的 KV 缓存策略，用于减少内存带宽消耗
 * 返回 0 表示该槽位未激活 SnapKV
 * 
 * @param slot_idx 槽位索引
 * @return 逻辑序列位置（0 表示未激活）
 */
uint32_t Qwen35moeForwardPass::snapkv_get_seq_pos(uint32_t slot_idx) const {
    return (slot_idx < snapkv_seq_pos_.size()) ? snapkv_seq_pos_[slot_idx] : 0;
}

/**
 * @brief 推进 SnapKV 的逻辑序列位置
 * 
 * 与 advance_cache 配合使用，更新 SnapKV 的位置计数器
 * 
 * @param slot_idx 槽位索引
 * @param n_tokens 推进的 token 数量
 */
void Qwen35moeForwardPass::snapkv_advance_seq_pos(uint32_t slot_idx, uint32_t n_tokens) {
    // 只有当槽位有效且 SnapKV 已激活（位置 > 0）时才推进
    if (slot_idx < snapkv_seq_pos_.size() && snapkv_seq_pos_[slot_idx] > 0)
        snapkv_seq_pos_[slot_idx] += n_tokens;
}

/**
 * @brief 从 GPU 获取输出 logits
 * 
 * 从计算图中提取 "logits" 张量，并将其数据从 GPU 复制到 CPU
 * 
 * @param gf 计算图
 * @return logits 数据（浮点数组）
 */
std::vector<float> Qwen35moeForwardPass::get_output_logits(
    ggml_cgraph* gf,
    ggml_backend_t compute_backend
) {
    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        throw std::runtime_error("logits tensor not found in graph");
    }

    const size_t logits_size = static_cast<size_t>(ggml_nbytes(logits));
    if (logits_size == 0) {
        return {};
    }

#ifdef QWEN35MOE_USE_CUDA
    if (compute_backend && ggml_backend_is_cuda(compute_backend) &&
        logits->buffer && !ggml_backend_buffer_is_host(logits->buffer)) {
        ensure_output_pinned(logits_size);
        ggml_backend_tensor_get_async(
            compute_backend, logits, output_pinned_.data(), 0, logits_size);
        ggml_backend_synchronize(compute_backend);
        std::vector<float> logits_result(logits_size / sizeof(float));
        std::memcpy(logits_result.data(), output_pinned_.data(), logits_size);
        return logits_result;
    }
#else
    (void)compute_backend;
#endif

    std::vector<float> logits_result(logits_size / sizeof(float));
    ggml_backend_tensor_get(logits, logits_result.data(), 0, logits_size);
    return logits_result;
}

TopKSampleCandidates Qwen35moeForwardPass::get_output_topk_candidates(
    ggml_cgraph* gf,
    uint32_t token_col,
    ggml_backend_t compute_backend
) {
    ggml_tensor* topk_idx = ggml_graph_get_tensor(gf, "sample_topk_idx");
    if (!topk_idx) {
        throw std::runtime_error("sample_topk_idx tensor not found in graph");
    }
    if (topk_idx->type != GGML_TYPE_I32) {
        throw std::runtime_error("sample_topk_idx tensor type mismatch");
    }
    if (token_col >= static_cast<uint32_t>(topk_idx->ne[1])) {
        throw std::runtime_error("token column out of range for sample_topk_idx");
    }

    const uint32_t k = static_cast<uint32_t>(topk_idx->ne[0]);
    TopKSampleCandidates result;
    result.token_ids.resize(k);
    result.logits.resize(k);

    const size_t idx_offset = static_cast<size_t>(token_col) * topk_idx->nb[1];
    const size_t idx_bytes = static_cast<size_t>(k) * sizeof(int32_t);

    ggml_tensor* topk_logits = ggml_graph_get_tensor(gf, "sample_topk_logits");
    const size_t logits_offset = topk_logits
        ? static_cast<size_t>(token_col) * topk_logits->nb[1]
        : 0;
    const size_t logits_bytes = topk_logits ? static_cast<size_t>(k) * sizeof(float) : 0;

#ifdef QWEN35MOE_USE_CUDA
    const bool use_cuda_d2h = compute_backend && ggml_backend_is_cuda(compute_backend) &&
        topk_idx->buffer && !ggml_backend_buffer_is_host(topk_idx->buffer);
    if (use_cuda_d2h) {
        const size_t pinned_bytes = idx_bytes + logits_bytes;
        ensure_output_pinned(pinned_bytes);
        uint8_t* pinned = output_pinned_.data();
        ggml_backend_tensor_get_async(compute_backend, topk_idx, pinned, idx_offset, idx_bytes);
        if (topk_logits) {
            if (topk_logits->type != GGML_TYPE_F32) {
                throw std::runtime_error("sample_topk_logits tensor type mismatch");
            }
            if (token_col >= static_cast<uint32_t>(topk_logits->ne[1]) ||
                topk_logits->ne[0] != static_cast<int64_t>(k)) {
                throw std::runtime_error("sample_topk_logits tensor shape mismatch");
            }
            ggml_backend_tensor_get_async(
                compute_backend, topk_logits, pinned + idx_bytes, logits_offset, logits_bytes);
        }
        ggml_backend_synchronize(compute_backend);
        std::memcpy(result.token_ids.data(), pinned, idx_bytes);
        if (topk_logits) {
            std::memcpy(result.logits.data(), pinned + idx_bytes, logits_bytes);
        } else {
            result.logits.clear();
        }
        return result;
    }
#else
    (void)compute_backend;
#endif

    ggml_backend_tensor_get(topk_idx, result.token_ids.data(), idx_offset, idx_bytes);

    if (topk_logits) {
        if (topk_logits->type != GGML_TYPE_F32) {
            throw std::runtime_error("sample_topk_logits tensor type mismatch");
        }
        if (token_col >= static_cast<uint32_t>(topk_logits->ne[1]) ||
            topk_logits->ne[0] != static_cast<int64_t>(k)) {
            throw std::runtime_error("sample_topk_logits tensor shape mismatch");
        }
        ggml_backend_tensor_get(topk_logits, result.logits.data(), logits_offset, logits_bytes);
    } else {
        result.logits.clear();
    }

    return result;
}
