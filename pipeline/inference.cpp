// pipeline/inference.cpp
// Qwen3.5 MoE forward pass — CPU, per-layer execution with CPU-side MoE routing.
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
//
// MoE routing is done entirely in pure C++ (compute_moe_routing_cpu).
// Each transformer layer is executed as a separate ggml sub-graph.
#include "pipeline/inference.hpp"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef QWEN35MOE_USE_CUDA
#include <ggml-cuda.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

// ============================================================
// Helpers
// ============================================================

static inline bool is_attn_layer(int il) { return (il % 4) == 3; }
// Metadata-only ggml contexts for GPU tensor descriptors (not tensor payloads).
// 64 MiB is enough for full-model tensor metadata with current model sizes.
static constexpr size_t kGpuWeightsCtxBytesNonExpert = 32u * 1024u * 1024u;
static constexpr size_t kGpuWeightsCtxBytesAll       = 64u * 1024u * 1024u;

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

InferenceEngine::InferenceEngine(Qwen35moeModel& model, GGUFReader* reader,
                                 int n_threads, int max_seq_len, GpuMode gpu_mode)
    : model_(model), reader_(reader), n_threads_(n_threads), max_seq_len_(max_seq_len),
      gpu_mode_(gpu_mode) {
    backend_cpu_ = ggml_backend_cpu_init();
    if (!backend_cpu_) {
        fprintf(stderr, "[Inference] ERROR: failed to init CPU backend\n");
        return;
    }
    ggml_backend_cpu_set_n_threads(backend_cpu_, n_threads_);
#ifdef QWEN35MOE_USE_CUDA
    if (gpu_mode_ != GpuMode::Off) {
        ggml_backend_t temp_cuda_backend = ggml_backend_cuda_init(0); // device 0
        if (!temp_cuda_backend) {
            fprintf(stderr, "[Inference] WARNING: CUDA init failed, falling back to CPU\n");
            gpu_mode_ = GpuMode::Off;
        } else {
            backend_gpu_ = temp_cuda_backend;
            const bool upload_ok = (gpu_mode_ == GpuMode::Full)
                ? upload_all_weights_to_gpu()
                : upload_non_expert_weights_to_gpu();
            if (!upload_ok) {
                fprintf(stderr, "[Inference] WARNING: failed to upload tensors to CUDA backend, falling back to CPU\n");
                ggml_backend_free(temp_cuda_backend);
                backend_gpu_ = backend_cpu_;
                gpu_mode_ = GpuMode::Off;
            } else {
                use_gpu_ = true;
                if (gpu_mode_ == GpuMode::Full) {
                    fprintf(stderr, "[Inference] CUDA backend ready (all tensors on GPU)\n");
                } else {
                    fprintf(stderr, "[Inference] CUDA backend ready (non-expert tensors on GPU)\n");
                }
            }
        }
    }
#else
    if (gpu_mode_ != GpuMode::Off) {
        fprintf(stderr, "[Inference] WARNING: CUDA not compiled in, using CPU\n");
        gpu_mode_ = GpuMode::Off;
    }
#endif
    if (!use_gpu_) {
        backend_gpu_ = backend_cpu_;
        fprintf(stderr, "[Inference] CPU backend ready (%d threads)\n", n_threads_);
    }
    init_state();
}

InferenceEngine::~InferenceEngine() {
    if (gpu_weights_buf_) {
        ggml_backend_buffer_free(gpu_weights_buf_);
        gpu_weights_buf_ = nullptr;
    }
    if (gpu_weights_ctx_) {
        ggml_free(gpu_weights_ctx_);
        gpu_weights_ctx_ = nullptr;
    }
    if (backend_gpu_ && backend_gpu_ != backend_cpu_) {
        ggml_backend_free(backend_gpu_);
    }
    if (backend_cpu_) {
        ggml_backend_free(backend_cpu_);
    }
}

bool InferenceEngine::upload_non_expert_weights_to_gpu() {
    if (!backend_gpu_ || backend_gpu_ == backend_cpu_) {
        return true;
    }

    struct UploadItem {
        ggml_tensor** tensor;
        const char* name;
    };
    std::vector<UploadItem> items;

    auto add_item = [&items](ggml_tensor*& t, const char* name) {
        if (t) {
            items.push_back(UploadItem{&t, name});
        }
    };

    auto& weights = model_.weights;
    add_item(weights.token_embd, "token_embd");
    add_item(weights.output, "output");
    add_item(weights.output_norm, "output_norm");

    for (auto& lyr : weights.layers) {
        add_item(lyr.attn_norm, "attn_norm");
        add_item(lyr.post_attention_norm, "post_attention_norm");

        add_item(lyr.attn_q, "attn_q");
        add_item(lyr.attn_k, "attn_k");
        add_item(lyr.attn_v, "attn_v");
        add_item(lyr.attn_output, "attn_output");
        add_item(lyr.attn_q_norm, "attn_q_norm");
        add_item(lyr.attn_k_norm, "attn_k_norm");

        add_item(lyr.attn_qkv, "attn_qkv");
        add_item(lyr.attn_gate, "attn_gate");
        add_item(lyr.ssm_a, "ssm_a");
        add_item(lyr.ssm_alpha, "ssm_alpha");
        add_item(lyr.ssm_beta, "ssm_beta");
        add_item(lyr.ssm_conv1d, "ssm_conv1d");
        add_item(lyr.ssm_dt_b, "ssm_dt_b");
        add_item(lyr.ssm_norm, "ssm_norm");
        add_item(lyr.ssm_out, "ssm_out");
    }

    // Metadata-only context for non-expert tensor descriptors before backend alloc.
    ggml_init_params p = { kGpuWeightsCtxBytesNonExpert, nullptr, true };
    gpu_weights_ctx_ = ggml_init(p);
    if (!gpu_weights_ctx_) {
        fprintf(stderr, "[Inference] ERROR: failed to init GPU weights context\n");
        return false;
    }

    std::vector<ggml_tensor*> dst_tensors;
    dst_tensors.reserve(items.size());
    for (const auto& item : items) {
        ggml_tensor* src = *item.tensor;
        ggml_tensor* dst = ggml_new_tensor(gpu_weights_ctx_, src->type, ggml_n_dims(src), src->ne);
        if (!dst) {
            fprintf(stderr, "[Inference] ERROR: failed to allocate GPU tensor for %s\n", item.name);
            return false;
        }
        if (src->name[0] != '\0') {
            ggml_set_name(dst, src->name);
        }
        dst_tensors.push_back(dst);
    }

    gpu_weights_buf_ = ggml_backend_alloc_ctx_tensors(gpu_weights_ctx_, backend_gpu_);
    if (!gpu_weights_buf_) {
        fprintf(stderr, "[Inference] ERROR: failed to allocate GPU buffer for non-expert tensors\n");
        return false;
    }

    for (size_t i = 0; i < items.size(); ++i) {
        ggml_tensor* src = *items[i].tensor;
        ggml_tensor* dst = dst_tensors[i];
        const size_t nbytes = ggml_nbytes(src);
        if (nbytes > 0) {
            if (!src->data) {
                fprintf(stderr, "[Inference] ERROR: source tensor has no CPU data: %s\n", items[i].name);
                return false;
            }
            ggml_backend_tensor_set(dst, src->data, 0, nbytes);
        }
        *items[i].tensor = dst; // Rebind model weight pointer to the GPU-backed tensor.
    }

    return true;
}

bool InferenceEngine::upload_all_weights_to_gpu() {
    if (!backend_gpu_ || backend_gpu_ == backend_cpu_) {
        return true;
    }

    struct UploadItem {
        ggml_tensor** tensor;
        const char* name;
    };
    std::vector<UploadItem> items;

    auto add_item = [&items](ggml_tensor*& t, const char* name) {
        if (t) {
            items.push_back(UploadItem{&t, name});
        }
    };

    auto& weights = model_.weights;
    add_item(weights.token_embd, "token_embd");
    add_item(weights.output, "output");
    add_item(weights.output_norm, "output_norm");

    for (auto& lyr : weights.layers) {
        add_item(lyr.attn_norm, "attn_norm");
        add_item(lyr.post_attention_norm, "post_attention_norm");

        add_item(lyr.attn_q, "attn_q");
        add_item(lyr.attn_k, "attn_k");
        add_item(lyr.attn_v, "attn_v");
        add_item(lyr.attn_output, "attn_output");
        add_item(lyr.attn_q_norm, "attn_q_norm");
        add_item(lyr.attn_k_norm, "attn_k_norm");

        add_item(lyr.attn_qkv, "attn_qkv");
        add_item(lyr.attn_gate, "attn_gate");
        add_item(lyr.ssm_a, "ssm_a");
        add_item(lyr.ssm_alpha, "ssm_alpha");
        add_item(lyr.ssm_beta, "ssm_beta");
        add_item(lyr.ssm_conv1d, "ssm_conv1d");
        add_item(lyr.ssm_dt_b, "ssm_dt_b");
        add_item(lyr.ssm_norm, "ssm_norm");
        add_item(lyr.ssm_out, "ssm_out");

        add_item(lyr.ffn_gate_exps, "ffn_gate_exps");
        add_item(lyr.ffn_up_exps, "ffn_up_exps");
        add_item(lyr.ffn_down_exps, "ffn_down_exps");
        add_item(lyr.ffn_gate_shexp, "ffn_gate_shexp");
        add_item(lyr.ffn_up_shexp, "ffn_up_shexp");
        add_item(lyr.ffn_down_shexp, "ffn_down_shexp");
        add_item(lyr.ffn_gate_inp, "ffn_gate_inp");
        add_item(lyr.ffn_gate_inp_shexp, "ffn_gate_inp_shexp");
    }

    // Metadata-only context for tensor descriptors before backend alloc.
    ggml_init_params p = { kGpuWeightsCtxBytesAll, nullptr, true };
    gpu_weights_ctx_ = ggml_init(p);
    if (!gpu_weights_ctx_) {
        fprintf(stderr, "[Inference] ERROR: failed to init GPU weights context\n");
        return false;
    }

    std::vector<ggml_tensor*> dst_tensors;
    dst_tensors.reserve(items.size());
    for (const auto& item : items) {
        ggml_tensor* src = *item.tensor;
        ggml_tensor* dst = ggml_new_tensor(gpu_weights_ctx_, src->type, ggml_n_dims(src), src->ne);
        if (!dst) {
            fprintf(stderr, "[Inference] ERROR: failed to allocate GPU tensor for %s\n", item.name);
            return false;
        }
        if (src->name[0] != '\0') {
            ggml_set_name(dst, src->name);
        }
        dst_tensors.push_back(dst);
    }

    gpu_weights_buf_ = ggml_backend_alloc_ctx_tensors(gpu_weights_ctx_, backend_gpu_);
    if (!gpu_weights_buf_) {
        fprintf(stderr, "[Inference] ERROR: failed to allocate GPU buffer for tensors\n");
        return false;
    }

    for (size_t i = 0; i < items.size(); ++i) {
        ggml_tensor* src = *items[i].tensor;
        ggml_tensor* dst = dst_tensors[i];
        const size_t nbytes = ggml_nbytes(src);
        if (nbytes > 0) {
            if (!src->data) {
                fprintf(stderr, "[Inference] ERROR: source tensor has no CPU data: %s\n", items[i].name);
                return false;
            }
            ggml_backend_tensor_set(dst, src->data, 0, nbytes);
        }
        *items[i].tensor = dst; // Rebind model weight pointer to the GPU-backed tensor.
    }

    return true;
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

    // Per-layer MoE routing result buffers (empty; resized in compute_moe_routing_cpu)
    moe_routes_.resize(n_layer);
    moe_gate_inp_cpu_cache_.resize(n_layer);

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
    if (!backend_gpu_ || !backend_cpu_) return {};

    const auto& cfg  = model_.config.qwen35moe;
    const int n_tokens   = (int)tokens.size();
    const int pos        = pos_;
    const int n_layer    = (int)cfg.block_count;
    const int n_embd     = (int)cfg.embedding_length;

    // Safecheck: don't exceed the KV cache capacity
    if (pos + n_tokens > max_seq_len_) {
        fprintf(stderr, "[Inference] ERROR: sequence length %d exceeds max_seq_len %d\n",
                pos + n_tokens, max_seq_len_);
        return {};
    }

    // Validate single-token constraint for incremental steps
    if (pos_ > 0 && n_tokens != 1) {
        fprintf(stderr, "[Inference] ERROR: incremental mode requires exactly 1 token "
                "(got %d; call reset_state() before a new prompt)\n", n_tokens);
        return {};
    }

    // ── Step 1: Token embedding ──
    std::vector<float> cur = exec_token_embd(tokens, n_tokens);
    if (cur.empty()) {
        fprintf(stderr, "[Inference] ERROR: exec_token_embd failed\n");
        return {};
    }

    // ── Step 2: Per-layer execution ──
    for (int il = 0; il < n_layer; il++) {
        const auto& lyr = model_.weights.layers[il];
        if (!lyr.attn_norm) {
            fprintf(stderr, "[Inference] ERROR: layer %d missing attn_norm\n", il);
            return {};
        }

        // 2a. attn_norm(cur) + attn/SSM → attn_out  [n_embd * n_tokens]
        std::vector<float> attn_out = exec_attn_or_ssm(cur, il, n_tokens, pos);
        if (attn_out.empty()) {
            fprintf(stderr, "[Inference] ERROR: exec_attn_or_ssm failed at layer %d\n", il);
            return {};
        }

        // 2b. Residual: cur += attn_out
        for (int i = 0; i < n_embd * n_tokens; i++) cur[i] += attn_out[i];

        // 2c. post_attn_norm(cur) → ffn_in
        if (!lyr.post_attention_norm) {
            fprintf(stderr, "[Inference] ERROR: layer %d missing post_attention_norm\n", il);
            return {};
        }
        std::vector<float> ffn_in = exec_rms_norm(cur, lyr.post_attention_norm, n_tokens);
        if (ffn_in.empty()) {
            fprintf(stderr, "[Inference] ERROR: exec_rms_norm failed at layer %d\n", il);
            return {};
        }

        // 2d. CPU-side MoE routing: softmax(W^T * ffn_in) → top-k → normalize
        if (!lyr.ffn_gate_inp) {
            fprintf(stderr, "[Inference] ERROR: layer %d missing ffn_gate_inp\n", il);
            return {};
        }
        const int n_expert = (int)cfg.expert_count;
        const float* gate_w = (const float*)lyr.ffn_gate_inp->data;
        if (!gate_w) {
            if (gpu_mode_ != GpuMode::Full) {
                fprintf(stderr, "[Inference] ERROR: layer %d ffn_gate_inp has no CPU data\n", il);
                return {};
            }
            auto& gate_cache = moe_gate_inp_cpu_cache_[il];
            if (gate_cache.empty()) {
                const size_t gate_count = (size_t)n_embd * n_expert;
                gate_cache.resize(gate_count);
                ggml_backend_tensor_get(lyr.ffn_gate_inp, gate_cache.data(), 0, gate_count * sizeof(float));
            }
            gate_w = gate_cache.data();
        }

        compute_moe_routing_cpu(gate_w, ffn_in.data(), il, n_tokens);

        // 2e. MoE FFN sub-graph (selected + weights are pre-filled leaf tensors)
        std::vector<float> moe_out = exec_moe_ffn(ffn_in, il, n_tokens);
        if (moe_out.empty()) {
            fprintf(stderr, "[Inference] ERROR: exec_moe_ffn failed at layer %d\n", il);
            return {};
        }

        // 2f. Residual: cur += moe_out
        for (int i = 0; i < n_embd * n_tokens; i++) cur[i] += moe_out[i];
    }

    // Advance position counter (after all layers have been processed)
    pos_ += n_tokens;

    // ── Step 3: final_norm + lm_head ──
    return exec_lm_head(cur, n_tokens);
}

// ============================================================
// exec_token_embd()
// ============================================================
std::vector<float> InferenceEngine::exec_token_embd(
    const std::vector<int32_t>& tokens, int n_tokens)
{
    const int n_embd = (int)model_.config.qwen35moe.embedding_length;
    struct ggml_tensor* embd = model_.weights.token_embd;
    if (!embd) return {};

    // CUDA get_rows currently does not support some quantized source types
    // (e.g. Q5_K). For GPU mode, fetch quantized embedding rows and
    // dequantize on CPU.
    if (use_gpu_) {
        const ggml_type_traits* traits = ggml_get_type_traits(embd->type);
        const ggml_to_float_t to_float = traits ? traits->to_float : nullptr;

        const size_t row_bytes_q = ggml_row_size(embd->type, n_embd);
        const size_t row_bytes_f = (size_t)n_embd * sizeof(float);
        const int64_t n_vocab = embd->ne[1];

        std::vector<float> result((size_t)n_embd * n_tokens);
        std::vector<char> qrow(row_bytes_q);

        for (int t = 0; t < n_tokens; ++t) {
            const int32_t tok = tokens[t];
            if (tok < 0 || (int64_t) tok >= n_vocab) {
                fprintf(stderr, "[Inference] ERROR: token id out of range: %d\n", tok);
                return {};
            }

            const size_t off = (size_t)tok * row_bytes_q;
            ggml_backend_tensor_get(embd, qrow.data(), off, row_bytes_q);

            if (embd->type == GGML_TYPE_F32) {
                std::memcpy(result.data() + (size_t)t * n_embd, qrow.data(), row_bytes_f);
            } else {
                if (!to_float) {
                    fprintf(stderr, "[Inference] ERROR: no to_float for embedding type %d\n", (int)embd->type);
                    return {};
                }
                to_float(qrow.data(), result.data() + (size_t)t * n_embd, n_embd);
            }
        }
        return result;
    }

    struct ggml_init_params p = { 32*1024*1024, nullptr, true };
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return {};
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 256, false);

    struct ggml_tensor* inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp, "inp_tokens");
    struct ggml_tensor* out = ggml_get_rows(ctx, embd, inp);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_gpu_));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return {};
    }

    // Fill inp_tokens
    {
        struct ggml_tensor* t = ggml_graph_get_tensor(gf, "inp_tokens");
        if (t)
            ggml_backend_tensor_set(t, tokens.data(), 0, (size_t)n_tokens * sizeof(int32_t));
    }

    ggml_backend_graph_compute(backend_gpu_, gf);

    std::vector<float> result((size_t)n_embd * n_tokens);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

// ============================================================
// exec_attn_or_ssm()
// ============================================================
std::vector<float> InferenceEngine::exec_attn_or_ssm(
    const std::vector<float>& cur_data, int il, int n_tokens, int pos)
{
    const auto& cfg   = model_.config.qwen35moe;
    const auto& lyr   = model_.weights.layers[il];
    const int n_embd  = (int)cfg.embedding_length;

    const int64_t d_inner = cfg.inner_size;
    const int64_t d_state = cfg.state_size;
    const int64_t n_group = cfg.group_count;
    const int64_t dt_rank = cfg.time_step_rank;
    const int64_t head_v  = d_inner / dt_rank;
    const int64_t qkv_ch  = d_inner + 2 * n_group * d_state;
    const int64_t conv_k  = cfg.conv_kernel;
    const int64_t head_dim = cfg.key_length;
    const int64_t n_kv_h  = cfg.head_count_kv;
    const float rms_eps   = cfg.layer_norm_rms_epsilon > 0.0f
                            ? cfg.layer_norm_rms_epsilon : 1e-6f;

    struct ggml_init_params p = { 64*1024*1024, nullptr, true };
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return {};
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false);

    // cur leaf: [n_embd, n_tokens] F32
    struct ggml_tensor* cur_leaf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(cur_leaf, "sub_cur");

    // Apply attn_norm inside the sub-graph
    struct ggml_tensor* normed = rms_norm(ctx, cur_leaf, lyr.attn_norm, rms_eps);

    // Build attn or SSM sub-graph
    struct ggml_tensor* attn_out = nullptr;
    struct ggml_tensor* inp_pos  = nullptr;
    struct ggml_tensor* kq_mask  = nullptr;

    if (is_attn_layer(il)) {
        inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(inp_pos, "inp_pos");

        if (pos == 0) {
            kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_tokens, n_tokens);
        } else {
            kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, pos + n_tokens, n_tokens);
        }
        ggml_set_name(kq_mask, "kq_mask");

        attn_out = build_attn_layer(ctx, gf, normed, inp_pos, kq_mask, il, n_tokens, pos);
    } else {
        attn_out = build_ssm_layer(ctx, gf, normed, il, n_tokens, pos);
    }

    if (!attn_out) {
        ggml_free(ctx);
        return {};
    }
    ggml_build_forward_expand(gf, attn_out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_gpu_));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return {};
    }

    // Fill all leaf tensors by direct name lookup (ggml stores leaves in
    // gf->leafs[], not gf->nodes[]; ggml_graph_get_tensor searches both).
    char leaf_nm[64];

    // sub_cur: current residual stream [n_embd, n_tokens]
    {
        struct ggml_tensor* t = ggml_graph_get_tensor(gf, "sub_cur");
        if (t)
            ggml_backend_tensor_set(t, cur_data.data(), 0, (size_t)n_embd * n_tokens * sizeof(float));
    }

    if (is_attn_layer(il)) {
        // inp_pos: position indices [4 * n_tokens] I32
        {
            struct ggml_tensor* t = ggml_graph_get_tensor(gf, "inp_pos");
            if (t) {
                std::vector<int32_t> pos_data(4 * n_tokens);
                for (int s = 0; s < 4; s++)
                    for (int pp = 0; pp < n_tokens; pp++)
                        pos_data[s * n_tokens + pp] = pos + pp;
                ggml_backend_tensor_set(t, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
            }
        }
        // kq_mask: causal attention mask FP16
        {
            struct ggml_tensor* t = ggml_graph_get_tensor(gf, "kq_mask");
            if (t) {
                if (pos == 0) {
                    std::vector<ggml_fp16_t> mask_buf(n_tokens * n_tokens);
                    for (int q = 0; q < n_tokens; q++)
                        for (int k = 0; k < n_tokens; k++) {
                            float val = (k <= q) ? 0.0f : -INFINITY;
                            mask_buf[q * n_tokens + k] = ggml_fp32_to_fp16(val);
                        }
                    ggml_backend_tensor_set(t, mask_buf.data(), 0, mask_buf.size() * sizeof(ggml_fp16_t));
                } else {
                    int kv_len = pos + n_tokens;
                    std::vector<ggml_fp16_t> mask_buf(kv_len, ggml_fp32_to_fp16(0.0f));
                    ggml_backend_tensor_set(t, mask_buf.data(), 0, mask_buf.size() * sizeof(ggml_fp16_t));
                }
            }
        }
        // KV cache leaves (only created when pos > 0)
        if (pos > 0) {
            int ai = attn_cache_idx(il);
            snprintf(leaf_nm, sizeof(leaf_nm), "kv_k_cache_%d", il);
            {
                struct ggml_tensor* t = ggml_graph_get_tensor(gf, leaf_nm);
                if (t) {
                    std::vector<float> staging((size_t)head_dim * pos * n_kv_h);
                    const float* buf = kv_caches_[ai].k.data();
                    for (int64_t h = 0; h < n_kv_h; h++)
                        memcpy(staging.data() + h * head_dim * pos,
                               buf + h * head_dim * max_seq_len_,
                               (size_t)head_dim * pos * sizeof(float));
                    ggml_backend_tensor_set(t, staging.data(), 0, staging.size() * sizeof(float));
                }
            }
            snprintf(leaf_nm, sizeof(leaf_nm), "kv_v_cache_%d", il);
            {
                struct ggml_tensor* t = ggml_graph_get_tensor(gf, leaf_nm);
                if (t) {
                    std::vector<float> staging((size_t)head_dim * pos * n_kv_h);
                    const float* buf = kv_caches_[ai].v.data();
                    for (int64_t h = 0; h < n_kv_h; h++)
                        memcpy(staging.data() + h * head_dim * pos,
                               buf + h * head_dim * max_seq_len_,
                               (size_t)head_dim * pos * sizeof(float));
                    ggml_backend_tensor_set(t, staging.data(), 0, staging.size() * sizeof(float));
                }
            }
        }
    } else {
        // SSM layer leaves
        int si = ssm_state_idx(il);

        snprintf(leaf_nm, sizeof(leaf_nm), "ssm_conv_pad_%d", il);
        {
            struct ggml_tensor* t = ggml_graph_get_tensor(gf, leaf_nm);
            if (t) {
                size_t sz = (size_t)(conv_k - 1) * (size_t)qkv_ch * sizeof(float);
                ggml_backend_tensor_set(t, ssm_conv_states_[si].data(), 0, sz);
            }
        }
        snprintf(leaf_nm, sizeof(leaf_nm), "ssm_state_in_%d", il);
        {
            struct ggml_tensor* t = ggml_graph_get_tensor(gf, leaf_nm);
            if (t) {
                size_t sz = (size_t)head_v * (size_t)head_v * (size_t)dt_rank * sizeof(float);
                ggml_backend_tensor_set(t, ssm_recurrent_states_[si].data(), 0, sz);
            }
        }
        // gate_zero / beta_ones: only present when alpha/beta weights are absent
        snprintf(leaf_nm, sizeof(leaf_nm), "gate_zero_%d", il);
        {
            struct ggml_tensor* t = ggml_graph_get_tensor(gf, leaf_nm);
            if (t) {
                std::vector<char> zeros(ggml_nbytes(t), 0);
                ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
            }
        }
        snprintf(leaf_nm, sizeof(leaf_nm), "beta_ones_%d", il);
        {
            struct ggml_tensor* t = ggml_graph_get_tensor(gf, leaf_nm);
            if (t) {
                int64_t n = ggml_nelements(t);
                std::vector<float> ones(n, 1.0f);
                ggml_backend_tensor_set(t, ones.data(), 0, n * sizeof(float));
            }
        }
    }

    ggml_backend_graph_compute(backend_gpu_, gf);

    // Read output BEFORE ggml_free (data lives in galloc buffer)
    std::vector<float> result((size_t)n_embd * n_tokens);
    ggml_backend_tensor_get(attn_out, result.data(), 0, result.size() * sizeof(float));

    // Save SSM states back to persistent buffers
    if (!is_attn_layer(il)) {
        const int64_t sv_h = head_v * dt_rank;
        int si = ssm_state_idx(il);
        auto& sout = tmp_ssm_outs_[si];

        if (sout.conv_state_out) {
            size_t sz = (size_t)(conv_k - 1) * (size_t)qkv_ch * sizeof(float);
            ggml_backend_tensor_get(sout.conv_state_out, ssm_conv_states_[si].data(), 0, sz);
        }
        if (sout.gdn_out) {
            size_t byte_offset = (size_t)sout.n_tokens * sv_h * sizeof(float);
            size_t sz = (size_t)head_v * (size_t)head_v * (size_t)dt_rank * sizeof(float);
            ggml_backend_tensor_get(sout.gdn_out, ssm_recurrent_states_[si].data(), byte_offset, sz);
        }
        sout.gdn_out        = nullptr;
        sout.conv_state_out = nullptr;
        sout.n_tokens       = 0;
    }

    // Save KV cache entries back to persistent buffers
    if (is_attn_layer(il)) {
        int ai = attn_cache_idx(il);
        auto& kout = tmp_kv_outs_[ai];

        if (kout.k_new) {
            std::vector<float> staging((size_t)head_dim * n_tokens * n_kv_h);
            ggml_backend_tensor_get(kout.k_new, staging.data(), 0, staging.size() * sizeof(float));
            float* buf = kv_caches_[ai].k.data();
            const float* src = staging.data();
            for (int64_t h = 0; h < n_kv_h; h++) {
                memcpy(buf + h * head_dim * max_seq_len_ + (size_t)pos * head_dim,
                       src + h * head_dim * n_tokens,
                       (size_t)head_dim * n_tokens * sizeof(float));
            }
        }
        if (kout.v_new) {
            std::vector<float> staging((size_t)head_dim * n_tokens * n_kv_h);
            ggml_backend_tensor_get(kout.v_new, staging.data(), 0, staging.size() * sizeof(float));
            float* buf = kv_caches_[ai].v.data();
            const float* src = staging.data();
            for (int64_t h = 0; h < n_kv_h; h++) {
                memcpy(buf + h * head_dim * max_seq_len_ + (size_t)pos * head_dim,
                       src + h * head_dim * n_tokens,
                       (size_t)head_dim * n_tokens * sizeof(float));
            }
        }
        kv_caches_[ai].len = pos + n_tokens;
        kout.k_new = nullptr;
        kout.v_new = nullptr;
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

// ============================================================
// exec_rms_norm()
// ============================================================
std::vector<float> InferenceEngine::exec_rms_norm(
    const std::vector<float>& cur_data, ggml_tensor* norm_w, int n_tokens)
{
    const int   n_embd = (int)model_.config.qwen35moe.embedding_length;
    const float eps    = model_.config.qwen35moe.layer_norm_rms_epsilon > 0.0f
                         ? model_.config.qwen35moe.layer_norm_rms_epsilon : 1e-6f;

    struct ggml_init_params p = { 16*1024*1024, nullptr, true };
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return {};
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 64, false);

    struct ggml_tensor* inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(inp, "rms_inp");
    struct ggml_tensor* out = ggml_rms_norm(ctx, inp, eps);
    out = ggml_mul(ctx, out, norm_w);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_gpu_));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return {};
    }

    {
        struct ggml_tensor* t = ggml_graph_get_tensor(gf, "rms_inp");
        if (t)
            ggml_backend_tensor_set(t, cur_data.data(), 0, (size_t)n_embd * n_tokens * sizeof(float));
    }

    ggml_backend_graph_compute(backend_gpu_, gf);

    std::vector<float> result((size_t)n_embd * n_tokens);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

// ============================================================
// compute_moe_routing_cpu()
// ============================================================
void InferenceEngine::compute_moe_routing_cpu(
    const float* W,   // [n_embd, n_expert] F32 column-major
    const float* X,   // [n_embd, n_tokens] F32 column-major
    int il, int n_tokens)
{
    const auto& cfg   = model_.config.qwen35moe;
    const int n_embd  = (int)cfg.embedding_length;   // 2048
    const int n_expert = (int)cfg.expert_count;       // 256
    const int n_top_k  = (int)cfg.expert_used_count;  // 8

    auto& route = moe_routes_[il];
    route.selected.resize((size_t)n_top_k * n_tokens);
    route.weights.resize((size_t)n_top_k * n_tokens);

    std::vector<float> logits(n_expert);
    std::vector<int>   order(n_expert);

    for (int t = 0; t < n_tokens; t++) {
        const float* x = X + (size_t)t * n_embd;

        // router logit[e] = dot(W[:, e], x)
        // W column-major [n_embd, n_expert]: col e starts at W + e * n_embd
        for (int e = 0; e < n_expert; e++) {
            const float* we = W + (size_t)e * n_embd;
            float s = 0.0f;
            for (int i = 0; i < n_embd; i++) s += we[i] * x[i];
            logits[e] = s;
        }

        // Softmax over all experts
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum_e = 0.0f;
        for (int e = 0; e < n_expert; e++) {
            logits[e] = std::exp(logits[e] - max_l);
            sum_e += logits[e];
        }
        for (int e = 0; e < n_expert; e++) logits[e] /= sum_e;

        // Partial sort to find top-k
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + n_top_k, order.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });

        // Normalize top-k weights to sum to 1
        float w_sum = 0.0f;
        for (int k = 0; k < n_top_k; k++) w_sum += logits[order[k]];
        constexpr float MIN_WEIGHT_SUM = 1e-9f;
        if (w_sum < MIN_WEIGHT_SUM) w_sum = MIN_WEIGHT_SUM;

        // Store in ggml column-major layout: index = k + t * n_top_k
        for (int k = 0; k < n_top_k; k++) {
            route.selected[k + t * n_top_k] = order[k];
            route.weights [k + t * n_top_k] = logits[order[k]] / w_sum;
        }
    }
}

// ============================================================
// exec_moe_ffn()
// ============================================================
std::vector<float> InferenceEngine::exec_moe_ffn(
    const std::vector<float>& ffn_in_data, int il, int n_tokens)
{
    const auto& cfg   = model_.config.qwen35moe;
    const auto& lyr   = model_.weights.layers[il];
    const int n_embd  = (int)cfg.embedding_length;
    const int n_top_k = (int)cfg.expert_used_count;

    if (!lyr.ffn_gate_exps || !lyr.ffn_up_exps || !lyr.ffn_down_exps) {
        fprintf(stderr, "[Inference] layer %d (moe_ffn): missing expert weights\n", il);
        return {};
    }

    struct ggml_init_params p = { 64*1024*1024, nullptr, true };
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return {};
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 2048, false);

    // Input leaf: [n_embd, n_tokens] F32
    struct ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(cur, "moe_ffn_in");

    // Selected leaf: [n_top_k, n_tokens] I32 (ggml column-major: index k+t*n_top_k)
    struct ggml_tensor* selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_top_k, n_tokens);
    ggml_set_name(selected, "moe_sel");

    // Weights leaf: [1, n_top_k, n_tokens] F32 (broadcasts over n_embd)
    struct ggml_tensor* weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_top_k, n_tokens);
    ggml_set_name(weights, "moe_wts");

    // Expert computation via ggml_mul_mat_id
    // cur_3d: [n_embd, 1, n_tokens]
    struct ggml_tensor* cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

    struct ggml_tensor* gate_out = ggml_mul_mat_id(ctx, lyr.ffn_gate_exps, cur_3d, selected);
    struct ggml_tensor* up_out   = ggml_mul_mat_id(ctx, lyr.ffn_up_exps,   cur_3d, selected);
    // gate_out, up_out: [ff_dim, n_top_k, n_tokens]

    // SwiGLU: silu(gate) * up
    struct ggml_tensor* hidden = ggml_mul(ctx, ggml_silu(ctx, gate_out), up_out);
    // hidden: [ff_dim, n_top_k, n_tokens]

    // Down projection
    struct ggml_tensor* experts = ggml_mul_mat_id(ctx, lyr.ffn_down_exps, hidden, selected);
    // experts: [n_embd, n_top_k, n_tokens]

    // Scale by expert weights: [1, n_top_k, n_tokens] broadcasts over n_embd
    experts = ggml_mul(ctx, experts, weights);

    // Sum over top-k slots: build views per slot and add
    struct ggml_tensor* moe_out = ggml_view_2d(ctx, experts, n_embd, n_tokens,
                                                experts->nb[2], 0);
    ggml_build_forward_expand(gf, moe_out);
    for (int i = 1; i < n_top_k; i++) {
        struct ggml_tensor* slot = ggml_view_2d(ctx, experts, n_embd, n_tokens,
                                                 experts->nb[2],
                                                 (size_t)i * experts->nb[1]);
        ggml_build_forward_expand(gf, slot);
        moe_out = ggml_add(ctx, moe_out, slot);
    }
    // moe_out: [n_embd, n_tokens]

    // Shared expert (present in all layers)
    if (lyr.ffn_up_shexp && lyr.ffn_gate_shexp && lyr.ffn_down_shexp) {
        struct ggml_tensor* sh_gate = ggml_mul_mat(ctx, lyr.ffn_gate_shexp, cur);
        struct ggml_tensor* sh_up   = ggml_mul_mat(ctx, lyr.ffn_up_shexp,   cur);
        struct ggml_tensor* sh_act  = ggml_mul(ctx, ggml_silu(ctx, sh_gate), sh_up);
        struct ggml_tensor* sh_out  = ggml_mul_mat(ctx, lyr.ffn_down_shexp, sh_act);

        if (lyr.ffn_gate_inp_shexp) {
            struct ggml_tensor* sh_g = ggml_mul_mat(ctx, lyr.ffn_gate_inp_shexp, cur);
            sh_g   = ggml_sigmoid(ctx, sh_g);
            sh_out = ggml_mul(ctx, sh_out, sh_g);
        }
        moe_out = ggml_add(ctx, moe_out, sh_out);
    }

    ggml_build_forward_expand(gf, moe_out);

    ggml_backend_t moe_backend = (gpu_mode_ == GpuMode::Full) ? backend_gpu_ : backend_cpu_;
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(moe_backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return {};
    }

    // Fill leaf tensors by direct name lookup
    {
        struct ggml_tensor* t = ggml_graph_get_tensor(gf, "moe_ffn_in");
        if (t)
            ggml_backend_tensor_set(t, ffn_in_data.data(), 0,
                                    (size_t)n_embd * n_tokens * sizeof(float));
    }
    {
        struct ggml_tensor* t = ggml_graph_get_tensor(gf, "moe_sel");
        if (t)
            ggml_backend_tensor_set(t, moe_routes_[il].selected.data(), 0,
                                    (size_t)n_top_k * n_tokens * sizeof(int32_t));
    }
    {
        // weights layout is [1, n_top_k, n_tokens] stored as [n_top_k * n_tokens]
        // moe_routes_[il].weights uses the same column-major layout
        struct ggml_tensor* t = ggml_graph_get_tensor(gf, "moe_wts");
        if (t)
            ggml_backend_tensor_set(t, moe_routes_[il].weights.data(), 0,
                                    (size_t)n_top_k * n_tokens * sizeof(float));
    }

    ggml_backend_graph_compute(moe_backend, gf);

    std::vector<float> result((size_t)n_embd * n_tokens);
    ggml_backend_tensor_get(moe_out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

// ============================================================
// exec_lm_head()
// ============================================================
std::vector<float> InferenceEngine::exec_lm_head(
    const std::vector<float>& cur, int n_tokens)
{
    const auto& cfg      = model_.config.qwen35moe;
    const auto& w        = model_.weights;
    const int   n_embd   = (int)cfg.embedding_length;
    const int   vocab_sz = (int)w.token_embd->ne[1];
    const float eps      = cfg.layer_norm_rms_epsilon > 0.0f
                           ? cfg.layer_norm_rms_epsilon : 1e-6f;

    struct ggml_init_params p = { 32*1024*1024, nullptr, true };
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return {};
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 256, false);

    // Input: last token's embedding [n_embd]
    struct ggml_tensor* inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_set_name(inp, "lm_inp");

    // Final RMS norm
    struct ggml_tensor* normed = ggml_rms_norm(ctx, inp, eps);
    normed = ggml_mul(ctx, normed, w.output_norm);

    // LM head: output.weight [n_embd, vocab_size]
    struct ggml_tensor* logits = ggml_mul_mat(ctx, w.output, normed);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_gpu_));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return {};
    }

    // Fill last-token embedding
    {
        struct ggml_tensor* t = ggml_graph_get_tensor(gf, "lm_inp");
        if (t) {
            const float* last = cur.data() + (size_t)(n_tokens - 1) * n_embd;
            ggml_backend_tensor_set(t, last, 0, (size_t)n_embd * sizeof(float));
        }
    }

    ggml_backend_graph_compute(backend_gpu_, gf);

    std::vector<float> result(vocab_sz);
    ggml_backend_tensor_get(logits, result.data(), 0, (size_t)vocab_sz * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
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
