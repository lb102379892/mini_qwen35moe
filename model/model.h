#pragma once

#include <memory>
#include <string>
#include <vector>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>
#include "model/metadata.h"
#include "model/weights.h"
#include "model/gguf_mmap.h"
#include "common.h"

#ifndef QWEN_DEFAULT_GRAPH_SIZE
#define QWEN_DEFAULT_GRAPH_SIZE 16384
#endif

class Qwen35moeModel {
public:
    Qwen35moeModel();
    ~Qwen35moeModel();

    bool init(const std::string& model_path_, DevMode dev_mode = DevMode::CPU_MODE, int n_threads = 1, size_t gpu_layer = 0, bool no_mmap = false);
    ggml_backend_t get_curr_backend();
    ggml_backend_sched_t get_scheduler() const;

    // Returns true when both GPU and CPU layers are present (AUTO_MODE with partial GPU offload).
    // In this case the cached-decode-graph optimisation must be disabled to avoid device
    // mismatch errors and incorrect CUDA-graph capture/replay.
    bool is_mixed_mode() const;

    // Compute a stable hash of the current layer→device assignment.
    // Used as part of the decode-graph signature to prevent reuse across different mappings.
    uint64_t compute_device_map_hash() const;
    const std::vector<ggml_backend_t>& get_layer_device_map() const;
    ggml_backend_t get_layer_backend(uint32_t layer_idx) const;
    ggml_tensor* get_weight_tensor(const EN_WEIGHT_TYPE weight_type);
    ggml_tensor* get_weight_layer_tensor(const EN_WEIGHT_TYPE layer_type, const int layer_idx);
    struct ggml_tensor* get_token_embedding_weight();
    struct ggml_tensor* get_output_weight();
    struct ggml_tensor* get_output_norm_weight();
    struct ggml_tensor* get_attn_k_weight(const int layer_idx);
    struct ggml_tensor* get_attn_k_norm_weight(const int layer_idx);
    struct ggml_tensor* get_attn_norm_weight(const int layer_idx);
    struct ggml_tensor* get_attn_gate_weight(const int layer_idx);
    struct ggml_tensor* get_attn_qkv_weight(const int layer_idx);
    struct ggml_tensor* get_attn_q_weight(const int layer_idx);
    struct ggml_tensor* get_attn_q_norm_weight(const int layer_idx);
    struct ggml_tensor* get_attn_v_weight(const int layer_idx);
    struct ggml_tensor* get_ffn_down_exps_weight(const int layer_idx);
    struct ggml_tensor* get_ffn_gate_exps_weight(const int layer_idx);
    struct ggml_tensor* get_ffn_gate_inp_weight(const int layer_idx);
    struct ggml_tensor* get_ffn_up_exps_weight(const int layer_idx);
    struct ggml_tensor* get_ffn_down_shexp_weight(const int layer_idx);
    struct ggml_tensor* get_ffn_gate_inp_shexp_weight(const int layer_idx);
    struct ggml_tensor* get_ffn_gate_shexp_weight(const int layer_idx);
    struct ggml_tensor* get_ffn_up_shexp_weight(const int layer_idx);
    struct ggml_tensor* get_ssm_a_weight(const int layer_idx);
    struct ggml_tensor* get_ssm_alpha_weight(const int layer_idx);
    struct ggml_tensor* get_ssm_beta_weight(const int layer_idx);
    struct ggml_tensor* get_ssm_conv1d_weight(const int layer_idx);
    struct ggml_tensor* get_ssm_dt_weight(const int layer_idx);
    struct ggml_tensor* get_ssm_norm_weight(const int layer_idx);
    struct ggml_tensor* get_ssm_out_weight(const int layer_idx);
    struct ggml_tensor* get_post_attention_norm_weight(const int layer_idx);
    struct ggml_tensor* get_attn_output_weight(const int layer_idx);

private:
    bool init_cpu();
    bool init_gpu();
    bool is_full_attention_layer(uint32_t layer_idx) const;
    bool init_auto_cpu(const std::set<int, std::less<int>>& gpu_layer_set);
    bool init_auto_gpu(size_t& free_mem);

    bool load_metadata();
    int get_ctx_size();
    void dequant_set(ggml_tensor* dst);
    void print_context_info(gguf_context* gguf_ctx, ggml_context* ctx_);
    void rebuild_layer_device_map();

public:
    DevMode dev_mode_ = DevMode::CPU_MODE;
    std::shared_ptr<GGUFLoader> loader_ = nullptr;
    std::shared_ptr<MetaDataInfo> meta_ = nullptr;
    std::shared_ptr<Qwen35moeWeights> gpu_weights_ = nullptr;
    std::shared_ptr<Qwen35moeWeights> cpu_weights_ = nullptr;

    int n_threads_ = 1;
    size_t gpu_layer_ = 0;
    ggml_backend_t backend_gpu_ = nullptr;
    ggml_backend_t backend_cpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;

    ggml_backend_buffer_t gpu_buf_ = nullptr;
    ggml_backend_buffer_t cpu_buf_ = nullptr;
    ggml_context* gpu_ctx_ = nullptr;
    ggml_context* cpu_ctx_ = nullptr;
    std::vector<ggml_backend_t> layer_device_map_;

    bool no_mmap_ = true;
};
