#pragma once

#include <memory>
#include <string>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>
#include "model/metadata.h"
#include "model/weights.h"
#include "model/gguf_mmap.h"

#ifndef QWEN_DEFAULT_GRAPH_SIZE
#define QWEN_DEFAULT_GRAPH_SIZE 16384
#endif

enum class DevMode {
    CPU_MODE,
    GPU_MODE,
    AUTO_MODE,
};

class Qwen35moeModel {
public:
    Qwen35moeModel();
    ~Qwen35moeModel();

    bool init(const std::string& model_path_, DevMode dev_mode = DevMode::CPU_MODE, int n_threads = 1, size_t gpu_layer = 0);
    ggml_backend_t get_curr_backend();
    ggml_backend_sched_t get_scheduler() const;
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
    bool init_auto_cpu(std::vector<struct tensor_info>::iterator& enditer);
    bool init_auto_gpu(size_t& free_mem);

    bool load_metadata();
    int get_ctx_size();
    void dequant_set(ggml_tensor* dst);
    void print_context_info(gguf_context* gguf_ctx, ggml_context* ctx_);

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
};
