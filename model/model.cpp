#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>
#include "model/model.h"

static const std::unordered_map<EN_LAYER_TYPE, const char *> g_layer_tensor_names = {
    {EN_LAYER_TYPE_ATTN_GATE, ".attn_gate.weight"},
    {EN_LAYER_TYPE_ATTN_NORM, ".attn_norm.weight"},
    {EN_LAYER_TYPE_ATTN_QKV, ".attn_qkv.weight"},
    {EN_LAYER_TYPE_FFN_DOWN_EXPS, ".ffn_down_exps.weight"},
    {EN_LAYER_TYPE_FFN_DOWN_SHEXP, ".ffn_down_shexp.weight"},
    {EN_LAYER_TYPE_FFN_GATE_EXPS, ".ffn_gate_exps.weight"},
    {EN_LAYER_TYPE_FFN_GATE_INP, ".ffn_gate_inp.weight"},
    {EN_LAYER_TYPE_FFN_GATE_INP_SHEXP, ".ffn_gate_inp_shexp.weight"},
    {EN_LAYER_TYPE_FFN_GATE_SHEXP, ".ffn_gate_shexp.weight"},
    {EN_LAYER_TYPE_FFN_UP_EXPS, ".ffn_up_exps.weight"},
    {EN_LAYER_TYPE_FFN_UP_SHEXP, ".ffn_up_shexp.weight"},
    {EN_LAYER_TYPE_POST_ATTENTION_NORM, ".post_attention_norm.weight"},
    {EN_LAYER_TYPE_SSM_A, ".ssm_a"},
    {EN_LAYER_TYPE_SSM_ALPHA, ".ssm_alpha.weight"},
    {EN_LAYER_TYPE_SSM_BETA, ".ssm_beta.weight"},
    {EN_LAYER_TYPE_SSM_CONV1D, ".ssm_conv1d.weight"},
    {EN_LAYER_TYPE_SSM_DT, ".ssm_dt.bias"},
    {EN_LAYER_TYPE_SSM_NORM, ".ssm_norm.weight"},
    {EN_LAYER_TYPE_SSM_OUT, ".ssm_out.weight"},
    {EN_LAYER_TYPE_ATTN_K, ".attn_k.weight"},
    {EN_LAYER_TYPE_ATTN_K_NORM, ".attn_k_norm.weight"},
    {EN_LAYER_TYPE_ATTN_Q, ".attn_q.weight"},
    {EN_LAYER_TYPE_ATTN_Q_NORM, ".attn_q_norm.weight"},
    {EN_LAYER_TYPE_ATTN_V, ".attn_v.weight"},
    {EN_LAYER_TYPE_ATTN_OUTPUT, ".attn_output.weight"}
};

static const std::unordered_map<EN_WEIGHT_TYPE, const char *> g_weight_tensor_names = {
    {EN_WEIGHT_TYPE_TOKEN_EMBD, "token_embd.weight"},
    {EN_WEIGHT_TYPE_OUTPUT, "output.weight"},
    {EN_WEIGHT_TYPE_OUTPUT_NORM, "output_norm.weight"}
};

Qwen35moeModel::Qwen35moeModel() {
}

Qwen35moeModel::~Qwen35moeModel() {
    if (gpu_buf_) {
        ggml_backend_buffer_free(gpu_buf_);
        gpu_buf_ = nullptr;
    }
    if (cpu_buf_) {
        ggml_backend_buffer_free(cpu_buf_);
        cpu_buf_ = nullptr;
    }
    if (gpu_ctx_) {
        ggml_free(gpu_ctx_);
        gpu_ctx_ = nullptr;
    }
    if (cpu_ctx_) {
        ggml_free(cpu_ctx_);
        cpu_ctx_ = nullptr;
    }
    if (sched_) {
        ggml_backend_sched_free(sched_);
    }
    if (backend_gpu_) {
        ggml_backend_free(backend_gpu_);
        backend_gpu_ = nullptr;
    }
    if (backend_cpu_) {
        ggml_backend_free(backend_cpu_);
        backend_cpu_ = nullptr;
    }
}

bool Qwen35moeModel::init(const std::string& model_path_, DevMode dev_mode, int n_threads, int gpu_layer) {
    printf("\n===================Loading Qwen35moe Model=====================\n");
    dev_mode_ = dev_mode;
    n_threads_ = n_threads;
    gpu_layer_ = gpu_layer;
    loader_ = std::make_shared<GGUFLoader>();
    meta_ = std::make_shared<MetaDataInfo>();

    // 0: model file loader
    if (!loader_->load(model_path_)) {
        printf("[Loader] Tensor loading failed\n");
        return false;
    }

    // 1: metadata
    if (!load_metadata()) {
        printf("[Loader] Config loading failed\n");
        return false;
    }

    if (dev_mode == DevMode::CPU_MODE) {
        init_cpu();
    }
    else {
        init_gpu();
    }

    if (backend_gpu_) {
        ggml_backend_t backends[] = {backend_gpu_, backend_cpu_};
        sched_ = ggml_backend_sched_new(backends, nullptr, 2, QWEN_DEFAULT_GRAPH_SIZE, true, false);
        fprintf(stderr, "Scheduler created with CPU + GPU backends\n");
    } else {
        ggml_backend_t backends[] = {backend_cpu_};
        sched_ = ggml_backend_sched_new(backends, nullptr, 1, QWEN_DEFAULT_GRAPH_SIZE, false, false);
        fprintf(stderr, "Scheduler created with CPU backend only\n");
    }
    
    if (!sched_) {
        fprintf(stderr, "Failed to create backend scheduler\n");
        return false;
    }

    printf("==================Loading Complete!======================\n");
    return true;
}

bool Qwen35moeModel::init_cpu() {
    printf("\n===================Loading Qwen35moe Model to CPU=====================\n");
    backend_cpu_ = ggml_backend_cpu_init();
    if (!backend_cpu_) {
        fprintf(stderr, "[Loader] ERROR: failed to init CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend_cpu_, n_threads_);

    const size_t ctx_size = (meta_->head.tensor_count + 1) * ggml_tensor_overhead();
    ggml_init_params cpu_p = { ctx_size, nullptr, true };
    cpu_ctx_ = ggml_init(cpu_p);
    if (!cpu_ctx_) {
        fprintf(stderr, "[Loader] ERROR: failed to init weight contexts\n");
        return false;
    }

    cpu_weights_ = std::make_shared<Qwen35moeWeights>();
    for (auto& weight_info : g_weight_tensor_names) {
        auto it = loader_->tensor_index_map_.find(weight_info.second);
        if (it == loader_->tensor_index_map_.end()) {
            fprintf(stderr, "[Loader] ERROR: failed to find tensor %s in model file\n", weight_info.second);
            return false;
        }

        auto* tensor_info = &loader_->tensors_[it->second];
        struct ggml_tensor* cur = ggml_new_tensor(cpu_ctx_, tensor_info->type, tensor_info->n_dims, tensor_info->dims);
        ggml_set_name(cur, tensor_info->name.c_str());
        if (NULL != cur) {
            cpu_weights_->heads[weight_info.first] = cur;
        }
    }

    for (int layer_idx = 0; layer_idx < meta_->qwen35moe.block_count; layer_idx++) {
        std::string prefix = "blk." + std::to_string(layer_idx);
        std::shared_ptr<Qwen35moeLayer> layer = std::make_shared<Qwen35moeLayer>();
        for (auto& layer_info : g_layer_tensor_names) {
            std::string name = prefix + layer_info.second;
            auto it = loader_->tensor_index_map_.find(name);
            if (it == loader_->tensor_index_map_.end()) {
                continue;
            }
            
            auto* tensor_info = &loader_->tensors_[it->second];
            struct ggml_tensor* cur = ggml_new_tensor(cpu_ctx_, tensor_info->type, tensor_info->n_dims, tensor_info->dims);
            ggml_set_name(cur, tensor_info->name.c_str());
            if (NULL != cur) {
                layer->tensors[layer_info.first] = cur;
            }
        }
        cpu_weights_->layers.push_back(layer);
    }

    // 在 CPU 上分配 buffer
    cpu_buf_ = ggml_backend_alloc_ctx_tensors(cpu_ctx_, backend_cpu_);
    if (!cpu_buf_) {
        fprintf(stderr, "[Loader] ERROR: failed to allocate CPU weight buffer\n");
        return false;
    }

    for (auto& head_iter : cpu_weights_->heads) {
        loader_->load_tensor_data(head_iter.second);
    }
    for (auto layer_iter : cpu_weights_->layers) {
        for (auto& layer_info : layer_iter->tensors) {
            loader_->load_tensor_data(layer_info.second);
        }
    }

    printf("[Loader] CPU weight loading complete\n");
    return true;
}

bool Qwen35moeModel::init_gpu() {
    printf("\n===================Loading Qwen35moe Model to GPU=====================\n");
    backend_gpu_ = ggml_backend_init_best();
    if (!backend_gpu_) {
        fprintf(stderr, "[Loader] ERROR: failed to init GPU backend\n");
        return false;
    }

    int device_count = ggml_backend_cuda_get_device_count();
    if (device_count == 0) {
        printf("No CUDA devices found\n");
        return false;
    }
    int gpu_id = 0;
    if (gpu_id >= device_count) {
        printf("Invalid device %d (only %d available)\n", gpu_id, device_count);
        return false;
    }

    size_t free_mem, total_mem;
    ggml_backend_cuda_get_device_memory(gpu_id, &free_mem, &total_mem);
    printf("Device %d: %.2f GB free / %.2f GB total\n", gpu_id, free_mem / 1e9, total_mem / 1e9);

    gpu_weights_ = std::make_shared<Qwen35moeWeights>();
    for (auto& weight_info : g_weight_tensor_names) {
        auto it = loader_->tensor_index_map_.find(weight_info.second);
        if (it == loader_->tensor_index_map_.end()) {
            fprintf(stderr, "[Loader] ERROR: failed to find tensor %s in model file\n", weight_info.second);
            return false;
        }

        auto* tensor_info = &loader_->tensors_[it->second];
        struct ggml_tensor* cur = ggml_new_tensor(cpu_ctx_, tensor_info->type, tensor_info->n_dims, tensor_info->dims);
        ggml_set_name(cur, tensor_info->name.c_str());
        if (NULL != cur) {
            gpu_weights_->heads[weight_info.first] = cur;
        }
    }

    for (int layer_idx = 0; layer_idx < meta_->qwen35moe.block_count; layer_idx++) {
        std::string prefix = "blk." + std::to_string(layer_idx);
        std::shared_ptr<Qwen35moeLayer> layer = std::make_shared<Qwen35moeLayer>();
        for (auto& layer_info : g_layer_tensor_names) {
            std::string name = prefix + layer_info.second;
            auto it = loader_->tensor_index_map_.find(name);
            if (it == loader_->tensor_index_map_.end()) {
                continue;
            }
            
            auto* tensor_info = &loader_->tensors_[it->second];
            struct ggml_tensor* cur = ggml_new_tensor(cpu_ctx_, tensor_info->type, tensor_info->n_dims, tensor_info->dims);
            ggml_set_name(cur, tensor_info->name.c_str());
            if (NULL != cur) {
                layer->tensors[layer_info.first] = cur;
            }
        }
        gpu_weights_->layers.push_back(layer);
    }

    gpu_buf_ = ggml_backend_alloc_ctx_tensors(gpu_ctx_, backend_gpu_);
    if (!gpu_buf_) {
        fprintf(stderr, "[Loader] ERROR: failed to allocate GPU weight buffer\n");
        return false;
    }

    for (auto head_iter : gpu_weights_->heads) {
        loader_->load_tensor_data(head_iter.second);
    }
    for (auto layer_iter : gpu_weights_->layers) {
        for (auto& layer_info : layer_iter->tensors) {
            loader_->load_tensor_data(layer_info.second);
        }
    }
    
    printf("[Loader] CPU weight loading complete\n");
    return true;
}

ggml_backend_t Qwen35moeModel::get_curr_backend() {
    if (dev_mode_ == DevMode::CPU_MODE)
        return backend_cpu_;
    else
        return backend_gpu_;
}

ggml_backend_sched_t Qwen35moeModel::get_scheduler() const { 
    return sched_; 
}

ggml_tensor* Qwen35moeModel::get_weight_tensor(const EN_WEIGHT_TYPE weight_type) {
    if (dev_mode_ == DevMode::CPU_MODE)
        return cpu_weights_->heads[weight_type];
    else
        return gpu_weights_->heads[weight_type];
}
ggml_tensor* Qwen35moeModel::get_weight_layer_tensor(const EN_LAYER_TYPE layer_type, const int layer_idx) {
    if (dev_mode_ == DevMode::CPU_MODE)
        return cpu_weights_->layers[layer_idx]->tensors[layer_type];
    if (dev_mode_ == DevMode::GPU_MODE)
        return gpu_weights_->layers[layer_idx]->tensors[layer_type];
    if (layer_idx < gpu_layer_) {
        return gpu_weights_->layers[layer_idx]->tensors[layer_type];
    } else {
        return cpu_weights_->layers[layer_idx - gpu_layer_]->tensors[layer_type];
    }
}

struct ggml_tensor* Qwen35moeModel::get_token_embedding_weight()
{
    return get_weight_tensor(EN_WEIGHT_TYPE_TOKEN_EMBD);
}

struct ggml_tensor* Qwen35moeModel::get_output_weight()
{
    return get_weight_tensor(EN_WEIGHT_TYPE_OUTPUT);
}

struct ggml_tensor* Qwen35moeModel::get_output_norm_weight()
{
    return get_weight_tensor(EN_WEIGHT_TYPE_OUTPUT_NORM);
}

struct ggml_tensor* Qwen35moeModel::get_attn_k_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_ATTN_K, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_k_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_ATTN_K_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_ATTN_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_gate_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_ATTN_GATE, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_qkv_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_ATTN_QKV, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_q_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_ATTN_Q, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_q_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_ATTN_Q_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_v_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_ATTN_V, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_down_exps_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_FFN_DOWN_EXPS, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_gate_exps_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_FFN_GATE_EXPS, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_gate_inp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_FFN_GATE_INP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_up_exps_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_FFN_UP_EXPS, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_down_shexp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_FFN_DOWN_SHEXP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_gate_inp_shexp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_FFN_GATE_INP_SHEXP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_gate_shexp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_FFN_GATE_SHEXP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_up_shexp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_FFN_UP_SHEXP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_a_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_SSM_A, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_alpha_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_SSM_ALPHA, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_beta_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_SSM_BETA, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_conv1d_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_SSM_CONV1D, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_dt_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_SSM_DT, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_SSM_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_out_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_SSM_OUT, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_post_attention_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_POST_ATTENTION_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_output_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_LAYER_TYPE_ATTN_OUTPUT, layer_idx);
}

bool Qwen35moeModel::load_metadata() {
    return meta_->load_from_gguf(loader_.get());
}

int Qwen35moeModel::get_ctx_size() {
    return meta_->qwen35moe.context_length;
}
