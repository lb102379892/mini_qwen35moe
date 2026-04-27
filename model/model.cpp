#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef QWEN35MOE_USE_CUDA
#include <ggml-cuda.h>
#endif
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
    reader_ = std::make_shared<GGUFReader>();
    meta_ = std::make_shared<MetaDataInfo>();
    if (dev_mode == DevMode::CPU_MODE || dev_mode == DevMode::AURO_MODE)
        cpu_weights_ = std::make_shared<Qwen35moeWeights>();
    if (dev_mode == DevMode::GPU_MODE || dev_mode == DevMode::AURO_MODE)
        gpu_weights_ = std::make_shared<Qwen35moeWeights>();
    tensor_loader_ = std::make_shared<GGUFMmapTensorLoader>();
    if (false == reader_->open(model_path_)) {
        printf("[Loader] ERROR: failed open modelfile(%s)\n", model_path_.c_str());
        return false;
    }

    // 1: metadata
    if (!load_metadata()) {
        printf("[Loader] Config loading failed\n");
        return false;
    }

    if (dev_mode == DevMode::CPU_MODE || dev_mode == DevMode::AURO_MODE) {
        cpu_ctx_ = reader_->ggml_ctx_;
    }
#ifdef QWEN35MOE_USE_CUDA
    if (dev_mode == DevMode::GPU_MODE || dev_mode == DevMode::AURO_MODE) {
        auto mem_size = ggml_get_mem_size(cpu_ctx_);
        ggml_init_params gpu_p = { mem_size, nullptr, true };
        gpu_ctx_ = ggml_init(gpu_p);
        if (!gpu_ctx_) {
            fprintf(stderr, "[Loader] ERROR: failed to init weight contexts\n");
            return false;
        }
    }
#endif

    // 2: model weights
    if (!load_qwen35moe(meta_->qwen35moe.block_count)) {
        printf("[Loader] Tensor loading failed\n");
        return false;
    }

    // 3: tensor loader
    if (!tensor_loader_->load(model_path_)) {
        printf("[Loader] Tensor loading failed\n");
        return false;
    }
    
    if (dev_mode == DevMode::CPU_MODE || dev_mode == DevMode::AURO_MODE) {
        backend_cpu_ = ggml_backend_cpu_init();
        if (!backend_cpu_) {
            fprintf(stderr, "[Loader] ERROR: failed to init CPU backend\n");
            return false;
        }
        ggml_backend_cpu_set_n_threads(backend_cpu_, n_threads_);

        cpu_buf_ = ggml_backend_alloc_ctx_tensors(cpu_ctx_, backend_cpu_);
        if (!cpu_buf_) {
            fprintf(stderr, "[Loader] ERROR: failed to allocate CPU weight buffer\n");
            return false;
        }
    }
    printf("success backend_cpu initialized\n");

#ifdef QWEN35MOE_USE_CUDA
    if (dev_mode == DevMode::GPU_MODE || dev_mode == DevMode::AURO_MODE) {
        backend_gpu_ = ggml_backend_cuda_init(0);
        if (!backend_gpu_) {
            fprintf(stderr, "[Loader] ERROR: failed to init GPU backend\n");
            return false;
        }

        gpu_buf_ = ggml_backend_alloc_ctx_tensors(gpu_ctx_, backend_gpu_);
        if (!gpu_buf_) {
            fprintf(stderr, "[Loader] ERROR: failed to allocate GPU weight buffer\n");
            return false;
        }
    }
    printf("success backend_gpu initialized\n");
#endif

    // set weights data
    if (!set_tensors_data(meta_->qwen35moe.block_count)) {
        printf("[Loader] set_tensors_data failed\n");
        return false;
    }

    //print_context_info(cpu_ctx_);

    printf("==================Loading Complete!======================\n");
    return true;
}

ggml_backend_t Qwen35moeModel::get_curr_backend() {
    if (dev_mode_ == DevMode::CPU_MODE)
        return backend_cpu_;
    else
        return backend_gpu_;
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

bool Qwen35moeModel::load_metadata() {
    return meta_->load_from_gguf(reader_->gguf_ctx_);
}

bool Qwen35moeModel::set_tensor_data(const size_t data_offset, ggml_tensor* tensor) {
    auto gguf_ctx = reader_->gguf_ctx_;
    int64_t tensor_idx = gguf_find_tensor(reader_->gguf_ctx_, tensor->name);

    const size_t nbytes = ggml_nbytes(tensor);
    const size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx, tensor_idx);
    const size_t file_offset = data_offset + tensor_offset;

    std::vector<uint8_t> tensor_data;
    tensor_loader_->get_tensor_data(file_offset, nbytes, tensor_data);
    ggml_backend_tensor_set(tensor, tensor_data.data(), 0, tensor_data.size());

    return true;
}

bool Qwen35moeModel::load_qwen35moe(const int layer_count) {
    printf("[Loader] load_qwen35moe...\n");

    if (dev_mode_ == DevMode::CPU_MODE) {
        for (auto& weight : g_weight_tensor_names) {
            struct ggml_tensor *tensor = ggml_get_tensor(cpu_ctx_, weight.second);
            if (NULL != tensor) {
                cpu_weights_->heads[weight.first] = tensor;
            }
        }

        for (uint32_t i = 0; i < layer_count; ++i) {
            std::shared_ptr<Qwen35moeLayer> layer = std::make_shared<Qwen35moeLayer>();
            load_qwen35moe_layer(layer, i);
            cpu_weights_->layers.push_back(layer);
        }
    } else {
        for (auto& weight : g_weight_tensor_names) {
            struct ggml_tensor *tensor = ggml_get_tensor(cpu_ctx_, weight.second);
            if (NULL == tensor) {
                continue;
            }
            ggml_tensor* new_tensor = ggml_dup_tensor(gpu_ctx_, tensor);
            if (NULL != new_tensor) {
                ggml_set_name(new_tensor, tensor->name);
                gpu_weights_->heads[weight.first] = new_tensor;
            }
        }

        for (uint32_t i = 0; i < layer_count; ++i) {
            std::shared_ptr<Qwen35moeLayer> layer = std::make_shared<Qwen35moeLayer>();
            load_qwen35moe_layer(layer, i);
            if (i < gpu_layer_) {
                gpu_weights_->layers.push_back(layer);
            } else {
                cpu_weights_->layers.push_back(layer);
            }
        }
    }

    printf("[Loader] LLM loaded tensors [OK]\n");
    return true;
}

void Qwen35moeModel::load_qwen35moe_layer(std::shared_ptr<Qwen35moeLayer>& layer, int layer_idx) {
    auto& layer_tensors = layer->tensors;
    std::string prefix = "blk." + std::to_string(layer_idx);

    std::string name = "";
    for (auto& layer : g_layer_tensor_names) {
        name = prefix + layer.second;
        struct ggml_tensor *tensor = ggml_get_tensor(cpu_ctx_, name.c_str());
        if (NULL == tensor) {
            continue;
        }
        
        if (dev_mode_ == DevMode::CPU_MODE || layer_idx >= gpu_layer_) {
            layer_tensors[layer.first] = tensor;
        } else {
            ggml_tensor* new_tensor = ggml_dup_tensor(gpu_ctx_, tensor);
            if (NULL != new_tensor) {
                ggml_set_name(new_tensor, tensor->name);
                layer_tensors[layer.first] = new_tensor;
            }
        }
    }
}

bool Qwen35moeModel::set_tensors_data(const int layer_count) {
    printf("[Loader] set_tensors_data...\n");

    auto gguf_ctx = reader_->gguf_ctx_;
    const size_t data_offset = gguf_get_data_offset(gguf_ctx);

    if (dev_mode_ == DevMode::CPU_MODE) {
        std::shared_ptr<Qwen35moeWeights> weights = dev_mode_ == DevMode::CPU_MODE ? cpu_weights_ : gpu_weights_;
        ggml_context* ctx = dev_mode_ == DevMode::CPU_MODE ? cpu_ctx_ : gpu_ctx_;

        for (auto& weight : g_weight_tensor_names) {
            if (weights->heads.find(weight.first) != weights->heads.end()) {
                set_tensor_data(data_offset, weights->heads[weight.first]);
            }
        }

        for (uint32_t i = 0; i < layer_count; ++i) {
            set_tensors_layer_data(ctx, weights->layers[i], data_offset, i);
        }
    } else {
        for (auto& weight : g_weight_tensor_names) {
            if (gpu_weights_->heads.find(weight.first) != gpu_weights_->heads.end()) {
                set_tensor_data(data_offset, gpu_weights_->heads[weight.first]);
            }
        }

        for (uint32_t i = 0; i < layer_count; ++i) {
            if (i < gpu_layer_) {
                set_tensors_layer_data(gpu_ctx_, gpu_weights_->layers[i], data_offset, i);
            } else {
                set_tensors_layer_data(cpu_ctx_, cpu_weights_->layers[i - gpu_layer_], data_offset, i);
            }
        }
    }

    printf("[Loader] set_tensors_data [OK]\n");
    return true;
}

bool Qwen35moeModel::set_tensors_layer_data(ggml_context* ctx, std::shared_ptr<Qwen35moeLayer>& layer, const size_t data_offset, int layer_idx) {
    auto& layer_tensors = layer->tensors;

    std::string name = "";
    for (auto& layer : g_layer_tensor_names) {
        if (layer_tensors.find(layer.first) != layer_tensors.end()) {
            set_tensor_data(data_offset, layer_tensors[layer.first]);
        }
    }

    return true;
}

int Qwen35moeModel::get_ctx_size() {
    return meta_->qwen35moe.context_length;
}

void Qwen35moeModel::print_context_info(ggml_context* ctx_) {
    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx_); cur != NULL; cur = ggml_get_next_tensor(ctx_, cur)) {
        const char* name = ggml_get_name(cur);
        size_t n_size = ggml_nbytes(cur);
        ggml_type type = cur->type;
        ggml_op op = cur->op;
        size_t offs = cur->view_offs;
        size_t offs1 = gguf_get_data_offset(reader_->gguf_ctx_) + gguf_get_tensor_offset(reader_->gguf_ctx_, gguf_find_tensor(reader_->gguf_ctx_, name));      
        void* data = ggml_get_data(cur);
        printf("Tensor: %s, size: %d, type: %d, op: %d, offs: %lu, offs1: %lu, data: %p, view_src: %p\n", name, n_size, type, op, offs, offs1, data, cur->view_src);
    }
}