#include <algorithm>
#include <cstdlib>
#include <set>
#include <string>
#include "model/model.h"

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

bool Qwen35moeModel::init(const std::string& model_path_, DevMode dev_mode, int n_threads, size_t gpu_layer, bool no_mmap) {
    printf("\n===================Loading Qwen35moe Model=====================\n");
    dev_mode_ = dev_mode;
    n_threads_ = n_threads;
    gpu_layer_ = gpu_layer;
    no_mmap_ = no_mmap;
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
    if (gpu_layer_ > loader_->n_tensors_) {
        gpu_layer_ = loader_->n_tensors_;
    }

    init_cpu();
    init_gpu();
    rebuild_layer_device_map();

    if (no_mmap_) {
        // 权重加载完成释放模型文件占用的内存映射，避免占用过多内存
        loader_->unload();
    }

    if (dev_mode_ == DevMode::GPU_MODE) {
        ggml_backend_t backends[] = {backend_gpu_, backend_cpu_};
        sched_ = ggml_backend_sched_new(backends, nullptr, 2, QWEN_DEFAULT_GRAPH_SIZE, false, true);
        if (!sched_) {
            fprintf(stderr, "Failed to create backend scheduler\n");
            return false;
        }
        fprintf(stderr, "Scheduler created with GPU backends\n");
    } else if (dev_mode_ == DevMode::AUTO_MODE) {
        ggml_backend_t backends[] = {backend_gpu_, backend_cpu_};
        sched_ = ggml_backend_sched_new(backends, nullptr, 2, QWEN_DEFAULT_GRAPH_SIZE, true, true);
        if (!sched_) {
            fprintf(stderr, "Failed to create backend scheduler\n");
            return false;
        }
        fprintf(stderr, "Scheduler created with CPU + GPU backends\n");
    } else {
        ggml_backend_t backends[] = {backend_cpu_};
        sched_ = ggml_backend_sched_new(backends, nullptr, 1, QWEN_DEFAULT_GRAPH_SIZE, false, false);
        if (!sched_) {
            fprintf(stderr, "Failed to create backend scheduler\n");
            return false;
        }
        fprintf(stderr, "Scheduler created with CPU backends\n");
    }
    
    printf("==================Loading Complete!======================\n");
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

bool Qwen35moeModel::is_mixed_mode() const {
    return dev_mode_ == DevMode::AUTO_MODE &&
           gpu_weights_ != nullptr && !gpu_weights_->layers.empty() &&
           cpu_weights_ != nullptr && !cpu_weights_->layers.empty();
}

uint64_t Qwen35moeModel::compute_device_map_hash() const {
    if (layer_device_map_.empty()) {
        return static_cast<uint64_t>(dev_mode_);
    }

    uint64_t hash = 0xcbf29ce484222325ULL;  // FNV offset basis
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;
    for (size_t il = 0; il < layer_device_map_.size(); ++il) {
        uint64_t tag = 2ULL;
        if (layer_device_map_[il] == backend_gpu_) {
            tag = 0ULL;
        } else if (layer_device_map_[il] == backend_cpu_) {
            tag = 1ULL;
        }
        uint64_t v = (static_cast<uint64_t>(il) << 2) | tag;
        hash = (hash ^ v) * FNV_PRIME;
    }
    return hash;
}

const std::vector<ggml_backend_t>& Qwen35moeModel::get_layer_device_map() const {
    return layer_device_map_;
}

ggml_backend_t Qwen35moeModel::get_layer_backend(uint32_t layer_idx) const {
    if (layer_idx < layer_device_map_.size() && layer_device_map_[layer_idx] != nullptr) {
        return layer_device_map_[layer_idx];
    }
    return dev_mode_ == DevMode::CPU_MODE ? backend_cpu_ : backend_gpu_;
}

void Qwen35moeModel::rebuild_layer_device_map() {
    const size_t n_layers = meta_ ? static_cast<size_t>(meta_->qwen35moe.block_count) : 0;
    layer_device_map_.assign(n_layers, backend_cpu_);
    if (dev_mode_ == DevMode::GPU_MODE) {
        std::fill(layer_device_map_.begin(), layer_device_map_.end(), backend_gpu_);
        return;
    }
    if (dev_mode_ == DevMode::CPU_MODE) {
        std::fill(layer_device_map_.begin(), layer_device_map_.end(), backend_cpu_);
        return;
    }
    for (size_t il = 0; il < n_layers; ++il) {
        ggml_backend_t backend = backend_cpu_;
        if (gpu_weights_ && gpu_weights_->layers.find(static_cast<int>(il)) != gpu_weights_->layers.end()) {
            backend = backend_gpu_;
        } else if (cpu_weights_ && cpu_weights_->layers.find(static_cast<int>(il)) != cpu_weights_->layers.end()) {
            backend = backend_cpu_;
        }
        layer_device_map_[il] = backend;
    }
}

ggml_tensor* Qwen35moeModel::get_weight_tensor(const EN_WEIGHT_TYPE weight_type) {
    if (dev_mode_ == DevMode::CPU_MODE)
        return cpu_weights_->heads[weight_type];
    else if (dev_mode_ == DevMode::GPU_MODE)
        return gpu_weights_->heads[weight_type];
    else {
        if (gpu_weights_->heads.find(weight_type) != gpu_weights_->heads.end()) {
            return gpu_weights_->heads[weight_type];
        }
        
        return cpu_weights_->heads[weight_type];
    }
}
ggml_tensor* Qwen35moeModel::get_weight_layer_tensor(const EN_WEIGHT_TYPE layer_type, const int layer_idx) {
    if (dev_mode_ == DevMode::CPU_MODE)
        return cpu_weights_->layers[layer_idx]->tensors[layer_type];
    else if (dev_mode_ == DevMode::GPU_MODE)
        return gpu_weights_->layers[layer_idx]->tensors[layer_type];
    else {
        if (gpu_weights_->layers.find(layer_idx) != gpu_weights_->layers.end()) {
            if (gpu_weights_->layers[layer_idx]->tensors.find(layer_type) != gpu_weights_->layers[layer_idx]->tensors.end()) {
                return gpu_weights_->layers[layer_idx]->tensors[layer_type];
            }
        }
        
        return cpu_weights_->layers[layer_idx]->tensors[layer_type];
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
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_ATTN_K, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_k_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_ATTN_K_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_ATTN_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_gate_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_ATTN_GATE, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_qkv_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_ATTN_QKV, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_q_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_ATTN_Q, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_q_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_ATTN_Q_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_v_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_ATTN_V, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_down_exps_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_FFN_DOWN_EXPS, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_gate_exps_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_FFN_GATE_EXPS, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_gate_inp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_FFN_GATE_INP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_up_exps_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_FFN_UP_EXPS, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_down_shexp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_FFN_DOWN_SHEXP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_gate_inp_shexp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_FFN_GATE_INP_SHEXP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_gate_shexp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_FFN_GATE_SHEXP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ffn_up_shexp_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_FFN_UP_SHEXP, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_a_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_SSM_A, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_alpha_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_SSM_ALPHA, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_beta_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_SSM_BETA, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_conv1d_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_SSM_CONV1D, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_dt_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_SSM_DT, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_SSM_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_ssm_out_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_SSM_OUT, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_post_attention_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_POST_ATTENTION_NORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_attn_output_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_ATTN_OUTPUT, layer_idx);
}

bool Qwen35moeModel::init_cpu() {
    printf("===================Loading Qwen35moe Model to CPU=====================\n");
    backend_cpu_ = ggml_backend_cpu_init();
    if (!backend_cpu_) {
        fprintf(stderr, "[Loader] ERROR: failed to init CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend_cpu_, n_threads_);

    if (dev_mode_ == DevMode::AUTO_MODE) {
        return true;
    }

    const size_t ctx_size = loader_->get_all_tensor_bytesize();
    ggml_init_params cpu_p = { ctx_size, nullptr, true };
    cpu_ctx_ = ggml_init(cpu_p);
    if (!cpu_ctx_) {
        fprintf(stderr, "[Loader] ERROR: failed to init weight contexts\n");
        return false;
    }

    if (dev_mode_ == DevMode::GPU_MODE) {
        printf("[Loader] AURO_MODE: skipping CPU weight loading\n");
        return true;
    }

    cpu_weights_ = std::make_shared<Qwen35moeWeights>();
    for (auto& tensor_iter : loader_->tensors_head_) {
        auto& tensor_info = tensor_iter.second;

        struct ggml_tensor* cur = ggml_new_tensor(cpu_ctx_, tensor_info.type, tensor_info.n_dims, tensor_info.dims);
        if (NULL != cur) {
            ggml_set_name(cur, tensor_info.name.c_str());
            cpu_weights_->heads[tensor_info.weight_type] = cur;
        }
    }

    for (auto& tensor_info : loader_->tensors_layer_) {
        struct ggml_tensor* cur = ggml_new_tensor(cpu_ctx_, tensor_info.type, tensor_info.n_dims, tensor_info.dims);
        if (NULL != cur) {
            ggml_set_name(cur, tensor_info.name.c_str());
            if (cpu_weights_->layers.find(tensor_info.layer_idx) == cpu_weights_->layers.end()) {
                cpu_weights_->layers[tensor_info.layer_idx] = std::make_shared<Qwen35moeLayer>();
            }
            cpu_weights_->layers[tensor_info.layer_idx]->tensors[tensor_info.weight_type] = cur;
        }
    }

    // 在 CPU 上分配 buffer
    // Try mmap zero-copy path first: bind tensor data pointers directly to the mmap region.
    // Controlled by MINI_QWEN_MMAP_ZEROCOPY env var (default: enabled).
    bool zerocopy_ok = false;
    if (false == no_mmap_ && loader_->is_mmap_active()) {
        cpu_buf_ = loader_->create_cpu_mmap_buffer();
        if (cpu_buf_) {
            bool all_ok = true;
            for (auto& head_iter : cpu_weights_->heads) {
                if (!loader_->bind_tensor_to_mmap(cpu_buf_, head_iter.second, true)) {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok) {
                for (auto layer_iter = cpu_weights_->layers.begin();
                     layer_iter != cpu_weights_->layers.end() && all_ok; ++layer_iter) {
                    for (auto& layer_info : layer_iter->second->tensors) {
                        if (!loader_->bind_tensor_to_mmap(cpu_buf_, layer_info.second, false)) {
                            all_ok = false;
                            break;
                        }
                    }
                }
            }
            if (all_ok) {
                zerocopy_ok = true;
                printf("[Loader] CPU mmap zero-copy enabled -- skipping weight copy\n");
            } else {
                // At least one tensor failed; undo the partial buffer and fall through to copy path.
                ggml_backend_buffer_free(cpu_buf_);
                cpu_buf_ = nullptr;
            }
        }
    }

    if (!zerocopy_ok) {
        cpu_buf_ = ggml_backend_alloc_ctx_tensors(cpu_ctx_, backend_cpu_);
        if (!cpu_buf_) {
            fprintf(stderr, "[Loader] ERROR: failed to allocate CPU weight buffer\n");
            return false;
        }

        for (auto& head_iter : cpu_weights_->heads) {
            loader_->load_tensor_head_data(head_iter.second);
        }
        auto layer_iter = cpu_weights_->layers.begin();
        for (; layer_iter != cpu_weights_->layers.end(); ++layer_iter) {
            for (auto& layer_info : layer_iter->second->tensors) {
                loader_->load_tensor_layer_data(layer_info.second);
            }
            printf("Loaded layer %d to CPU\n", layer_iter->first);
        }
        printf("[Loader] CPU weight loading complete\n");
    }

    return true;
}

bool Qwen35moeModel::init_gpu() {
    if (dev_mode_ == DevMode::CPU_MODE) {
        return false;
    }
    printf("===================Loading Qwen35moe Model to GPU=====================\n");
    int gpu_id = 0;
    backend_gpu_ = ggml_backend_cuda_init(gpu_id);//ggml_backend_init_best();
    if (!backend_gpu_) {
        fprintf(stderr, "[Loader] ERROR: failed to init GPU backend\n");
        return false;
    }

    int device_count = ggml_backend_cuda_get_device_count();
    if (device_count == 0) {
        printf("No CUDA devices found\n");
        return false;
    }
    
    if (gpu_id >= device_count) {
        printf("Invalid device %d (only %d available)\n", gpu_id, device_count);
        return false;
    }

    size_t free_mem, total_mem;
    ggml_backend_cuda_get_device_memory(gpu_id, &free_mem, &total_mem);
    printf("Device %d: %.2f GB free / %.2f GB total\n", gpu_id, free_mem / 1e9, total_mem / 1e9);

    if (dev_mode_ == DevMode::AUTO_MODE) {
        return init_auto_gpu(free_mem);
    }

    const size_t ctx_size = loader_->get_all_tensor_bytesize();
    ggml_init_params gpu_p = { ctx_size, nullptr, true };
    gpu_ctx_ = ggml_init(gpu_p);
    if (!gpu_ctx_) {
        fprintf(stderr, "[Loader] ERROR: failed to init weight contexts\n");
        return false;
    }

    gpu_weights_ = std::make_shared<Qwen35moeWeights>();
    for (auto& tensor_iter : loader_->tensors_head_) {
        auto* tensor_info = &tensor_iter.second;

        struct ggml_tensor* cur = nullptr;
        if (tensor_info->weight_type == EN_WEIGHT_TYPE_TOKEN_EMBD) {
            cur = ggml_new_tensor_2d(gpu_ctx_, GGML_TYPE_F16, tensor_info->dims[0], tensor_info->dims[1]);
        } else {
            cur = ggml_new_tensor(gpu_ctx_, tensor_info->type, tensor_info->n_dims, tensor_info->dims);
        }
        if (NULL != cur) {
            ggml_set_name(cur, tensor_info->name.c_str());
            gpu_weights_->heads[tensor_info->weight_type] = cur;
        }
    }

    for (auto& tensor_info : loader_->tensors_layer_) {
        struct ggml_tensor* cur = ggml_new_tensor(gpu_ctx_, tensor_info.type, tensor_info.n_dims, tensor_info.dims);
        if (NULL != cur) {
            ggml_set_name(cur, tensor_info.name.c_str());
            if (gpu_weights_->layers.find(tensor_info.layer_idx) == gpu_weights_->layers.end()) {
                gpu_weights_->layers[tensor_info.layer_idx] = std::make_shared<Qwen35moeLayer>();
            }
            gpu_weights_->layers[tensor_info.layer_idx]->tensors[tensor_info.weight_type] = cur;
        }
    }

    gpu_buf_ = ggml_backend_alloc_ctx_tensors(gpu_ctx_, backend_gpu_);
    if (!gpu_buf_) {
        fprintf(stderr, "[Loader] ERROR: failed to allocate GPU weight buffer\n");
        return false;
    }

    for (auto& head_iter : gpu_weights_->heads) {
        if (head_iter.first == EN_WEIGHT_TYPE_TOKEN_EMBD) {
            dequant_set(head_iter.second);
        } else {
            loader_->load_tensor_head_data(head_iter.second);
        }
    }
    auto layer_iter = gpu_weights_->layers.begin();
    for (; layer_iter != gpu_weights_->layers.end(); ++layer_iter) {
        for (auto& layer_info : layer_iter->second->tensors) {
            loader_->load_tensor_layer_data(layer_info.second);
        }
        printf("Loaded layer %d to GPU\n", layer_iter->first);
    }
    printf("[Loader] GPU weight loading complete\n");
    
    return true;
}

bool Qwen35moeModel::init_auto_cpu(const std::set<int, std::less<int>>& gpu_layer_set) {
    size_t use_byte = 0;
    auto beginiter = loader_->tensor_layer_index_list_.begin();
    for (; beginiter != loader_->tensor_layer_index_list_.end(); ++beginiter) {
        if (gpu_layer_set.find(beginiter->first) != gpu_layer_set.end()) {
            continue;
        }
        for (auto& index : beginiter->second) {
            use_byte += loader_->get_tensor_bytesize(loader_->tensors_layer_[index]);
        }
    }

    ggml_init_params cpu_p = { use_byte, nullptr, true };
    cpu_ctx_ = ggml_init(cpu_p);
    if (!cpu_ctx_) {
        fprintf(stderr, "[Loader] ERROR: failed to init weight contexts\n");
        return false;
    }

    cpu_weights_ = std::make_shared<Qwen35moeWeights>();
    beginiter = loader_->tensor_layer_index_list_.begin();
    for (; beginiter != loader_->tensor_layer_index_list_.end(); ++beginiter) {
        if (gpu_layer_set.find(beginiter->first) != gpu_layer_set.end()) {
            continue;
        }
        for (auto& index : beginiter->second) {
            auto* iter = &loader_->tensors_layer_[index];
            struct ggml_tensor* cur = ggml_new_tensor(cpu_ctx_, iter->type, iter->n_dims, iter->dims);
            if (NULL != cur) {
                ggml_set_name(cur, iter->name.c_str());
                if (cpu_weights_->layers.find(iter->layer_idx) == cpu_weights_->layers.end()) {
                    cpu_weights_->layers[iter->layer_idx] = std::make_shared<Qwen35moeLayer>();
                }
                cpu_weights_->layers[iter->layer_idx]->tensors[iter->weight_type] = cur;
            }
        }
    }

    // 在 CPU 上分配 buffer
    // Try mmap zero-copy for CPU layer tensors in AUTO_MODE.
    bool zerocopy_ok = false;
    if (false == no_mmap_ && loader_->is_mmap_active()) {
        cpu_buf_ = loader_->create_cpu_mmap_buffer();
        if (cpu_buf_) {
            bool all_ok = true;
            for (auto layer_iter = cpu_weights_->layers.begin();
                 layer_iter != cpu_weights_->layers.end() && all_ok; ++layer_iter) {
                for (auto& layer_info : layer_iter->second->tensors) {
                    if (!loader_->bind_tensor_to_mmap(cpu_buf_, layer_info.second, false)) {
                        all_ok = false;
                        break;
                    }
                }
            }
            if (all_ok) {
                zerocopy_ok = true;
                printf("[Loader] auto-CPU mmap zero-copy enabled -- skipping weight copy\n");
            } else {
                ggml_backend_buffer_free(cpu_buf_);
                cpu_buf_ = nullptr;
            }
        }
    }

    if (!zerocopy_ok) {
        cpu_buf_ = ggml_backend_alloc_ctx_tensors(cpu_ctx_, backend_cpu_);
        if (!cpu_buf_) {
            fprintf(stderr, "[Loader] ERROR: failed to allocate auto CPU weight buffer\n");
            return false;
        }

        auto layer_iter = cpu_weights_->layers.begin();
        for (; layer_iter != cpu_weights_->layers.end(); ++layer_iter) {
            for (auto& layer_info : layer_iter->second->tensors) {
                loader_->load_tensor_layer_data(layer_info.second);
            }
            printf("Loaded layer blk.%d.* to auto CPU\n", layer_iter->first);
        }
        printf("[Loader] auto CPU weight loading complete\n");
    }

    return true;
}

bool Qwen35moeModel::is_full_attention_layer(uint32_t layer_idx) const {
    return (layer_idx % meta_->qwen35moe.full_attention_interval) == (meta_->qwen35moe.full_attention_interval - 1);
}

bool Qwen35moeModel::init_auto_gpu(size_t& free_mem) {
    // 留出 10% 的余量，避免显存占满导致系统不稳定
    free_mem = (size_t)(free_mem * 0.9); 

    size_t use_byte = 0;
    size_t curr_gpu_layer = 0;
    size_t curr_tensor_byte = 0;
    // 输出头，必须优先放GPU
    for (auto& tensor_iter : loader_->tensors_head_) {
        auto* tensor_info = &tensor_iter.second;
        curr_tensor_byte = loader_->get_tensor_bytesize(*tensor_info);
        if (use_byte + curr_tensor_byte > free_mem) {
            fprintf(stderr, "[Loader] ERROR: failed to allocate tensor %s due to insufficient GPU memory\n", tensor_iter.first.c_str());
            return false;
        }
        use_byte += curr_tensor_byte;
        ++curr_gpu_layer;
        if (gpu_layer_ > 0 && curr_gpu_layer > gpu_layer_) {
            fprintf(stderr, "[Loader] ERROR: failed to allocate tensor %s due to insufficient GPU memory\n", tensor_iter.first.c_str());
            return false;
        }
    }

    // 末尾连续层的其余DeltaNet+间隔全注意力，优先全放 GPU
    std::set<int, std::less<int>> gpu_layer_set;
    auto iter = loader_->tensor_layer_index_list_.rbegin();
    for (; iter != loader_->tensor_layer_index_list_.rend(); ++iter) {
        size_t layer_use_byte = 0;
        size_t gpu_layer_num = iter->second.size();
        for (auto& index : iter->second) {
            layer_use_byte += loader_->get_tensor_bytesize(loader_->tensors_layer_[index]);
        }
        if (
            (use_byte + layer_use_byte > free_mem) || 
            (gpu_layer_ > 0 && (curr_gpu_layer + gpu_layer_num > gpu_layer_))
        ) {
            break;
        }
        use_byte += layer_use_byte;
        curr_gpu_layer += gpu_layer_num;
        gpu_layer_set.insert(iter->first);
    }

    ggml_init_params gpu_p = { use_byte, nullptr, true };
    gpu_ctx_ = ggml_init(gpu_p);
    if (!gpu_ctx_) {
        fprintf(stderr, "[Loader] ERROR: failed to init weight contexts\n");
        return false;
    }

    gpu_weights_ = std::make_shared<Qwen35moeWeights>();
    for (auto& tensor_iter : loader_->tensors_head_) {
        auto* tensor_info = &tensor_iter.second;

        struct ggml_tensor* cur = nullptr;
        if (tensor_info->weight_type == EN_WEIGHT_TYPE_TOKEN_EMBD) {
            cur = ggml_new_tensor_2d(gpu_ctx_, GGML_TYPE_F16, tensor_info->dims[0], tensor_info->dims[1]);
        } else {
            cur = ggml_new_tensor(gpu_ctx_, tensor_info->type, tensor_info->n_dims, tensor_info->dims);
        }
        if (NULL != cur) {
            ggml_set_name(cur, tensor_info->name.c_str());
            gpu_weights_->heads[tensor_info->weight_type] = cur;
        }
    }

    for(auto& layer_index : gpu_layer_set) {
        auto layer_list = loader_->tensor_layer_index_list_.find(layer_index);
        for (auto& index : layer_list->second) {
            auto* iter = &loader_->tensors_layer_[index];
            struct ggml_tensor* cur = ggml_new_tensor(gpu_ctx_, iter->type, iter->n_dims, iter->dims);
            if (NULL != cur) {
                ggml_set_name(cur, iter->name.c_str());
                if (gpu_weights_->layers.find(iter->layer_idx) == gpu_weights_->layers.end()) {
                    gpu_weights_->layers[iter->layer_idx] = std::make_shared<Qwen35moeLayer>();
                }
                gpu_weights_->layers[iter->layer_idx]->tensors[iter->weight_type] = cur;
            }
        }
    }

    gpu_buf_ = ggml_backend_alloc_ctx_tensors(gpu_ctx_, backend_gpu_);
    if (!gpu_buf_) {
        fprintf(stderr, "[Loader] ERROR: failed to allocate auto GPU weight buffer\n");
        return false;
    }

    for (auto& head_iter : gpu_weights_->heads) {
        if (head_iter.first == EN_WEIGHT_TYPE_TOKEN_EMBD) {
            dequant_set(head_iter.second);
        } else {
            loader_->load_tensor_head_data(head_iter.second);
        }
        printf("Loaded auto head %s to GPU\n", head_iter.second->name);
    }
    auto layer_iter = gpu_weights_->layers.begin();
    for (; layer_iter != gpu_weights_->layers.end(); ++layer_iter) {
        for (auto& layer_info : layer_iter->second->tensors) {
            loader_->load_tensor_layer_data(layer_info.second);
        }
        printf("Loaded auto layer blk.%d.* to GPU\n", layer_iter->first);
    }
    printf("[Loader] auto GPU weight loading complete\n");

    if (gpu_layer_set.size() == loader_->tensor_layer_index_list_.size()) {
        return true;
    }
    
    return init_auto_cpu(gpu_layer_set);
}

bool Qwen35moeModel::load_metadata() {
    return meta_->load_from_gguf(loader_.get());
}

int Qwen35moeModel::get_ctx_size() {
    return meta_->qwen35moe.context_length;
}

void Qwen35moeModel::dequant_set(ggml_tensor* dst) {
    if (!dst) 
        return;

    auto* tensor = &loader_->tensors_head_[dst->name];

    const int64_t ncols = tensor->dims[0];
    int64_t nrows = tensor->dims[1];
    for (int64_t i = 2; i < tensor->n_dims; ++i) {
        nrows *= tensor->dims[i];
    }
    const auto* tr = ggml_get_type_traits(tensor->type);

    std::vector<uint8_t> src_data(tensor->byte_size);
    loader_->get_tensor_data(tensor, src_data);

    std::vector<float> row32(ncols);
    std::vector<ggml_fp16_t> row16(ncols);
    for (int64_t r = 0; r < nrows; ++r) {
        const uint8_t * rp = src_data.data() + r * tensor->type_size*(ncols/tensor->blck_size);
        for (int64_t c = 0; c < ncols; c += tensor->blck_size) {
            tr->to_float(rp + (c / tensor->blck_size) * tensor->type_size, row32.data() + c, tensor->blck_size);
        }
        for (int64_t i = 0; i < ncols; ++i)
            row16[i] = ggml_fp32_to_fp16(row32[i]);
        ggml_backend_tensor_set(dst, row16.data(), (size_t)r * dst->nb[1], (size_t)ncols * sizeof(ggml_fp16_t));
    }
}

void Qwen35moeModel::print_context_info(gguf_context* gguf_ctx, ggml_context* ctx_) {
    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx_); cur != NULL; cur = ggml_get_next_tensor(ctx_, cur)) {
        const char* name = ggml_get_name(cur);
        size_t n_size = ggml_nbytes(cur);
        ggml_type type = cur->type;
        ggml_op op = cur->op;
        size_t data_offs = gguf_get_tensor_offset(gguf_ctx, gguf_find_tensor(gguf_ctx, name));
        size_t file_offs = gguf_get_data_offset(gguf_ctx) + data_offs;      
        void* data = ggml_get_data(cur);
        printf("Tensor: %s, size: %d, type: %d, op: %d, data_offs: %lu, file_offs: %lu, data: %p, view_src: %p\n", name, n_size, type, op, data_offs, file_offs, data, cur->view_src);
    }
}
