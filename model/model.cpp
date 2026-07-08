#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "model/model.h"

namespace {

bool auto_boundary_align_disabled() {
    const char* v = std::getenv("QWEN35MOE_AUTO_BOUNDARY_DISABLE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

}  // namespace

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

bool Qwen35moeModel::init(
    const std::string& model_path_,
    DevMode dev_mode,
    int n_threads,
    size_t gpu_layer,
    bool no_mmap,
    int gpu_id,
    const std::vector<std::string>& tensor_overrides,
    int n_threads_batch
) {
    printf("\n===================Loading Qwen35moe Model=====================\n");
    dev_mode_ = dev_mode;
    n_threads_ = std::max(1, n_threads);
    n_threads_batch_ = std::max(0, n_threads_batch);
    gpu_layer_ = gpu_layer;
    no_mmap_ = no_mmap;
    gpu_id_ = gpu_id;
    tensor_overrides_.clear();
    for (const std::string& spec : tensor_overrides) {
        std::string error;
        if (!tensor_overrides_.add_rule(spec, &error)) {
            fprintf(stderr, "[Loader] ERROR: %s\n", error.c_str());
            return false;
        }
    }
    if (!tensor_overrides_.empty()) {
        tensor_overrides_.log_rules();
    }
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
    const size_t n_model_layers = loader_->tensor_layer_index_list_.size();
    if (gpu_layer_ > n_model_layers) {
        gpu_layer_ = n_model_layers;
    }

    init_cpu();
    if (!init_gpu() && dev_mode_ != DevMode::CPU_MODE) {
        return false;
    }
    rebuild_layer_device_map();
    maybe_release_model_file();

    if (dev_mode_ == DevMode::GPU_MODE) {
        ggml_backend_t backends[] = {backend_gpu_, backend_cpu_};
        const bool sched_offload = uses_tensor_overrides() || is_mixed_mode();
        sched_ = ggml_backend_sched_new(backends, nullptr, 2, QWEN_DEFAULT_GRAPH_SIZE, sched_offload, true);
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

ggml_backend_sched_t Qwen35moeModel::create_scheduler() const {
    if (dev_mode_ == DevMode::CPU_MODE) {
        ggml_backend_t backends[] = { backend_cpu_ };
        return ggml_backend_sched_new(backends, nullptr, 1, QWEN_DEFAULT_GRAPH_SIZE, false, false);
    }
    ggml_backend_t backends[] = { backend_gpu_, backend_cpu_ };
    if (dev_mode_ == DevMode::GPU_MODE) {
        const bool parallel = uses_tensor_overrides() || is_mixed_mode();
        return ggml_backend_sched_new(backends, nullptr, 2, QWEN_DEFAULT_GRAPH_SIZE, parallel, true);
    }
    return ggml_backend_sched_new(backends, nullptr, 2, QWEN_DEFAULT_GRAPH_SIZE, true, true);
}

bool Qwen35moeModel::is_mixed_mode() const {
    const auto has_gpu_weights = [&]() -> bool {
        if (!gpu_weights_) {
            return false;
        }
        return !gpu_weights_->heads.empty() || !gpu_weights_->layers.empty();
    };
    const auto has_cpu_weights = [&]() -> bool {
        if (!cpu_weights_) {
            return false;
        }
        return !cpu_weights_->heads.empty() || !cpu_weights_->layers.empty();
    };

    if (dev_mode_ == DevMode::CPU_MODE) {
        return false;
    }
    if (dev_mode_ == DevMode::GPU_MODE) {
        return has_gpu_weights() && has_cpu_weights();
    }
    return dev_mode_ == DevMode::AUTO_MODE &&
           has_gpu_weights() &&
           has_cpu_weights();
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
    if (uses_tensor_overrides()) {
        hash = (hash ^ 0x6c6f745f6f74ULL) * FNV_PRIME;
        hash = (hash ^ static_cast<uint64_t>(tensor_overrides_.size())) * FNV_PRIME;
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

WeightDevice Qwen35moeModel::resolve_weight_device(
    const std::string& tensor_name,
    WeightDevice default_device
) const {
    return tensor_overrides_.resolve(tensor_name, default_device);
}

WeightDevice Qwen35moeModel::default_head_weight_device() const {
    if (dev_mode_ == DevMode::CPU_MODE) {
        return WeightDevice::CPU;
    }
    return WeightDevice::GPU;
}

WeightDevice Qwen35moeModel::default_layer_weight_device(
    int layer_idx,
    const std::set<int, std::less<int>>* gpu_layer_set
) const {
    if (dev_mode_ == DevMode::CPU_MODE) {
        return WeightDevice::CPU;
    }
    if (dev_mode_ == DevMode::GPU_MODE) {
        return WeightDevice::GPU;
    }
    if (gpu_layer_set != nullptr &&
        gpu_layer_set->find(layer_idx) != gpu_layer_set->end()) {
        return WeightDevice::GPU;
    }
    return WeightDevice::CPU;
}

bool Qwen35moeModel::layer_has_split_devices(int layer_idx) const {
    bool saw_cpu = false;
    bool saw_gpu = false;
    auto note = [&](WeightDevice device) {
        if (device == WeightDevice::CPU) {
            saw_cpu = true;
        } else if (device == WeightDevice::GPU) {
            saw_gpu = true;
        }
    };

    if (gpu_weights_ && gpu_weights_->layers.count(layer_idx)) {
        for (const auto& kv : gpu_weights_->layers[layer_idx]->tensors) {
            if (kv.second != nullptr) {
                note(WeightDevice::GPU);
            }
        }
    }
    if (cpu_weights_ && cpu_weights_->layers.count(layer_idx)) {
        for (const auto& kv : cpu_weights_->layers[layer_idx]->tensors) {
            if (kv.second != nullptr) {
                note(WeightDevice::CPU);
            }
        }
    }
    return saw_cpu && saw_gpu;
}

bool Qwen35moeModel::has_any_split_device_layers() const {
    if (!meta_) {
        return false;
    }
    const int n_layers = static_cast<int>(trunk_layer_count());
    for (int il = 0; il < n_layers; ++il) {
        if (layer_has_split_devices(il)) {
            return true;
        }
    }
    return false;
}

void Qwen35moeModel::maybe_log_weight_override(
    const std::string& tensor_name,
    WeightDevice planned,
    WeightDevice resolved
) const {
    if (planned == resolved) {
        return;
    }
    std::fprintf(stderr,
        "[Loader] override-tensor: %s buffer -> %s (default %s)\n",
        tensor_name.c_str(),
        resolved == WeightDevice::CPU ? "CPU" : "GPU",
        planned == WeightDevice::CPU ? "CPU" : "GPU");
}

void Qwen35moeModel::rebuild_layer_device_map() {
    const size_t n_layers = meta_ ? static_cast<size_t>(trunk_layer_count()) : 0;
    layer_device_map_.assign(n_layers, backend_cpu_);
    if (dev_mode_ == DevMode::CPU_MODE) {
        std::fill(layer_device_map_.begin(), layer_device_map_.end(), backend_cpu_);
        return;
    }

    for (size_t il = 0; il < n_layers; ++il) {
        const int layer_idx = static_cast<int>(il);
        bool on_gpu = false;
        if (gpu_weights_ && gpu_weights_->layers.count(layer_idx)) {
            for (const auto& kv : gpu_weights_->layers[layer_idx]->tensors) {
                if (kv.second != nullptr) {
                    on_gpu = true;
                    break;
                }
            }
        }
        layer_device_map_[il] = on_gpu ? backend_gpu_ : backend_cpu_;
    }
}

ggml_tensor* Qwen35moeModel::get_weight_tensor(const EN_WEIGHT_TYPE weight_type) {
    if (dev_mode_ == DevMode::CPU_MODE) {
        return cpu_weights_->heads[weight_type];
    }
    if (gpu_weights_ && gpu_weights_->heads.count(weight_type)) {
        return gpu_weights_->heads[weight_type];
    }
    if (cpu_weights_ && cpu_weights_->heads.count(weight_type)) {
        return cpu_weights_->heads[weight_type];
    }
    return nullptr;
}
ggml_tensor* Qwen35moeModel::get_weight_layer_tensor(const EN_WEIGHT_TYPE layer_type, const int layer_idx) {
    if (dev_mode_ == DevMode::CPU_MODE) {
        return cpu_weights_->layers[layer_idx]->tensors[layer_type];
    }
    // override-tensor: split layers may have expert weights on CPU while the layer
    // is otherwise GPU-resident — prefer the CPU copy when present.
    if (layer_has_split_devices(layer_idx) &&
        cpu_weights_ &&
        cpu_weights_->layers.count(layer_idx) &&
        cpu_weights_->layers[layer_idx]->tensors.count(layer_type)) {
        return cpu_weights_->layers[layer_idx]->tensors[layer_type];
    }
    if (gpu_weights_ &&
        gpu_weights_->layers.count(layer_idx) &&
        gpu_weights_->layers[layer_idx]->tensors.count(layer_type)) {
        return gpu_weights_->layers[layer_idx]->tensors[layer_type];
    }
    if (cpu_weights_ &&
        cpu_weights_->layers.count(layer_idx) &&
        cpu_weights_->layers[layer_idx]->tensors.count(layer_type)) {
        return cpu_weights_->layers[layer_idx]->tensors[layer_type];
    }
    return nullptr;
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

struct ggml_tensor* Qwen35moeModel::get_nextn_eh_proj_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_NEXTN_EH_PROJ, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_nextn_enorm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_NEXTN_ENORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_nextn_hnorm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_NEXTN_HNORM, layer_idx);
}

struct ggml_tensor* Qwen35moeModel::get_nextn_shared_head_norm_weight(const int layer_idx)
{
    return get_weight_layer_tensor(EN_WEIGHT_TYPE_NEXTN_SHARED_HEAD_NORM, layer_idx);
}

bool Qwen35moeModel::has_mtp() const
{
    return meta_ && meta_->qwen35moe.nextn_predict_layers > 0;
}

uint32_t Qwen35moeModel::trunk_layer_count() const
{
    if (!meta_) {
        return 0;
    }
    const uint32_t all = meta_->qwen35moe.block_count;
    const uint32_t mtp_layers = meta_->qwen35moe.nextn_predict_layers;
    if (mtp_layers == 0 || mtp_layers >= all) {
        return all;
    }
    return all - mtp_layers;
}

uint32_t Qwen35moeModel::mtp_layer_index() const
{
    return trunk_layer_count();
}

void Qwen35moeModel::apply_cpu_thread_pool(bool for_batch) const {
    if (!backend_cpu_) {
        return;
    }
    const int n = for_batch ? n_threads_batch() : n_threads_;
    ggml_backend_cpu_set_n_threads(backend_cpu_, std::max(1, n));
}

void Qwen35moeModel::parallel_load_cpu_tensors(
    const std::vector<std::pair<ggml_tensor*, bool>>& jobs
) {
    printf("%lu [Loader] Starting parallel CPU tensor loading: %lu tensors, %d threads\n", time(NULL), jobs.size(), n_threads_);
    const size_t n_jobs = jobs.size();
    if (n_jobs == 0) {
        return;
    }

    // Pick worker count: cap by tensor count and a reasonable upper bound, since
    // memory copies are bandwidth-bound and more threads quickly saturate.
    int desired = std::max(1, n_threads_);
    desired = std::min(desired, 16);
    int n_workers = static_cast<int>(std::min<size_t>(static_cast<size_t>(desired), n_jobs));
    if (n_workers < 1) n_workers = 1;

    std::atomic<size_t> next_idx{0};
    std::atomic<size_t> failed{0};
    auto worker = [&]() {
        for (;;) {
            const size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (idx >= n_jobs) break;
            const auto& job = jobs[idx];
            const bool ok = job.second
                ? loader_->load_tensor_head_data(job.first)
                : loader_->load_tensor_layer_data(job.first);
            if (!ok) {
                failed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_workers - 1));
    for (int i = 0; i < n_workers - 1; ++i) {
        workers.emplace_back(worker);
    }
    worker();
    for (auto& t : workers) {
        t.join();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const size_t bad = failed.load(std::memory_order_relaxed);
    if (bad != 0) {
        fprintf(stderr,
            "[Loader] WARNING: %zu/%zu CPU tensors failed to load (parallel copy)\n",
            bad, n_jobs);
    }
    printf("[Loader] CPU weight copy: %zu tensors via %d threads in %.0f ms\n",
        n_jobs, n_workers, dt_ms);
}

bool Qwen35moeModel::init_cpu() {
    printf("===================Loading Qwen35moe Model to CPU=====================\n");
    backend_cpu_ = ggml_backend_cpu_init();
    if (!backend_cpu_) {
        fprintf(stderr, "[Loader] ERROR: failed to init CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend_cpu_, n_threads_);
    if (n_threads_batch() != n_threads_) {
        printf("[Loader] CPU thread pools: decode=%d batch=%d\n", n_threads_, n_threads_batch());
    }

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
        const WeightDevice planned = default_head_weight_device();
        const WeightDevice resolved = resolve_weight_device(tensor_info.name, planned);
        maybe_log_weight_override(tensor_info.name, planned, resolved);
        if (resolved != WeightDevice::CPU) {
            continue;
        }

        struct ggml_tensor* cur = ggml_new_tensor(cpu_ctx_, tensor_info.type, tensor_info.n_dims, tensor_info.dims);
        if (NULL != cur) {
            ggml_set_name(cur, tensor_info.name.c_str());
            cpu_weights_->heads[tensor_info.weight_type] = cur;
        }
    }

    for (auto& tensor_info : loader_->tensors_layer_) {
        const WeightDevice planned = WeightDevice::CPU;
        const WeightDevice resolved = resolve_weight_device(tensor_info.name, planned);
        maybe_log_weight_override(tensor_info.name, planned, resolved);
        if (resolved != WeightDevice::CPU) {
            continue;
        }

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
                cpu_mmap_zerocopy_ = true;
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

        if (no_mmap_) {
            printf("[Loader] NOTE: --no-mmap is set; CPU weights will be copied "
                   "(consumes ~2x file size at peak; for fastest startup, drop --no-mmap)\n");
        }

        auto now = time(NULL);
        std::vector<std::pair<ggml_tensor*, bool>> jobs;
        jobs.reserve(cpu_weights_->heads.size());
        for (auto& head_iter : cpu_weights_->heads) {
            jobs.emplace_back(head_iter.second, /*is_head=*/true);
        }
        for (auto& layer_iter : cpu_weights_->layers) {
            for (auto& layer_info : layer_iter.second->tensors) {
                jobs.emplace_back(layer_info.second, /*is_head=*/false);
            }
        }
        parallel_load_cpu_tensors(jobs);
        auto end = time(NULL);
        printf("[Loader] CPU weight loading complete in %ld seconds\n", end - now);
    }

    return true;
}

bool Qwen35moeModel::init_gpu() {
    if (dev_mode_ == DevMode::CPU_MODE) {
        return true;
    }
#ifndef QWEN35MOE_USE_CUDA
    fprintf(stderr, "[Loader] ERROR: CUDA backend was not compiled. Rebuild with -DQWEN35MOE_CUDA=ON for GPU/AUTO mode.\n");
    return false;
#else
    printf("===================Loading Qwen35moe Model to GPU=====================\n");
    backend_gpu_ = ggml_backend_cuda_init(gpu_id_);//ggml_backend_init_best();
    if (!backend_gpu_) {
        fprintf(stderr, "[Loader] ERROR: failed to init GPU backend\n");
        return false;
    }

    int device_count = ggml_backend_cuda_get_device_count();
    if (device_count == 0) {
        printf("No CUDA devices found\n");
        return false;
    }
    
    if (gpu_id_ >= device_count) {
        printf("Invalid device %d (only %d available)\n", gpu_id_, device_count);
        return false;
    }

    size_t free_mem, total_mem;
    ggml_backend_cuda_get_device_memory(gpu_id_, &free_mem, &total_mem);
    printf("Device %d: %.2f GB free / %.2f GB total\n", gpu_id_, free_mem / 1e9, total_mem / 1e9);

    if (dev_mode_ == DevMode::AUTO_MODE) {
        return init_auto_gpu(free_mem);
    }

    const size_t ctx_size = [&]() {
        size_t bytes = 0;
        for (auto& tensor_iter : loader_->tensors_head_) {
            const WeightDevice planned = default_head_weight_device();
            if (resolve_weight_device(tensor_iter.first, planned) == WeightDevice::GPU) {
                bytes += loader_->get_tensor_bytesize(tensor_iter.second);
            }
        }
        for (auto& tensor_info : loader_->tensors_layer_) {
            if (resolve_weight_device(tensor_info.name, WeightDevice::GPU) == WeightDevice::GPU) {
                bytes += loader_->get_tensor_bytesize(tensor_info);
            }
        }
        return bytes;
    }();
    ggml_init_params gpu_p = { ctx_size, nullptr, true };
    gpu_ctx_ = ggml_init(gpu_p);
    if (!gpu_ctx_) {
        fprintf(stderr, "[Loader] ERROR: failed to init weight contexts\n");
        return false;
    }

    gpu_weights_ = std::make_shared<Qwen35moeWeights>();
    for (auto& tensor_iter : loader_->tensors_head_) {
        auto* tensor_info = &tensor_iter.second;
        const WeightDevice planned = default_head_weight_device();
        const WeightDevice resolved = resolve_weight_device(tensor_info->name, planned);
        maybe_log_weight_override(tensor_info->name, planned, resolved);
        if (resolved != WeightDevice::GPU) {
            continue;
        }

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
        const WeightDevice planned = WeightDevice::GPU;
        const WeightDevice resolved = resolve_weight_device(tensor_info.name, planned);
        maybe_log_weight_override(tensor_info.name, planned, resolved);
        if (resolved != WeightDevice::GPU) {
            continue;
        }

        struct ggml_tensor* cur = ggml_new_tensor(gpu_ctx_, tensor_info.type, tensor_info.n_dims, tensor_info.dims);
        if (NULL != cur) {
            ggml_set_name(cur, tensor_info.name.c_str());
            if (gpu_weights_->layers.find(tensor_info.layer_idx) == gpu_weights_->layers.end()) {
                gpu_weights_->layers[tensor_info.layer_idx] = std::make_shared<Qwen35moeLayer>();
            }
            gpu_weights_->layers[tensor_info.layer_idx]->tensors[tensor_info.weight_type] = cur;
        }
    }

    if (gpu_weights_->heads.empty() && gpu_weights_->layers.empty()) {
        fprintf(stderr, "[Loader] ERROR: no GPU tensors selected after override-tensor planning\n");
        return false;
    }

    gpu_buf_ = ggml_backend_alloc_ctx_tensors(gpu_ctx_, backend_gpu_);
    if (!gpu_buf_) {
        fprintf(stderr, "[Loader] ERROR: failed to allocate GPU weight buffer\n");
        return false;
    }

    for (auto& head_iter : gpu_weights_->heads) {
        if (head_iter.first == EN_WEIGHT_TYPE_TOKEN_EMBD) {
            dequant_set_to_backend(head_iter.second, backend_gpu_);
        } else {
            loader_->load_tensor_head_data(head_iter.second, backend_gpu_);
        }
    }
    auto layer_iter = gpu_weights_->layers.begin();
    for (; layer_iter != gpu_weights_->layers.end(); ++layer_iter) {
        for (auto& layer_info : layer_iter->second->tensors) {
            loader_->load_tensor_layer_data(layer_info.second, backend_gpu_);
        }
        printf("Loaded layer %d to GPU\n", layer_iter->first);
    }
    ggml_backend_synchronize(backend_gpu_);
    printf("[Loader] GPU weight loading complete\n");

    if (!init_cpu_weights_for_plan(nullptr)) {
        return false;
    }
    
    return true;
#endif
}

bool Qwen35moeModel::init_cpu_weights_for_plan(const std::set<int, std::less<int>>* gpu_layer_set) {
    struct PendingCpuTensor {
        bool is_head = false;
        const struct tensor_info* info = nullptr;
    };
    std::vector<PendingCpuTensor> pending;
    size_t use_byte = 0;

    for (auto& tensor_iter : loader_->tensors_head_) {
        const WeightDevice planned = default_head_weight_device();
        const WeightDevice resolved = resolve_weight_device(tensor_iter.first, planned);
        if (resolved != WeightDevice::CPU) {
            continue;
        }
        maybe_log_weight_override(tensor_iter.first, planned, resolved);
        use_byte += loader_->get_tensor_bytesize(tensor_iter.second);
        pending.push_back({true, &tensor_iter.second});
    }

    for (auto& tensor_info : loader_->tensors_layer_) {
        const WeightDevice planned = default_layer_weight_device(tensor_info.layer_idx, gpu_layer_set);
        const WeightDevice resolved = resolve_weight_device(tensor_info.name, planned);
        if (resolved != WeightDevice::CPU) {
            continue;
        }
        maybe_log_weight_override(tensor_info.name, planned, resolved);
        use_byte += loader_->get_tensor_bytesize(tensor_info);
        pending.push_back({false, &tensor_info});
    }

    if (pending.empty()) {
        return true;
    }

    if (!cpu_weights_) {
        cpu_weights_ = std::make_shared<Qwen35moeWeights>();
    }

    ggml_init_params cpu_p = { use_byte, nullptr, true };
    cpu_ctx_ = ggml_init(cpu_p);
    if (!cpu_ctx_) {
        fprintf(stderr, "[Loader] ERROR: failed to init CPU override weight context\n");
        return false;
    }

    for (const PendingCpuTensor& item : pending) {
        const struct tensor_info* info = item.info;
        struct ggml_tensor* cur = ggml_new_tensor(cpu_ctx_, info->type, info->n_dims, info->dims);
        if (cur == nullptr) {
            fprintf(stderr, "[Loader] ERROR: failed to create CPU tensor %s\n", info->name.c_str());
            return false;
        }
        ggml_set_name(cur, info->name.c_str());
        if (item.is_head) {
            cpu_weights_->heads[info->weight_type] = cur;
        } else {
            if (cpu_weights_->layers.find(info->layer_idx) == cpu_weights_->layers.end()) {
                cpu_weights_->layers[info->layer_idx] = std::make_shared<Qwen35moeLayer>();
            }
            cpu_weights_->layers[info->layer_idx]->tensors[info->weight_type] = cur;
        }
    }

    bool zerocopy_ok = false;
    const bool allow_mmap =
        !no_mmap_ &&
        loader_->is_mmap_active() &&
        (dev_mode_ == DevMode::AUTO_MODE ? gpu_layer_set != nullptr : dev_mode_ == DevMode::GPU_MODE);
    if (allow_mmap) {
        cpu_buf_ = loader_->create_cpu_mmap_buffer();
        if (cpu_buf_) {
            bool all_ok = true;
            for (auto& head_iter : cpu_weights_->heads) {
                if (!loader_->bind_tensor_to_mmap(cpu_buf_, head_iter.second, true)) {
                    all_ok = false;
                    break;
                }
            }
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
                cpu_mmap_zerocopy_ = true;
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
            fprintf(stderr, "[Loader] ERROR: failed to allocate CPU override weight buffer\n");
            return false;
        }

        if (no_mmap_) {
            printf("[Loader] NOTE: --no-mmap is set; CPU override tensors will be copied\n");
        }

        std::vector<std::pair<ggml_tensor*, bool>> jobs;
        for (auto& head_iter : cpu_weights_->heads) {
            jobs.emplace_back(head_iter.second, true);
        }
        for (auto& layer_iter : cpu_weights_->layers) {
            for (auto& layer_info : layer_iter.second->tensors) {
                jobs.emplace_back(layer_info.second, false);
            }
        }
        parallel_load_cpu_tensors(jobs);
        printf("[Loader] CPU override weight loading complete (%zu tensors)\n", jobs.size());
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
    size_t curr_tensor_byte = 0;
    // 输出头：override 到 CPU 的不占 GPU 显存预算
    for (auto& tensor_iter : loader_->tensors_head_) {
        auto* tensor_info = &tensor_iter.second;
        const WeightDevice planned = default_head_weight_device();
        const WeightDevice resolved = resolve_weight_device(tensor_info->name, planned);
        maybe_log_weight_override(tensor_info->name, planned, resolved);
        if (resolved != WeightDevice::GPU) {
            continue;
        }
        curr_tensor_byte = loader_->get_tensor_bytesize(*tensor_info);
        if (use_byte + curr_tensor_byte > free_mem) {
            fprintf(stderr, "[Loader] ERROR: failed to allocate tensor %s due to insufficient GPU memory\n", tensor_iter.first.c_str());
            return false;
        }
        use_byte += curr_tensor_byte;
    }

    // 末尾连续层的其余 DeltaNet + 间隔全注意力，优先全放 GPU。
    //
    // --gpu-layer / --n-gpu-layers 语义（与 llama.cpp --n-gpu-layers 对齐）：
    //   - 0：不限制层数，仅按显存预算从尾部 fill；启用 block_align 时按 attention block 对齐
    //   - N>0：从尾部精确 offload N 个 transformer 层（受显存约束时可少于 N）
    //
    // --override-tensor 在层计划之后继续生效：仅计入仍落在 GPU 的 tensor 字节。
    auto layer_byte_size = [&](int layer_index) -> size_t {
        size_t layer_use_byte = 0;
        auto it = loader_->tensor_layer_index_list_.find(layer_index);
        if (it != loader_->tensor_layer_index_list_.end()) {
            for (auto& index : it->second) {
                auto& ti = loader_->tensors_layer_[index];
                if (resolve_weight_device(ti.name, WeightDevice::GPU) == WeightDevice::GPU) {
                    layer_use_byte += loader_->get_tensor_bytesize(ti);
                }
            }
        }
        return layer_use_byte;
    };

    std::set<int, std::less<int>> gpu_layer_set;
    const int total_layers = static_cast<int>(loader_->tensor_layer_index_list_.size());
    const uint32_t interval = meta_->qwen35moe.full_attention_interval;
    const bool layer_limit_enabled = gpu_layer_ > 0;
    const size_t max_gpu_layers = layer_limit_enabled
        ? std::min(gpu_layer_, static_cast<size_t>(total_layers))
        : static_cast<size_t>(total_layers);
    size_t gpu_block_layers = 0;
    const bool block_align = !layer_limit_enabled &&
        (interval > 0) && (total_layers > 0) && !auto_boundary_align_disabled();

    auto can_add_layers = [&](size_t layer_count, size_t layer_bytes) -> bool {
        if (layer_limit_enabled && gpu_block_layers + layer_count > max_gpu_layers) {
            return false;
        }
        return use_byte + layer_bytes <= free_mem;
    };

    auto add_layer = [&](int layer_index) {
        use_byte += layer_byte_size(layer_index);
        gpu_block_layers++;
        gpu_layer_set.insert(layer_index);
    };

    if (layer_limit_enabled) {
        // llama.cpp 风格：从尾部逐层 fill，精确受 --gpu-layer 限制
        for (auto iter = loader_->tensor_layer_index_list_.rbegin();
             iter != loader_->tensor_layer_index_list_.rend();
             ++iter) {
            if (gpu_block_layers >= max_gpu_layers) {
                break;
            }
            const size_t layer_bytes = layer_byte_size(iter->first);
            if (!can_add_layers(1, layer_bytes)) {
                break;
            }
            add_layer(iter->first);
        }
    } else if (block_align) {
        // 以 attention block 为单位从尾部 fill。模型 block 自然边界为
        // {0, K, 2K, ..., floor(N/K)*K, N}（K=interval, N=total_layers）。
        const int K = static_cast<int>(interval);
        int boundary = total_layers;
        auto prev_block_start = [&](int b) {
            if (b <= 0) return -1;
            int aligned = ((b - 1) / K) * K;
            return aligned;
        };
        for (int block_l0 = prev_block_start(boundary);
             block_l0 >= 0;
             block_l0 = prev_block_start(boundary)) {
            const int block_l1 = boundary;
            const size_t block_layer_count = static_cast<size_t>(block_l1 - block_l0);
            size_t block_use_byte = 0;
            for (int il = block_l0; il < block_l1; ++il) {
                block_use_byte += layer_byte_size(il);
            }
            if (!can_add_layers(block_layer_count, block_use_byte)) {
                break;
            }
            use_byte += block_use_byte;
            gpu_block_layers += block_layer_count;
            for (int il = block_l0; il < block_l1; ++il) {
                gpu_layer_set.insert(il);
            }
            boundary = block_l0;
        }
        if (gpu_layer_set.empty()) {
            // 当一整 block 都装不下时，按层 fill 兜底：尽量利用剩余显存
            for (auto iter = loader_->tensor_layer_index_list_.rbegin();
                 iter != loader_->tensor_layer_index_list_.rend();
                 ++iter) {
                const size_t layer_bytes = layer_byte_size(iter->first);
                if (!can_add_layers(1, layer_bytes)) {
                    break;
                }
                add_layer(iter->first);
            }
        }
    } else {
        // block_align 不可用：按层从尾部 fill，仅受显存约束
        for (auto iter = loader_->tensor_layer_index_list_.rbegin();
             iter != loader_->tensor_layer_index_list_.rend();
             ++iter) {
            const size_t layer_bytes = layer_byte_size(iter->first);
            if (!can_add_layers(1, layer_bytes)) {
                break;
            }
            add_layer(iter->first);
        }
    }

    // 诊断日志：展示实际 GPU/CPU 划分，便于用户确认 boundary 是否对齐
    if (!gpu_layer_set.empty()) {
        const int gpu_first = *gpu_layer_set.begin();
        const int gpu_last = *gpu_layer_set.rbegin();
        const int gpu_count = static_cast<int>(gpu_layer_set.size());
        const int cpu_count = total_layers - gpu_count;
        const bool aligned =
            (interval > 0) && (gpu_first == 0 || (gpu_first % static_cast<int>(interval)) == 0);
        if (layer_limit_enabled && static_cast<size_t>(gpu_count) < max_gpu_layers) {
            fprintf(stderr,
                "[Loader] WARNING: --gpu-layer %zu requested but only %d layer(s) fit in GPU memory\n",
                max_gpu_layers,
                gpu_count);
        }
        const size_t gpu_layer_cap_log = layer_limit_enabled ? max_gpu_layers : 0;
        printf(
            "[Loader] auto layout: GPU=[%d,%d] (%d layers, %.2f GB), CPU=[0,%d) (%d layers), "
            "block_align=%s gpu_layer_cap=%zu%s%s\n",
            gpu_first,
            gpu_last,
            gpu_count,
            use_byte / 1e9,
            gpu_first,
            cpu_count,
            block_align ? "on" : "off",
            gpu_layer_cap_log,
            layer_limit_enabled ? "" : " (VRAM-only)",
            (block_align && aligned) ? "" : (block_align ? " (NOT aligned, check budget)" : "")
        );
    } else {
        printf(
            "[Loader] auto layout: all %d layers on CPU (no transformer layer fit GPU budget%s)\n",
            total_layers,
            layer_limit_enabled ? "; check --gpu-layer or VRAM" : ""
        );
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
        const WeightDevice planned = default_head_weight_device();
        if (resolve_weight_device(tensor_info->name, planned) != WeightDevice::GPU) {
            continue;
        }

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

    for (auto& layer_index : gpu_layer_set) {
        auto layer_list = loader_->tensor_layer_index_list_.find(layer_index);
        for (auto& index : layer_list->second) {
            auto* iter = &loader_->tensors_layer_[index];
            if (resolve_weight_device(iter->name, WeightDevice::GPU) != WeightDevice::GPU) {
                continue;
            }
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

    if (gpu_weights_->heads.empty() && gpu_weights_->layers.empty()) {
        fprintf(stderr, "[Loader] ERROR: auto GPU layout has no GPU tensors after override-tensor planning\n");
        return false;
    }

    gpu_buf_ = ggml_backend_alloc_ctx_tensors(gpu_ctx_, backend_gpu_);
    if (!gpu_buf_) {
        fprintf(stderr, "[Loader] ERROR: failed to allocate auto GPU weight buffer\n");
        return false;
    }

    for (auto& head_iter : gpu_weights_->heads) {
        if (head_iter.first == EN_WEIGHT_TYPE_TOKEN_EMBD) {
            dequant_set_to_backend(head_iter.second, backend_gpu_);
        } else {
            loader_->load_tensor_head_data(head_iter.second, backend_gpu_);
        }
        printf("Loaded auto head %s to GPU\n", head_iter.second->name);
    }
    auto layer_iter = gpu_weights_->layers.begin();
    for (; layer_iter != gpu_weights_->layers.end(); ++layer_iter) {
        for (auto& layer_info : layer_iter->second->tensors) {
            loader_->load_tensor_layer_data(layer_info.second, backend_gpu_);
        }
        printf("Loaded auto layer blk.%d.* to GPU\n", layer_iter->first);
    }
    ggml_backend_synchronize(backend_gpu_);
    printf("[Loader] auto GPU weight loading complete\n");

    if (uses_tensor_overrides()) {
        for (int il = 0; il < total_layers; ++il) {
            if (layer_has_split_devices(il)) {
                std::fprintf(stderr,
                    "[Loader] override-tensor: layer blk.%d has split CPU/GPU weights (scheduler path required)\n",
                    il);
            }
        }
    }

    return init_cpu_weights_for_plan(&gpu_layer_set);
}

bool Qwen35moeModel::load_metadata() {
    return meta_->load_from_gguf(loader_.get());
}

int Qwen35moeModel::get_ctx_size() {
    return meta_->qwen35moe.context_length;
}

void Qwen35moeModel::dequant_set(ggml_tensor* dst) {
    dequant_set_to_backend(dst, nullptr);
}

void Qwen35moeModel::dequant_set_to_backend(ggml_tensor* dst, ggml_backend_t upload_backend) {
    if (!dst) {
        return;
    }

    auto* tensor = &loader_->tensors_head_[dst->name];
    const uint8_t* src_base = loader_->get_tensor_file_ptr(*tensor);
    if (!src_base) {
        throw std::runtime_error("Qwen35moeModel::dequant_set_to_backend: model file mapping is not available");
    }

    const int64_t ncols = tensor->dims[0];
    int64_t nrows = tensor->dims[1];
    for (int64_t i = 2; i < tensor->n_dims; ++i) {
        nrows *= tensor->dims[i];
    }
    const auto* tr = ggml_get_type_traits(tensor->type);
    if (!tr || !tr->to_float) {
        throw std::runtime_error("Qwen35moeModel::dequant_set_to_backend: unsupported source type");
    }

    const int64_t n_blocks_per_row = ncols / tensor->blck_size;
    const size_t row_src_stride = static_cast<size_t>(n_blocks_per_row) * tensor->type_size;

    std::vector<ggml_fp16_t> fp16_all(static_cast<size_t>(nrows * ncols));
    std::vector<float> row32(static_cast<size_t>(ncols));

    for (int64_t r = 0; r < nrows; ++r) {
        const uint8_t* rp = src_base + static_cast<size_t>(r) * row_src_stride;
        tr->to_float(rp, row32.data(), ncols);
        ggml_fp32_to_fp16_row(row32.data(), fp16_all.data() + static_cast<size_t>(r) * ncols, ncols);
    }

    const size_t dst_bytes = ggml_nbytes(dst);
    if (upload_backend != nullptr) {
        ggml_backend_tensor_set_async(upload_backend, dst, fp16_all.data(), 0, dst_bytes);
    } else {
        ggml_backend_tensor_set(dst, fp16_all.data(), 0, dst_bytes);
    }
}

void Qwen35moeModel::maybe_release_model_file() {
    if (cpu_mmap_zerocopy_) {
        printf("[Loader] Keeping GGUF mmap for CPU zero-copy weights\n");
        return;
    }

    loader_->unload();
    if (no_mmap_) {
        printf("[Loader] Released GGUF mmap (--no-mmap)\n");
    } else {
        printf("[Loader] Released GGUF mmap after weight upload (GPU-only / copied CPU weights)\n");
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
        printf("Tensor: %s, size: %lu, type: %d, op: %d, data_offs: %lu, file_offs: %lu, data: %p, view_src: %p\n", name, n_size, type, op, data_offs, file_offs, data, cur->view_src);
    }
}
