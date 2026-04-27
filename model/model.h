#pragma once

#include <memory>
#include <string>
#include "model/metadata.h"
#include "model/weights.h"
#include "core/gguf_reader.h"
#include "model/gguf_mmap.h"

enum class DevMode {
    CPU_MODE,
    GPU_MODE,
    AURO_MODE,
};

class Qwen35moeModel {
public:
    Qwen35moeModel();
    ~Qwen35moeModel();

    bool init(const std::string& model_path_, DevMode dev_mode = DevMode::CPU_MODE, int n_threads = 1, int gpu_layer = 0);
    ggml_backend_t get_curr_backend();
    ggml_tensor* get_weight_tensor(const EN_WEIGHT_TYPE weight_type);
    ggml_tensor* get_weight_layer_tensor(const EN_LAYER_TYPE layer_type, const int layer_idx);

private:
    bool load_metadata();
    bool set_tensor_data(const size_t data_offset, ggml_tensor* tensor);
    bool load_qwen35moe(const int layer_count);
    void load_qwen35moe_layer(std::shared_ptr<Qwen35moeLayer>& layer, int layer_idx);
    bool set_tensors_data(const int layer_count);
    bool set_tensors_layer_data(ggml_context* ctx, std::shared_ptr<Qwen35moeLayer>& layer, const size_t data_offset, int layer_idx);
    int get_ctx_size();
    void print_context_info(ggml_context* ctx_);

public:
    DevMode dev_mode_ = DevMode::CPU_MODE;
    std::shared_ptr<MetaDataInfo> meta_ = nullptr;
    std::shared_ptr<Qwen35moeWeights> gpu_weights_ = nullptr;
    std::shared_ptr<Qwen35moeWeights> cpu_weights_ = nullptr;
    std::shared_ptr<GGUFReader> reader_ = nullptr;

    std::shared_ptr<GGUFMmapTensorLoader> tensor_loader_ = nullptr;

    int n_threads_ = 1;
    int gpu_layer_ = 0;
    ggml_backend_t backend_gpu_ = nullptr;
    ggml_backend_t backend_cpu_ = nullptr;

    ggml_backend_buffer_t gpu_buf_ = nullptr;
    ggml_backend_buffer_t cpu_buf_ = nullptr;
    ggml_context* gpu_ctx_ = nullptr;
    ggml_context* cpu_ctx_ = nullptr;
};
