// pipeline/recognizer.hpp
// Recognizer — loads a GGUF model and exposes it for inference
//
#ifndef FUNASR_PIPELINE_RECOGNIZER_HPP
#define FUNASR_PIPELINE_RECOGNIZER_HPP

#include "core/gguf_reader.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "pipeline/inference.hpp"
#include <memory>
#include <string>

class Recognizer {
public:
    Recognizer() = default;
    ~Recognizer() = default;

    // Not copyable (owns GGUFReader)
    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // Load model from GGUF file
    /**
     * @brief 初始化识别器并加载模型
     * 
     * 该方法负责初始化识别器，加载指定路径的模型文件，并根据指定的GPU模式
     * 选择合适的加载方式。
     * 
     * @param model_path 模型文件路径
     * @param verbose 是否输出详细信息（默认为true）
     * @param gpu_mode GPU模式（默认为Off）
     * @return bool 是否成功初始化
     */
    bool init(const std::string& model_path, bool verbose = true, GpuMode gpu_mode = GpuMode::Off) {
        // 打印加载模型信息
        if (verbose) printf("[Recognizer] Loading model: %s\n", model_path.c_str());
        
        // 根据GPU模式打印相应信息
        if (verbose && gpu_mode == GpuMode::Hybrid) {
            printf("[Recognizer] GPU mode: metadata-only load for hybrid CPU-GPU direct backend loading\n");
        } else if (verbose && gpu_mode == GpuMode::Full) {
            printf("[Recognizer] GPU mode: metadata-only load for full GPU direct backend loading\n");
        }

        // 创建GGUFReader实例
        reader_ = std::make_unique<GGUFReader>();
        
        // 根据GPU模式选择打开方式
        // Off模式：直接打开并分配内存
        // Hybrid/Full模式：仅加载元数据，不分配内存
        const bool ok = (gpu_mode == GpuMode::Off) ? reader_->open(model_path) : reader_->open_no_alloc(model_path);
        
        // 处理打开失败的情况
        if (!ok) {
            last_error_ = "Failed to open model file";
            return false;
        }

        // 创建Qwen35moeModel实例
        model_ = std::make_unique<Qwen35moeModel>();
        
        // 加载模型权重
        if (!ModelLoader::load(*reader_, *model_)) {
            last_error_ = "Failed to load model weights";
            return false;
        }

        // 设置就绪状态并打印信息
        ready_ = true;
        if (verbose) printf("[Recognizer] Ready!\n");
        return true;
    }

    bool is_ready() const { return ready_; }
    const std::string& last_error() const { return last_error_; }
    const ModelConfig& config() const { return model_->config; }
    // Expose the full model for inference engine
    Qwen35moeModel* model() { return model_.get(); }
    const Qwen35moeModel* model() const { return model_.get(); }
    GGUFReader* reader() { return reader_.get(); }
    const GGUFReader* reader() const { return reader_.get(); }

private:
    std::unique_ptr<GGUFReader> reader_ = nullptr;
    std::unique_ptr<Qwen35moeModel> model_ = nullptr;
    bool ready_ = false;
    std::string last_error_ = "";
};

#endif // FUNASR_PIPELINE_RECOGNIZER_HPP
