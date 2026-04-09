// pipeline/recognizer.hpp
// Recognizer — 面向用户的一站式语音识别 API
//
#ifndef FUNASR_PIPELINE_RECOGNIZER_HPP
#define FUNASR_PIPELINE_RECOGNIZER_HPP

#include "core/gguf_reader.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include <memory>
#include <string>

class Recognizer {
public:
    Recognizer() = default;
    ~Recognizer() = default;

    // 不可拷贝（持有 GGUFReader）
    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // ============================================================
    // 初始化: 加载模型 + 词表 + 创建 Pipeline
    // ============================================================
    bool init(const std::string& model_path, bool verbose = true) {
        if (verbose) printf("[Recognizer] Loading model: %s\n", model_path.c_str());

        // 1. 打开 GGUF
        reader_ = std::make_unique<GGUFReader>();
        if (!reader_->open(model_path)) {
            last_error_ = "Failed to open model file";
            return false;
        }

        // 2. 加载模型权重
        model_ = std::make_unique<Qwen35moeModel>();
        if (!ModelLoader::load(*reader_, *model_)) {
            last_error_ = "Failed to load model weights";
            return false;
        }

        ready_ = true;
        if (verbose) {
            printf("[Recognizer] Ready!\n");
        }
        return true;
    }

    // 查询
    bool is_ready()                   const { return ready_; }
    const std::string& last_error()   const { return last_error_; }
    const ModelConfig& config()       const { return model_->config; }

private:
    std::unique_ptr<GGUFReader>   reader_;
    std::unique_ptr<Qwen35moeModel>  model_;

    bool ready_ = false;
    std::string last_error_;
};

#endif // FUNASR_PIPELINE_RECOGNIZER_HPP