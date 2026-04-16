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
    bool init(const std::string& model_path, bool verbose = true, GpuMode gpu_mode = GpuMode::Off) {
        if (verbose) printf("[Recognizer] Loading model: %s\n", model_path.c_str());
        if (verbose && gpu_mode == GpuMode::Hybrid) {
            printf("[Recognizer] GPU mode: loading CPU weights for hybrid CPU-GPU offload\n");
        } else if (verbose && gpu_mode == GpuMode::Full) {
            printf("[Recognizer] GPU mode: loading CPU weights for full GPU offload\n");
        }

        reader_ = std::make_unique<GGUFReader>();
        const bool ok = reader_->open(model_path);
        if (!ok) {
            last_error_ = "Failed to open model file";
            return false;
        }

        model_ = std::make_unique<Qwen35moeModel>();
        if (!ModelLoader::load(*reader_, *model_)) {
            last_error_ = "Failed to load model weights";
            return false;
        }

        ready_ = true;
        if (verbose) printf("[Recognizer] Ready!\n");
        return true;
    }

    bool                       is_ready()   const { return ready_; }
    const std::string&         last_error() const { return last_error_; }
    const ModelConfig&         config()     const { return model_->config; }
    // Expose the full model for inference engine
    Qwen35moeModel*            model()            { return model_.get(); }
    const Qwen35moeModel*      model()      const { return model_.get(); }
    GGUFReader*                reader()            { return reader_.get(); }
    const GGUFReader*          reader()      const { return reader_.get(); }

private:
    std::unique_ptr<GGUFReader>      reader_;
    std::unique_ptr<Qwen35moeModel>  model_;
    bool        ready_      = false;
    std::string last_error_;
};

#endif // FUNASR_PIPELINE_RECOGNIZER_HPP
