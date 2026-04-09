// pipeline/recognizer.hpp
// Recognizer — loads a GGUF model and exposes it for inference
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

    // Not copyable (owns GGUFReader)
    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // Load model from GGUF file
    bool init(const std::string& model_path, bool verbose = true) {
        if (verbose) printf("[Recognizer] Loading model: %s\n", model_path.c_str());

        reader_ = std::make_unique<GGUFReader>();
        if (!reader_->open(model_path)) {
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
    const Qwen35moeModel*      model()      const { return model_.get(); }

private:
    std::unique_ptr<GGUFReader>      reader_;
    std::unique_ptr<Qwen35moeModel>  model_;
    bool        ready_      = false;
    std::string last_error_;
};

#endif // FUNASR_PIPELINE_RECOGNIZER_HPP