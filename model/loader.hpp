#ifndef FUNASR_MODEL_LOADER_HPP
#define FUNASR_MODEL_LOADER_HPP

#include "core/gguf_reader.hpp"
#include "core/config.hpp"
#include "model/weights.hpp"
#include "model/model.hpp"
#include <string>

class ModelLoader {
public:
    static bool load(GGUFReader& reader, Qwen35moeModel& model);
    static bool load_config(GGUFReader& reader, ModelConfig& config);
    static bool load_qwen35moe(GGUFReader& reader, Qwen35moeWeights& weights, const int layer_count = 40);

private:
    static void load_qwen35moe_layer(GGUFReader& reader, Qwen35moeLayer& layer, int layer_idx);
};

#endif