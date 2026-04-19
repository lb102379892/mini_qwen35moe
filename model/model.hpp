#ifndef FUNASR_MODEL_MODEL_HPP
#define FUNASR_MODEL_MODEL_HPP

#include "core/config.hpp"
#include "model/weights.hpp"

struct Qwen35moeModel {
    ModelConfig config;
    Qwen35moeWeights weights;
};

#endif // FUNASR_MODEL_MODEL_HPP