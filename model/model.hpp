// funasr/model/model.hpp
// 聚合模型结构体 — 把 Config + 三大模块权重打包
#ifndef FUNASR_MODEL_MODEL_HPP
#define FUNASR_MODEL_MODEL_HPP

#include "core/config.hpp"
#include "model/weights.hpp"

struct Qwen35moeModel {
    ModelConfig config;
    Qwen35moeWeights weights;

    int tensor_count () {
        return 0;
    }
};

#endif // FUNASR_MODEL_MODEL_HPP