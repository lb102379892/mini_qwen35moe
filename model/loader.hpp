// funasr/model/loader.hpp
// 模型加载器 — 从 GGUFReader 绑定权重指针到结构体
//
// 设计原则：
//   1. 全部 static 方法，不持有状态
//   2. 接收 GGUFReader 引用，reader 必须在 model 使用期间保持存活
//   3. 利用 reader.require_tensor() 收集缺失的 tensor
//   4. 层数从 Config 读取，不硬编码
//
#ifndef FUNASR_MODEL_LOADER_HPP
#define FUNASR_MODEL_LOADER_HPP

#include "core/gguf_reader.hpp"
#include "core/config.hpp"
#include "model/weights.hpp"
#include "model/model.hpp"
#include <string>

class ModelLoader {
public:
    // ============================================================
    // 一站式加载
    // ============================================================
    static bool load(GGUFReader& reader, Qwen35moeModel& model);

    // ============================================================
    // 分步加载（可单独验证每个模块）
    // ============================================================
    static bool load_config(GGUFReader& reader, ModelConfig& config);

    static bool load_qwen35moe(GGUFReader& reader, Qwen35moeWeights& weights, const int layer_count = 40);

private:
    // 加载单层
    static void load_qwen35moe_layer(GGUFReader& reader, Qwen35moeLayer& layer, int layer_idx);
};

#endif