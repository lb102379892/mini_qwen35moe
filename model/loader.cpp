// funasr/model/loader.cpp
// 模型加载实现
//
// 这里是 "GGUF tensor name" 和 "结构体字段" 的映射关系所在。
// GGUF 里的名字是转换脚本写入时决定的（不可改），
// 结构体字段名是我们的内部命名（随便叫）。
// 这个文件负责把两者连接起来。
//
#include "model/loader.hpp"
#include <cstdio>
#include <string>

// ============================================================
// 加载单层
// ============================================================
void ModelLoader::load_qwen35moe_layer(GGUFReader& reader, Qwen35moeLayer& layer, int layer_idx) {
    std::string prefix = "blk." + std::to_string(layer_idx);

    layer.attn_gate       = reader.get_tensor(prefix + ".attn_gate.weight");
    layer.attn_norm       = reader.require_tensor(prefix + ".attn_norm.weight");
    layer.attn_qkv        = reader.get_tensor(prefix + ".attn_qkv.weight");
    layer.ffn_down_exps   = reader.require_tensor(prefix + ".ffn_down_exps.weight");
    layer.ffn_down_shexp  = reader.require_tensor(prefix + ".ffn_down_shexp.weight");
    layer.ffn_gate_exps   = reader.require_tensor(prefix + ".ffn_gate_exps.weight");
    layer.ffn_gate_inp    = reader.require_tensor(prefix + ".ffn_gate_inp.weight");
    layer.ffn_gate_inp_shexp = reader.require_tensor(prefix + ".ffn_gate_inp_shexp.weight");
    layer.ffn_gate_shexp  = reader.require_tensor(prefix + ".ffn_gate_shexp.weight");
    layer.ffn_up_exps     = reader.require_tensor(prefix + ".ffn_up_exps.weight");
    layer.ffn_up_shexp    = reader.require_tensor(prefix + ".ffn_up_shexp.weight");
    layer.post_attention_norm = reader.require_tensor(prefix + ".post_attention_norm.weight");
    layer.ssm_a           = reader.get_tensor(prefix + ".ssm_a.weight");
    layer.ssm_alpha       = reader.get_tensor(prefix + ".ssm_alpha.weight");
    layer.ssm_beta        = reader.get_tensor(prefix + ".ssm_beta.weight");
    layer.ssm_conv1d      = reader.get_tensor(prefix + ".ssm_conv1d.weight");
    layer.ssm_dt_b          = reader.get_tensor(prefix + ".ssm_dt.bias");
    layer.ssm_norm        = reader.get_tensor(prefix + ".ssm_norm.weight");
    layer.ssm_out         = reader.get_tensor(prefix + ".ssm_out.weight");

    layer.attn_k         = reader.get_tensor(prefix + ".attn_k.weight");
    layer.attn_k_norm    = reader.get_tensor(prefix + ".attn_k_norm.weight");
    layer.attn_q         = reader.get_tensor(prefix + ".attn_q.weight");
    layer.attn_q_norm    = reader.get_tensor(prefix + ".attn_q_norm.weight");
    layer.attn_v         = reader.get_tensor(prefix + ".attn_v.weight");
    layer.attn_output    = reader.get_tensor(prefix + ".attn_output.weight");
}

// ============================================================
// 加载 Config（从 GGUF metadata）
// ============================================================
bool ModelLoader::load_config(GGUFReader& reader, ModelConfig& config) {
    if (!reader.is_open()) {
        printf("[Loader] ERROR: reader not open\n");
        return false;
    }
    return config.load_from_gguf(reader.gguf_ctx());
}

// ============================================================
// 加载 LLM Decoder
// ============================================================
bool ModelLoader::load_qwen35moe(GGUFReader& reader, Qwen35moeWeights& weights, const int layer_count) {
    printf("[Loader] load_qwen35moe...\n");
    reader.clear_errors();

    // token_embd
    weights.token_embd = reader.require_tensor("token_embd.weight");

    // Layers
    weights.layers.resize(layer_count);
    for (int i = 0; i < layer_count; i++) {
        load_qwen35moe_layer(reader, weights.layers[i], i);
    }

    // 
    weights.output = reader.require_tensor("output.weight");
    weights.output_norm    = reader.require_tensor("output_norm.weight");

    if (reader.has_errors()) {
        printf("[Loader] LLM: %zu tensor(s) missing\n", reader.missing_tensors().size());
        reader.print_errors();
        return false;
    }

    printf("[Loader] LLM loaded tensors [OK]\n");
    return true;
}

// ============================================================
// 一站式加载
// ============================================================
bool ModelLoader::load(GGUFReader& reader, Qwen35moeModel& model) {
    printf("\n========================================\n");
    printf("  Loading Qwen35moe Model\n");
    printf("========================================\n");

    if (!reader.is_open()) {
        printf("[Loader] ERROR: reader not open\n");
        return false;
    }

    printf("File: %s\n", reader.path().c_str());
    printf("Tensors: %d, KV pairs: %d\n\n", reader.tensor_count(), reader.kv_count());

    // Step 1: Config
    if (!load_config(reader, model.config)) {
        printf("[Loader] Config loading failed\n");
        return false;
    }

    // Step 2: modle
    if (!load_qwen35moe(reader, model.weights)) {
        printf("[Loader] Tensor loading failed\n");
        return false;
    }

    // Summary
    printf("\n========================================\n");
    printf("  Loading Complete!\n");
    printf("========================================\n");

    // 验证总数
    int expected = reader.tensor_count();
    int actual = model.tensor_count();
    if (actual != expected) {
        printf("[Loader] WARNING: loaded %d tensors, GGUF has %d\n", actual, expected);
    }

    return true;
}
