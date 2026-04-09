// 下面是使用Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf模型的参数：
//     metadata:
//         version	3
//         tensor_count	733
//         kv_count	43
//         general.architecture	qwen35moe
//         general.type	model
//         general.sampling.top_k	20
//         general.sampling.top_p	0.949999988079071
//         general.sampling.temp	1
//         general.name	Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
//         general.finetune	5RefKL0.042
//         general.quantization_version	2
//         general.file_type	Q5_K_M
//         general.size_label	35B-A3B
//         qwen35moe.block_count	40
//         qwen35moe.context_length	262144
//         qwen35moe.embedding_length	2048
//         qwen35moe.attention.head_count	16
//         qwen35moe.attention.head_count_kv	2
//         qwen35moe.attention.layer_norm_rms_epsilon	9.999999974752427e-7
//         qwen35moe.attention.key_length	256
//         qwen35moe.attention.value_length	256
//         qwen35moe.rope.dimension_sections	[11, 11, 10, 0]
//         qwen35moe.rope.freq_base	10000000
//         qwen35moe.rope.dimension_count	64
//         qwen35moe.expert_count	256
//         qwen35moe.expert_used_count	8
//         qwen35moe.expert_feed_forward_length	512
//         qwen35moe.expert_shared_feed_forward_length	512
//         qwen35moe.ssm.conv_kernel	4
//         qwen35moe.ssm.state_size	128
//         qwen35moe.ssm.group_count	16
//         qwen35moe.ssm.time_step_rank	32
//         qwen35moe.ssm.inner_size	4096
//         qwen35moe.full_attention_interval	4
//         tokenizer.ggml.model	gpt2
//         tokenizer.ggml.pre	qwen35
//         tokenizer.ggml.tokens	[!, ", #, $, %, ...]
//         tokenizer.ggml.token_type	[1, 1, 1, 1, 1, ...]
//         tokenizer.ggml.merges	[Ġ Ġ, ĠĠ ĠĠ, i n, Ġ t, ĠĠĠĠ ĠĠĠĠ, ...]
//         tokenizer.ggml.eos_token_id	248046
//         tokenizer.ggml.padding_token_id	248044
//         tokenizer.chat_template {%- set image_count = ......
//         quantize.imatrix.file	Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive.imatrix
//         quantize.imatrix.dataset	/home/hauhaucs/llama.cpp/groups_merged.txt
//         quantize.imatrix.entries_count	510
//         quantize.imatrix.chunks_count	93

#ifndef FUNASR_CORE_CONFIG_HPP
#define FUNASR_CORE_CONFIG_HPP

#include <gguf.h>
#include <string>
#include <vector>
#include <cstdio>

// ============================================================
// GGUF metadata 安全读取辅助函数
// 找不到 key 时返回 default，不会崩溃
// ============================================================
inline int32_t gguf_get_i32_or(struct gguf_context* ctx, const char* key, int default_val) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    return gguf_get_val_i32(ctx, idx);
}

inline uint32_t gguf_get_u32_or(struct gguf_context* ctx, const char* key, uint32_t default_val) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    return gguf_get_val_u32(ctx, idx);
}

inline float gguf_get_f32_or(struct gguf_context* ctx, const char* key, float default_val) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx, idx);
}

inline std::string gguf_get_str_or(struct gguf_context* ctx, const char* key, const char* default_val) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    return gguf_get_val_str(ctx, idx);
}

inline std::vector<std::string> gguf_get_arrary_string_or(struct gguf_context* ctx, const char* key) {
    std::vector<std::string> result;
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return result;
    int arr_n = gguf_get_arr_n(ctx, idx);
    const enum gguf_type arr_type = gguf_get_arr_type(ctx, idx);
    for (int j = 0; j < arr_n; j++) {
        if (arr_type == GGUF_TYPE_STRING) {
            result.push_back(gguf_get_arr_str(ctx, idx, j));
        }
        else {
            printf("[Config] WARNING: array type '%d' not supported\n", arr_type);
        }
    }

    return result;
}

inline std::vector<uint32_t> gguf_get_arrary_u32_or(struct gguf_context* ctx, const char* key) {
    std::vector<uint32_t> result;
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return result;
    int n = gguf_get_arr_n(ctx, idx);
    result.resize(n);
    const uint32_t * values = (const uint32_t *)gguf_get_arr_data(ctx, idx);
    for (int i = 0; i < n; ++i) {
        result[i] = values[i];
    }

    return result;
}

inline std::vector<int32_t> gguf_get_arrary_i32_or(struct gguf_context* ctx, const char* key) {
    std::vector<int32_t> result;
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return result;
    int n = gguf_get_arr_n(ctx, idx);
    result.resize(n);
    const int32_t * values = (const int32_t *)gguf_get_arr_data(ctx, idx);
    for (int i = 0; i < n; ++i) {
        result[i] = values[i];
    }

    return result;
}

// 必须存在的 key，找不到返回 false 并打印错误
inline bool gguf_require_i32(struct gguf_context* ctx, const char* key, int& out) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        printf("[Config] ERROR: required key '%s' not found in GGUF\n", key);
        return false;
    }
    out = gguf_get_val_i32(ctx, idx);
    return true;
}

inline bool gguf_require_u32(struct gguf_context* ctx, const char* key, uint32_t& out) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        printf("[Config] ERROR: required key '%s' not found in GGUF\n", key);
        return false;
    }
    out = gguf_get_val_u32(ctx, idx);
    return true;
}

// ============================================================
// general 配置
// GGUF keys: general.*
// ============================================================
struct GeneralConfig {
    std::string type = "";
    std::string architecture = "";
    uint32_t quantization_version = 0;
    uint32_t file_type = 0;
    int32_t sampling_top_k = 0;
    float sampling_top_p = 0.0;
    float sampling_temp = 0.0;
    std::string name = "";

    void load_from_gguf(struct gguf_context* gctx) {
        type = gguf_get_str_or(gctx, "general.type", "");
        architecture = gguf_get_str_or(gctx, "general.architecture", "");
        quantization_version = gguf_get_u32_or(gctx, "general.quantization_version", 0);
        file_type = gguf_get_u32_or(gctx, "general.file_type", 0);
        sampling_top_k = gguf_get_i32_or(gctx, "general.sampling.top_k", 0);
        sampling_top_p = gguf_get_f32_or(gctx, "general.sampling.top_p", 0.0);
        sampling_temp = gguf_get_f32_or(gctx, "general.sampling.temp", 0.0);
        name = gguf_get_str_or(gctx, "general.name", "");
    }

    void print() const {
        printf("\n--- general ---\n");
        printf(" general.type:%s\n", type.c_str());
        printf(" general.architecture:%s\n", architecture.c_str());
        printf(" general.quantization_version:%d\n", quantization_version);
        printf(" general.file_type:%d\n", file_type);
        printf(" general.sampling.top_k:%d\n", sampling_top_k);
        printf(" general.sampling.top_p:%f\n", sampling_top_p);
        printf(" general.sampling.temp:%f\n", sampling_temp);
        printf(" general.name:%s\n", name.c_str());
    }
};

// ============================================================
// qwen35moe 配置
// GGUF keys: qwen35moe.*
// ============================================================
struct Qwen35moeConfig {
    uint32_t context_length = 0;
    uint32_t embedding_length = 0;
    uint32_t block_count = 0;
    uint32_t expert_feed_forward_length = 0;
    uint32_t expert_shared_feed_forward_length = 0;
    uint32_t expert_count = 0;
    uint32_t expert_used_count = 0;
    uint32_t full_attention_interval = 0;
    uint32_t head_count = 0;
    uint32_t head_count_kv = 0;
    uint32_t key_length = 0;
    uint32_t value_length = 0;
    float layer_norm_rms_epsilon = 0.0;
    uint32_t dimension_count = 0;
    std::vector<int32_t> dimension_sections;
    float freq_base = 0.0;
    uint32_t conv_kernel = 0;
    uint32_t inner_size = 0;
    uint32_t state_size = 0;
    uint32_t time_step_rank = 0;
    uint32_t group_count = 0;

    void load_from_gguf(struct gguf_context* gctx) {
        context_length = gguf_get_u32_or(gctx, "qwen35moe.context_length", 0);
        embedding_length = gguf_get_u32_or(gctx, "qwen35moe.embedding_length", 0);
        block_count = gguf_get_u32_or(gctx, "qwen35moe.block_count", 0);
        expert_feed_forward_length = gguf_get_u32_or(gctx, "qwen35moe.expert_feed_forward_length", 0);
        expert_shared_feed_forward_length = gguf_get_u32_or(gctx, "qwen35moe.expert_shared_feed_forward_length", 0);
        expert_count = gguf_get_u32_or(gctx, "qwen35moe.expert_count", 0);
        expert_used_count = gguf_get_u32_or(gctx, "qwen35moe.expert_used_count", 0);
        full_attention_interval = gguf_get_u32_or(gctx, "qwen35moe.full_attention_interval", 0);
        head_count = gguf_get_u32_or(gctx, "qwen35moe.attention.head_count" , 0);
        head_count_kv = gguf_get_u32_or(gctx, "qwen35moe.attention.head_count_kv", 0);
        key_length = gguf_get_u32_or(gctx, "qwen35moe.attention.key_length", 0);
        value_length = gguf_get_u32_or(gctx, "qwen35moe.attention.value_length", 0);
        layer_norm_rms_epsilon = gguf_get_f32_or(gctx, "qwen35moe.attention.layer_norm_rms_epsilon", 0.0);
        dimension_count = gguf_get_u32_or(gctx, "qwen35moe.rope.dimension_count", 0);
        dimension_sections = gguf_get_arrary_i32_or(gctx, "qwen35moe.rope.dimension_sections");
        freq_base = gguf_get_f32_or(gctx, "qwen35moe.rope.freq_base", 0.0);
        conv_kernel = gguf_get_u32_or(gctx, "qwen35moe.ssm.conv_kernel", 0);
        inner_size = gguf_get_u32_or(gctx, "qwen35moe.ssm.inner_size", 0);
        state_size = gguf_get_u32_or(gctx, "qwen35moe.ssm.state_size", 0);
        time_step_rank = gguf_get_u32_or(gctx, "qwen35moe.ssm.time_step_rank", 0);
        group_count = gguf_get_u32_or(gctx, "qwen35moe.ssm.group_count", 0);
    }

    void print() const {
        printf("\n--- qwen35moe ---\n");
        printf(" qwen35moe.context_length:%d\n", context_length);
        printf(" qwen35moe.embedding_length:%d\n", embedding_length);
        printf(" qwen35moe.block_count:%d\n", block_count);
        printf(" qwen35moe.expert_feed_forward_length:%d\n", expert_feed_forward_length);
        printf(" qwen35moe.expert_shared_feed_forward_length:%d\n", expert_shared_feed_forward_length);
        printf(" qwen35moe.expert_count:%d\n", expert_count);
        printf(" qwen35moe.expert_used_count:%d\n", expert_used_count);
        printf(" qwen35moe.full_attention_interval:%d\n", full_attention_interval);
        printf(" qwen35moe.head_count:%d\n", head_count);
        printf(" qwen35moe.head_count_kv:%d\n", head_count_kv);
        printf(" qwen35moe.key_length:%d\n", key_length);
        printf(" qwen35moe.value_length:%d\n", value_length);
        printf(" qwen35moe.layer_norm_rms_epsilon:%f\n", layer_norm_rms_epsilon);
        printf(" qwen35moe.dimension_count:%d\n", dimension_count);
        for (size_t i = 0; i < dimension_sections.size(); ++i) {
            printf(" qwen35moe.dimension_sections[%zu]:%d\n", i, dimension_sections[i]);
        }
        printf(" qwen35moe.freq_base:%f\n", freq_base);
        printf(" qwen35moe.conv_kernel:%d\n", conv_kernel);
        printf(" qwen35moe.inner_size:%d\n", inner_size);
        printf(" qwen35moe.state_size:%d\n", state_size);
        printf(" qwen35moe.time_step_rank:%d\n", time_step_rank);
        printf(" qwen35moe.group_count:%d\n", group_count);
    }
};

// ============================================================
// tokenizer 配置
// GGUF metadata 中没有 adaptor 参数
// 这些值从 encoder/LLM 配置推导
// ============================================================
struct TokenizerConfig {
    std::string ggml_model = "";
    std::string ggml_pre = "";
    std::vector<std::string> ggml_tokens;
    std::vector<int32_t> ggml_token_type;
    std::vector<std::string> ggml_merges;
    uint32_t ggml_eos_token_id = 0;
    uint32_t ggml_padding_token_id = 0;
    std::string chat_template = "";

    void load_from_gguf(struct gguf_context* gctx) {
        ggml_model = gguf_get_str_or(gctx, "tokenizer.ggml.model", "");
        ggml_pre = gguf_get_str_or(gctx, "tokenizer.ggml.pre", "");
        ggml_tokens = gguf_get_arrary_string_or(gctx, "tokenizer.ggml.tokens");
        ggml_token_type = gguf_get_arrary_i32_or(gctx, "tokenizer.ggml.token_type");
        ggml_merges = gguf_get_arrary_string_or(gctx, "tokenizer.ggml.merges");
        ggml_eos_token_id = gguf_get_u32_or(gctx, "tokenizer.ggml.eos_token_id", 0);
        ggml_padding_token_id = gguf_get_u32_or(gctx, "tokenizer.ggml.padding_token_id", 0);
        chat_template = gguf_get_str_or(gctx, "tokenizer.chat_template", "");
    }

    // 打印所有配置参数（验证用）
    void print() const {
        printf("\n--- tokenizer ---\n");
        printf(" tokenizer.ggml.model:%s\n", ggml_model.c_str());
        printf(" tokenizer.ggml.pre:%s\n", ggml_pre.c_str());
        printf(" tokenizer.ggml.tokens[size:%zu]\n", ggml_tokens.size());
        // for (size_t i = 0; i < ggml_tokens.size(); ++i) {
        //     printf(" tokenizer.ggml.tokens[%d]:%d\n", i, ggml_tokens[i]);
        // }
        printf(" tokenizer.ggml.token_type[size:%zu]\n", ggml_token_type.size());
        // for (size_t i = 0; i < ggml_token_type.size(); ++i) {
        //     printf(" tokenizer.ggml.token_type[%d]:%d\n", i, ggml_token_type[i]);
        // }
        printf(" tokenizer.ggml.merges[size:%zu]\n", ggml_merges.size());
        // for (size_t i = 0; i < ggml_merges.size(); ++i) {
        //     printf(" tokenizer.ggml.merges[%d]:%s\n", i, ggml_merges[i].c_str());
        // }
        printf(" tokenizer.ggml.eos_token_id:%d\n", ggml_eos_token_id);
        printf(" tokenizer.ggml.padding_token_id:%d\n", ggml_padding_token_id);
        //printf(" tokenizer.chat_template:%s\n", chat_template.c_str());
    }
};

// ============================================================
// quantize 配置
// GGUF metadata 中没有 adaptor 参数
// 这些值从 encoder/LLM 配置推导
// ============================================================
struct QuantizeConfig {
    std::string imatrix_file = "";
    std::string imatrix_dataset = "";
    uint32_t entries_count = 0;
    uint32_t chunks_count = 0;

    void load_from_gguf(struct gguf_context* gctx) {
        imatrix_file = gguf_get_str_or(gctx, "quantize.imatrix.file", "");
        imatrix_dataset = gguf_get_str_or(gctx, "quantize.imatrix.dataset", "");
        entries_count = gguf_get_u32_or(gctx, "quantize.imatrix.entries_count", 0);
        chunks_count = gguf_get_u32_or(gctx, "quantize.imatrix.chunks_count", 0);
    }

    // 打印所有配置参数（验证用）
    void print() const {
        printf("\n--- tokenizer ---\n");
        printf(" quantize.imatrix.file:%s\n", imatrix_file.c_str());
        printf(" quantize.imatrix.dataset:%s\n", imatrix_dataset.c_str());
        printf(" quantize.imatrix.entries_count:%d\n", entries_count);
        printf(" quantize.imatrix.chunks_count:%d\n", chunks_count);
    }
};

// ============================================================
// 完整模型配置
// ============================================================
struct ModelConfig {
    GeneralConfig general;
    Qwen35moeConfig qwen35moe;
    TokenizerConfig tokenizer;
    QuantizeConfig quantize;

    // 从 GGUF metadata 填充所有配置
    // 返回 false 表示缺少关键参数
    bool load_from_gguf(struct gguf_context* gctx) {
        if (!gctx) {
            printf("[Config] ERROR: gguf_context is null\n");
            return false;
        }

        general.load_from_gguf(gctx);
        qwen35moe.load_from_gguf(gctx);
        tokenizer.load_from_gguf(gctx);
        quantize.load_from_gguf(gctx);
        print();
        return true;
    }

    // 打印所有配置参数（验证用）
    void print() const {
        printf("\n");
        printf("========================================\n");
        printf("  Qwen35moe Model Configuration\n");
        printf("========================================\n");

        general.print();
        qwen35moe.print();
        tokenizer.print();
        quantize.print();
        printf("========================================\n\n");
    }
};

#endif // FUNASR_CORE_CONFIG_HPP