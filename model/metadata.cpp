#include "model/metadata.h"
#include "core/gguf_reader.h"

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
        } else {
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

void GeneralInfo::load_from_gguf(struct gguf_context* gctx) {
    type = gguf_get_str_or(gctx, "general.type", "");
    architecture = gguf_get_str_or(gctx, "general.architecture", "");
    quantization_version = gguf_get_u32_or(gctx, "general.quantization_version", 0);
    file_type = gguf_get_u32_or(gctx, "general.file_type", 0);
    sampling_top_k = gguf_get_i32_or(gctx, "general.sampling.top_k", 0);
    sampling_top_p = gguf_get_f32_or(gctx, "general.sampling.top_p", 0.0);
    sampling_temp = gguf_get_f32_or(gctx, "general.sampling.temp", 0.0);
    name = gguf_get_str_or(gctx, "general.name", "");
}

void GeneralInfo::print() const {
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

void Qwen35moeInfo::load_from_gguf(struct gguf_context* gctx) {
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

void Qwen35moeInfo::print() const {
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

void TokenizerInfo::load_from_gguf(struct gguf_context* gctx) {
    ggml_model = gguf_get_str_or(gctx, "tokenizer.ggml.model", "");
    ggml_pre = gguf_get_str_or(gctx, "tokenizer.ggml.pre", "");
    ggml_tokens = gguf_get_arrary_string_or(gctx, "tokenizer.ggml.tokens");
    ggml_token_type = gguf_get_arrary_i32_or(gctx, "tokenizer.ggml.token_type");
    ggml_merges = gguf_get_arrary_string_or(gctx, "tokenizer.ggml.merges");
    ggml_eos_token_id = gguf_get_u32_or(gctx, "tokenizer.ggml.eos_token_id", 0);
    ggml_padding_token_id = gguf_get_u32_or(gctx, "tokenizer.ggml.padding_token_id", 0);
    chat_template = gguf_get_str_or(gctx, "tokenizer.chat_template", "");
}

void TokenizerInfo::print() const {
    printf("\n--- tokenizer ---\n");
    printf(" tokenizer.ggml.model:%s\n", ggml_model.c_str());
    printf(" tokenizer.ggml.pre:%s\n", ggml_pre.c_str());
    printf(" tokenizer.ggml.tokens[size:%zu,list(", ggml_tokens.size());
    for (size_t i = 0; i < ggml_tokens.size(); ++i) {
        if (i < 50 || i >= (ggml_tokens.size() - 50)) {
            printf("%s ", ggml_tokens[i].c_str());
        } else if (50 == i) {
            printf(" ...... ");
        }
    }
    printf(")]\n");
    printf(" tokenizer.ggml.token_type[size:%zu,list(", ggml_token_type.size());
    for (size_t i = 0; i < ggml_token_type.size(); ++i) {
        if (i < 50 || i >= (ggml_token_type.size() - 200)) {
            printf("%d ", ggml_token_type[i]);
        } else if (50 == i) {
            printf(" ...... ");
        }
    }
    printf(")]\n");
    printf(" tokenizer.ggml.merges[size:%zu,list(", ggml_merges.size());
    for (size_t i = 0; i < ggml_merges.size(); ++i) {
        if (i < 10 || i >= (ggml_merges.size() - 10)) {
            printf("%s ", ggml_merges[i].c_str());
        } else if (10 == i) {
            printf(" ...... ");
        }
    }
    printf(")]\n");
    printf(" tokenizer.ggml.eos_token_id:%d\n", ggml_eos_token_id);
    printf(" tokenizer.ggml.padding_token_id:%d\n", ggml_padding_token_id);
    printf(" tokenizer.chat_template:%s\n", chat_template.c_str());
}

void QuantizeInfo::load_from_gguf(struct gguf_context* gctx) {
    imatrix_file = gguf_get_str_or(gctx, "quantize.imatrix.file", "");
    imatrix_dataset = gguf_get_str_or(gctx, "quantize.imatrix.dataset", "");
    entries_count = gguf_get_u32_or(gctx, "quantize.imatrix.entries_count", 0);
    chunks_count = gguf_get_u32_or(gctx, "quantize.imatrix.chunks_count", 0);
}

void QuantizeInfo::print() const {
    printf("\n--- tokenizer ---\n");
    printf(" quantize.imatrix.file:%s\n", imatrix_file.c_str());
    printf(" quantize.imatrix.dataset:%s\n", imatrix_dataset.c_str());
    printf(" quantize.imatrix.entries_count:%d\n", entries_count);
    printf(" quantize.imatrix.chunks_count:%d\n", chunks_count);
}

bool MetaDataInfo::load_from_gguf(struct gguf_context* gctx) {
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

void MetaDataInfo::print() const {
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