#include "model/metadata.h"

void HeadInfo::print() const {
    printf("\n--- head ---\n");
    printf(" magic:%s\n", magic.c_str());
    printf(" version:%u\n", version);
    printf(" tensor_count:%lld\n", tensor_count);
    printf(" kv_count:%lld\n", kv_count);
}

void GeneralInfo::load_from_gguf(GGUFLoader* load) {
    architecture = load->get_str_or("general.architecture", "");
    type = load->get_str_or("general.type", "");
    sampling_top_k = load->get_i32_or("general.sampling.top_k", 0);
    sampling_top_p = load->get_f32_or("general.sampling.top_p", 0.0);
    sampling_temp = load->get_f32_or("general.sampling.temp", 0.0);
    name = load->get_str_or("general.name", "");
    finetune = load->get_str_or("general.finetune", "");
    quantization_version = load->get_u32_or("general.quantization_version", 0);
    file_type = load->get_u32_or("general.file_type", 0);
    size_label = load->get_str_or("general.size_label", "");
}

void GeneralInfo::print() const {
    printf("\n--- general ---\n");
    printf(" general.architecture:%s\n", architecture.c_str());
    printf(" general.type:%s\n", type.c_str());
    printf(" general.sampling.top_k:%d\n", sampling_top_k);
    printf(" general.sampling.top_p:%f\n", sampling_top_p);
    printf(" general.sampling.temp:%f\n", sampling_temp);
    printf(" general.name:%s\n", name.c_str());
    printf(" general.finetune:%s\n", finetune.c_str());
    printf(" general.quantization_version:%d\n", quantization_version);
    printf(" general.file_type:%d\n", file_type);
    printf(" general.size_label:%s\n", size_label.c_str());
}

void Qwen35moeInfo::load_from_gguf(GGUFLoader* load) {
    block_count = load->get_u32_or("qwen35moe.block_count", 0);
    context_length = load->get_u32_or("qwen35moe.context_length", 0);
    embedding_length = load->get_u32_or("qwen35moe.embedding_length", 0);
    head_count = load->get_u32_or("qwen35moe.attention.head_count" , 0);
    head_count_kv = load->get_u32_or("qwen35moe.attention.head_count_kv", 0);
    layer_norm_rms_epsilon = load->get_f32_or("qwen35moe.attention.layer_norm_rms_epsilon", 0.0);
    key_length = load->get_u32_or("qwen35moe.attention.key_length", 0);
    value_length = load->get_u32_or("qwen35moe.attention.value_length", 0);
    dimension_sections = load->get_arrary_i32_or("qwen35moe.rope.dimension_sections");
    freq_base = load->get_f32_or("qwen35moe.rope.freq_base", 0.0);
    dimension_count = load->get_u32_or("qwen35moe.rope.dimension_count", 0);
    expert_count = load->get_u32_or("qwen35moe.expert_count", 0);
    expert_used_count = load->get_u32_or("qwen35moe.expert_used_count", 0);
    expert_feed_forward_length = load->get_u32_or("qwen35moe.expert_feed_forward_length", 0);
    expert_shared_feed_forward_length = load->get_u32_or("qwen35moe.expert_shared_feed_forward_length", 0);
    conv_kernel = load->get_u32_or("qwen35moe.ssm.conv_kernel", 0);
    state_size = load->get_u32_or("qwen35moe.ssm.state_size", 0);
    group_count = load->get_u32_or("qwen35moe.ssm.group_count", 0);
    time_step_rank = load->get_u32_or("qwen35moe.ssm.time_step_rank", 0);
    inner_size = load->get_u32_or("qwen35moe.ssm.inner_size", 0);
    full_attention_interval = load->get_u32_or("qwen35moe.full_attention_interval", 0);
}

void Qwen35moeInfo::print() const {
    printf("\n--- qwen35moe ---\n");
    printf(" qwen35moe.block_count:%d\n", block_count);
    printf(" qwen35moe.context_length:%d\n", context_length);
    printf(" qwen35moe.embedding_length:%d\n", embedding_length);
    printf(" qwen35moe.head_count:%d\n", head_count);
    printf(" qwen35moe.head_count_kv:%d\n", head_count_kv);
    printf(" qwen35moe.layer_norm_rms_epsilon:%f\n", layer_norm_rms_epsilon);
    printf(" qwen35moe.key_length:%d\n", key_length);
    printf(" qwen35moe.value_length:%d\n", value_length);
    for (size_t i = 0; i < dimension_sections.size(); ++i) {
        printf(" qwen35moe.dimension_sections[%zu]:%d\n", i, dimension_sections[i]);
    }
    printf(" qwen35moe.freq_base:%f\n", freq_base);
    printf(" qwen35moe.dimension_count:%d\n", dimension_count);
    printf(" qwen35moe.expert_count:%d\n", expert_count);
    printf(" qwen35moe.expert_used_count:%d\n", expert_used_count);
    printf(" qwen35moe.expert_feed_forward_length:%d\n", expert_feed_forward_length);
    printf(" qwen35moe.expert_shared_feed_forward_length:%d\n", expert_shared_feed_forward_length);
    printf(" qwen35moe.conv_kernel:%d\n", conv_kernel);
    printf(" qwen35moe.state_size:%d\n", state_size);
    printf(" qwen35moe.group_count:%d\n", group_count);
    printf(" qwen35moe.time_step_rank:%d\n", time_step_rank);
    printf(" qwen35moe.inner_size:%d\n", inner_size);
    printf(" qwen35moe.full_attention_interval:%d\n", full_attention_interval);
}

void TokenizerInfo::load_from_gguf(GGUFLoader* load) {
    ggml_model = load->get_str_or("tokenizer.ggml.model", "");
    ggml_pre = load->get_str_or("tokenizer.ggml.pre", "");
    ggml_tokens = load->get_arrary_string_or("tokenizer.ggml.tokens");
    ggml_token_type = load->get_arrary_i32_or("tokenizer.ggml.token_type");
    ggml_merges = load->get_arrary_string_or("tokenizer.ggml.merges");
    ggml_eos_token_id = load->get_u32_or("tokenizer.ggml.eos_token_id", 0);
    ggml_padding_token_id = load->get_u32_or("tokenizer.ggml.padding_token_id", 0);
    ggml_bos_token_id = load->get_u32_or("tokenizer.ggml.bos_token_id", 0);
    chat_template = load->get_str_or("tokenizer.chat_template", "");
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
    printf(" tokenizer.ggml.bos_token_id:%d\n", ggml_bos_token_id);
    printf(" tokenizer.chat_template:%s\n", chat_template.c_str());
}

void QuantizeInfo::load_from_gguf(GGUFLoader* load) {
    imatrix_file = load->get_str_or("quantize.imatrix.file", "");
    imatrix_dataset = load->get_str_or("quantize.imatrix.dataset", "");
    entries_count = load->get_u32_or("quantize.imatrix.entries_count", 0);
    chunks_count = load->get_u32_or("quantize.imatrix.chunks_count", 0);
}

void QuantizeInfo::print() const {
    printf("\n--- tokenizer ---\n");
    printf(" quantize.imatrix.file:%s\n", imatrix_file.c_str());
    printf(" quantize.imatrix.dataset:%s\n", imatrix_dataset.c_str());
    printf(" quantize.imatrix.entries_count:%d\n", entries_count);
    printf(" quantize.imatrix.chunks_count:%d\n", chunks_count);
}

bool MetaDataInfo::load_from_gguf(GGUFLoader* load) {
    if (!load) {
        printf("[Config] ERROR: GGUFLoader is null\n");
        return false;
    }

    head.magic = load->magic_;
    head.version = load->version_;
    head.tensor_count = load->n_tensors_;
    head.kv_count = load->n_kv_;
    general.load_from_gguf(load);
    qwen35moe.load_from_gguf(load);
    tokenizer.load_from_gguf(load);
    quantize.load_from_gguf(load);
    //print();
    return true;
}

void MetaDataInfo::print() const {
    printf("\n");
    printf("========================================\n");
    printf("  Qwen35moe Model Configuration\n");
    printf("========================================\n");

    head.print();
    general.print();
    qwen35moe.print();
    tokenizer.print();
    quantize.print();
    printf("========================================\n\n");
}