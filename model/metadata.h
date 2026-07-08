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
// 下面是使用Qwen3.5-35B-A3B-UD-Q5_K_M.gguf模型的参数：
//     metadata:
//         version	3
//         tensor_count	753
//         kv_count	54
//         general.architecture	qwen35moe
//         general.type	model
//         general.sampling.top_k	20
//         general.sampling.top_p	0.949999988079071
//         general.sampling.temp	1
//         general.name	Qwen3.5-35B-A3B
//         general.basename	Qwen3.5-35B-A3B
//         general.quantized_by	Unsloth
//         general.size_label	35B-A3B
//         general.license	apache-2.0
//         general.license.link	https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE
//         general.repo_url	https://huggingface.co/unsloth
//         general.base_model.count	1
//         general.base_model.0.name	Qwen3.5 35B A3B
//         general.base_model.0.organization	Qwen
//         general.base_model.0.repo_url	https://huggingface.co/Qwen/Qwen3.5-35B-A3B
//         general.tags	[unsloth, image-text-to-text]
//         general.quantization_version	2
//         general.file_type	Q5_K_M
//         qwen35moe.block_count	41
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
//         qwen35moe.nextn_predict_layers	1
//         tokenizer.ggml.model	gpt2
//         tokenizer.ggml.pre	qwen35
//         tokenizer.ggml.tokens	[!, ", #, $, %, ...]
//         tokenizer.ggml.token_type	[1, 1, 1, 1, 1, ...]
//         tokenizer.ggml.merges	[Ġ Ġ, ĠĠ ĠĠ, i n, Ġ t, ĠĠĠĠ ĠĠĠĠ, ...]
//         tokenizer.ggml.eos_token_id	248046
//         tokenizer.ggml.padding_token_id	248055
//         tokenizer.ggml.add_bos_token	false
//         tokenizer.chat_template	
//         quantize.imatrix.file	Qwen3.5-35B-A3B-GGUF/imatrix_unsloth.gguf
//         quantize.imatrix.dataset	unsloth_calibration_Qwen3.5-35B-A3B.txt
//         quantize.imatrix.entries_count	510
//         quantize.imatrix.chunks_count	77

#pragma once

#include <gguf.h>
#include <string>
#include <vector>
#include <cstdio>
#include "model/gguf_mmap.h"

class HeadInfo {
public:
    std::string magic = "";
    uint32_t version = 0;
    int64_t tensor_count = 0;
    int64_t kv_count = 0;

    void print() const;
};

// ============================================================
// general 配置
// GGUF keys: general.*
// ============================================================
class GeneralInfo {
public:
    std::string architecture = "";
    std::string type = "";
    int32_t sampling_top_k = 0;
    float sampling_top_p = 0.0;
    float sampling_temp = 0.0;
    std::string name = "";
    std::string basename = "";
    std::string quantized_by = "";
    std::string finetune = "";
    uint32_t quantization_version = 0;
    uint32_t file_type = 0;
    std::string size_label = "";
    std::string license = "";
    std::string license_link = "";
    std::string repo_url = "";
    uint32_t base_model_count = 0;
    std::string base_model_0_name = "";
    std::string base_model_0_organization = "";
    std::string base_model_0_repo_url = "";
    std::vector<std::string> tags;

    void load_from_gguf(GGUFLoader* load);
    void print() const;
};

// ============================================================
// qwen35moe 配置
// GGUF keys: qwen35moe.*
// ============================================================
class Qwen35moeInfo {
public:
    uint32_t block_count = 0;
    uint32_t context_length = 0;
    uint32_t embedding_length = 0;
    uint32_t head_count = 0;
    uint32_t head_count_kv = 0;
    float layer_norm_rms_epsilon = 0.0;
    uint32_t key_length = 0;
    uint32_t value_length = 0;
    std::vector<int32_t> dimension_sections;
    float freq_base = 0.0;
    uint32_t dimension_count = 0;
    uint32_t expert_count = 0;
    uint32_t expert_used_count = 0;
    uint32_t expert_feed_forward_length = 0;
    uint32_t expert_shared_feed_forward_length = 0;
    uint32_t conv_kernel = 0;
    uint32_t state_size = 0;
    uint32_t group_count = 0;
    uint32_t time_step_rank = 0;
    uint32_t inner_size = 0;
    uint32_t full_attention_interval = 0;
    uint32_t nextn_predict_layers = 0;
    
    void load_from_gguf(GGUFLoader* load);
    void print() const;
};

// ============================================================
// tokenizer 配置
// ============================================================
class TokenizerInfo {
public:
    std::string ggml_model = "";
    std::string ggml_pre = "";
    std::vector<std::string> ggml_tokens;
    std::vector<int32_t> ggml_token_type;
    std::vector<std::string> ggml_merges;
    uint32_t ggml_eos_token_id = 0;
    uint32_t ggml_padding_token_id = 0;
    uint32_t ggml_bos_token_id = 0;
    bool ggml_add_bos_token = false;
    std::string chat_template = "";

    void load_from_gguf(GGUFLoader* load);
    void print() const;
};

// ============================================================
// quantize 配置
// ============================================================
class QuantizeInfo {
public:
    std::string imatrix_file = "";
    std::string imatrix_dataset = "";
    uint32_t entries_count = 0;
    uint32_t chunks_count = 0;

    void load_from_gguf(GGUFLoader* load);
    void print() const;
};

// ============================================================
// 完整模型配置
// ============================================================
class MetaDataInfo {
public:
    HeadInfo head;
    GeneralInfo general;
    Qwen35moeInfo qwen35moe;
    TokenizerInfo tokenizer;
    QuantizeInfo quantize;

    bool load_from_gguf(GGUFLoader* load);
    void print() const;
};
