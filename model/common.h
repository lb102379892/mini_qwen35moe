#pragma once

#include <unordered_map>
#include <map>

enum EN_WEIGHT_TYPE {
    EN_WEIGHT_TYPE_TOKEN_EMBD = 0,
    EN_WEIGHT_TYPE_OUTPUT = 1,
    EN_WEIGHT_TYPE_OUTPUT_NORM = 2,

    EN_WEIGHT_TYPE_ATTN_K = 3,
    EN_WEIGHT_TYPE_ATTN_K_NORM = 4,
    EN_WEIGHT_TYPE_ATTN_NORM = 5,
    EN_WEIGHT_TYPE_ATTN_GATE = 6,
    EN_WEIGHT_TYPE_ATTN_QKV = 7,
    EN_WEIGHT_TYPE_ATTN_Q = 8,
    EN_WEIGHT_TYPE_ATTN_Q_NORM = 9,
    EN_WEIGHT_TYPE_ATTN_V = 10,
    EN_WEIGHT_TYPE_FFN_DOWN_EXPS = 11,
    EN_WEIGHT_TYPE_FFN_GATE_EXPS = 12,
    EN_WEIGHT_TYPE_FFN_GATE_INP = 13,
    EN_WEIGHT_TYPE_FFN_UP_EXPS = 14,
    EN_WEIGHT_TYPE_FFN_DOWN_SHEXP = 15,
    EN_WEIGHT_TYPE_FFN_GATE_INP_SHEXP = 16,
    EN_WEIGHT_TYPE_FFN_GATE_SHEXP = 17,
    EN_WEIGHT_TYPE_FFN_UP_SHEXP = 18,
    EN_WEIGHT_TYPE_SSM_A = 19,
    EN_WEIGHT_TYPE_SSM_ALPHA = 20,
    EN_WEIGHT_TYPE_SSM_BETA = 21,
    EN_WEIGHT_TYPE_SSM_CONV1D = 22,
    EN_WEIGHT_TYPE_SSM_DT = 23,
    EN_WEIGHT_TYPE_SSM_NORM = 24,
    EN_WEIGHT_TYPE_SSM_OUT = 25,
    EN_WEIGHT_TYPE_POST_ATTENTION_NORM = 26,
    EN_WEIGHT_TYPE_ATTN_OUTPUT = 27,

    EN_WEIGHT_TYPE_COUNT = 28
};

const std::string WEIGHT_TOKEN_EMBD_NAME = "token_embd.weight";
const std::string WEIGHT_OUTPUT_NAME = "output.weight";
const std::string WEIGHT_OUTPUT_NORM_NAME = "output_norm.weight";

static const std::unordered_map<std::string, EN_WEIGHT_TYPE> g_tensor_layer_names = {
    {".attn_gate.weight", EN_WEIGHT_TYPE_ATTN_GATE},
    {".attn_norm.weight", EN_WEIGHT_TYPE_ATTN_NORM},
    {".attn_qkv.weight", EN_WEIGHT_TYPE_ATTN_QKV},
    {".ffn_down_exps.weight", EN_WEIGHT_TYPE_FFN_DOWN_EXPS},
    {".ffn_down_shexp.weight", EN_WEIGHT_TYPE_FFN_DOWN_SHEXP},
    {".ffn_gate_exps.weight", EN_WEIGHT_TYPE_FFN_GATE_EXPS},
    {".ffn_gate_inp.weight", EN_WEIGHT_TYPE_FFN_GATE_INP},
    {".ffn_gate_inp_shexp.weight", EN_WEIGHT_TYPE_FFN_GATE_INP_SHEXP},
    {".ffn_gate_shexp.weight", EN_WEIGHT_TYPE_FFN_GATE_SHEXP},
    {".ffn_up_exps.weight", EN_WEIGHT_TYPE_FFN_UP_EXPS},
    {".ffn_up_shexp.weight", EN_WEIGHT_TYPE_FFN_UP_SHEXP},
    {".post_attention_norm.weight", EN_WEIGHT_TYPE_POST_ATTENTION_NORM},
    {".ssm_a", EN_WEIGHT_TYPE_SSM_A},
    {".ssm_alpha.weight", EN_WEIGHT_TYPE_SSM_ALPHA},
    {".ssm_beta.weight", EN_WEIGHT_TYPE_SSM_BETA},
    {".ssm_conv1d.weight", EN_WEIGHT_TYPE_SSM_CONV1D},
    {".ssm_dt.bias", EN_WEIGHT_TYPE_SSM_DT},
    {".ssm_norm.weight", EN_WEIGHT_TYPE_SSM_NORM},
    {".ssm_out.weight", EN_WEIGHT_TYPE_SSM_OUT},
    {".attn_k.weight", EN_WEIGHT_TYPE_ATTN_K},
    {".attn_k_norm.weight", EN_WEIGHT_TYPE_ATTN_K_NORM},
    {".attn_q.weight", EN_WEIGHT_TYPE_ATTN_Q},
    {".attn_q_norm.weight", EN_WEIGHT_TYPE_ATTN_Q_NORM},
    {".attn_v.weight", EN_WEIGHT_TYPE_ATTN_V},
    {".attn_output.weight", EN_WEIGHT_TYPE_ATTN_OUTPUT}
};

static std::map<std::string, EN_WEIGHT_TYPE> g_tensor_head_names = {
    {WEIGHT_TOKEN_EMBD_NAME, EN_WEIGHT_TYPE_TOKEN_EMBD},
    {WEIGHT_OUTPUT_NAME, EN_WEIGHT_TYPE_OUTPUT},
    {WEIGHT_OUTPUT_NORM_NAME, EN_WEIGHT_TYPE_OUTPUT_NORM}
};

enum class DevMode {
    CPU_MODE,
    GPU_MODE,
    AUTO_MODE,
};

struct CParam {
    std::string model_path = "";
    std::string prompt = "";
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    int n_threads = 4;
    int ctx_size = 4096;
    int n_batch = -1;
    int n_ubatch = -1;
    size_t gpu_layer = 0;
    bool use_chat = true;
    bool verbose = true;
    bool repl_mode = false;
    bool flash_attention = false;
    bool enable_paged_kv = false;
    uint32_t paged_kv_block_size = 16;
    DevMode dev_mode = DevMode::CPU_MODE;
    uint64_t rng_seed = 79977733;
    bool no_mmap = false;
};

extern int extractNumber(const std::string& str);
extern std::string getAfterNumber(const std::string& str);