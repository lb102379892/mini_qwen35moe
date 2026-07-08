#pragma once

#include <string>
#include <unordered_map>
#include <vector>
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

    EN_WEIGHT_TYPE_NEXTN_SHARED_HEAD_NORM = 28,
    EN_WEIGHT_TYPE_NEXTN_EH_PROJ = 29,
    EN_WEIGHT_TYPE_NEXTN_ENORM = 30,
    EN_WEIGHT_TYPE_NEXTN_HNORM = 31,

    EN_WEIGHT_TYPE_COUNT = 32
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
    {".attn_output.weight", EN_WEIGHT_TYPE_ATTN_OUTPUT},
    {".nextn.shared_head_norm.weight", EN_WEIGHT_TYPE_NEXTN_SHARED_HEAD_NORM},
    {".nextn.eh_proj.weight", EN_WEIGHT_TYPE_NEXTN_EH_PROJ},
    {".nextn.enorm.weight", EN_WEIGHT_TYPE_NEXTN_ENORM},
    {".nextn.hnorm.weight", EN_WEIGHT_TYPE_NEXTN_HNORM}
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

// Tracks CLI / API fields set explicitly; unset fields may receive dev-mode defaults.
struct CParamUserFlags {
    bool temperature = false;
    bool top_p = false;
    bool top_k = false;
    bool flash_attention = false;
    bool enable_paged_kv = false;
    bool max_sequences = false;
    bool n_threads = false;
    bool n_threads_batch = false;
    bool dev_mode = false;
};

struct CParam {
    std::string model_path = "";
    std::string prompt = "";
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    int n_threads = 4;
    // llama.cpp --threads-batch: CPU threads for prefill / batched decode.
    // 0 means use n_threads (same pool for everything).
    int n_threads_batch = 0;
    int ctx_size = 4096;
    int n_batch = -1;
    int n_ubatch = -1;
    int max_sequences = 4;
    size_t gpu_layer = 0;
    // llama.cpp --override-tensor / -ot: "regex=CPU" per entry; last match wins.
    std::vector<std::string> tensor_overrides;
    bool use_chat = true;
    bool verbose = true;
    bool repl_mode = false;
    bool flash_attention = false;
    bool enable_paged_kv = false;
    uint32_t paged_kv_block_size = 16;
    DevMode dev_mode = DevMode::CPU_MODE;
    uint64_t rng_seed = 79977733;
    bool no_mmap = false;
    // MTP (Multi-Token Prediction / NextN) speculative decoding. Uses the
    // model's built-in nextn block (blk.<trunk_layer_count>) as a draft head:
    // draft up to `spec_draft_n_max` tokens, then verify them in one batched
    // trunk forward. Only active when the GGUF actually carries an MTP block
    // (nextn_predict_layers > 0), dev-mode is cpu/gpu (not mixed/auto),
    // --parallel 1, and greedy sampling (temp<=0 or top_k==1). Otherwise the
    // engine transparently falls back to standard single-token decode.
    bool mtp = false;
    int spec_draft_n_max = 6;
    // HTTP server bind address & port. Defaults preserve legacy 0.0.0.0:6666;
    // override via --host / --port so multiple servers (e.g. one mini server
    // + one llama-server, or several mini servers for different dev-modes)
    // can coexist on the same host during testing.
    std::string http_host = "0.0.0.0";
    int http_port = 6666;
    int gpu_id = 0;
    // Populated by CLI parsers; unset fields may receive dev-mode defaults.
    CParamUserFlags user_flags;
};

extern int extractNumber(const std::string& str);
extern std::string getAfterNumber(const std::string& str);

// ggml's default logger prints GGML_LOG_DEBUG unconditionally; install a
// level-aware callback so CUDA graph reuse spam stays quiet unless needed.
void init_ggml_logging();
