// 下面是使用Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf模型的参数：
// tensors:
//     token_embd.weight	[2048, 248320]	Q5_K
//     output.weight	[2048, 248320]	Q6_K
//     output_norm.weight	[2048]	F32
//     blk.*，*是0/1/2/4/5/6/8/9/10/12/13/14/17/18/20/21/22/24/25/26/28/29/30/32/33/34/36/37/38时是下面的参数：
//     blk
//         blk.0
//             blk.0.attn_gate.weight	[2048, 4096]	Q5_K
//             blk.0.attn_norm.weight	[2048]	F32
//             blk.0.attn_qkv.weight	[2048, 8192]	Q6_K
//             blk.0.ffn_down_exps.weight	[512, 2048, 256]	Q6_K
//             blk.0.ffn_down_shexp.weight	[512, 2048]	Q6_K
//             blk.0.ffn_gate_exps.weight	[2048, 512, 256]	Q5_K
//             blk.0.ffn_gate_inp.weight	[2048, 256]	F32
//             blk.0.ffn_gate_inp_shexp.weight	[2048]	F32
//             blk.0.ffn_gate_shexp.weight	[2048, 512]	Q5_K
//             blk.0.ffn_up_exps.weight	[2048, 512, 256]	Q5_K
//             blk.0.ffn_up_shexp.weight	[2048, 512]	Q5_K
//             blk.0.post_attention_norm.weight	[2048]	F32
//             blk.0.ssm_a	[32]	F32
//             blk.0.ssm_alpha.weight	[2048, 32]	Q5_K
//             blk.0.ssm_beta.weight	[2048, 32]	Q5_K
//             blk.0.ssm_conv1d.weight	[4, 8192]	F32
//             blk.0.ssm_dt.bias	[32]	F32
//             blk.0.ssm_norm.weight	[128]	F32
//             blk.0.ssm_out.weight	[4096, 2048]	Q5_K

//     blk.*，*是3/7/11/15/19/23/27/31/35/39时是下面的参数：
//     blk
//         blk.3
//             blk.3.attn_k.weight	[2048, 512]	Q5_K
//             blk.3.attn_k_norm.weight	[256]	F32
//             blk.3.attn_norm.weight	[2048]	F32
//             blk.3.attn_q.weight	[2048, 8192]	Q5_K
//             blk.3.attn_q_norm.weight	[256]	F32
//             blk.3.attn_v.weight	[2048, 512]	Q6_K
//             blk.3.ffn_down_exps.weight	[512, 2048, 256]	Q6_K
//             blk.3.ffn_down_shexp.weight	[512, 2048]	Q6_K
//             blk.3.ffn_gate_exps.weight	[2048, 512, 256]	Q5_K
//             blk.3.ffn_gate_inp.weight	[2048, 256]	F32
//             blk.3.ffn_gate_inp_shexp.weight	[2048]	F32
//             blk.3.ffn_gate_shexp.weight	[2048, 512]	Q5_K
//             blk.3.ffn_up_exps.weight	[2048, 512, 256]	Q5_K
//             blk.3.ffn_up_shexp.weight	[2048, 512]	Q5_K
//             blk.3.post_attention_norm.weight	[2048]	F32
//             blk.3.attn_output.weight	[4096, 2048]	Q5_K
#ifndef FUNASR_MODEL_WEIGHTS_HPP
#define FUNASR_MODEL_WEIGHTS_HPP

#include <ggml.h>
#include <vector>
#include <cstdio>

// 判断某层是否为 Attention 层（每 4 层一个，index % 4 == 3）
inline bool is_attn_layer(int idx) { return (idx % 4) == 3; }

// ============================================================
// 模型文件中每层的权重结构
// ============================================================
struct Qwen35moeLayer {
    struct ggml_tensor * attn_k          = nullptr;
    struct ggml_tensor * attn_k_norm     = nullptr;
    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * attn_gate       = nullptr;
    struct ggml_tensor * attn_qkv        = nullptr;
    struct ggml_tensor * attn_q          = nullptr;
    struct ggml_tensor * attn_q_norm     = nullptr;
    struct ggml_tensor * attn_v          = nullptr;
    
    struct ggml_tensor * ffn_down_exps     = nullptr;
    struct ggml_tensor * ffn_gate_exps     = nullptr;
    struct ggml_tensor * ffn_gate_inp      = nullptr;
    struct ggml_tensor * ffn_up_exps       = nullptr;
  
    struct ggml_tensor * ffn_down_shexp     = nullptr;
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp     = nullptr;
    struct ggml_tensor * ffn_up_shexp       = nullptr;

    struct ggml_tensor * ssm_a              = nullptr;
    struct ggml_tensor * ssm_alpha          = nullptr;
    struct ggml_tensor * ssm_beta           = nullptr;
    struct ggml_tensor * ssm_conv1d         = nullptr;
    struct ggml_tensor * ssm_dt_b           = nullptr;
    struct ggml_tensor * ssm_norm           = nullptr;
    struct ggml_tensor * ssm_out            = nullptr;
    
    struct ggml_tensor * post_attention_norm = nullptr;
    struct ggml_tensor * attn_output        = nullptr;
};

// ============================================================
// 完整权重
// ============================================================
struct Qwen35moeWeights {
    ggml_tensor* token_embd = nullptr;
    std::vector<Qwen35moeLayer> layers;
    ggml_tensor* output = nullptr;
    ggml_tensor* output_norm = nullptr;

    bool is_valid() const {
        bool layer_ok = true;
        size_t layer_count = layers.size();
        for (size_t i = 0; i < layer_count; i++) {
            const Qwen35moeLayer& lyr = layers[i];
            if (is_attn_layer(i)) {
                bool ok = lyr.attn_q && lyr.attn_k && lyr.attn_v && lyr.attn_output;
                if (!ok) layer_ok = false;
            } else {
                bool ok = lyr.attn_qkv && lyr.attn_gate && lyr.ssm_out;
                if (!ok) layer_ok = false;
            }
        }
        if (false == layer_ok) {
            printf("[Loader] ERROR: one or more layers are missing required tensors\n");
            return false;
        }

        return true;
    }
};

#endif // FUNASR_MODEL_WEIGHTS_HPP