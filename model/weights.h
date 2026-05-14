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
#pragma once

#include <ggml.h>
#include <vector>
#include <cstdio>
#include <memory>
#include <string>
#include <set>
#include <unordered_map>
#include "common.h"

// 判断某层是否为 Attention 层（每 4 层一个，index % 4 == 3）
inline bool is_attn_layer(int idx) { return (idx % 4) == 3; }

class Qwen35moeLayer {
public:
    std::unordered_map<EN_WEIGHT_TYPE, ggml_tensor*> tensors;
};

class Qwen35moeWeights {
public:
    std::unordered_map<EN_WEIGHT_TYPE, ggml_tensor*> heads;
    std::map<int, std::shared_ptr<Qwen35moeLayer>, std::less<int>> layers;
};
