// funasr/compute/llm_ops.hpp
// LLM Decoder 前向计算 (Qwen3-0.6B with GQA + KV Cache)
//
// 数据流:
//   hidden_states [1024, seq_len]
//     → 28 × Transformer Layer:
//         RMSNorm → GQA (16Q/8KV, RoPE, CausalMask, KVCache) → Residual
//         RMSNorm → SwiGLU MLP (1024→3072→1024) → Residual
//     → Final RMSNorm
//     → LM Head (1024 → 151936)
//     → logits [151936, seq_len]
//
#ifndef FUNASR_COMPUTE_LLM_OPS_HPP
#define FUNASR_COMPUTE_LLM_OPS_HPP

#include <ggml.h>
#include "core/config.hpp"
#include "model/weights.hpp"

ggml_tensor* build_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, float eps = 1e-5f) const;
ggml_tensor* build_norm_rms(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, float eps = 1e-5f) const;
ggml_tensor* llm_build_qwen35moe::build_norm_gated(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weights, ggml_tensor* gate);

#endif // FUNASR_COMPUTE_LLM_OPS_HPP