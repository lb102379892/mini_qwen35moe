// compute/ops_cpu.hpp
// Primitive ggml graph-building helpers for Qwen3.5 MoE inference.
//
#ifndef FUNASR_COMPUTE_OPS_CPU_HPP
#define FUNASR_COMPUTE_OPS_CPU_HPP

#include <ggml.h>

// RMSNorm: rms_norm(x) * weight
ggml_tensor* ops_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps = 1e-6f);

// LayerNorm: norm(x) * weight + bias  (bias may be nullptr)
ggml_tensor* ops_layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias = nullptr, float eps = 1e-5f);

// SwiGLU FFN: down( silu(gate @ x) * up @ x )
ggml_tensor* ops_swiglu_ffn(ggml_context* ctx, ggml_tensor* x,
                             ggml_tensor* w_gate, ggml_tensor* w_up, ggml_tensor* w_down);

#endif // FUNASR_COMPUTE_OPS_CPU_HPP