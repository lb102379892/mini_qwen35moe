#include "compute/ops_cpu.hpp"
#include <cmath>

ggml_tensor* ops_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    ggml_tensor* cur = ggml_rms_norm(ctx, x, eps);
    if (weight) {
        cur = ggml_mul(ctx, cur, weight);
    }
    return cur;
}

ggml_tensor* ops_layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, float eps) {
    ggml_tensor* cur = ggml_norm(ctx, x, eps);
    if (weight) {
        cur = ggml_mul(ctx, cur, weight);
    }
    if (bias) {
        cur = ggml_add(ctx, cur, bias);
    }
    return cur;
}

ggml_tensor* ops_swiglu_ffn(ggml_context* ctx, ggml_tensor* x,
                              ggml_tensor* w_gate, ggml_tensor* w_up, ggml_tensor* w_down) {
    ggml_tensor* gate   = ggml_mul_mat(ctx, w_gate, x);
    ggml_tensor* up     = ggml_mul_mat(ctx, w_up,   x);
    ggml_tensor* hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    return ggml_mul_mat(ctx, w_down, hidden);
}
