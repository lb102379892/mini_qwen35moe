#include "compute/ops_cpu.hpp"
#include <cmath>
#include <cstdio>

ggml_tensor* build_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, float eps) const {
    cur = ggml_norm(ctx, x, eps);
    if (weight) {
        cur = ggml_mul(ctx, cur, weight);
    }

    if (bias) {
        cur = ggml_add(ctx, cur, bias);
    }

    return cur;
}

ggml_tensor* build_norm_rms(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, float eps) const {
    cur = ggml_rms_norm(ctx, x, eps);
    if (weight) {
        cur = ggml_mul(ctx, cur, weight);
    }

    if (bias) {
        cur = ggml_add(ctx, cur, bias);
    }

    return cur;
}

ggml_tensor* build_norm_gated(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weights, ggml_tensor* gate) {
    ggml_tensor* normalized = build_norm_rms(ctx, x, weights, nullptr);
    ggml_tensor* gated_silu = ggml_silu(ctx, gate);

    return ggml_mul(ctx, normalized, gated_silu);
}
