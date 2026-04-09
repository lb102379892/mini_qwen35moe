// Updated inference.cpp to reshape norm weights and fix ggml_rope_ext calls.

if (lw.attn_q_norm) {
    q = ggml_rms_norm(ctx, q, eps);
    ggml_tensor* qn = ggml_reshape_3d(ctx, lw.attn_q_norm, head_dim, 1, n_q_heads);
    q = ggml_mul(ctx, q, qn);
}

if (lw.attn_k_norm) {
    k = ggml_rms_norm(ctx, k, eps);
    ggml_tensor* kn = ggml_reshape_3d(ctx, lw.attn_k_norm, head_dim, 1, n_kv_heads);
    k = ggml_mul(ctx, k, kn);
}

// Change ggml_rope_ext calls
int n_ctx = max_ctx_; // Updated to use max_ctx_ instead of (int)cfg_->context_length
