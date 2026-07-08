#include "deltanet-decode-proj.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"

void ggml_cuda_op_deltanet_decode_proj(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * w_qkv   = dst->src[1];
    const ggml_tensor * w_gate  = dst->src[2];
    const ggml_tensor * w_beta  = dst->src[3];
    const ggml_tensor * w_alpha = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_quantized(w_qkv->type));
    GGML_ASSERT(ggml_is_quantized(w_gate->type));
    GGML_ASSERT(ggml_is_quantized(w_beta->type));
    GGML_ASSERT(ggml_is_quantized(w_alpha->type));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_embd  = x->ne[0];
    const int64_t n_qkv   = w_qkv->ne[1];
    const int64_t n_gate  = w_gate->ne[1];
    const int64_t n_beta  = w_beta->ne[1];
    const int64_t n_alpha = w_alpha->ne[1];

    GGML_ASSERT(w_qkv->ne[0] == n_embd);
    GGML_ASSERT(w_gate->ne[0] == n_embd);
    GGML_ASSERT(w_beta->ne[0] == n_embd);
    GGML_ASSERT(w_alpha->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[0] == n_qkv + n_gate + n_beta + n_alpha);
    GGML_ASSERT(x->ne[1] == 1 || ggml_is_vector(x));

    cudaStream_t stream = ctx.stream();

    const float * x_d = (const float *) x->data;
    float *       dst_d = (float *) dst->data;

    const int64_t ne10_padded = GGML_PAD(n_embd, MATRIX_ROW_PADDING);
    const int64_t s11 = x->nb[1] / sizeof(float);
    const int64_t s12 = x->nb[2] / sizeof(float);
    const int64_t s13 = x->nb[3] / sizeof(float);
    ggml_cuda_pool_alloc<char> x_q8_1(ctx.pool(), x->ne[1] * x->ne[2] * x->ne[3] * ne10_padded * sizeof(block_q8_1) / QK8_1);
    quantize_row_q8_1_cuda(
        x_d, nullptr, x_q8_1.get(), w_qkv->type,
        n_embd, s11, s12, s13,
        ne10_padded, x->ne[1], x->ne[2], x->ne[3], stream);

    float * out_qkv   = dst_d;
    float * out_gate  = dst_d + n_qkv;
    float * out_beta  = out_gate + n_gate;
    float * out_alpha = out_beta + n_beta;

    ggml_cuda_mul_mat_vec_q_preq(ctx, w_qkv,   x_q8_1.get(), ne10_padded, out_qkv,   stream);
    ggml_cuda_mul_mat_vec_q_preq(ctx, w_gate,  x_q8_1.get(), ne10_padded, out_gate,  stream);
    ggml_cuda_mul_mat_vec_q_preq(ctx, w_beta,  x_q8_1.get(), ne10_padded, out_beta,  stream);
    ggml_cuda_mul_mat_vec_q_preq(ctx, w_alpha, x_q8_1.get(), ne10_padded, out_alpha, stream);
}
