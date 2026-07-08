#include "common.cuh"
#include "ssm-conv-step.cuh"
#include "unary.cuh"

template<bool apply_silu, size_t d_conv>
static __global__ void ssm_conv_step_f32(
        const float * __restrict__ state,
        const float * __restrict__ x,
        const float * __restrict__ w,
        const int state_nb1,
        const int w_nb1,
        float * __restrict__ out_conv,
        float * __restrict__ out_state,
        const int64_t nr) {
    ggml_cuda_pdl_lc();
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;

    const int64_t row = static_cast<int64_t>(bidx) * blockDim.x + tid;
    if (row >= nr) {
        return;
    }

    const float * s = (const float *) ((const char *) state + row * state_nb1);
    const float * c = (const float *) ((const char *) w + row * w_nb1);
    const float   xv = x[row];

    float sumf = 0.0f;
#pragma unroll
    for (size_t j = 0; j < d_conv - 1; ++j) {
        sumf += s[j] * c[j];
    }
    sumf += xv * c[d_conv - 1];
    out_conv[row] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;

#pragma unroll
    for (size_t j = 0; j < d_conv - 2; ++j) {
        out_state[j + row * (d_conv - 1)] = s[j + 1];
    }
    out_state[(d_conv - 2) + row * (d_conv - 1)] = xv;
}

template<bool apply_silu>
static void ssm_conv_step_f32_cuda(
        const float * state,
        const float * x,
        const float * w,
        const int state_nb1,
        const int w_nb1,
        float * out_conv,
        float * out_state,
        const int64_t nc,
        const int64_t nr,
        cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch = [&](auto DC) {
        constexpr int kDC = decltype(DC)::value;
        const dim3 blocks((nr + threads - 1) / threads, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
        ggml_cuda_kernel_launch(ssm_conv_step_f32<apply_silu, kDC>, launch_params,
            state, x, w, state_nb1, w_nb1, out_conv, out_state, nr);
    };

    switch (nc) {
        case 3:  launch(std::integral_constant<int, 3 >{}); break;
        case 4:  launch(std::integral_constant<int, 4 >{}); break;
        case 5:  launch(std::integral_constant<int, 5 >{}); break;
        case 9:  launch(std::integral_constant<int, 9 >{}); break;
        case 15: launch(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("ssm_conv_step: unsupported kernel size");
    }
}

void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    const ggml_tensor * state = dst->src[0];
    const ggml_tensor * x     = dst->src[1];
    const ggml_tensor * w     = dst->src[2];
    const bool fuse_silu = silu_dst != nullptr;
    const ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = w->ne[0];
    const int64_t nr  = w->ne[1];

    GGML_ASSERT(state->ne[0] == nc - 1);
    GGML_ASSERT(state->ne[1] == nr);
    GGML_ASSERT(x->ne[0] == nr);
    GGML_ASSERT(dst->ne[0] == nr + (nc - 1) * nr);
    GGML_ASSERT(state->type == GGML_TYPE_F32);
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    GGML_ASSERT(state->nb[0] == sizeof(float));
    GGML_ASSERT(x->nb[0] == sizeof(float));
    GGML_ASSERT(w->nb[0] == sizeof(float));
    GGML_ASSERT(state->nb[1] == state->ne[0] * sizeof(float));

    const float * state_d = (const float *) state->data;
    const float * x_d     = (const float *) x->data;
    const float * w_d     = (const float *) w->data;
    float *       conv_d  = (float *) dst->data;
    float *       state_out_d = conv_d + nr;
    float *       out_d = (float *) out->data;

    cudaStream_t stream = ctx.stream();

    if (fuse_silu) {
        ssm_conv_step_f32_cuda<true>(state_d, x_d, w_d, (int) state->nb[1], (int) w->nb[1],
            out_d, state_out_d, nc, nr, stream);
    } else {
        ssm_conv_step_f32_cuda<false>(state_d, x_d, w_d, (int) state->nb[1], (int) w->nb[1],
            conv_d, state_out_d, nc, nr, stream);
    }
}
