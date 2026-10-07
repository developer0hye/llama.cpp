#include "dsv41-fused.cuh"

// RMS_NORM + MUL_MAT (F32 weight, few rows). Replaces a single-block rms_norm over the 20480-wide hc stream
// (~5.7 us) and a 24-block mul_mat_vec_f (~5.5 us) per sublayer with one 24-block kernel (each block
// recomputes the sum of squares from L2; x is 80 KB). The norm scale is applied after the dot product
// instead of before, which changes rounding at the 1e-7 level.
template <int block_size>
static __global__ void dsv41_rms_mul_mat_f32(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ dst,
        const int ncols, const int64_t sx_t, const int64_t sw_r, const int64_t sd_r, const int64_t sd_t, const float eps) {
    const int r = blockIdx.x;
    const int t = blockIdx.y;
    const float4 * x4 = (const float4 *) (x + t*sx_t);
    const float4 * w4 = (const float4 *) (w + r*sw_r);
    float ss = 0.0f;
    float dt = 0.0f;
    for (int i = threadIdx.x; i < ncols/4; i += block_size) {
        const float4 xv = x4[i];
        const float4 wv = w4[i];
        ss += xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w;
        dt += xv.x*wv.x + xv.y*wv.y + xv.z*wv.z + xv.w*wv.w;
    }
    ss = warp_reduce_sum(ss);
    dt = warp_reduce_sum(dt);
    __shared__ float s_ss[32];
    __shared__ float s_dt[32];
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        s_ss[warp] = ss;
        s_dt[warp] = dt;
    }
    __syncthreads();
    if (warp == 0) {
        ss = lane < block_size/WARP_SIZE ? s_ss[lane] : 0.0f;
        dt = lane < block_size/WARP_SIZE ? s_dt[lane] : 0.0f;
        ss = warp_reduce_sum(ss);
        dt = warp_reduce_sum(dt);
        if (lane == 0) {
            const float scale = rsqrtf(ss / ncols + eps);
            dst[r*sd_r + t*sd_t] = dt * scale;
        }
    }
}

bool ggml_cuda_dsv41_can_fuse_rms_mul_mat(const ggml_tensor * rms, const ggml_tensor * mm) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_DSV41_RMS_MM");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!enabled || rms->op != GGML_OP_RMS_NORM || mm->op != GGML_OP_MUL_MAT || mm->src[1] != rms) {
        return false;
    }
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mm->src[0];
    return x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 && rms->type == GGML_TYPE_F32 &&
        w->ne[1] <= 64 && w->ne[0] >= 4096 && w->ne[0] % 4 == 0 && w->ne[0] == x->ne[0] &&
        x->ne[2] == 1 && x->ne[3] == 1 && w->ne[2] == 1 && w->ne[3] == 1 &&
        ggml_is_contiguous_rows(x) && x->nb[0] == sizeof(float) && x->nb[1] % 16 == 0 && ((uintptr_t) x->data % 16) == 0 &&
        ggml_is_contiguous(w) && ((uintptr_t) w->data % 16) == 0 && mm->nb[0] == sizeof(float);
}

void ggml_cuda_dsv41_rms_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, ggml_tensor * mm) {
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mm->src[0];
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    const int     ncols = (int) x->ne[0];
    const int64_t nt    = x->ne[1];
    const dim3 grid((unsigned) w->ne[1], (unsigned) nt, 1);
    constexpr int bs = 512;
    dsv41_rms_mul_mat_f32<bs><<<grid, bs, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (float *) mm->data,
            ncols, x->nb[1]/sizeof(float), w->nb[1]/sizeof(float), mm->nb[0]/sizeof(float), mm->nb[1]/sizeof(float), eps);
    CUDA_CHECK(cudaGetLastError());
}
