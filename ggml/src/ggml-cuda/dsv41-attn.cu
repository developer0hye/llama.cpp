#include "dsv41-attn.cuh"

// DeepSeek-V4.1 decode attention over the sliding-window latent rows and the top-k picked compressed rows
// (see ggml_dsv41_attn_decode). Replaces, per layer and token: concat of the (padded) raw + compressed K into
// a new tensor, the top-k mask build (fill + set_rows + add + concat) and flash_attn_ext with its mask ->
// sparse-index conversion and stream-k fixup (~27 us on TP8). Cost here does not depend on the compressed
// stream length or on the n_kv padding.
//
// Kernel 1: one block per (chunk of CH rows, group of 8 heads, token): the chunk's K rows (== V) are staged
// in shared memory once and shared by the 8 warps (one head each, K is a single latent head); online softmax
// per warp, partial (max, sum, acc[D]) written out. Kernel 2: per (head, token) merge of the chunk partials
// with the head's value-less sink logit.

#define DSV41_ATTN_D      512
#define DSV41_ATTN_CH     16
#define DSV41_ATTN_WARPS  8

struct dsv41_attn_args {
    const float * q;  int64_t q_nb1, q_nb2, q_nb3;                 // bytes
    const half  * raw_k; int64_t rk_nb2, rk_nb3; int n_raw;
    const half  * raw_m; int64_t rm_nb0, rm_nb1, rm_nb3;
    const half  * comp_k; int64_t ck_nb2, ck_nb3;
    const half  * comp_m; int64_t cm_nb0, cm_nb1, cm_nb3;
    const int32_t * top_k; int64_t tk_nb0, tk_nb1, tk_nb3; int n_top;
    const float * sinks;
    float * dst; int64_t d_nb1, d_nb2, d_nb3;
    float * part;                                                   // [NT*NS][H][n_chunk][D+4] (16 B aligned rows)
    int H, NT, n_chunk;
    float scale;
};

static __global__ void __launch_bounds__(DSV41_ATTN_WARPS*WARP_SIZE)
dsv41_attn_partial(const dsv41_attn_args a) {
    constexpr int D = DSV41_ATTN_D, CH = DSV41_ATTN_CH, PL = D/WARP_SIZE;   // 16 values per lane
    __shared__ __align__(16) half s_k[CH][D];
    __shared__ float s_m[CH];

    const int chunk = blockIdx.x;
    const int tk    = blockIdx.z;                 // token*NS + seq
    const int t     = tk % a.NT;
    const int sq    = tk / a.NT;
    const int warp  = threadIdx.x / WARP_SIZE;
    const int lane  = threadIdx.x % WARP_SIZE;
    const int h     = blockIdx.y*DSV41_ATTN_WARPS + warp;
    const int r0    = chunk*CH;
    const int n_rows = a.n_raw + a.n_top;

    // stage the chunk: masks first, then the rows that are visible
    if (threadIdx.x < CH) {
        const int r = r0 + threadIdx.x;
        float mv = -INFINITY;
        if (r < a.n_raw) {
            mv = __half2float(*(const half *) ((const char *) a.raw_m + r*a.rm_nb0 + t*a.rm_nb1 + sq*a.rm_nb3));
        } else if (r < n_rows) {
            const int idx = *(const int32_t *) ((const char *) a.top_k + (r - a.n_raw)*a.tk_nb0 + t*a.tk_nb1 + sq*a.tk_nb3);
            mv = __half2float(*(const half *) ((const char *) a.comp_m + idx*a.cm_nb0 + t*a.cm_nb1 + sq*a.cm_nb3));
        }
        s_m[threadIdx.x] = mv;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < CH*(D/8); i += blockDim.x) {
        const int rr = i / (D/8);
        const int c  = i % (D/8);
        const int r  = r0 + rr;
        if (s_m[rr] == -INFINITY) {
            continue;
        }
        const half * src;
        if (r < a.n_raw) {
            src = (const half *) ((const char *) a.raw_k + r*a.rk_nb2 + sq*a.rk_nb3);
        } else {
            const int idx = *(const int32_t *) ((const char *) a.top_k + (r - a.n_raw)*a.tk_nb0 + t*a.tk_nb1 + sq*a.tk_nb3);
            src = (const half *) ((const char *) a.comp_k + idx*a.ck_nb2 + sq*a.ck_nb3);
        }
        ((uint4 *) s_k[rr])[c] = ((const uint4 *) src)[c];
    }
    __syncthreads();

    if (h >= a.H) {
        return;
    }
    const float * qv = (const float *) ((const char *) a.q + h*a.q_nb1 + t*a.q_nb2 + sq*a.q_nb3);
    float qr[PL];
#pragma unroll
    for (int i = 0; i < PL; i += 4) {
        const float4 v = *(const float4 *) (qv + lane*PL + i);
        qr[i] = v.x; qr[i+1] = v.y; qr[i+2] = v.z; qr[i+3] = v.w;
    }
    float m = -INFINITY, l = 0.0f, acc[PL];
#pragma unroll
    for (int i = 0; i < PL; ++i) {
        acc[i] = 0.0f;
    }
    for (int rr = 0; rr < CH; ++rr) {
        const float mv = s_m[rr];
        if (mv == -INFINITY) {
            continue;
        }
        float kr[PL];
        const half2 * k2 = (const half2 *) (s_k[rr] + lane*PL);
#pragma unroll
        for (int i = 0; i < PL/2; ++i) {
            const float2 f = __half22float2(k2[i]);
            kr[2*i] = f.x; kr[2*i+1] = f.y;
        }
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            d += qr[i]*kr[i];
        }
        d = warp_reduce_sum(d);
        const float s     = d*a.scale + mv;
        const float m_new = fmaxf(m, s);
        const float corr  = expf(m - m_new);
        const float p     = expf(s - m_new);
        l = l*corr + p;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            acc[i] = acc[i]*corr + p*kr[i];
        }
        m = m_new;
    }
    float * pp = a.part + (((int64_t) tk*a.H + h)*a.n_chunk + chunk)*(D + 4);
#pragma unroll
    for (int i = 0; i < PL; i += 4) {
        *(float4 *) (pp + lane*PL + i) = make_float4(acc[i], acc[i+1], acc[i+2], acc[i+3]);
    }
    if (lane == 0) {
        pp[D]   = m;
        pp[D+1] = l;
    }
}

// one block per (head, token): chunk weights are computed once in shared memory, then every thread merges
// one output dimension over the chunks that saw a visible row
static __global__ void __launch_bounds__(DSV41_ATTN_D)
dsv41_attn_combine(const dsv41_attn_args a) {
    constexpr int D = DSV41_ATTN_D;
    constexpr int MAXC = 1024;
    __shared__ float s_w[MAXC];
    __shared__ float s_red[32];
    __shared__ float s_M, s_L;
    const int h  = blockIdx.x;
    const int tk = blockIdx.y;
    const int t  = tk % a.NT;
    const int sq = tk / a.NT;
    const float * pp = a.part + ((int64_t) tk*a.H + h)*a.n_chunk*(D + 4);
    const float sink = a.sinks ? a.sinks[h] : -INFINITY;
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;

    // max over chunk maxima and the sink
    float mx = sink;
    for (int c = threadIdx.x; c < a.n_chunk; c += blockDim.x) {
        mx = fmaxf(mx, pp[c*(D + 4) + D]);
    }
    mx = warp_reduce_max(mx);
    if (lane == 0) { s_red[warp] = mx; }
    __syncthreads();
    if (warp == 0) {
        mx = lane < (int) (blockDim.x/WARP_SIZE) ? s_red[lane] : -INFINITY;
        mx = warp_reduce_max(mx);
        if (lane == 0) { s_M = mx; }
    }
    __syncthreads();
    const float M = s_M;
    // chunk weights and the normaliser
    float lsum = 0.0f;
    for (int c = threadIdx.x; c < a.n_chunk; c += blockDim.x) {
        const float mc = pp[c*(D + 4) + D];
        const float w  = mc == -INFINITY ? 0.0f : expf(mc - M);
        s_w[c] = w;
        lsum += w*pp[c*(D + 4) + D + 1];
    }
    lsum = warp_reduce_sum(lsum);
    __syncthreads();
    if (lane == 0) { s_red[warp] = lsum; }
    __syncthreads();
    if (warp == 0) {
        lsum = lane < (int) (blockDim.x/WARP_SIZE) ? s_red[lane] : 0.0f;
        lsum = warp_reduce_sum(lsum);
        if (lane == 0) { s_L = lsum + (M == -INFINITY ? 0.0f : expf(sink - M)); }
    }
    __syncthreads();
    const float L  = s_L;
    const float iL = L > 0.0f ? 1.0f/L : 0.0f;
    float * o = (float *) ((char *) a.dst + h*a.d_nb1 + t*a.d_nb2 + sq*a.d_nb3);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = 0.0f;
        for (int c = 0; c < a.n_chunk; ++c) {
            const float w = s_w[c];
            if (w != 0.0f) {
                v += pp[c*(D + 4) + i]*w;
            }
        }
        o[i] = v*iL;
    }
}

bool ggml_cuda_dsv41_attn_decode_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    return q->ne[0] == DSV41_ATTN_D && q->nb[0] == sizeof(float) && q->nb[1] % 16 == 0 &&
        dst->src[1]->nb[0] == sizeof(half) && dst->src[3]->nb[0] == sizeof(half) &&
        dst->src[1]->nb[2] % 16 == 0 && dst->src[3]->nb[2] % 16 == 0 && dst->nb[0] == sizeof(float);
}

void ggml_cuda_op_dsv41_attn_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    dsv41_attn_args a;
    a.q = (const float *) q->data; a.q_nb1 = q->nb[1]; a.q_nb2 = q->nb[2]; a.q_nb3 = q->nb[3];
    const ggml_tensor * rk = dst->src[1]; a.raw_k = (const half *) rk->data; a.rk_nb2 = rk->nb[2]; a.rk_nb3 = rk->nb[3]; a.n_raw = (int) rk->ne[2];
    const ggml_tensor * rm = dst->src[2]; a.raw_m = (const half *) rm->data; a.rm_nb0 = rm->nb[0]; a.rm_nb1 = rm->nb[1]; a.rm_nb3 = rm->nb[3];
    const ggml_tensor * ck = dst->src[3]; a.comp_k = (const half *) ck->data; a.ck_nb2 = ck->nb[2]; a.ck_nb3 = ck->nb[3];
    const ggml_tensor * cm = dst->src[4]; a.comp_m = (const half *) cm->data; a.cm_nb0 = cm->nb[0]; a.cm_nb1 = cm->nb[1]; a.cm_nb3 = cm->nb[3];
    const ggml_tensor * tk = dst->src[5]; a.top_k = (const int32_t *) tk->data; a.tk_nb0 = tk->nb[0]; a.tk_nb1 = tk->nb[1]; a.tk_nb3 = tk->nb[3]; a.n_top = (int) tk->ne[0];
    a.sinks = dst->src[6] ? (const float *) dst->src[6]->data : nullptr;
    a.dst = (float *) dst->data; a.d_nb1 = dst->nb[1]; a.d_nb2 = dst->nb[2]; a.d_nb3 = dst->nb[3];
    a.H = (int) q->ne[1]; a.NT = (int) q->ne[2];
    const int NS = (int) q->ne[3];
    a.n_chunk = (a.n_raw + a.n_top + DSV41_ATTN_CH - 1) / DSV41_ATTN_CH;
    a.scale = ggml_get_op_params_f32(dst, 0);
    {
        static int n_dbg = getenv("DSV41_ATTN_DEBUG") ? 4 : 0;
        if (n_dbg > 0) {
            n_dbg--;
            GGML_LOG_INFO("dsv41_attn_decode: H=%d NT=%d NS=%d n_raw=%d n_top=%d n_chunk=%d\n", a.H, a.NT, NS, a.n_raw, a.n_top, a.n_chunk);
        }
    }

    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) a.NT*NS*a.H*a.n_chunk*(DSV41_ATTN_D + 4));
    a.part = part.get();

    const dim3 g1(a.n_chunk, (a.H + DSV41_ATTN_WARPS - 1)/DSV41_ATTN_WARPS, a.NT*NS);
    dsv41_attn_partial<<<g1, DSV41_ATTN_WARPS*WARP_SIZE, 0, ctx.stream()>>>(a);
    const dim3 g2(a.H, a.NT*NS, 1);
    GGML_ASSERT(a.n_chunk <= 1024);
    dsv41_attn_combine<<<g2, DSV41_ATTN_D, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}
