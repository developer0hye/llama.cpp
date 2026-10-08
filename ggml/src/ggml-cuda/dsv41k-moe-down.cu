// Fused MoE down projection for DeepSeek-V4.1-Flash under TP=8 (per-GPU slice: K = 2304/8 = 288).
//   out[t][r] = sum_{s=0..5} w[t][s] * (Wdown[ids[t][s]] (r,:) . act[t][s][:])  (+ Wsh(r,:) . act_sh[t][:])
// Routed experts: MXFP4 (ggml type 39), native ggml layout [n_expert][rows][K/32 blocks of 17 B].
// Shared expert (optional): Q8_0, native layout [rows][K/32 blocks of 34 B], weight 1.
// act: f32 (SiLU(gate)*up), quantized in-kernel to q8_1 exactly like ggml quantize_q8_1.
// n_tok 1..8 (decode / spec-verify batch). Experts used by several tokens are read and decoded ONCE.
//
// v3 layout: block = 64 output rows (80 blocks for 5120 rows ~ 1 per SM, so the act prologue is paid once per SM).
// Work entry = 2448 B: (routed expert, 16-row sub-tile: 144 MXFP4 blocks) or (shared, 8-row sub-tile: 72 Q8_0 blocks).
// Lane handles blocks lane + 32 i. Per (token, slot, row): fixed 9-term sum; slot sum in fixed order 0..5, + shared
// -> batch invariant (token t's result is bit-identical for any n_tok / co-batched tokens).
#include "dsv41k-tile.cuh"
#include "dsv41k-kernels.h"

namespace dsv41 {

constexpr int DN_RB = 64;    // rows per block
constexpr int DN_BPR = 9;    // K = 288

template <int NT, int NW>
__global__ void __launch_bounds__(NW * 32, 1)
moe_down_kernel(const uint8_t * __restrict__ W, size_t expert_stride, const uint8_t * __restrict__ Wsh,
                const float * __restrict__ act, const float * __restrict__ act_sh,
                const int32_t * __restrict__ ids, const int ids_stride, const float * __restrict__ wts,
                float * __restrict__ out, int n_tok, int n_used, int rows) {
    constexpr int NQ = NT * 7 * DN_BPR;    // q8_1 act blocks: routed [t][s][b], shared [t][b] at NT*6*9
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r0 = blockIdx.x * DN_RB;

    __shared__ int4 s_aq[NQ][2];
    __shared__ float s_ad[NQ];
    __shared__ float s_part[NW][160];
    __shared__ float s_res[NT][7][DN_RB];      // per (token, slot, row); slot 6 = shared
    __shared__ int8_t s_slot[NT * 6][NT];

    const int nsel = n_tok * n_used;
    // ids rows may be strided (ggml view of the router's top-k): element i = (token i/n_used, slot i%n_used)
    const int v0 = lane < nsel ? __ldg(ids + (lane / n_used) * ids_stride + lane % n_used) : -1;
    const int v1 = lane + 32 < nsel ? __ldg(ids + ((lane + 32) / n_used) * ids_stride + (lane + 32) % n_used) : -1;

    constexpr int MAXQ = (NQ + NW - 1) / NW;
    const int nq_r = n_tok * n_used * DN_BPR, nq = nq_r + (Wsh ? n_tok * DN_BPR : 0);
    float xv[MAXQ];
#pragma unroll
    for (int j = 0; j < MAXQ; ++j) {
        const int i = warp + NW * j;
        xv[j] = i < nq_r ? act[32 * i + lane] : (i < nq ? act_sh[32 * (i - nq_r) + lane] : 0.0f);
    }

    bool f0 = v0 >= 0, f1 = v1 >= 0;
    for (int k = 0; k < nsel; ++k) {
        const int vk = __shfl_sync(0xffffffff, k < 32 ? v0 : v1, k & 31);
        f0 &= !(vk == v0 && k < lane);
        f1 &= !(vk == v1 && k < lane + 32);
    }
    const unsigned b0 = __ballot_sync(0xffffffff, f0), b1 = __ballot_sync(0xffffffff, f1);
    const int nrouted = __popc(b0) + __popc(b1);
    const int nent = nrouted * 4 + (Wsh ? 8 : 0);
    auto entry_expert = [&](int uu) -> int {
        unsigned m = b0; int base = 0;
        if (uu >= __popc(b0)) { m = b1; uu -= __popc(b0); base = 32; }
        for (int k = 0; k < uu; ++k) m &= m - 1;
        const int pos = __ffs(m) - 1 + base;
        const int a0 = __shfl_sync(0xffffffff, v0, pos & 31), a1 = __shfl_sync(0xffffffff, v1, pos & 31);  // per-lane pos may differ
        return pos < 32 ? a0 : a1;
    };

#pragma unroll
    for (int j = 0; j < MAXQ; ++j) {
        const int i = warp + NW * j;
        if (i < nq) {
            float d;
            const int8_t q = quant_q8_1_lane(xv[j], d);
            const int slot = i < nq_r ? i : NT * 6 * DN_BPR + (i - nq_r);
            ((int8_t *) s_aq[slot])[lane] = q;
            if (lane == 0) s_ad[slot] = d;
        }
    }
    if (warp == 0) {
        for (int u0 = 0; u0 < nrouted; u0 += 32) {
            const int uu = u0 + lane;
            const int e = entry_expert(min(uu, nrouted - 1));
            if (uu < nrouted) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    int sl = -1;
                    if (t < n_tok) for (int k = 0; k < n_used; ++k) if (__ldg(ids + t * ids_stride + k) == e) sl = k;
                    s_slot[uu][t] = (int8_t) sl;
                }
            }
        }
    }

    // raw words: routed 5 blocks x 5 words (b < 144), shared 3 blocks x 9 words (b < 72)
    uint32_t rw[27];
    auto issue = [&](int en) {
        if (en < nrouted * 4) {
            const int e = entry_expert(en >> 2);
            const uint8_t * tile = W + (size_t) e * expert_stride + (size_t) (r0 + 16 * (en & 3)) * DN_BPR * 17;
#pragma unroll
            for (int i = 0; i < 5; ++i) {
                const int b = lane + 32 * i;
                const uint32_t * p = (const uint32_t *) (tile + ((b * 17) & ~3));
#pragma unroll
                for (int j = 0; j < 5; ++j) rw[5 * i + j] = b < 144 ? __ldg(p + j) : 0u;
            }
        } else if (en < nent) {
            const uint8_t * tile = Wsh + (size_t) (r0 + 8 * (en - nrouted * 4)) * DN_BPR * 34;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const int b = lane + 32 * i;
                const uint32_t * p = (const uint32_t *) (tile + ((b * 34) & ~3));
#pragma unroll
                for (int j = 0; j < 9; ++j) rw[9 * i + j] = b < 72 ? __ldg(p + j) : 0u;
            }
        }
    };
    int en = warp;
    issue(en);
    __syncthreads();

    int col[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) col[i] = (lane + 32 * i) % DN_BPR;

    for (; en < nent; en += NW) {
        if (en < nrouted * 4) {
            const int u = en >> 2, sub = en & 3;
            decblk db[5];
            const uint32_t s0 = 8 * (lane & 3), s1 = s0 + 8;
#pragma unroll
            for (int i = 0; i < 5; ++i) {
                const uint32_t ee = __funnelshift_rc(rw[5 * i], rw[5 * i + 1], s0) & 0xff;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int2 v = mxfp4_lut(__funnelshift_rc(rw[5 * i + l], rw[5 * i + l + 1], s1));
                    db[i].v[l] = v.x; db[i].v[l + 4] = v.y;
                }
                db[i].d = e8m0_half_fast(ee);
            }
            if (en + NW < nent) issue(en + NW);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t >= n_tok) break;
                const int sl = s_slot[u][t];
                if (sl < 0) continue;
                const int ab = (t * 6 + sl) * DN_BPR;
#pragma unroll
                for (int i = 0; i < 5; ++i) {
                    const int b = lane + 32 * i;
                    if (b < 144) s_part[warp][b] = dot_blk(db[i], (const int *) s_aq[ab + col[i]], s_ad[ab + col[i]]);
                }
                __syncwarp();
                if (lane < 16) {
                    float rs = 0.0f;
#pragma unroll
                    for (int c = 0; c < DN_BPR; ++c) rs += s_part[warp][lane * DN_BPR + c];
                    s_res[t][sl][16 * sub + lane] = rs;
                }
                __syncwarp();
            }
        } else {
            const int sub = en - nrouted * 4;
            decblk db[3];
            const uint32_t sq = (lane & 1) ? 2 : 0, s2 = 8 * (sq + 2);
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                db[i].d = __half2float(__ushort_as_half((unsigned short) ((sq ? (rw[9 * i] >> 16) : rw[9 * i]) & 0xffff)));
#pragma unroll
                for (int l = 0; l < 8; ++l) db[i].v[l] = (int) __funnelshift_rc(rw[9 * i + l], rw[9 * i + l + 1], s2);
            }
            if (en + NW < nent) issue(en + NW);
            for (int t = 0; t < n_tok; ++t) {
                const int ab = NT * 6 * DN_BPR + t * DN_BPR;
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    const int b = lane + 32 * i;
                    if (b < 72) s_part[warp][b] = dot_blk(db[i], (const int *) s_aq[ab + col[i]], s_ad[ab + col[i]]);
                }
                __syncwarp();
                if (lane < 8) {
                    float rs = 0.0f;
#pragma unroll
                    for (int c = 0; c < DN_BPR; ++c) rs += s_part[warp][lane * DN_BPR + c];
                    s_res[t][6][8 * sub + lane] = rs;
                }
                __syncwarp();
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_tok * DN_RB; i += NW * 32) {
        const int t = i / DN_RB, r = i % DN_RB;
        float s = 0.0f;
        for (int sl = 0; sl < n_used; ++sl) s += __ldg(wts + t * n_used + sl) * s_res[t][sl][r];
        if (Wsh) s += s_res[t][6][r];
        out[(size_t) t * rows + r0 + r] = s;
    }
}

template <int NT, int NW>
static void launch_down(const dsv41_moe_down_args & a, cudaStream_t st) {
    moe_down_kernel<NT, NW><<<a.rows / DN_RB, NW * 32, 0, st>>>(
        (const uint8_t *) a.w, a.expert_stride, (const uint8_t *) a.w_shared, a.act, a.act_shared, a.ids, a.ids_stride > 0 ? a.ids_stride : a.n_used, a.weights,
        a.out, a.n_tok, a.n_used, a.rows);
}

}  // namespace dsv41

void dsv41_moe_down(const dsv41_moe_down_args & a, cudaStream_t st) {
    using namespace dsv41;
    if (a.n_used != 6 || a.rows % DN_RB || a.K != 288 || a.expert_stride % 16) { fprintf(stderr, "moe_down: unsupported shape\n"); abort(); }
    switch (a.n_tok) {
        case 1: launch_down<1, 16>(a, st); break;
        case 2: launch_down<2, 16>(a, st); break;
        case 3: case 4: launch_down<4, 16>(a, st); break;
        case 5: case 6: case 7: case 8: launch_down<8, 16>(a, st); break;
        default: fprintf(stderr, "moe_down: n_tok must be 1..8\n"); abort();
    }
}
