// Lane-per-quant-block tile engine on the NATIVE ggml layouts (no load-time repack).
// A "tile" is R consecutive rows of one weight matrix = R*BPR contiguous quant blocks.
// Lane l handles blocks l, l+32, ... of the tile. Each block is fetched with aligned 32-bit __ldg loads
// covering its (unaligned) byte range and realigned with funnel shifts, so a warp instruction touches one
// contiguous ~32-block span (good coalescing) without any shared-memory staging.
#pragma once
#include "dsv41k-common.cuh"

namespace dsv41 {

// optional per-warp phase timestamps (compile with -DDSV41_PROF): g_prof[warp_global][k]
#ifdef DSV41_PROF
__device__ unsigned long long g_prof[1 << 14][4];
static __device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define PROF_MARK(k) do { if ((threadIdx.x & 31) == 0) g_prof[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32][k] = gtime(); } while (0)
#else
#define PROF_MARK(k) do {} while (0)
#endif

enum wtype : int { W_Q8_0 = 8, W_MXFP4 = 39 };

template <int T> struct wtraits;
template <> struct wtraits<W_MXFP4> { static constexpr int bytes = 17; static constexpr int nwords = 5; };  // s+17 <= 20
template <> struct wtraits<W_Q8_0>  { static constexpr int bytes = 34; static constexpr int nwords = 9; };  // s in {0,2}, s+34 <= 36

static __device__ __forceinline__ float e8m0_half_fast(uint32_t e) {
    // == ggml_cuda_e8m0_to_fp32(e) * 0.5f bit-exactly (ggml's CUDART<12.8 path); avoids the cvt library CALL on sm_86
    return __uint_as_float(e == 0 ? 0x00400000u : (e << 23)) * 0.5f;
}

// raw words of one block (issued early so many loads are in flight)
template <int T> struct rawblk { uint32_t w[wtraits<T>::nwords]; uint32_t s; };

template <int T>
static __device__ __forceinline__ void load_blk(rawblk<T> & r, const uint8_t * base, size_t blk) {
    const size_t off = blk * wtraits<T>::bytes;
    const uint32_t * p = (const uint32_t *) (base + (off & ~(size_t) 3));
    r.s = (uint32_t) (off & 3);
#pragma unroll
    for (int i = 0; i < wtraits<T>::nwords; ++i) r.w[i] = __ldg(p + i);
}

// decoded block: 8 ints of int8 weights in element order 0..31 (ready for dp4a with q8_1 qs) + float weight scale
struct decblk { int v[8]; float d; };

static __device__ __forceinline__ void decode(const rawblk<W_MXFP4> & r, decblk & o) {
    const uint32_t e = __funnelshift_rc(r.w[0], r.w[1], 8 * r.s) & 0xff;
    const uint32_t s1 = 8 * (r.s + 1);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t q4 = __funnelshift_rc(r.w[l], r.w[l + 1], s1);
        const int2 v = mxfp4_lut(q4);
        o.v[l] = v.x;      // elements 4l..4l+3      (low nibbles)
        o.v[l + 4] = v.y;  // elements 16+4l..16+4l+3 (high nibbles)
    }
    o.d = e8m0_half_fast(e);
}

static __device__ __forceinline__ void decode(const rawblk<W_Q8_0> & r, decblk & o) {
    const uint32_t dh = (r.s ? (r.w[0] >> 16) : r.w[0]) & 0xffff;
    const uint32_t s2 = 8 * (r.s + 2);
#pragma unroll
    for (int l = 0; l < 8; ++l) o.v[l] = (int) __funnelshift_rc(r.w[l], r.w[l + 1], s2);
    o.d = __half2float(__ushort_as_half((unsigned short) dh));
}

static __device__ __forceinline__ float dot_blk(const decblk & w, const int * aq, float ad) {
    const int4 a0 = *(const int4 *) &aq[0];
    const int4 a1 = *(const int4 *) &aq[4];
    int sumi = 0;
    sumi = __dp4a(w.v[0], a0.x, sumi); sumi = __dp4a(w.v[1], a0.y, sumi);
    sumi = __dp4a(w.v[2], a0.z, sumi); sumi = __dp4a(w.v[3], a0.w, sumi);
    sumi = __dp4a(w.v[4], a1.x, sumi); sumi = __dp4a(w.v[5], a1.y, sumi);
    sumi = __dp4a(w.v[6], a1.z, sumi); sumi = __dp4a(w.v[7], a1.w, sumi);
    return (w.d * ad) * (float) sumi;
}

// q8_1 activation block held in shared memory: 32 int8 + float d (d already rounded through half like ggml)
struct __align__(16) actblk { int q[8]; float d; float pad[3]; };

static __device__ __forceinline__ float dot_blk(const decblk & w, const actblk & a) {
    return dot_blk(w, a.q, a.d);
}

// Warp quantizes nblk q8_1 blocks of f32 input (block i = x[32*i .. 32*i+31]) into smem, ggml-exact.
static __device__ __forceinline__ void warp_quantize(const float * __restrict__ x, actblk * dst, int nblk, int warp, int nwarps, int lane) {
    for (int i = warp; i < nblk; i += nwarps) {
        const float xi = x[32 * i + lane];
        float d;
        const int8_t q = quant_q8_1_lane(xi, d);
        ((int8_t *) dst[i].q)[lane] = q;
        if (lane == 0) dst[i].d = d;
    }
}


// ======================================================================================================
// 128-bit coalesced fetch of native ggml blocks + in-register realignment (warp shuffles + barrel shift)
// ======================================================================================================
// MXFP4: a group = 16 consecutive blocks = 272 B = 17 x 16 B chunks (group start 16-B aligned).
//   Half-warp handles one group; lane j = lane & 15 -> block j. Lane loads chunk j; lane 15 also chunk 16.
// Q8_0:  a group = 8 consecutive blocks = 272 B = 17 chunks. Quarter-warp handles one group; lane j = lane & 7
//   -> block j. Lane loads chunks 2j, 2j+1; lane 7 also chunk 16.
// Both: the block's bytes are completed with the next lane's first chunk (shfl_down 1).
struct mx_raw { uint4 c, x; };
struct q8_raw { uint4 a, b, x; };

static __device__ __forceinline__ uint4 ldg_u4(const uint8_t * p) { return __ldg((const uint4 *) p); }

static __device__ __forceinline__ void mx_issue(mx_raw & r, const uint8_t * gbase, int j) {
    r.c = ldg_u4(gbase + 16 * j);
    if (j == 15) r.x = ldg_u4(gbase + 256);
}
static __device__ __forceinline__ void q8_issue(q8_raw & r, const uint8_t * gbase, int j) {
    r.a = ldg_u4(gbase + 32 * j);
    r.b = ldg_u4(gbase + 32 * j + 16);
    if (j == 7) r.x = ldg_u4(gbase + 256);
}

// must be called by the full warp (shuffles)
static __device__ __forceinline__ void mx_decode(const mx_raw & r, int j, decblk & o) {
    uint32_t n0 = __shfl_down_sync(0xffffffff, r.c.x, 1), n1 = __shfl_down_sync(0xffffffff, r.c.y, 1),
             n2 = __shfl_down_sync(0xffffffff, r.c.z, 1), n3 = __shfl_down_sync(0xffffffff, r.c.w, 1);
    if (j == 15) { n0 = r.x.x; n1 = r.x.y; n2 = r.x.z; n3 = r.x.w; }
    uint32_t W[8] = {r.c.x, r.c.y, r.c.z, r.c.w, n0, n1, n2, n3};
    // block starts at byte j: rotate the window left by A = j/4 words (2-stage barrel), keep R[0..4]
    const int A = j >> 2;
    uint32_t T[6], R[5];
#pragma unroll
    for (int i = 0; i < 6; ++i) T[i] = (A & 2) ? W[i + 2] : W[i];
#pragma unroll
    for (int i = 0; i < 5; ++i) R[i] = (A & 1) ? T[i + 1] : T[i];
    const uint32_t sb = 8 * (j & 3);
    const uint32_t e = (R[0] >> sb) & 0xff;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = mxfp4_lut(__funnelshift_rc(R[l], R[l + 1], sb + 8));
        o.v[l] = v.x; o.v[l + 4] = v.y;
    }
    o.d = e8m0_half_fast(e);
}

static __device__ __forceinline__ void q8_decode(const q8_raw & r, int j, decblk & o) {
    uint32_t n0 = __shfl_down_sync(0xffffffff, r.a.x, 1), n1 = __shfl_down_sync(0xffffffff, r.a.y, 1),
             n2 = __shfl_down_sync(0xffffffff, r.a.z, 1), n3 = __shfl_down_sync(0xffffffff, r.a.w, 1);
    if (j == 7) { n0 = r.x.x; n1 = r.x.y; n2 = r.x.z; n3 = r.x.w; }
    uint32_t W[12] = {r.a.x, r.a.y, r.a.z, r.a.w, r.b.x, r.b.y, r.b.z, r.b.w, n0, n1, n2, n3};
    // block starts at byte 2j: scale word A = j/2; rotate left by A (2-stage), keep R[0..9]
    const int A = j >> 1;
    uint32_t T[11], R[10];
#pragma unroll
    for (int i = 0; i < 10; ++i) T[i] = (A & 2) ? W[i + 2] : W[i];
    T[10] = (A & 2) ? 0u : W[10];
#pragma unroll
    for (int i = 0; i < 10; ++i) R[i] = (A & 1) ? T[i + 1] : T[i];
    const bool odd = j & 1;
    o.d = __half2float(__ushort_as_half((unsigned short) (odd ? (R[0] >> 16) : (R[0] & 0xffff))));
#pragma unroll
    for (int l = 0; l < 8; ++l) o.v[l] = (int) (odd ? R[l + 1] : __funnelshift_r(R[l], R[l + 1], 16));
}



// ------------------------------------------------------------------------------------------------------
// "Stream" mapping (no extra loads, no wasted registers): a row (or any 16-B aligned run of whole groups)
// is read as a stream of units; lane l of slot i holds unit u = l + 32 i.
//   MXFP4: unit = 16 B chunk; 16 blocks per 17 units; unit u (k = u % 17 < 16) starts block 16*(u/17) + k
//          at byte offset k.  Window = own unit + next unit (lane l+1, or lane 0 of slot i+1 for lane 31).
//   Q8_0:  unit = 32 B (2 chunks); 16 blocks per 17 units; unit u (k < 16) starts block 16*(u/17) + k at
//          byte offset 2k.  Window = own 32 B + next unit's 32 B.
// k == 16 units start no block (lane idles in that slot: 1/17 waste).
// The "next" value for lane 31 comes from slot i+1, so a slot can be refilled only after slot i-1 is decoded.
static __device__ __forceinline__ uint32_t next_unit_word(uint32_t cur, uint32_t nxt, int lane) {
    const uint32_t p = lane == 0 ? nxt : cur;   // lane 31 reads lane 0's next-slot word; others read lane+1's cur
    return __shfl_sync(0xffffffff, p, (lane + 1) & 31);
}

static __device__ __forceinline__ void mxs_decode(const uint4 cur, const uint4 nxt, int lane, int k, decblk & o) {
    const uint32_t W[8] = {cur.x, cur.y, cur.z, cur.w,
                           next_unit_word(cur.x, nxt.x, lane), next_unit_word(cur.y, nxt.y, lane),
                           next_unit_word(cur.z, nxt.z, lane), next_unit_word(cur.w, nxt.w, lane)};
    const int A = (k >> 2) & 3;
    uint32_t T[6], R[5];
#pragma unroll
    for (int i = 0; i < 6; ++i) T[i] = (A & 2) ? W[i + 2] : W[i];
#pragma unroll
    for (int i = 0; i < 5; ++i) R[i] = (A & 1) ? T[i + 1] : T[i];
    const uint32_t sb = 8 * (k & 3);
    const uint32_t e = (R[0] >> sb) & 0xff;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = mxfp4_lut(__funnelshift_rc(R[l], R[l + 1], sb + 8));
        o.v[l] = v.x; o.v[l + 4] = v.y;
    }
    o.d = e8m0_half_fast(e);
}

struct u8x32 { uint4 a, b; };
static __device__ __forceinline__ void q8s_decode(const u8x32 cur, const u8x32 nxt, int lane, int k, decblk & o) {
    const uint32_t W[16] = {cur.a.x, cur.a.y, cur.a.z, cur.a.w, cur.b.x, cur.b.y, cur.b.z, cur.b.w,
                            next_unit_word(cur.a.x, nxt.a.x, lane), next_unit_word(cur.a.y, nxt.a.y, lane),
                            next_unit_word(cur.a.z, nxt.a.z, lane), next_unit_word(cur.a.w, nxt.a.w, lane),
                            next_unit_word(cur.b.x, nxt.b.x, lane), next_unit_word(cur.b.y, nxt.b.y, lane),
                            next_unit_word(cur.b.z, nxt.b.z, lane), next_unit_word(cur.b.w, nxt.b.w, lane)};
    const int A = (k >> 1) & 7;   // block starts at byte 2k: word k/2
    uint32_t T[12], U[10], R[9];
#pragma unroll
    for (int i = 0; i < 12; ++i) T[i] = (A & 4) ? W[i + 4] : W[i];
#pragma unroll
    for (int i = 0; i < 10; ++i) U[i] = (A & 2) ? T[i + 2] : T[i];
#pragma unroll
    for (int i = 0; i < 9; ++i) R[i] = (A & 1) ? U[i + 1] : U[i];
    const bool odd = k & 1;
    o.d = __half2float(__ushort_as_half((unsigned short) (odd ? (R[0] >> 16) : (R[0] & 0xffff))));
#pragma unroll
    for (int l = 0; l < 8; ++l) o.v[l] = (int) (odd ? R[l + 1] : __funnelshift_r(R[l], R[l + 1], 16));
}

// Fast ggml-exact q8_1 quantization: 8 lanes per q8 block, 4 consecutive values per lane (float4 load),
// amax over the 8-lane group with 3 xor-shuffles. Same d / q values as ggml quantize_q8_1 (max is order-free,
// d = amax/127, q = roundf(x/d)). Writes the split smem layout q0[blk] (quants 0..15), q1[blk] (16..31), d[blk].
// blk i's 32 floats are at x + xoff(i). Must be called by whole warps.
template <int U = 4, typename XOFF>
static __device__ __forceinline__ void quantize_split(const float * __restrict__ x, XOFF xoff, int nblk,
                                                      int4 * s_q0, int4 * s_q1, float * s_d, int warp, int nwarps, int lane) {
    const int sub = lane >> 3, m = lane & 7;   // 4 blocks per warp-step, lane covers elements 4m..4m+3
    for (int i0 = 4 * warp; i0 < nblk; i0 += 4 * nwarps * U) {
        float4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = i0 + 4 * nwarps * u + sub;
            v[u] = i < nblk ? *(const float4 *) (x + xoff(i) + 4 * m) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = i0 + 4 * nwarps * u + sub;
            float a = fmaxf(fmaxf(fabsf(v[u].x), fabsf(v[u].y)), fmaxf(fabsf(v[u].z), fabsf(v[u].w)));
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 1));
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 2));
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 4));
            const float d = a / 127.0f;
            int q0 = 0, q1 = 0, q2 = 0, q3 = 0;
            if (a != 0.0f) { q0 = (int) roundf(v[u].x / d); q1 = (int) roundf(v[u].y / d); q2 = (int) roundf(v[u].z / d); q3 = (int) roundf(v[u].w / d); }
            const uint32_t packed = (uint32_t) (q0 & 0xff) | ((uint32_t) (q1 & 0xff) << 8) | ((uint32_t) (q2 & 0xff) << 16) | ((uint32_t) (q3 & 0xff) << 24);
            if (i < nblk) {
                ((uint32_t *) (m < 4 ? &s_q0[i] : &s_q1[i]))[m & 3] = packed;
                if (m == 0) s_d[i] = __half2float(__float2half(d));
            }
        }
    }
}

// Block-cooperative ggml-exact q8_1 quantization of nblk blocks of 32 floats; block i reads x + xoff(i).
// Loads are issued 8 at a time per warp so L2 latency is paid ~nblk/(8*nwarps) times, not nblk/nwarps times.
template <typename XOFF>
static __device__ __forceinline__ void block_quantize(const float * __restrict__ x, XOFF xoff, int nblk, int4 (*s_aq)[2], float * s_ad,
                                                      int warp, int nwarps, int lane) {
    for (int i0 = warp; i0 < nblk; i0 += 8 * nwarps) {
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { const int i = i0 + j * nwarps; v[j] = i < nblk ? x[xoff(i) + lane] : 0.0f; }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int i = i0 + j * nwarps;
            if (i < nblk) {
                float d;
                const int8_t q = quant_q8_1_lane(v[j], d);
                ((int8_t *) s_aq[i])[lane] = q;
                if (lane == 0) s_ad[i] = d;
            }
        }
    }
}

}  // namespace dsv41
