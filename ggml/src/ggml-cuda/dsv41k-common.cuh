// DeepSeek-V4.1-Flash decode GEMV kernels for sm_86 (RTX A6000) — shared primitives.
// Numerics primitives are copied verbatim from ggml-cuda (llama.cpp-tp @ 7d5b3fc):
//   get_int_from_table_16 (vecdotq.cuh), ggml_cuda_e8m0_to_fp32 (common.cuh), quantize_q8_1 (quantize.cu),
//   kvalues_mxfp4 (ggml-common.h), ggml_cuda_op_silu_single (unary.cuh).
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace dsv41 {

constexpr int QK = 32;  // Q8_0, Q8_1, MXFP4 block size

struct block_q8_0  { half d; int8_t qs[32]; };              // 34 B (GGML_TYPE_Q8_0 = 8)
struct block_q8_1  { half2 ds; int8_t qs[32]; };            // 36 B (only .x = d is used by Q8_0/MXFP4 dots)
struct block_mxfp4 { uint8_t e; uint8_t qs[16]; };          // 17 B (GGML_TYPE_MXFP4 = 39)
static_assert(sizeof(block_q8_0) == 34, "q8_0");
static_assert(sizeof(block_q8_1) == 36, "q8_1");
static_assert(sizeof(block_mxfp4) == 17, "mxfp4");

__device__ __constant__ static const int8_t kvalues_mxfp4_c[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// packed LUT as 4 u32 (little endian) for __byte_perm lookups
#define DSV41_MXFP4_LUT0 0x03020100u
#define DSV41_MXFP4_LUT1 0x0C080604u
#define DSV41_MXFP4_LUT2 0xFDFEFF00u
#define DSV41_MXFP4_LUT3 0xF4F8FAFCu

// == ggml get_int_from_table_16 (CUDA branch) with the table in registers ==
// returns .x = LUT[low nibbles of the 4 bytes], .y = LUT[high nibbles]
static __device__ __forceinline__ int2 mxfp4_lut(const uint32_t q4) {
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(DSV41_MXFP4_LUT0, DSV41_MXFP4_LUT1, q4 >> shift);
        const uint32_t high = __byte_perm(DSV41_MXFP4_LUT2, DSV41_MXFP4_LUT3, q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

static __device__ __forceinline__ float e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float) e;
#else
    uint32_t bits = x == 0 ? 0x00400000u : ((uint32_t) x << 23);
    return __uint_as_float(bits);
#endif
}

static __device__ __forceinline__ float silu1(float x) { return x / (1.0f + expf(-x)); }  // ggml_cuda_op_silu_single

template <int width = 32>
static __device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = width / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, width);
    return x;
}
template <int width = 32>
static __device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = width / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, width));
    return x;
}

// One warp quantizes 32 floats (one value per lane) to a q8_1 block exactly like ggml quantize_q8_1.
// Returns this lane's int8 and writes d (as stored in half, then read back as float) via d_out.
static __device__ __forceinline__ int8_t quant_q8_1_lane(float xi, float & d_as_half) {
    const float amax = warp_max(fabsf(xi));
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    d_as_half = __half2float(__float2half(d));  // ggml stores make_half2(d, sum) and the dots read __low2float(ds)
    return q;
}

// aligned 32-bit read-only global load (non-coherent path)
static __device__ __forceinline__ uint32_t ldg32(const void * p) { return __ldg((const unsigned int *) p); }
static __device__ __forceinline__ int4 ldg128(const void * p) { return __ldg((const int4 *) p); }

// funnel: bytes [s, s+4) of the 8-byte little-endian (lo, hi); s in 0..3 (runtime ok)
static __device__ __forceinline__ uint32_t fsh(uint32_t lo, uint32_t hi, uint32_t s) { return __funnelshift_r(lo, hi, 8 * s); }

}  // namespace dsv41
