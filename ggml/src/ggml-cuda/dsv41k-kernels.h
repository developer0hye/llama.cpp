// Public API of the DeepSeek-V4.1-Flash sm_86 decode GEMV kernels (workstream B).
// All weights are in the NATIVE ggml block layout of the per-GPU TP slice (no load-time repack needed).
// All activations are f32 in global memory; q8_1 quantization (ggml-exact) happens inside the kernels.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---- pre-quantized activations ("split q8" layout, values bit-identical to ggml quantize_q8_1) ----
// q0[t*nb + b] = quants 0..15 of block b of token t, q1 = quants 16..31, d = scale (rounded through half like ggml)
struct dsv41_q8act { int4 * q0; int4 * q1; float * d; };
// ---- MoE routing table: unique experts of a (1..8)-token batch in first-occurrence order ----
// nu[0] = #unique; uexp[u] = expert id; uslot[u*8 + t] = slot of that expert in token t's top-k (-1 if unused)
struct dsv41_route { int * nu; int * uexp; int8_t * uslot; };
size_t dsv41_prep_workspace_bytes(int n_tok_max, int K);   // q8act + route
void dsv41_prep_carve(void * ws, int n_tok_max, int K, dsv41_q8act & q, dsv41_route & r);
// one tiny kernel: quantize x[n_tok][K] and (if ids) build the routing table
void dsv41_prep(const float * x, int n_tok, int K, const dsv41_q8act & q, const int32_t * ids, int n_used,
                const dsv41_route & r, cudaStream_t st);

// ---- MoE down (routed MXFP4 experts + optional Q8_0 shared expert), weighted expert sum fused ----
// out[t][r] = sum_s weights[t][s] * dot(W[ids[t][s]] row r, act[t][s]) + dot(W_shared row r, act_shared[t])
struct dsv41_moe_down_args {
    const void * w;            // MXFP4 experts [n_expert][rows][K/32] blocks (17 B), per-GPU slice
    size_t expert_stride;      // bytes between experts (= rows*K/32*17 for a contiguous slice)
    const void * w_shared;     // Q8_0 [rows][K/32] (34 B blocks) or nullptr
    const float * act;         // [n_tok][n_used][K]
    const float * act_shared;  // [n_tok][K] (if w_shared)
    const int32_t * ids;       // [n_tok][n_used]
    const float * weights;     // [n_tok][n_used] final routing weights
    float * out;               // [n_tok][rows]
    int n_tok, n_used, rows, K;   // K must be 288 (TP8 slice), n_used 6, n_tok 1..8
    int ids_stride = 0;           // int32 elements between tokens' id rows (0 = n_used, contiguous) [A: ggml views]
};
void dsv41_moe_down(const dsv41_moe_down_args & a, cudaStream_t st);

// ---- MoE gate+up (routed MXFP4 + optional Q8_0 shared expert), SiLU(gate)*up fused ----
// act[t][s][r] = silu(dot(Wg[ids[t][s]] row r, x[t])) * dot(Wu[ids[t][s]] row r, x[t]);  act_shared likewise
struct dsv41_moe_gate_up_args {
    const void * w_gate; const void * w_up;   // MXFP4 [n_expert][rows][K/32]
    size_t expert_stride;
    const void * w_gate_shared; const void * w_up_shared;  // Q8_0 [rows][K/32] or nullptr
    const float * x;           // [n_tok][K]
    const int32_t * ids;       // [n_tok][n_used]
    float * act;               // [n_tok][n_used][rows]
    float * act_shared;        // [n_tok][rows]
    int n_tok, n_used, rows, K;   // K = 5120, rows = 288 (TP8 slice)
};
void dsv41_moe_gate_up(const dsv41_moe_gate_up_args & a, cudaStream_t st);
// v4: prologue-free, takes the prep output (x quantized + routing table)
void dsv41_moe_gate_up_q(const dsv41_moe_gate_up_args & a, const dsv41_q8act & xq, const dsv41_route & r, cudaStream_t st);

// ---- Q8_0 GEMV, up to 4 matrices sharing the same input x (one launch), f32 in, f32 out ----
// out_s[t][r] = dot(W_s row r, x[t]); W_s native Q8_0 [rows_s][K/32] (34 B blocks). K in {1024,1280,2304,4096,5120,6144}
struct dsv41_gemv_seg { const void * w; float * out; int rows; };
struct dsv41_gemv_group { dsv41_gemv_seg seg[4]; int nseg; };
void dsv41_gemv_q8_0(const dsv41_gemv_group & g, const float * x, int K, int n_tok, cudaStream_t st);

// ---- BF16 / F32 GEMV: out[t][r] = dot(W row r, x[t]); optional deterministic split-K ----
struct dsv41_gemv_f_args {
    const void * w; bool is_bf16;
    const float * x; float * out;
    int K, rows, n_tok;
    int nsplit;              // 1 = no split; else K/(8|4)/nsplit chunks per split
    float * workspace;       // [n_tok*rows*nsplit] when nsplit > 1
    unsigned * ticket;       // zero-initialised device counter when nsplit > 1 (self re-arming)
};
void dsv41_gemv_f(const dsv41_gemv_f_args & a, cudaStream_t st);
