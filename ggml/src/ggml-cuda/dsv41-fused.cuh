#pragma once

#include "common.cuh"

// DeepSeek-V4.1 decode fusions (ai-8a6000 TP build). See dsv41-fused.cu.

// RMS_NORM (no weight) -> MUL_MAT with a small F32 weight (hyper-connection mixes: [20480, 24]):
// dst[r, t] = rsqrt(mean(x_t^2) + eps) * dot(W_r, x_t), one block per (r, t).
bool ggml_cuda_dsv41_can_fuse_rms_mul_mat(const ggml_tensor * rms, const ggml_tensor * mm);
void ggml_cuda_dsv41_rms_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, ggml_tensor * mm);
