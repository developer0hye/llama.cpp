#pragma once
#include "common.cuh"

bool ggml_cuda_dsv41_attn_decode_supported(const ggml_tensor * dst);
void ggml_cuda_op_dsv41_attn_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
