#pragma once

#include "common.cuh"

// One-shot peer-to-peer AllReduce for small tensors under tensor parallelism (DeepSeek-V4.1 decode:
// 80 AllReduces of 5120 f32 per token on 8 GPUs, where NCCL's ring costs ~25 us each).
// Each rank pushes its slice of the vector into a slot of every peer's receive buffer with P2P stores,
// raises a per-block flag on each peer, waits for the peers' flags and sums the n_rank slots in rank order,
// so every rank produces bit-identical results. Usable inside CUDA graph capture: the epoch counters live
// in device memory and advance by themselves on every replay.
struct ggml_cuda_tp_ar;

// returns nullptr if peer access cannot be enabled between all devices
ggml_cuda_tp_ar * ggml_cuda_tp_ar_init(const int * devices, int n_devices, int64_t max_elems);
void              ggml_cuda_tp_ar_free(ggml_cuda_tp_ar * ar);
int64_t           ggml_cuda_tp_ar_max_elems(const ggml_cuda_tp_ar * ar);

// data[r]: f32 vector of ne elements on device devices[r] (in place); zero[r]: rank contributes zeros;
// streams[r]: stream to enqueue on. All ranks must call this the same number of times in the same order.
void ggml_cuda_tp_ar_allreduce_f32(ggml_cuda_tp_ar * ar, float ** data, const bool * zero, int64_t ne, cudaStream_t * streams);
