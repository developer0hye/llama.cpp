#include "tp-allreduce.cuh"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

// see tp-allreduce.cuh
//
// Topology (ai-8a6000, 8x RTX A6000): NVLink only within the pairs (0,1) (2,3) (4,5) (6,7). PCIe P2P between
// other pairs is reported as supported but SILENTLY CORRUPTS data on this platform (stores and loads; checked with
// ~/dsv41/tp/p2ptest), so peer access is enabled only within NVLink pairs and the cross-pair step goes through
// pinned host memory:
//   A. pair exchange over NVLink: each rank pushes its slice into its partner's receive slot, s = x_even + x_odd
//   B. the even rank of each pair writes s to a host slot and raises a host flag
//   C. every rank waits for all pair flags and sums the pair slots in pair order -> bit-identical on all ranks
// Everything is per block of a fixed 16-block grid with per-block flags/epochs, so blocks never wait on each
// other inside a rank. Epoch counters live in device memory and advance on every execution (CUDA-graph safe).
// Two buffer sets (epoch parity) suffice: a rank can only start call k+1 after every pair published call k,
// which every rank does only after it finished reading call k-1.

#define TP_AR_MAX_RANKS 8
#define TP_AR_BLOCKS    16
#define TP_AR_THREADS   1024

struct ggml_cuda_tp_ar {
    int     n_rank;
    int     devices[TP_AR_MAX_RANKS];
    int64_t max_elems;
    // device memory of each rank, written by its NVLink partner
    float    * pslot[TP_AR_MAX_RANKS]; // [2][max_elems]
    uint32_t * pflag[TP_AR_MAX_RANKS]; // [2][TP_AR_BLOCKS]
    uint32_t * ctr  [TP_AR_MAX_RANKS]; // [TP_AR_BLOCKS]
    // pinned, mapped host memory shared by all ranks
    float    * hslot;                  // [2][n_pair][max_elems]
    uint32_t * hflag;                  // [2][n_pair][TP_AR_BLOCKS]
};

static __device__ __forceinline__ uint32_t tp_ar_ld_volatile(const uint32_t * p) {
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ void tp_ar_st_volatile(uint32_t * p, uint32_t v) {
    asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

static __device__ __forceinline__ float  tp_ar_add(float  a, float  b) { return a + b; }
static __device__ __forceinline__ float4 tp_ar_add(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// ne, max_elems in units of T
template <typename T>
static __global__ void __launch_bounds__(TP_AR_THREADS)
tp_ar_pair_host(T * __restrict__ data, const int64_t ne, const int rank, const int n_pair, const int64_t max_elems,
        const bool zero, const T * __restrict__ my_pslot, const uint32_t * __restrict__ my_pflag,
        T * __restrict__ partner_pslot, uint32_t * __restrict__ partner_pflag,
        T * __restrict__ hslot, uint32_t * __restrict__ hflag, uint32_t * __restrict__ ctr) {
    __shared__ uint32_t s_epoch;
    const int b = blockIdx.x;
    if (threadIdx.x == 0) {
        s_epoch = ctr[b] + 1;
    }
    __syncthreads();
    const uint32_t epoch = s_epoch;
    const int      set   = epoch & 1;
    const bool     even  = (rank & 1) == 0;
    const int      pair  = rank >> 1;

    const int64_t chunk = (ne + gridDim.x - 1) / gridDim.x;
    const int64_t lo    = b*chunk;
    const int64_t hi    = lo + chunk < ne ? lo + chunk : ne;

    // A. NVLink pair exchange
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const T v = zero ? T{} : data[i];
        partner_pslot[set*max_elems + i] = v;
        if (zero) {
            data[i] = v;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        tp_ar_st_volatile(partner_pflag + set*TP_AR_BLOCKS + b, epoch);
        while (tp_ar_ld_volatile(my_pflag + set*TP_AR_BLOCKS + b) != epoch) {
        }
        __threadfence();
    }
    __syncthreads();

    // B. pair leader publishes x_even + x_odd to host memory
    if (even) {
        for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            const T p = __ldcg(my_pslot + set*max_elems + i);
            hslot[((int64_t) set*n_pair + pair)*max_elems + i] = tp_ar_add(data[i], p);
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            tp_ar_st_volatile(hflag + (set*n_pair + pair)*TP_AR_BLOCKS + b, epoch);
        }
    }

    // C. wait for every pair, sum in pair order
    if (threadIdx.x < n_pair) {
        const uint32_t * f = hflag + (set*n_pair + threadIdx.x)*TP_AR_BLOCKS + b;
        while (tp_ar_ld_volatile(f) != epoch) {
        }
        __threadfence_system();
    }
    __syncthreads();
    const T * hs = hslot + (int64_t) set*n_pair*max_elems;
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        T acc = __ldcv(hs + i);
        for (int q = 1; q < n_pair; ++q) {
            acc = tp_ar_add(acc, __ldcv(hs + (int64_t) q*max_elems + i));
        }
        data[i] = acc;
    }
    if (threadIdx.x == 0) {
        ctr[b] = epoch;
    }
}

static bool tp_ar_is_nvlink_pair(int a, int b) {
    int access = 0, atomics = 0;
    if (cudaDeviceCanAccessPeer(&access, a, b) != cudaSuccess || !access) {
        (void) cudaGetLastError();
        return false;
    }
    // NVLink peers support native atomics; PCIe peers on this platform do not
    if (cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, a, b) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return atomics != 0;
}

ggml_cuda_tp_ar * ggml_cuda_tp_ar_init(const int * devices, int n_devices, int64_t max_elems) {
    if (n_devices < 2 || n_devices > TP_AR_MAX_RANKS || n_devices % 2 != 0) {
        return nullptr;
    }
    for (int i = 0; i < n_devices; i += 2) {
        if (!tp_ar_is_nvlink_pair(devices[i], devices[i + 1])) {
            GGML_LOG_WARN("%s: devices %d,%d are not an NVLink pair, pair/host AllReduce disabled\n", __func__, devices[i], devices[i + 1]);
            return nullptr;
        }
    }
    auto * ar = new ggml_cuda_tp_ar{};
    ar->n_rank    = n_devices;
    ar->max_elems = max_elems;
    for (int i = 0; i < n_devices; ++i) {
        ar->devices[i] = devices[i];
        ggml_cuda_set_device(devices[i]);
        const cudaError_t err = cudaDeviceEnablePeerAccess(devices[i ^ 1], 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            GGML_LOG_WARN("%s: cudaDeviceEnablePeerAccess %d -> %d failed: %s\n", __func__, devices[i], devices[i ^ 1], cudaGetErrorString(err));
            (void) cudaGetLastError();
            ggml_cuda_tp_ar_free(ar);
            return nullptr;
        }
        (void) cudaGetLastError();
        CUDA_CHECK(cudaMalloc(&ar->pslot[i], sizeof(float)*2*max_elems));
        CUDA_CHECK(cudaMalloc(&ar->pflag[i], sizeof(uint32_t)*2*TP_AR_BLOCKS));
        CUDA_CHECK(cudaMalloc(&ar->ctr[i],   sizeof(uint32_t)*TP_AR_BLOCKS));
        CUDA_CHECK(cudaMemset(ar->pflag[i], 0, sizeof(uint32_t)*2*TP_AR_BLOCKS));
        CUDA_CHECK(cudaMemset(ar->ctr[i],   0, sizeof(uint32_t)*TP_AR_BLOCKS));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const int n_pair = n_devices / 2;
    CUDA_CHECK(cudaHostAlloc((void **) &ar->hslot, sizeof(float)*2*n_pair*max_elems, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &ar->hflag, sizeof(uint32_t)*2*n_pair*TP_AR_BLOCKS, cudaHostAllocPortable | cudaHostAllocMapped));
    memset(ar->hflag, 0, sizeof(uint32_t)*2*n_pair*TP_AR_BLOCKS);
    GGML_LOG_INFO("%s: NVLink-pair + host AllReduce enabled for %d devices, up to %lld f32\n", __func__, n_devices, (long long) max_elems);
    return ar;
}

void ggml_cuda_tp_ar_free(ggml_cuda_tp_ar * ar) {
    if (ar == nullptr) {
        return;
    }
    for (int i = 0; i < ar->n_rank; ++i) {
        ggml_cuda_set_device(ar->devices[i]);
        if (ar->pslot[i]) { (void) cudaFree(ar->pslot[i]); }
        if (ar->pflag[i]) { (void) cudaFree(ar->pflag[i]); }
        if (ar->ctr[i])   { (void) cudaFree(ar->ctr[i]); }
    }
    if (ar->hslot) { (void) cudaFreeHost(ar->hslot); }
    if (ar->hflag) { (void) cudaFreeHost(ar->hflag); }
    delete ar;
}

int64_t ggml_cuda_tp_ar_max_elems(const ggml_cuda_tp_ar * ar) {
    return ar ? ar->max_elems : 0;
}

static void tp_ar_launch(ggml_cuda_tp_ar * ar, int r, float * data, bool zero, int64_t ne, cudaStream_t stream, bool vec) {
    const int n_pair = ar->n_rank / 2;
    const int pr = r ^ 1;
    // one 1024-thread block per ~1024 float4: fewest flag polls/fences for decode-sized vectors (5120 f32 ->
    // 1 block, ~21 us on 8 GPUs vs ~24 us NCCL ring + NCCL's graph host nodes; ~/dsv41/tp/arbench)
    const int64_t n_units = vec ? ne/4 : ne;
    const int nblk = (int) std::min<int64_t>(TP_AR_BLOCKS, std::max<int64_t>(1, (n_units + 1023)/1024));
    ggml_cuda_set_device(ar->devices[r]);
    if (vec) {
        tp_ar_pair_host<float4><<<nblk, TP_AR_THREADS, 0, stream>>>(
                (float4 *) data, ne/4, r, n_pair, ar->max_elems/4, zero,
                (const float4 *) ar->pslot[r], ar->pflag[r], (float4 *) ar->pslot[pr], ar->pflag[pr],
                (float4 *) ar->hslot, ar->hflag, ar->ctr[r]);
    } else {
        tp_ar_pair_host<float><<<nblk, TP_AR_THREADS, 0, stream>>>(
                data, ne, r, n_pair, ar->max_elems, zero,
                ar->pslot[r], ar->pflag[r], ar->pslot[pr], ar->pflag[pr],
                ar->hslot, ar->hflag, ar->ctr[r]);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_tp_ar_allreduce_f32(ggml_cuda_tp_ar * ar, float ** data, const bool * zero, int64_t ne, cudaStream_t * streams) {
    GGML_ASSERT(ne <= ar->max_elems);
    // the vector path must be chosen identically on every rank: decide from ne only (ggml tensors are 16B aligned)
    const bool vec = ne % 4 == 0 && ar->max_elems % 4 == 0;
    for (int r = 0; r < ar->n_rank; ++r) {
        GGML_ASSERT(!vec || (uintptr_t) data[r] % 16 == 0);
        tp_ar_launch(ar, r, data[r], zero[r], ne, streams[r], vec);
    }
}

void ggml_cuda_tp_ar_allreduce_f32_rank(ggml_cuda_tp_ar * ar, int rank, float * data, bool zero, int64_t ne, cudaStream_t stream) {
    GGML_ASSERT(ne <= ar->max_elems);
    const bool vec = ne % 4 == 0 && ar->max_elems % 4 == 0;
    GGML_ASSERT(!vec || (uintptr_t) data % 16 == 0);
    tp_ar_launch(ar, rank, data, zero, ne, stream, vec);
}
