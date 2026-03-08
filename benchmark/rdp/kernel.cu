#include <stdint.h>
#include <climits>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

namespace py = pybind11;

extern "C" {

// ============================== helpers =====================================

__host__ __device__ __forceinline__
int modS_ll(long long x, int S) {
    if (S <= 0) return 0;
    long long r = x % S;
    if (r < 0) r += S;
    return (int)r;
}

__device__ __forceinline__ unsigned short h2u(half h)  { return __half_as_ushort(h); }
__device__ __forceinline__ half u2h(unsigned short u)  { return __ushort_as_half(u); }

// ========================= pack (tokens-major) ===============================

__global__ void pack_scatter_kernel_full(
    const half* __restrict__ key_TN,     // [T, n]
    const half* __restrict__ val_TN,     // [T, n]
    unsigned short* __restrict__ shards, // [(nd+2) x S]
    int T, int n, int nd, int S)
{
    if (T <= 0 || n <= 0 || nd <= 0 || S <= 0) return;

    int token_id = blockIdx.x;
    if (token_id >= T) return;

    const int tid = threadIdx.x;
    const int BS  = blockDim.x;

    const half* k_row = key_TN + (long long)token_id * n;
    const half* v_row = val_TN + (long long)token_id * n;

    long long baseK = (long long)token_id * n;
    long long baseV = (long long)T * n + (long long)token_id * n;

    int rowK0 = (int)(baseK / S);
    int offK0 = (int)(baseK - (long long)rowK0 * S);
    int rowV0 = (int)(baseV / S);
    int offV0 = (int)(baseV - (long long)rowV0 * S);

    int i = tid;
    while (i < n) {
        unsigned short wk = h2u(k_row[i]);
        unsigned short wv = h2u(v_row[i]);

        int offK = offK0 + i, rowK = rowK0;
        if (offK >= S) { int bump = offK / S; offK -= bump * S; rowK += bump; }
        if (rowK < nd) shards[(long long)rowK * S + offK] = wk;

        int offV = offV0 + i, rowV = rowV0;
        if (offV >= S) { int bump2 = offV / S; offV -= bump2 * S; rowV += bump2; }
        if (rowV < nd) shards[(long long)rowV * S + offV] = wv;

        i += BS;
    }
}

// ========================= parity (P row, Q diag) ============================

__global__ void compute_rdp_parity_kernel(
    const unsigned short* __restrict__ shards_in,  // [(nd+2) x S]
    unsigned short* __restrict__ shards_out,
    int nd, int S, int star_base)
{
    if (nd <= 0 || S <= 0) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int Prow = nd;
    const int Qrow = nd + 1;

    // Row parity P
    for (int off = tid; off < S; off += blockDim.x * gridDim.x) {
        unsigned short acc = 0;
        #pragma unroll 4
        for (int r = 0; r < nd; ++r)
            acc = (unsigned short)(acc ^ shards_in[(long long)r * S + off]);
        shards_out[(long long)Prow * S + off] = acc;
    }

    // Diagonal parity Q (skip star for each row)
    for (int d = tid; d < S; d += blockDim.x * gridDim.x) {
        unsigned short acc = 0;
        #pragma unroll 4
        for (int r = 0; r < nd; ++r) {
            int c = modS_ll((long long)d - r, S);
            int star_r = (star_base + r) % S;
            if (c == star_r) continue;
            acc = (unsigned short)(acc ^ shards_in[(long long)r * S + c]);
        }
        shards_out[(long long)Qrow * S + d] = acc;
    }
}

// ================= single-missing (parallel) =================================

__global__ void rdp_reconstruct_single_kernel(
    unsigned short* __restrict__ shards,
    int nd, int S, int miss, int /*star_base*/)
{
    if (nd <= 0 || S <= 0 || miss < 0 || miss >= nd) return;
    const int Prow = nd;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int c = tid; c < S; c += blockDim.x * gridDim.x) {
        unsigned short acc = shards[(long long)Prow * S + c];
        #pragma unroll 4
        for (int r = 0; r < nd; ++r) {
            if (r == miss) continue;
            acc = (unsigned short)(acc ^ shards[(long long)r * S + c]);
        }
        shards[(long long)miss * S + c] = acc;
    }
}

// ============== double-missing (parallel: recover A, rebuild A/B) ============

static __device__ __forceinline__
unsigned short rhs_rowP_except_ab(const unsigned short* shards, int nd, int S, int a, int b, int c) {
    const int Prow = nd;
    unsigned short acc = shards[(long long)Prow * S + c];
    for (int r = 0; r < nd; ++r) {
        if (r == a || r == b) continue;
        acc = (unsigned short)(acc ^ shards[(long long)r * S + c]);
    }
    return acc; // equals a[c] ^ b[c]
}

static __device__ __forceinline__
unsigned short rhs_diagQ_except_ab(const unsigned short* shards, int nd, int S, int a, int b, int star_base, int d) {
    const int Qrow = nd + 1;
    unsigned short acc = shards[(long long)Qrow * S + d];
    for (int r = 0; r < nd; ++r) {
        if (r == a || r == b) continue;
        int c = modS_ll((long long)d - r, S);
        int star_r = (star_base + r) % S;
        if (c == star_r) continue;
        acc = (unsigned short)(acc ^ shards[(long long)r * S + c]);
    }
    return acc; // equals a[c_a] ^ b[c_b_prev]
}

#ifndef TILE_K
#define TILE_K 256
#endif

// --- Kernel 1: per-tile provisional solve for A + carry-out ------------------
__global__ void rdp_double_tile_provisional_A_only(
    unsigned short* __restrict__ shards, // in/out (write provisional A only)
    int nd, int S, int a, int b, int star_base,
    int num_tiles, int L, int pos0, int delta_ab,
    unsigned short* __restrict__ tile_carries) // [num_tiles], XOR(W) per tile
{
    int tile = blockIdx.x;
    if (tile >= num_tiles) return;

    int k0 = tile * TILE_K;
    int tile_len = min(TILE_K, L - k0);
    if (tile_len <= 0) { tile_carries[tile] = 0; return; }

    extern __shared__ unsigned short sh[];
    unsigned short* shW  = sh;                 // W = RP ^ RQex
    unsigned short* shPX = sh + TILE_K;        // prefixExclusive(W)
    unsigned short* shRQ = sh + 2 * TILE_K;    // RQex

    // pos at start of this tile in the ring walk
    int pos = (int)(( ( (long long)pos0 + ((long long)(k0 % S) * (long long)delta_ab) ) % (long long)S + (long long)S ) % (long long)S);

    int star_a = (star_base + a) % S;

    // Fill W and RQex
    for (int j = threadIdx.x; j < tile_len; j += blockDim.x) {
        int pos_j = (int)(( ( (long long)pos + ((long long)j * (long long)delta_ab) ) % (long long)S + (long long)S ) % (long long)S);
        int d_a   = modS_ll((long long)pos_j + a, S);

        unsigned short RP = rhs_rowP_except_ab(shards, nd, S, a, b, pos_j);
        unsigned short RQ = rhs_diagQ_except_ab(shards, nd, S, a, b, star_base, d_a);

        shW[j]  = (unsigned short)(RP ^ RQ);
        shRQ[j] = RQ;
    }
    __syncthreads();

    // Parallel XOR scan (exclusive) over W → shPX
    for (int j = threadIdx.x; j < tile_len; j += blockDim.x) shPX[j] = shW[j];
    __syncthreads();

    for (int offset = 1; offset < tile_len; offset <<= 1) {
        int idx = (threadIdx.x + 1) * (offset << 1) - 1;
        if (idx < tile_len) shPX[idx] = (unsigned short)(shPX[idx] ^ shPX[idx - offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0) shPX[tile_len - 1] = 0;
    __syncthreads();
    for (int offset = tile_len >> 1; offset > 0; offset >>= 1) {
        int idx = (threadIdx.x + 1) * (offset << 1) - 1;
        if (idx < tile_len) {
            unsigned short t = shPX[idx - offset];
            shPX[idx - offset] = shPX[idx];
            shPX[idx] = (unsigned short)(shPX[idx] ^ t);
        }
        __syncthreads();
    }

    // Write provisional A only (t_in assumed 0). Never touch star_a here.
    for (int j = threadIdx.x; j < tile_len; j += blockDim.x) {
        int pos_j = (int)(( ( (long long)pos + ((long long)j * (long long)delta_ab) ) % (long long)S + (long long)S ) % (long long)S);
        if (pos_j != star_a) {
            unsigned short a_prov = (unsigned short)(shRQ[j] ^ shPX[j]);
            shards[(long long)a * S + pos_j] = a_prov;
        }
    }
    __syncthreads();

    // Carry-out = XOR(W over tile)
    __shared__ unsigned int ssum;
    if (threadIdx.x == 0) ssum = 0;
    __syncthreads();

    unsigned int x = 0;
    for (int j = threadIdx.x; j < tile_len; j += blockDim.x) x ^= (unsigned int)shW[j];
    atomicXor(&ssum, x);
    __syncthreads();

    if (threadIdx.x == 0) tile_carries[tile] = (unsigned short)ssum;
}

// --- Kernel 2: apply carry-in per tile to A only -----------------------------

__global__ void rdp_double_tile_fix_apply_A(
    unsigned short* __restrict__ shards,
    int S, int a,
    int num_tiles, int L, int pos0, int delta_ab,
    const unsigned short* __restrict__ tile_tin)
{
    int tile = blockIdx.x;
    if (tile >= num_tiles) return;
    unsigned short t_in = tile_tin[tile];

    int k0 = tile * TILE_K;
    int tile_len = min(TILE_K, L - k0);
    if (tile_len <= 0) return;

    int pos = (int)(( ( (long long)pos0 + ((long long)(k0 % S) * (long long)delta_ab) ) % (long long)S + (long long)S ) % (long long)S);

    for (int j = threadIdx.x; j < tile_len; j += blockDim.x) {
        int pos_j = (int)(( ( (long long)pos + ((long long)j * (long long)delta_ab) ) % (long long)S + (long long)S ) % (long long)S);
        shards[(long long)a * S + pos_j] = (unsigned short)(shards[(long long)a * S + pos_j] ^ t_in);
    }
}

// --- Fill A’s star via P -----------------------------------------------------

__global__ void fill_a_star_from_P(
    unsigned short* __restrict__ shards, int nd, int S, int a, int star_base)
{
    const int Prow = nd;
    int star_a = (star_base + a) % S;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned short acc = shards[(long long)Prow * S + star_a];
        for (int r = 0; r < nd; ++r) {
            if (r == a) continue;
            acc = (unsigned short)(acc ^ shards[(long long)r * S + star_a]);
        }
        shards[(long long)a * S + star_a] = acc;
    }
}

// --- Rebuild an entire data row from P --------------------------------------

__global__ void rebuild_row_from_P(
    unsigned short* __restrict__ shards,
    int nd, int S, int target_row /*row index to rebuild*/)
{
    const int Prow = nd;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= S) return;

    unsigned short acc = shards[(long long)Prow * S + c];  // P[c]
    // XOR every data row except target_row to solve target_row[c]
    for (int r = 0; r < nd; ++r) {
        if (r == target_row) continue;
        acc = (unsigned short)(acc ^ shards[(long long)r * S + c]);
    }
    shards[(long long)target_row * S + c] = acc;
}

// ============================ load full (K|V) ================================

__global__ void load_full_kernel(
    const unsigned short* __restrict__ shards,
    half* __restrict__ out_half_TN,  // [T*n]K | [T*n]V
    int T, int n, int nd, int S)
{
    if (T <= 0 || n <= 0 || nd <= 0 || S <= 0) return;

    int token_id = blockIdx.x;
    if (token_id >= T) return;

    const int tid = threadIdx.x;
    const int BS  = blockDim.x;

    half* dK = out_half_TN + (long long)token_id * n;
    half* dV = out_half_TN + (long long)T * n + (long long)token_id * n;

    long long baseK = (long long)token_id * n;
    long long baseV = (long long)T * n + (long long)token_id * n;

    int rowK0 = (int)(baseK / S);
    int offK0 = (int)(baseK - (long long)rowK0 * S);
    int rowV0 = (int)(baseV / S);
    int offV0 = (int)(baseV - (long long)rowV0 * S);

    int i = tid;
    while (i < n) {
        int offK = offK0 + i, rowK = rowK0;
        if (offK >= S) { int bump = offK / S; offK -= bump * S; rowK += bump; }
        unsigned short uk = 0;
        if (rowK < nd) uk = shards[(long long)rowK * S + offK];
        dK[i] = u2h(uk);

        int offV = offV0 + i, rowV = rowV0;
        if (offV >= S) { int bump2 = offV / S; offV -= bump2 * S; rowV += bump2; }
        unsigned short uv = 0;
        if (rowV < nd) uv = shards[(long long)rowV * S + offV];
        dV[i] = u2h(uv);

        i += BS;
    }
}

} // extern "C"

// ============================ Host utilities =================================

static inline int calc_blocks(int work, int threads = 256) {
    if (work <= 0) return 0;
    long long blocks = ((long long)work + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (blocks > INT_MAX) blocks = INT_MAX;
    return (int)blocks;
}

static inline void checkLastKernel(const char* what) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

// ===================== Two exported host-side entrypoints ====================

void fused_backup_full_kv_launcher(torch::Tensor K_TN,
                                   torch::Tensor V_TN,
                                   torch::Tensor shards,
                                   int nd, int S, int T, int n, int star_base)
{
    // Pack
    {
        const int threads = 512;
        const int blocks  = std::max(1, T);
        pack_scatter_kernel_full<<<blocks, threads>>>(
            reinterpret_cast<half*>(K_TN.data_ptr<torch::Half>()),
            reinterpret_cast<half*>(V_TN.data_ptr<torch::Half>()),
            shards.data_ptr<unsigned short>(),
            T, n, nd, S
        );
        checkLastKernel("pack_scatter_kernel_full");
    }
    // Parities
    {
        const int threads = 256;
        const int work    = S;
        const int blocks  = std::min(calc_blocks(work, threads), 65535);
        compute_rdp_parity_kernel<<<blocks, threads>>>(
            shards.data_ptr<unsigned short>(),
            shards.data_ptr<unsigned short>(),
            nd, S, star_base
        );
        checkLastKernel("compute_rdp_parity_kernel");
    }
}

void fused_recovery_full_kv_launcher(torch::Tensor shards,
                                     torch::Tensor out_half_TN,
                                     int nd, int S, int T, int n,
                                     int star_base, int miss_a, int miss_b)
{
    auto norm = [&](int m)->int { return (m >= 0 && m < nd) ? m : -1; };
    miss_a = norm(miss_a);
    miss_b = norm(miss_b);

    // (1) Recovery
    if (miss_a >= 0 && miss_b >= 0 && miss_a != miss_b) {
        int a = miss_a, b = miss_b;
        if (b < a) { int t = a; a = b; b = t; } // normalize order

        // Ring parameters for A
        int star_b = (star_base + b) % S;
        int delta_ab = modS_ll((long long)b - a, S);

        int d_b   = modS_ll((long long)star_b + b, S);
        int pos0  = modS_ll((long long)d_b - a, S);   // first pos for A (exclude A's star)
        int L = (S > 0) ? (S - 1) : 0;                // exclude A's star

        const int TILE = TILE_K;
        int num_tiles = (L + TILE - 1) / TILE;

        // Phase 1: per-tile provisional for A + carries
        auto opts = shards.options();
        torch::Tensor tile_carries = torch::empty({std::max(1, num_tiles)}, opts);
        {
            const int threads = 256;
            const int blocks  = std::max(1, num_tiles);
            size_t shmem = sizeof(unsigned short) * (TILE * 3);
            rdp_double_tile_provisional_A_only<<<blocks, threads, shmem>>>(
                shards.data_ptr<unsigned short>(),
                nd, S, a, b, star_base,
                num_tiles, L, pos0, delta_ab,
                tile_carries.data_ptr<unsigned short>()
            );
            checkLastKernel("rdp_double_tile_provisional_A_only");
        }

        // Phase 2: exclusive prefix XOR of tile_carries on CPU (tiny array)
        torch::Tensor tile_tin = tile_carries.cpu();
        {
            auto acc = (unsigned short)0;
            auto* t = tile_tin.data_ptr<unsigned short>();
            for (int i = 0; i < num_tiles; ++i) {
                unsigned short tmp = t[i];
                t[i] = acc;    // exclusive prefix
                acc ^= tmp;
            }
        }
        tile_tin = tile_tin.to(shards.device());

        // Phase 3: apply carry-in to A only
        {
            const int threads = 256;
            const int blocks  = std::max(1, num_tiles);
            rdp_double_tile_fix_apply_A<<<blocks, threads>>>(
                shards.data_ptr<unsigned short>(),
                S, a,
                num_tiles, L, pos0, delta_ab,
                tile_tin.data_ptr<unsigned short>()
            );
            checkLastKernel("rdp_double_tile_fix_apply_A");
        }

        // Fill A's star from P
        {
            fill_a_star_from_P<<<1, 1>>>(
                shards.data_ptr<unsigned short>(), nd, S, a, star_base
            );
            checkLastKernel("fill_a_star_from_P");
        }

        // NEW: Snap A to parity — rebuild A from P
        {
            const int threads = 256;
            const int blocks  = std::max(1, (S + threads - 1) / threads);
            rebuild_row_from_P<<<blocks, threads>>>(
                shards.data_ptr<unsigned short>(), nd, S, a
            );
            checkLastKernel("rebuild_row_from_P(a)");
        }

        // Rebuild B entirely from P
        {
            const int threads = 256;
            const int blocks  = std::max(1, (S + threads - 1) / threads);
            rebuild_row_from_P<<<blocks, threads>>>(
                shards.data_ptr<unsigned short>(), nd, S, b
            );
            checkLastKernel("rebuild_row_from_P(b)");
        }

    } else if (miss_a >= 0 && miss_b < 0) {
        // single data row
        const int threads = 256;
        const int blocks  = std::min(calc_blocks(S, threads), 65535);
        rdp_reconstruct_single_kernel<<<blocks, threads>>>(
            shards.data_ptr<unsigned short>(), nd, S, miss_a, 0
        );
        checkLastKernel("rdp_reconstruct_single_kernel");
    } else if (miss_b >= 0 && miss_a < 0) {
        const int threads = 256;
        const int blocks  = std::min(calc_blocks(S, threads), 65535);
        rdp_reconstruct_single_kernel<<<blocks, threads>>>(
            shards.data_ptr<unsigned short>(), nd, S, miss_b, 0
        );
        checkLastKernel("rdp_reconstruct_single_kernel");
    }
    // else: nothing missing

    // (2) Load back full K|V
    {
        const int threads = 512;
        const int blocks  = std::max(1, T);
        load_full_kernel<<<blocks, threads>>>(
            shards.data_ptr<unsigned short>(),
            reinterpret_cast<half*>(out_half_TN.data_ptr<torch::Half>()),
            T, n, nd, S
        );
        checkLastKernel("load_full_kernel");
    }
}

// ============================== Python bindings ==============================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_backup_full_kv",  &fused_backup_full_kv_launcher,
          "Pack+encode full KV + compute RDP parities (one call)",
          py::arg("K_TN"), py::arg("V_TN"), py::arg("shards"),
          py::arg("num_data_shards"), py::arg("shard_size"),
          py::arg("T"), py::arg("n"), py::arg("star_base"));

    m.def("fused_recovery_full_kv", &fused_recovery_full_kv_launcher,
          "Recover 1 or 2 missing DATA rows (RDP, parallel) then load full packed TN (one call)",
          py::arg("shards"), py::arg("out_half_TN"),
          py::arg("num_data_shards"), py::arg("shard_size"),
          py::arg("T"), py::arg("n"), py::arg("star_base"),
          py::arg("missing_a"), py::arg("missing_b"));
}
