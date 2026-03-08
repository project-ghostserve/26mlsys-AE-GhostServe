#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace py = pybind11;

using gf_t = uint16_t;
using u64  = unsigned long long; // 64-bit for indexing

static constexpr int GF_SIZE  = (1 << 16);   // 65536
static constexpr int GF_ORDER = GF_SIZE - 1; // 65535

// ---------------- GF(2^16) arithmetic via log/exp tables ----------------
__device__ __forceinline__ gf_t gf_mul(gf_t a, gf_t b,
                                       const gf_t* __restrict__ gf_exp,
                                       const gf_t* __restrict__ gf_log) {
    if (a == 0 || b == 0) return 0;
    int s = (int)gf_log[a] + (int)gf_log[b];
    if (s >= GF_ORDER) s -= GF_ORDER;
    return gf_exp[s];
}

__device__ __forceinline__ gf_t gf_div(gf_t a, gf_t b,
                                       const gf_t* __restrict__ gf_exp,
                                       const gf_t* __restrict__ gf_log) {
    if (a == 0) return 0;
    if (b == 0) return 0; // guard; caller should avoid div by 0
    int d = (int)gf_log[a] - (int)gf_log[b];
    if (d < 0) d += GF_ORDER;
    return gf_exp[d];
}

// ---------------- Pack / Unpack FP16 <-> U16 ----------------
__global__ void pack_fp16_pair_to_u16_kernel(const __half* __restrict__ k,
                                             const __half* __restrict__ v,
                                             gf_t* __restrict__ out,
                                             u64 N_k, u64 N_v) {
    u64 idx = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (idx < N_k) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(k);
        out[idx] = p[idx];
    }
    if (idx < N_v) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(v);
        out[N_k + idx] = p[idx];
    }
}

__global__ void u16_to_fp16_split_kernel(const gf_t* __restrict__ in,
                                         __half* __restrict__ k_out,
                                         __half* __restrict__ v_out,
                                         u64 N_k, u64 N_v) {
    u64 idx = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (idx < N_k) {
        uint16_t u = in[idx];
        reinterpret_cast<uint16_t*>(k_out)[idx] = u;
    }
    if (idx < N_v) {
        uint16_t u = in[N_k + idx];
        reinterpret_cast<uint16_t*>(v_out)[idx] = u;
    }
}

// ---------------- RS ENCODE ----------------
__global__ void encode_rs_u16_kernel(const gf_t* __restrict__ data_padded,
                                     gf_t* __restrict__ shards,
                                     int data_shards,
                                     int parity_shards,
                                     u64 shard_size,
                                     const gf_t* __restrict__ gf_exp,
                                     const gf_t* __restrict__ gf_log) {

    u64 tid    = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    u64 stride = (u64)blockDim.x * gridDim.x;
    u64 total_data = (u64)data_shards * shard_size;

    // Copy data shards straight into shards[0 : total_data)
    for (u64 i = tid; i < total_data; i += stride) {
        shards[i] = data_padded[i];
    }

    // Compute parity shards position-wise
    for (u64 pos = tid; pos < shard_size; pos += stride) {
        for (int p = 0; p < parity_shards; ++p) {
            gf_t acc = 0;
            int base = (p + 1);
            for (int k = 0; k < data_shards; ++k) {
                u64 off = (u64)k * shard_size + pos;
                gf_t x = data_padded[off];
                if (x != 0) {
                    int e = (base * k) % GF_ORDER;
                    gf_t coef = gf_exp[e];
                    acc ^= gf_mul(x, coef, gf_exp, gf_log);
                }
            }
            u64 pout = ((u64)data_shards + (u64)p) * shard_size + pos;
            shards[pout] = acc;
        }
    }
}

// ---------------- RS RECONSTRUCT (general m missing data shards) ----------------
__global__ void reconstruct_rs_u16_kernel_inplace(
        gf_t* __restrict__ shards,
        u64 shard_size,
        int data_shards,
        int parity_shards,
        const int* __restrict__ missing_data,
        int m,
        const int* __restrict__ parity_used,
        const gf_t* __restrict__ gf_exp,
        const gf_t* __restrict__ gf_log) {

    if (m == 0) return;

    const int MAX_M = 16;
    if (m > MAX_M) return; // host should guard; kernel still safe

    __shared__ int  s_missing[MAX_M];
    __shared__ int  s_pused[MAX_M];
    __shared__ gf_t s_invV[MAX_M * MAX_M];

    if (threadIdx.x < m) {
        s_missing[threadIdx.x] = missing_data[threadIdx.x];
        s_pused[threadIdx.x]   = parity_used[threadIdx.x];
    }
    __syncthreads();

    u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (t >= shard_size) return;

    // Build and invert V (only once per block; tiny work)
    if (threadIdx.x == 0) {
        gf_t V[MAX_M * MAX_M];

        for (int r = 0; r < m; ++r) {
            int pr = s_pused[r] + 1;
            for (int c = 0; c < m; ++c) {
                int mc = s_missing[c];
                int e = (pr * mc) % GF_ORDER;
                V[r * m + c] = gf_exp[e];
            }
        }

        for (int r = 0; r < m; ++r)
            for (int c = 0; c < m; ++c)
                s_invV[r * m + c] = (r == c) ? (gf_t)1 : (gf_t)0;

        // Gauss-Jordan elimination in GF(2^16)
        for (int i = 0; i < m; ++i) {
            int pivot_row = i;
            while (pivot_row < m && V[pivot_row * m + i] == 0) ++pivot_row;
            if (pivot_row == m) continue;
            if (pivot_row != i) {
                for (int c = 0; c < m; ++c) {
                    gf_t tmp  = V[i * m + c];
                    V[i * m + c] = V[pivot_row * m + c];
                    V[pivot_row * m + c] = tmp;

                    gf_t tmp2 = s_invV[i * m + c];
                    s_invV[i * m + c] = s_invV[pivot_row * m + c];
                    s_invV[pivot_row * m + c] = tmp2;
                }
            }
            gf_t piv = V[i * m + i];
            if (piv == 0) continue;
            gf_t inv_piv = gf_div((gf_t)1, piv, gf_exp, gf_log);
            for (int c = 0; c < m; ++c) {
                V[i * m + c]      = gf_mul(V[i * m + c],      inv_piv, gf_exp, gf_log);
                s_invV[i * m + c] = gf_mul(s_invV[i * m + c], inv_piv, gf_exp, gf_log);
            }
            for (int r = 0; r < m; ++r) {
                if (r == i) continue;
                gf_t factor = V[r * m + i];
                if (factor) {
                    for (int c = 0; c < m; ++c) {
                        V[r * m + c]      ^= gf_mul(factor, V[i * m + c],      gf_exp, gf_log);
                        s_invV[r * m + c] ^= gf_mul(factor, s_invV[i * m + c], gf_exp, gf_log);
                    }
                }
            }
        }
    }
    __syncthreads();

    // Build RHS b(t) at position t and solve x = inv(V) * b
    gf_t b[16]; // MAX_M
    for (int r = 0; r < m; ++r) {
        int pr = s_pused[r];
        u64 off_par = ((u64)data_shards + (u64)pr) * shard_size + t;
        gf_t acc = shards[off_par];
        int base = pr + 1;
        for (int k = 0; k < data_shards; ++k) {
            bool is_missing = false;
            #pragma unroll
            for (int mi = 0; mi < m; ++mi) {
                if (k == s_missing[mi]) { is_missing = true; break; }
            }
            if (is_missing) continue;
            u64 off = (u64)k * shard_size + t;
            gf_t x = shards[off];
            if (x) {
                int e = (base * k) % GF_ORDER;
                gf_t coef = gf_exp[e];
                acc ^= gf_mul(x, coef, gf_exp, gf_log);
            }
        }
        b[r] = acc;
    }

    for (int c = 0; c < m; ++c) {
        gf_t sum = 0;
        for (int r = 0; r < m; ++r) {
            sum ^= gf_mul(s_invV[c * m + r], b[r], gf_exp, gf_log);
        }
        int shard_idx = s_missing[c];
        u64 off = (u64)shard_idx * shard_size + t;
        shards[off] = sum;
    }
}

// ---------------- C++ Bindings ----------------
void pack_fp16_pair_to_u16(torch::Tensor k, torch::Tensor v, torch::Tensor out) {
    TORCH_CHECK(k.is_cuda() && v.is_cuda() && out.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(k.dtype() == torch::kFloat16 && v.dtype() == torch::kFloat16, "k/v must be float16");
    TORCH_CHECK(out.dtype() == torch::kUInt16, "out must be uint16");

    const u64 N_k = (u64)k.numel();
    const u64 N_v = (u64)v.numel();

    int threads = 256;
    int blocks = (int)((std::max(N_k, N_v) + (u64)threads - 1ULL) / (u64)threads);
    if (blocks < 1) blocks = 1;

    pack_fp16_pair_to_u16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        out.data_ptr<gf_t>(),
        N_k, N_v
    );
}

void u16_to_fp16_split(torch::Tensor in, torch::Tensor k_out, torch::Tensor v_out, int64_t N_k, int64_t N_v) {
    TORCH_CHECK(in.is_cuda() && k_out.is_cuda() && v_out.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(in.dtype() == torch::kUInt16, "in must be uint16");
    TORCH_CHECK(k_out.dtype() == torch::kFloat16 && v_out.dtype() == torch::kFloat16, "outputs must be float16");

    u64 n_k = (u64)N_k;
    u64 n_v = (u64)N_v;

    int threads = 256;
    int blocks = (int)((std::max(n_k, n_v) + (u64)threads - 1ULL) / (u64)threads);
    if (blocks < 1) blocks = 1;

    u16_to_fp16_split_kernel<<<blocks, threads>>>(
        in.data_ptr<gf_t>(),
        reinterpret_cast<__half*>(k_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
        n_k, n_v
    );
}

void encode_rs_u16(torch::Tensor data_padded, torch::Tensor shards,
                   int data_shards, int parity_shards, int shard_size,
                   torch::Tensor gf_exp, torch::Tensor gf_log) {
    TORCH_CHECK(data_padded.is_cuda() && shards.is_cuda() && gf_exp.is_cuda() && gf_log.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(data_padded.dtype() == torch::kUInt16 && shards.dtype() == torch::kUInt16, "uint16 expected");
    TORCH_CHECK(gf_exp.dtype() == torch::kUInt16 && gf_log.dtype() == torch::kUInt16, "gf tables must be uint16");

    u64 sz = (u64)shard_size;

    int threads = 256;
    int blocks = (int)((sz + (u64)threads - 1ULL) / (u64)threads);
    if (blocks < 1) blocks = 1;

    encode_rs_u16_kernel<<<blocks, threads>>>(
        data_padded.data_ptr<gf_t>(),
        shards.data_ptr<gf_t>(),
        data_shards, parity_shards, sz,
        gf_exp.data_ptr<gf_t>(),
        gf_log.data_ptr<gf_t>()
    );
}

void reconstruct_rs_u16_inplace(torch::Tensor shards,
                                int shard_size, int data_shards, int parity_shards,
                                torch::Tensor missing_data, torch::Tensor parity_used,
                                torch::Tensor gf_exp, torch::Tensor gf_log) {
    TORCH_CHECK(shards.is_cuda() && gf_exp.is_cuda() && gf_log.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(shards.dtype() == torch::kUInt16, "shards must be uint16");
    TORCH_CHECK(missing_data.dtype() == torch::kInt32 && parity_used.dtype() == torch::kInt32, "indices must be int32");

    int m = (int)missing_data.numel();
    TORCH_CHECK(m <= 16, "m (missing data shards) must be <= 16");

    u64 sz = (u64)shard_size;

    int threads = 256;
    int blocks = (int)((sz + (u64)threads - 1ULL) / (u64)threads);
    if (blocks < 1) blocks = 1;

    reconstruct_rs_u16_kernel_inplace<<<blocks, threads>>>(
        shards.data_ptr<gf_t>(),
        sz, data_shards, parity_shards,
        missing_data.data_ptr<int>(), m,
        parity_used.data_ptr<int>(),
        gf_exp.data_ptr<gf_t>(),
        gf_log.data_ptr<gf_t>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_fp16_pair_to_u16", &pack_fp16_pair_to_u16, "Pack K/V FP16 -> one uint16 stream");
    m.def("u16_to_fp16_split", &u16_to_fp16_split, "Split uint16 stream -> K/V FP16",
          py::arg("in"), py::arg("k_out"), py::arg("v_out"), py::arg("N_k"), py::arg("N_v"));
    m.def("encode_rs_u16", &encode_rs_u16, "RS encode over GF(2^16)");
    m.def("reconstruct_rs_u16_inplace", &reconstruct_rs_u16_inplace, "RS reconstruct missing DATA shards in-place");
}
