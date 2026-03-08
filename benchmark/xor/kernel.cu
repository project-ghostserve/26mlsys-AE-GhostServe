// kernel.cu — XOR parity + FP16<->bytes (IEEE-754 binary16) pack/unpack
// 64-bit safe for sizes/offsets to handle >= 2 GiB byte streams.

#include <stdint.h>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <torch/extension.h>
namespace py = pybind11;

extern "C" {

// ======================= XOR ENCODE =======================
__global__ void encode_xor_kernel(
    const unsigned char* __restrict__ data_padded, // [data_shards * shard_size]
    unsigned char* __restrict__ shards,            // [(data_shards+1) x shard_size]
    int data_shards,
    size_t shard_size)
{
    const size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    const size_t total  = (size_t)data_shards * shard_size;

    // Copy data rows
    for (size_t i = tid; i < total; i += stride) {
        shards[i] = data_padded[i];
    }

    // Compute parity from the immutable input to avoid races
    for (size_t off = tid; off < shard_size; off += stride) {
        unsigned char acc = 0;
        #pragma unroll 4
        for (int k = 0; k < data_shards; ++k) {
            acc ^= data_padded[(size_t)k * shard_size + off];
        }
        shards[(size_t)data_shards * shard_size + off] = acc;
    }
}

// ======================= XOR RECONSTRUCT =======================
// missing_indices: len==1
//  - if missing < data_shards => reconstruct data
//  - if missing == data_shards => recompute parity
__global__ void reconstruct_xor_kernel(
    const unsigned char* __restrict__ encoded_shards, // [(data_shards+1) x shard_size]
    unsigned char* __restrict__ recovered_shards,     // in-place OK
    unsigned char* __restrict__ recovered_missing,    // [1 x shard_size]
    size_t shard_size,
    int data_shards,
    const int* __restrict__ missing_indices,
    int missing_count)
{
    if (missing_count == 0) return;
    if (missing_count > 1) return;

    const int missing = missing_indices[0];
    const size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

    if (missing >= 0 && missing < data_shards) {
        // Reconstruct a data shard: Dm = parity ^ XOR(other data)
        for (size_t off = tid; off < shard_size; off += stride) {
            unsigned char acc = encoded_shards[(size_t)data_shards * shard_size + off]; // parity
            for (int k = 0; k < data_shards; ++k) {
                if (k == missing) continue;
                acc ^= encoded_shards[(size_t)k * shard_size + off];
            }
            recovered_shards[(size_t)missing * shard_size + off] = acc;
            recovered_missing[off] = acc;
        }
        return;
    }

    if (missing == data_shards) {
        // Recompute parity: P = XOR(all data)
        for (size_t off = tid; off < shard_size; off += stride) {
            unsigned char acc = 0;
            for (int k = 0; k < data_shards; ++k) {
                acc ^= encoded_shards[(size_t)k * shard_size + off];
            }
            recovered_shards[(size_t)data_shards * shard_size + off] = acc;
            recovered_missing[off] = acc;
        }
    }
}

// ======================= PACK FP16->BYTES =======================
// Pack K and V into a single byte stream: [K_bytes | V_bytes]
// Each half is written as little-endian uint16 (low byte first, then high byte).
__global__ void pack_kv_kernel(
    const half* __restrict__ key_input,    // [T*n]
    const half* __restrict__ value_input,  // [T*n]
    unsigned char* __restrict__ kv_bytes,  // [T*n*2]K | [T*n*2]V
    int n, int seq_len)
{
    const size_t total  = (size_t)seq_len * (size_t)n;
    const size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

    const uint16_t* k16 = reinterpret_cast<const uint16_t*>(key_input);
    const uint16_t* v16 = reinterpret_cast<const uint16_t*>(value_input);

    const size_t k_off = 0;
    const size_t v_off = total * 2;

    for (size_t i = tid; i < total; i += stride) {
        const uint16_t kb = k16[i];
        const uint16_t vb = v16[i];

        kv_bytes[k_off + 2 * i + 0] = static_cast<unsigned char>(kb & 0xFF);
        kv_bytes[k_off + 2 * i + 1] = static_cast<unsigned char>((kb >> 8) & 0xFF);

        kv_bytes[v_off + 2 * i + 0] = static_cast<unsigned char>(vb & 0xFF);
        kv_bytes[v_off + 2 * i + 1] = static_cast<unsigned char>((vb >> 8) & 0xFF);
    }
}

// ======================= UNPACK BYTES->FP16 =======================
__global__ void unpack_kv_kernel(
    const unsigned char* __restrict__ kv_bytes, // [T*n*2]K | [T*n*2]V
    half* __restrict__ dequant_KV,              // [T*n]K | [T*n]V  (as halves)
    int n, int seq_len)
{
    const size_t total  = (size_t)seq_len * (size_t)n;
    const size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

    uint16_t* dK16 = reinterpret_cast<uint16_t*>(dequant_KV);
    uint16_t* dV16 = reinterpret_cast<uint16_t*>(dequant_KV + total);

    const size_t k_off = 0;
    const size_t v_off = total * 2;

    for (size_t i = tid; i < total; i += stride) {
        const uint16_t kb = (uint16_t)kv_bytes[k_off + 2 * i + 0]
                          | (uint16_t)kv_bytes[k_off + 2 * i + 1] << 8;
        const uint16_t vb = (uint16_t)kv_bytes[v_off + 2 * i + 0]
                          | (uint16_t)kv_bytes[v_off + 2 * i + 1] << 8;

        dK16[i] = kb;
        dV16[i] = vb;
    }
}

} // extern "C"

// ======================= Host launchers & bindings =======================
static inline int calc_blocks(size_t work, int threads = 256) {
    const size_t blocks64 = (work + (size_t)threads - 1) / (size_t)threads;
    // gridDim.x can be large on modern GPUs; cap to a big number to be safe
    const size_t cap = 1ULL << 31; // very high cap; adjust if needed
    return (int)std::min(blocks64, cap);
}

void encode_xor_launcher(torch::Tensor data_padded,
                         torch::Tensor shards,
                         int data_shards,
                         int64_t shard_size)
{
    const int threads = 256;
    const int blocks  = calc_blocks((size_t)std::max<int64_t>((int64_t)data_shards * shard_size, shard_size), threads);

    encode_xor_kernel<<<blocks, threads>>>(
        data_padded.data_ptr<unsigned char>(),
        shards.data_ptr<unsigned char>(),
        data_shards, (size_t)shard_size
    );
}

void reconstruct_xor_launcher(torch::Tensor encoded_data,
                              torch::Tensor recovered_data,
                              torch::Tensor recovered_missing_shards,
                              int64_t shard_size,
                              int data_shards,
                              torch::Tensor missing_indices)
{
    const int threads = 256;
    const int blocks  = calc_blocks((size_t)shard_size, threads);

    reconstruct_xor_kernel<<<blocks, threads>>>(
        encoded_data.data_ptr<unsigned char>(),
        recovered_data.data_ptr<unsigned char>(),
        recovered_missing_shards.data_ptr<unsigned char>(),
        (size_t)shard_size, data_shards,
        missing_indices.data_ptr<int>(),
        (int)missing_indices.numel()
    );
}

void pack_kv_launcher(torch::Tensor key_input, torch::Tensor value_input,
                      torch::Tensor kv_bytes,
                      int n, int seq_len)
{
    const int threads = 256;
    const size_t total = (size_t)seq_len * (size_t)n;
    const int blocks  = calc_blocks(total, threads);

    pack_kv_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(key_input.data_ptr<torch::Half>()),
        reinterpret_cast<const half*>(value_input.data_ptr<torch::Half>()),
        kv_bytes.data_ptr<unsigned char>(),
        n, seq_len
    );
}

void unpack_kv_launcher(torch::Tensor kv_bytes,
                        torch::Tensor dequant_KV,
                        int n, int seq_len)
{
    const int threads = 256;
    const size_t total = (size_t)seq_len * (size_t)n;
    const int blocks  = calc_blocks(total, threads);

    unpack_kv_kernel<<<blocks, threads>>>(
        kv_bytes.data_ptr<unsigned char>(),
        reinterpret_cast<half*>(dequant_KV.data_ptr<torch::Half>()),
        n, seq_len
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_xor", &encode_xor_launcher, "XOR Encode",
          py::arg("data_padded"), py::arg("shards"),
          py::arg("data_shards"), py::arg("shard_size")); // int64_t

    m.def("reconstruct_xor", &reconstruct_xor_launcher, "XOR Reconstruct (single missing shard)",
          py::arg("encoded_data"), py::arg("recovered_data"), py::arg("recovered_missing_shards"),
          py::arg("shard_size"), py::arg("data_shards"), py::arg("missing_indices")); // int64_t

    m.def("pack_kv", &pack_kv_launcher,
          "Pack FP16 KV -> bytes",
          py::arg("key_input"), py::arg("value_input"),
          py::arg("kv_bytes"),
          py::arg("n"), py::arg("seq_len"));

    m.def("unpack_kv", &unpack_kv_launcher,
          "Unpack bytes -> FP16 KV",
          py::arg("kv_bytes"), py::arg("dequant_KV"),
          py::arg("n"), py::arg("seq_len"));
}
