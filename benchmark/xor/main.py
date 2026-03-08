#!/usr/bin/env python3
import os
import sys
import math
import random
import argparse
import numpy as np
import torch
from torch.utils.cpp_extension import load

# ------------------------------- Utils ---------------------------------------

def tensor_memory_size(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024 ** 3)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def gbps(total_bytes: int, ms: float) -> float:
    if ms <= 0:
        return float("inf")
    return (total_bytes / 1e9) / (ms / 1e3)

# --------------------------- Load CUDA extension -----------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
xor_kernel = load(
    name='xor_kernel',
    sources=[os.path.join(current_dir, 'kernel.cu')],
    extra_cuda_cflags=["--expt-relaxed-constexpr", "--use_fast_math"],
    extra_cflags=["-std=c++17"],
    verbose=False,
)

# ------------------------------ Helpers --------------------------------------

def determine_shard_size(num_bytes: int, num_data_shards: int):
    pad = (num_data_shards - (num_bytes % num_data_shards)) % num_data_shards
    padded_total = num_bytes + pad
    shard_size = padded_total // num_data_shards
    return shard_size, pad

# ------------------------------ Forward path ---------------------------------

def forward_encode_kv(keys, values, num_data_shards, num_parity_shards):
    """
    Bit-cast FP16 K,V to bytes (IEEE-754 binary16 -> uint16 -> 2 bytes each),
    then XOR-encode into (num_data_shards + 1) shards.

    Byte layout: [K_bytes (T*n*2)] | [V_bytes (T*n*2)]
    """
    if num_parity_shards != 1:
        raise ValueError("XOR parity supports exactly 1 parity shard.")

    assert keys.is_cuda and values.is_cuda, "Run on CUDA for GPU timing."
    assert keys.dtype == torch.float16 and values.dtype == torch.float16
    assert keys.shape == values.shape

    # Ensure linear memory order matches our pack/unpack flattening
    keys   = keys.contiguous()
    values = values.contiguous()

    device = keys.device
    B, H, T, D = keys.shape
    n = B * H * D

    total_bytes = 4 * T * n  # 2 bytes per half, for K and V

    kv_bytes = torch.empty(total_bytes, dtype=torch.uint8, device=device)

    # --- Pack timing ---
    # torch.cuda.synchronize()
    # p_start = torch.cuda.Event(enable_timing=True)
    # p_end   = torch.cuda.Event(enable_timing=True)
    # inner_loops = 10
    # p_start.record()
    # for _ in range(inner_loops):
    xor_kernel.pack_kv(keys, values, kv_bytes, int(n), int(T))
    # p_end.record()
    # torch.cuda.synchronize()
    # pack_ms = p_start.elapsed_time(p_end)

    # --- XOR encode timing ---
    shard_size, pad = determine_shard_size(kv_bytes.numel(), num_data_shards)
    data_bytes = num_data_shards * shard_size

    data_padded = torch.zeros(data_bytes, dtype=torch.uint8, device=device)
    data_padded[:kv_bytes.numel()] = kv_bytes

    total_shards = num_data_shards + num_parity_shards
    shards_KV = torch.zeros((total_shards, shard_size), dtype=torch.uint8, device=device)

    e_start = torch.cuda.Event(enable_timing=True)
    e_end   = torch.cuda.Event(enable_timing=True)
    
    
    # e_start.record()
    # for _ in range(inner_loops):
    xor_kernel.encode_xor(data_padded, shards_KV, int(num_data_shards), int(shard_size))
    # e_end.record()
    # torch.cuda.synchronize()
    # encode_ms = e_start.elapsed_time(e_end)

    # I/O estimates (approx)
    pack_reads  = 4 * T * n
    pack_writes = total_bytes
    pack_io     = pack_reads + pack_writes

    encode_reads  = data_bytes
    encode_writes = data_bytes + shard_size
    encode_io     = encode_reads + encode_writes

    return {
        "shards_KV": shards_KV,
        "seq_len": T,
        "n": n,
        "shard_size": shard_size,
        "device": device,
        "shape_BHTD": (B, H, T, D),
        # "pack_ms": pack_ms,
        # "encode_ms": encode_ms,
        "pack_io": pack_io,
        "encode_io": encode_io,
        "num_data_shards": num_data_shards,
        "num_parity_shards": num_parity_shards,
        "pad": pad,
        "total_bytes": total_bytes,
        # keep originals for exactness compare
        "orig_keys": keys,
        "orig_vals": values,
    }

# ------------------------------ Backward path --------------------------------

def backwards_decode_kv(pkg, missing_shards):
    """
    Reconstruct (if 1 missing shard), then unpack bytes back to fp16 K,V (bit-exact).
    """
    shards_KV        = pkg["shards_KV"]
    T                = pkg["seq_len"]
    n                = pkg["n"]
    shard_size       = pkg["shard_size"]
    device           = pkg["device"]
    B, H, _, D       = pkg["shape_BHTD"]
    num_data_shards  = pkg["num_data_shards"]
    num_parity_shards= pkg["num_parity_shards"]
    total_shards     = num_data_shards + num_parity_shards

    if num_parity_shards != 1:
        raise ValueError("XOR parity supports exactly 1 parity shard.")
    if len(missing_shards) > 1:
        raise ValueError("XOR parity can recover only a single missing shard.")
    for idx in missing_shards:
        if idx < 0 or idx >= total_shards:
            raise ValueError(f"Missing shard {idx} out of range 0..{total_shards-1}")

    # Zero the explicitly missing shard for clarity
    for idx in missing_shards:
        shards_KV[idx].zero_()

    torch.cuda.synchronize()
    r_ms = 0.0
    if len(missing_shards) == 1:
        missing_idx_tensor = torch.tensor(missing_shards, dtype=torch.int32, device=device)
        recovered_missing = torch.empty((1, shard_size), dtype=torch.uint8, device=device)

        # r_start = torch.cuda.Event(enable_timing=True)
        # r_end   = torch.cuda.Event(enable_timing=True)
        # inner_loops = 10
        # r_start.record()
        # for _ in range(inner_loops):
        xor_kernel.reconstruct_xor(
            shards_KV, shards_KV, recovered_missing,
            int(shard_size), int(num_data_shards), missing_idx_tensor
        )
        # r_end.record()
        # torch.cuda.synchronize()
        # r_ms = r_start.elapsed_time(r_end)

    # Flatten data region back to byte stream
    byte_region = shards_KV[:num_data_shards].contiguous().view(-1)
    de_bytes_needed = 4 * T * n
    if byte_region.numel() < de_bytes_needed:
        raise RuntimeError("Not enough bytes in reconstructed data region.")

    dequant_KV = torch.empty(2 * T * n, dtype=torch.float16, device=device)

    # d_start = torch.cuda.Event(enable_timing=True)
    # d_end   = torch.cuda.Event(enable_timing=True)
    # d_start.record()
    # for _ in range(inner_loops):
    xor_kernel.unpack_kv(byte_region, dequant_KV, int(n), int(T))
    # d_end.record()
    # torch.cuda.synchronize()
    # unpack_ms = d_start.elapsed_time(d_end)

    # IO (approx)
    de_reads  = de_bytes_needed
    de_writes = 4 * T * n
    de_io     = de_reads + de_writes

    keys_out   = dequant_KV[:T * n].view(B, H, T, D)
    values_out = dequant_KV[T * n : 2 * T * n].view(B, H, T, D)

    return keys_out, values_out, {
        "reconstruct_ms": r_ms,
        # "unpack_ms": unpack_ms,
        "reconstruct_io": ((num_data_shards + 1) * shard_size + 2 * shard_size) if r_ms > 0 else 0,
        "unpack_io": de_io
    }

# ----------------------------------- Demo ------------------------------------

def run(args):
    set_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this example.")

    device = torch.device("cuda")

    B, nh, T, hd = args.batch, args.heads, args.seq, args.head_dim
    keys   = torch.randn(B, nh, T, hd, device=device, dtype=torch.float16).contiguous()
    values = torch.randn(B, nh, T, hd, device=device, dtype=torch.float16).contiguous()

    print(f'Batch={B}, Heads={nh}, Seq={T}, HeadDim={hd}, N={B*nh*hd}')
    print(f'KV tensor size (fp16) each: {tensor_memory_size(keys):.3f} GB')
    start_backup = torch.cuda.Event(True); stop_backup = torch.cuda.Event(True)
    start_recovery = torch.cuda.Event(True); stop_recovery = torch.cuda.Event(True)
    torch.cuda.synchronize()
    start_backup.record()
    for _ in range(5):
        pkg = forward_encode_kv(keys, values, args.num_data_shards, args.num_parity_shards)
    stop_backup.record()
    
    torch.cuda.synchronize()
    start_recovery.record()
    for _ in range(5):
        keys_rt, values_rt, back_prof = backwards_decode_kv(pkg, args.missing_shards)
    stop_recovery.record()
    # ---------------------- Profiling summary ---------------------------------
    # print("\n=== Profiling (CUDA events) ===")
    # print(f"Pack:   {(pkg['pack_ms'])/10:.3f} ms   ")
    # print(f"Encode: {(pkg['encode_ms'])/10:.3f} ms   ")
    # print(f"Pack+Encode:   {(pkg['pack_ms']+pkg['encode_ms'])/10:.3f} ms   ")
    
    # print(f"Reconstruct:   {(back_prof['reconstruct_ms'])/10:.3f} ms")
    # print(f"Unpack:   {(back_prof['unpack_ms'])/10:.3f} ms   ")
    # print(f"Reconstruct+unpack:   {(back_prof['reconstruct_ms']+back_prof['unpack_ms'])/10:.3f} ms")
    total_backup = start_backup.elapsed_time(stop_backup)
    total_recovery = start_recovery.elapsed_time(stop_recovery)
    print(f"total backup time: {total_backup:.3f} ms")
    print(f"total recovery  time: {total_recovery:.3f} ms")
    # ---------------------- Exactness checks ----------------------------------
    k_orig = pkg["orig_keys"]
    v_orig = pkg["orig_vals"]

    exact_keys = torch.equal(k_orig, keys_rt)
    exact_vals = torch.equal(v_orig, values_rt)

    # Byte-level proof (bit-for-bit)
    k_bytes_orig = k_orig.view(torch.uint16).contiguous().view(torch.uint8)
    v_bytes_orig = v_orig.view(torch.uint16).contiguous().view(torch.uint8)
    k_bytes_rt   = keys_rt.view(torch.uint16).contiguous().view(torch.uint8)
    v_bytes_rt   = values_rt.view(torch.uint16).contiguous().view(torch.uint8)

    # print("\n=== Exactness check ===")
    # print(f"keys   equal (tensor): {exact_keys}")
    # print(f"values equal (tensor): {exact_vals}")
    # print(f"bytes(keys)   equal:   {torch.equal(k_bytes_orig, k_bytes_rt)}")
    # print(f"bytes(values) equal:   {torch.equal(v_bytes_orig, v_bytes_rt)}")

    # ---------------------- Functional check in FP32 --------------------------
    query_fp32      = torch.randn((B, nh, 1, hd), device=device, dtype=torch.float32)
    keys_new_fp32   = torch.randn((B, nh, 1, hd), device=device, dtype=torch.float32)
    values_new_fp32 = torch.randn((B, nh, 1, hd), device=device, dtype=torch.float32)

    keys32     = k_orig.float()
    values32   = v_orig.float()
    keys_rt32  = keys_rt.float()
    values_rt32= values_rt.float()

    keys_full32   = torch.cat([keys32,   keys_new_fp32],   dim=2)
    values_full32 = torch.cat([values32, values_new_fp32], dim=2)
    keys_rt_full32   = torch.cat([keys_rt32,   keys_new_fp32],   dim=2)
    values_rt_full32 = torch.cat([values_rt32, values_new_fp32], dim=2)

    att_w  = torch.matmul(query_fp32, keys_full32.transpose(2, 3)) / math.sqrt(hd)
    att_w  = torch.softmax(att_w, dim=-1)
    att_o  = torch.matmul(att_w, values_full32)

    att_w_rt = torch.matmul(query_fp32, keys_rt_full32.transpose(2, 3)) / math.sqrt(hd)
    att_w_rt = torch.softmax(att_w_rt, dim=-1)
    att_o_rt = torch.matmul(att_w_rt, values_rt_full32)

    mse_o = torch.mean((att_o - att_o_rt) ** 2).item()
    mse_w = torch.mean((att_w - att_w_rt) ** 2).item()
    print(f"\nMSE(attn_output, fp32):  {mse_o:.6e}")
    print(f"MSE(attn_weights, fp32): {mse_w:.6e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_data_shards",   type=int, default=8, help="Number of data shards")
    parser.add_argument("--num_parity_shards", type=int, default=1, help="XOR supports exactly 1 parity shard")
    parser.add_argument("--missing_shards",    type=int, nargs="+", default=[3],
                        help="Single missing shard index (0..data+parity-1). XOR can recover one.")
    parser.add_argument("--batch",     type=int, default=1)
    parser.add_argument("--heads",     type=int, default=8)
    parser.add_argument("--seq",       type=int, default=256)
    parser.add_argument("--head_dim",  type=int, default=128)
    args = parser.parse_args()

    total = args.num_data_shards + args.num_parity_shards
    if args.num_parity_shards != 1:
        print("Error: XOR parity supports exactly 1 parity shard.")
        sys.exit(1)
    if len(args.missing_shards) > 1:
        print("Error: XOR parity can only recover a single missing shard.")
        sys.exit(1)
    if any((m < 0 or m >= total) for m in args.missing_shards):
        print(f"Error: missing shard index must be in [0, {total-1}].")
        sys.exit(1)

    run(args)

if __name__ == "__main__":
    main()
