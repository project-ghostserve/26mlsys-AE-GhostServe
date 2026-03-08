#!/usr/bin/env python3
import os
import sys
import math
import random
import argparse
from time import time

import numpy as np
import torch
from torch.utils.cpp_extension import load


def tensor_memory_size(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / (1024 ** 3)


def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---- Load CUDA extension ----
current_dir = os.path.dirname(os.path.abspath(__file__))
rs_kernel = load(
    name="rs_kernel16",
    sources=[os.path.join(current_dir, "kernel.cu")],
    extra_cuda_cflags=["--expt-relaxed-constexpr", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-std=c++17"],
    verbose=False,
)

GF_SIZE = 1 << 16           # 65536
GF_ORDER = GF_SIZE - 1      # 65535


def init_gf_tables():
    # Primitive polynomial for GF(2^16). Keep consistent across encode/decode.
    primitive_polynomial = 0x1100B
    gf_log = np.zeros(GF_SIZE, dtype=np.uint16)
    gf_exp = np.zeros(2 * GF_ORDER, dtype=np.uint16)

    x = 1
    for i in range(GF_ORDER):
        gf_exp[i] = x
        gf_log[x] = i
        x <<= 1
        if x & GF_SIZE:  # x >= 2^16
            x ^= primitive_polynomial

    gf_exp[GF_ORDER:] = gf_exp[:GF_ORDER]
    return gf_exp, gf_log


def shard_size_and_pad(total_symbols: int, num_data_shards: int):
    pad = (-total_symbols) % num_data_shards
    shard_size = (total_symbols + pad) // num_data_shards
    return shard_size, pad


def forward_encode_kv(keys: torch.Tensor,
                      values: torch.Tensor,
                      num_data_shards: int,
                      num_parity_shards: int):

    assert keys.dtype == torch.float16 and values.dtype == torch.float16
    device = keys.device
    assert device.type == "cuda"

    B, H, T, D = keys.shape
    N_k = keys.numel()
    N_v = values.numel()
    total = N_k + N_v


    assert total > 0, "Empty KV tensors"
    assert num_data_shards > 0 and num_parity_shards >= 0

    KV_u16 = torch.empty(total, dtype=torch.uint16, device=device)

    # Pack FP16 -> uint16 stream [K (N_k) | V (N_v)]
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    inner_loops = 10

    # start.record()
    # for _ in range(inner_loops):
    rs_kernel.pack_fp16_pair_to_u16(keys.contiguous().view(-1),
                                    values.contiguous().view(-1),
                                    KV_u16)
    # end.record()
    # torch.cuda.synchronize()
    # t0 = start.elapsed_time(end)

    # Shard sizing (pad to multiple of data shards)
    shard_sz, pad = shard_size_and_pad(total, num_data_shards)
    total_shards = num_data_shards + num_parity_shards

    # Guard: shard_sz must be representable; we use 64-bit in kernels.
    assert shard_sz > 0
    # data_padded holds only data shards; parity computed into `shards`.
    data_padded = torch.empty(num_data_shards * shard_sz,
                              dtype=torch.uint16, device=device)
    if pad:
        data_padded.fill_(0)
    data_padded[:total].copy_(KV_u16)

    shards = torch.empty((total_shards, shard_sz), dtype=torch.uint16, device=device)

    # GF tables (small)
    gf_exp_np, gf_log_np = init_gf_tables()
    gf_exp = torch.from_numpy(gf_exp_np).to(device=device, dtype=torch.uint16)
    gf_log = torch.from_numpy(gf_log_np).to(device=device, dtype=torch.uint16)

    # start.record()
    
    rs_kernel.encode_rs_u16(data_padded, shards, num_data_shards, num_parity_shards, shard_sz, gf_exp, gf_log)
    # end.record()
    # torch.cuda.synchronize()
    # t1 = start.elapsed_time(end)

    # print("pack_fp16_pair_to_u16 time:", t0/inner_loops, "ms")
    # print("encode_rs_u16 time:",       t1/inner_loops, "ms")
    # print("total time for forwardpass:", (t0 + t1)/inner_loops, "ms")

    meta = {
        "B": B, "H": H, "T": T, "D": D,
        "N_k": N_k, "N_v": N_v,
        "total": total,
        "pad": pad,
        "shard_sz": shard_sz,
        "num_data": num_data_shards,
        "num_parity": num_parity_shards,
    }
    return shards, meta


def backwards_decode_kv(shards: torch.Tensor,
                        meta: dict,
                        missing_shards: list[int]):

    device = shards.device
    num_data = meta["num_data"]
    num_parity = meta["num_parity"]
    shard_sz = meta["shard_sz"]

    # Partition what’s missing
    missing_data = sorted([i for i in missing_shards if i < num_data])
    missing_par  = sorted([i - num_data for i in missing_shards if num_data <= i < num_data + num_parity])
    avail_par    = [p for p in range(num_parity) if p not in missing_par]

    if len(missing_data) > len(avail_par):
        raise RuntimeError("Not enough parity shards to reconstruct the missing data shards.")

    # GF tables
    gf_exp_np, gf_log_np = init_gf_tables()
    gf_exp = torch.from_numpy(gf_exp_np).to(device=device, dtype=torch.uint16)
    gf_log = torch.from_numpy(gf_log_np).to(device=device, dtype=torch.uint16)

    missing_data_t = (torch.tensor(missing_data, dtype=torch.int32, device=device)
                      if missing_data else torch.empty(0, dtype=torch.int32, device=device))
    parity_used_t = (torch.tensor(avail_par[:len(missing_data)], dtype=torch.int32, device=device)
                     if missing_data else torch.empty(0, dtype=torch.int32, device=device))

    inner_loops = 10
    t_recon = 0.0

    if len(missing_data) > 0:
        if len(missing_data) > 16:
            raise RuntimeError("Reconstruction supports at most 16 missing data shards at once.")
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        # for _ in range(inner_loops):
        rs_kernel.reconstruct_rs_u16_inplace(
            shards, shard_sz, num_data, num_parity,
            missing_data_t, parity_used_t,
            gf_exp, gf_log
        )
        # end.record()
        # torch.cuda.synchronize()
        # t_recon = start.elapsed_time(end)

    # Take only the true data region (trim padding) and split back into K/V
    data_area = shards[:num_data].contiguous().view(-1)[:meta["total"]]

    K_fp16 = torch.empty(meta["N_k"], dtype=torch.float16, device=device)
    V_fp16 = torch.empty(meta["N_v"], dtype=torch.float16, device=device)

    # start = torch.cuda.Event(enable_timing=True)
    # end   = torch.cuda.Event(enable_timing=True)
    # start.record()
    # for _ in range(inner_loops):
    rs_kernel.u16_to_fp16_split(data_area, K_fp16, V_fp16, meta["N_k"], meta["N_v"])
    # end.record()
    # torch.cuda.synchronize()
    # t_split = start.elapsed_time(end)

    # print("reconstruction time:", (t_recon/inner_loops if len(missing_data) else 0.0), "ms")
    # print("u16_to_fp16_split time:", t_split/inner_loops, "ms")
    # print("total time for backwardpass:", (t_recon + t_split)/inner_loops, "ms")

    B, H, T, D = meta["B"], meta["H"], meta["T"], meta["D"]
    return K_fp16.view(B, H, T, D), V_fp16.view(B, H, T, D)


def main(num_data_shards: int, num_parity_shards: int, missing_shards: list[int]):
    set_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    device = torch.device("cuda")
    B, nh, T, hd = args.batch, args.heads, args.seq, args.head_dim

    keys = torch.randn(B, nh, T, hd, device=device, dtype=torch.float16)
    values = torch.randn_like(keys)

    print(f"Batch size: {B}, Seq length: {T}, Hidden size: {nh*hd}")
    print(f"Tensor size (one of K/V): {tensor_memory_size(keys):.3f} GB")
    
    start_backup = torch.cuda.Event(True); stop_backup = torch.cuda.Event(True)
    start_recovery = torch.cuda.Event(True); stop_recovery = torch.cuda.Event(True)
    torch.cuda.synchronize()
    start_backup.record()
    for _ in range(5):
        shards, meta = forward_encode_kv(keys, values, num_data_shards, num_parity_shards)
    stop_backup.record()
    
    # Erase requested shards (simulate failures)
    shards_erase = shards.clone()
    for m in missing_shards:
        if 0 <= m < shards_erase.size(0):
            shards_erase[m].zero_()
        else:
            raise ValueError(f"Missing shard index {m} out of range 0..{shards_erase.size(0)-1}")

    torch.cuda.synchronize()
    start_recovery.record()
    for _ in range(5):
        K_rec, V_rec = backwards_decode_kv(shards_erase, meta, missing_shards)
    stop_recovery.record()
    torch.cuda.synchronize()
    total_backup = start_backup.elapsed_time(stop_backup)
    total_recovery = start_recovery.elapsed_time(stop_recovery)
    print(f"total backup time: {total_backup/5:.3f} ms")
    print(f"total recovery  time: {total_recovery/5:.3f} ms")
    # Quick functional check with one-step extension of sequence
    q = torch.randn((B, nh, 1, hd), device=device, dtype=torch.float16)
    k_new = torch.randn((B, nh, 1, hd), device=device, dtype=torch.float16)
    v_new = torch.randn((B, nh, 1, hd), device=device, dtype=torch.float16)

    K_rec_cat = torch.cat([K_rec, k_new], dim=2)
    V_rec_cat = torch.cat([V_rec, v_new], dim=2)
    att_w_rec = torch.softmax(torch.matmul(q, K_rec_cat.transpose(2, 3)) / math.sqrt(hd), dim=-1)
    att_o_rec = torch.matmul(att_w_rec, V_rec_cat)

    K_full = torch.cat([keys, k_new], dim=2)
    V_full = torch.cat([values, v_new], dim=2)
    att_w = torch.softmax(torch.matmul(q, K_full.transpose(2, 3)) / math.sqrt(hd), dim=-1)
    att_o = torch.matmul(att_w, V_full)

    mse_attn_output = torch.mean((att_o - att_o_rec) ** 2).item()
    mse_attn_weights = torch.mean((att_w - att_w_rec) ** 2).item()
    mse_keys = torch.mean((K_full.float() - K_rec_cat.float()) ** 2).item()
    mse_values = torch.mean((V_full.float() - V_rec_cat.float()) ** 2).item()

    # print(f"MSE attn_output:  {mse_attn_output:.5e}")
    # print(f"MSE attn_weights: {mse_attn_weights:.5e}")
    # print(f"MSE keys:         {mse_keys:.5e}")
    # print(f"MSE values:       {mse_values:.5e}")

    gap_w = (att_w - att_w_rec) / (att_w.abs() + 1e-6)
    gap_o = (att_o - att_o_rec) / (att_o.abs() + 1e-6)
    # print(f"mean |rel gap| attn_w: {gap_w.abs().mean().item():.5e}")
    # print(f"mean |rel gap| attn_o: {gap_o.abs().mean().item():.5e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_data_shards", type=int, default=8)
    parser.add_argument("--num_parity_shards", type=int, default=2)
    parser.add_argument("--missing_shards", type=int, nargs="+", default=[0,1],
                        help="Indices (0..num_data+num_parity-1) to erase")
    parser.add_argument("--batch",     type=int, default=1)
    parser.add_argument("--heads",     type=int, default=8)
    parser.add_argument("--seq",       type=int, default=256)
    parser.add_argument("--head_dim",  type=int, default=128)
    args = parser.parse_args()

    if len(args.missing_shards) > (args.num_data_shards + args.num_parity_shards):
        print("Error: Too many missing shards!")
        sys.exit(1)

    main(args.num_data_shards, args.num_parity_shards, args.missing_shards)
