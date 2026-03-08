#!/usr/bin/env python3
import os
import sys
import math
import random
import argparse
import numpy as np
import torch
from torch.utils.cpp_extension import load

# =============================== Utils =======================================

def tensor_memory_size(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024 ** 3)

def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def determine_shard_size_words(total_words: int, num_data_shards: int):
    if num_data_shards <= 0:
        raise ValueError("num_data_shards must be > 0")
    pad = (num_data_shards - (total_words % num_data_shards)) % num_data_shards
    padded_total = total_words + pad
    shard_size = padded_total // num_data_shards
    return shard_size, pad

# ======================== Build/Load CUDA extension ===========================
# Adjust archs to your hardware
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")

current_dir = os.path.dirname(os.path.abspath(__file__))
rdp_kernel = load(
    name="rdp_kernel_rdp_parallel",
    sources=[os.path.join(current_dir, "kernel.cu")],
    extra_cuda_cflags=["--expt-relaxed-constexpr", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-std=c++17"],
    verbose=False,
)

# =========================== Two-call API (Python) ============================

def fused_backup_full_kv(keys_bhtd: torch.Tensor,
                         values_bhtd: torch.Tensor,
                         num_data_shards: int,
                         star_base: int = 0):
    """
    Packs full KV (fp16 bit patterns) and computes RDP parities (P, Q) in one call.
    Returns a dict with shards and geometry.
    """
    if not (keys_bhtd.is_cuda and values_bhtd.is_cuda):
        raise ValueError("keys/values must be CUDA tensors.")
    if keys_bhtd.dtype != torch.float16 or values_bhtd.dtype != torch.float16:
        raise ValueError("keys/values must be fp16.")
    if keys_bhtd.shape != values_bhtd.shape:
        raise ValueError("K and V shapes must match [B,H,T,D].")
    if num_data_shards <= 0:
        raise ValueError("num_data_shards must be > 0")

    B, H, T, D = keys_bhtd.shape
    n = B * H * D
    total_words = 2 * T * n  # fp16 words

    # [B,H,T,D] -> [T, n]
    K_TN = keys_bhtd.permute(2, 0, 1, 3).contiguous().view(T, n)
    V_TN = values_bhtd.permute(2, 0, 1, 3).contiguous().view(T, n)

    shard_size, pad_words = determine_shard_size_words(total_words, num_data_shards)
    nd = int(num_data_shards)
    total_shards = nd + 2

    shards = torch.zeros((total_shards, shard_size), dtype=torch.uint16, device=K_TN.device)
    inner_loops = 10
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True); t0.record()

    rdp_kernel.fused_backup_full_kv(K_TN, V_TN, shards, nd, int(shard_size), int(T), int(n), int(star_base))
    t1.record(); torch.cuda.synchronize()
    backup_ms = t0.elapsed_time(t1)

    return {
        "shards_KV": shards,
        "shard_size": int(shard_size),
        "seq_len": int(T),
        "n": int(n),
        "num_data_shards": nd,
        "star_base": int(star_base),
        "shape_BHTD": (B, H, T, D),
        "backup_ms": backup_ms,
        "pad_words": int(pad_words),
    }


def fused_recovery_full_kv(pkg: dict, missing_a: int, missing_b: int = -1):
    """
    Recovers 0/1/2 missing DATA rows (RDP) and loads the full packed K|V into fp16.
    Returns (keys_bhtd, values_bhtd, stats).
    """
    shards     = pkg["shards_KV"]
    shard_size = pkg["shard_size"]
    T          = pkg["seq_len"]
    n          = pkg["n"]
    nd         = pkg["num_data_shards"]
    star_base  = pkg["star_base"]
    B, H, _, D = pkg["shape_BHTD"]

    packed_TN = torch.empty(2 * T * n, dtype=torch.float16, device=shards.device)
    inner_loops = 10
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True); t0.record()

    rdp_kernel.fused_recovery_full_kv(
        shards, packed_TN, int(nd), int(shard_size), int(T), int(n),
        int(star_base), int(missing_a), int(missing_b)
    )
    t1.record(); torch.cuda.synchronize()
    recover_ms = t0.elapsed_time(t1)

    # unpack back to [B,H,T,D]
    dK = packed_TN[:T * n].view(T, n)
    dV = packed_TN[T * n:].view(T, n)
    keys   = dK.view(T, B, H, D).permute(1, 2, 0, 3).contiguous()
    values = dV.view(T, B, H, D).permute(1, 2, 0, 3).contiguous()

    return keys, values, {"recover_ms": recover_ms}

# ================================ Demo =======================================

def main(args):
    set_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this example.")
    device = torch.device("cuda")

    B, nh, T, hd = args.batch, args.heads, args.seq, args.head_dim
    print(f"Batch={B}, Heads={nh}, Seq={T}, HeadDim={hd}, N={B*nh*hd}")
    keys   = torch.randn(B, nh, T, hd, device=device, dtype=torch.float16)
    values = torch.randn(B, nh, T, hd, device=device, dtype=torch.float16)
    print(f"KV tensor size each: {tensor_memory_size(keys):.3f} GiB")

    
    start_backup = torch.cuda.Event(True); stop_backup = torch.cuda.Event(True)
    start_recovery = torch.cuda.Event(True); stop_recovery = torch.cuda.Event(True)
    torch.cuda.synchronize()
    start_backup.record()
    for _ in range(5):
        pkg = fused_backup_full_kv(keys, values, args.num_data_shards, star_base=0)
    stop_backup.record()
    # print(f"Kernel time: {pkg['backup_ms']/10:.3f} ms (pack+parity)")



    ms = [m for m in args.missing_shards if m != -1]
    if len(ms) > 2:
        raise ValueError("RDP can only recover up to two missing data shards.")
    miss_a = ms[0] if len(ms) >= 1 else -1
    miss_b = ms[1] if len(ms) == 2 else -1 
    torch.cuda.synchronize()
    start_recovery.record()
    for _ in range(5):
        keys_q, values_q, stats = fused_recovery_full_kv(pkg, miss_a, miss_b)
    stop_recovery.record()
    torch.cuda.synchronize()
    # print(f"kernel Recovery+Load time: {stats['recover_ms']/10:.3f} ms")
    total_backup = start_backup.elapsed_time(stop_backup)
    total_recovery = start_recovery.elapsed_time(stop_recovery)
    print(f"total backup time: {total_backup/5:.3f} ms")
    print(f"total recovery  time: {total_recovery/5:.3f} ms")

    # correctness check (should be bit-exact)
    query      = torch.randn((B, nh, 1, hd), device=device, dtype=torch.float16)
    keys_new   = torch.randn((B, nh, 1, hd), device=device, dtype=torch.float16)
    values_new = torch.randn((B, nh, 1, hd), device=device, dtype=torch.float16)

    keys_q_   = torch.cat([keys_q,   keys_new],   dim=2)
    values_q_ = torch.cat([values_q, values_new], dim=2)

    att_w_q = torch.matmul(query, keys_q_.transpose(2, 3)) / math.sqrt(hd)
    att_w_q = torch.softmax(att_w_q, dim=-1)
    att_o_q = torch.matmul(att_w_q, values_q_)

    keys_full   = torch.cat([keys,   keys_new],   dim=2)
    values_full = torch.cat([values, values_new], dim=2)
    att_w  = torch.matmul(query, keys_full.transpose(2, 3)) / math.sqrt(hd)
    att_w  = torch.softmax(att_w, dim=-1)
    att_o  = torch.matmul(att_w, values_full)

    mse_keys   = torch.mean((keys_full.float()   - keys_q_.float())   ** 2).item()
    mse_values = torch.mean((values_full.float() - values_q_.float()) ** 2).item()
    mse_w      = torch.mean((att_w - att_w_q) ** 2).item()
    mse_o      = torch.mean((att_o - att_o_q) ** 2).item()

    # print("\n=== Accuracy check ===")
    # print(f"MSE(attn_output):  {mse_o:.6e}")
    # print(f"MSE(attn_weights): {mse_w:.6e}")
    # print(f"MSE(keys):         {mse_keys:.6e}")
    # print(f"MSE(values):       {mse_values:.6e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-KV fp16-pack + RDP (parallel double-failure)")
    parser.add_argument("--num_data_shards",   type=int, default=8, help="Number of data shards")
    parser.add_argument("--missing_shards",    type=int, nargs="*", default=[0, 1],
                        help="Missing DATA shard indices (≤2). Use -1 for none.")
    parser.add_argument("--batch",     type=int, default=1)
    parser.add_argument("--heads",     type=int, default=8)
    parser.add_argument("--seq",       type=int, default=512)
    parser.add_argument("--head_dim",  type=int, default=128)
    args = parser.parse_args()

    if any((m != -1 and m >= args.num_data_shards) for m in args.missing_shards):
        print(f"Error: data shard indices must be -1 or in [0, {args.num_data_shards-1}].")
        sys.exit(1)

    main(args)
