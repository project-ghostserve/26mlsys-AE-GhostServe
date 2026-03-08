# GPU KV-Cache Backup/Recovery Kernels (XOR / RS / RDP) for LLM Inference

> CUDA + PyTorch prototypes for **KV-cache protection and recovery** using **XOR parity**, **Reed–Solomon (RS)**, and **Row-Diagonal Parity (RDP)** style encoding on `fp16` key/value tensors.

This repository contains experimental implementations of GPU-accelerated backup/recovery paths for LLM KV cache tensors, including:
- **XOR parity (single-failure recovery)** prototype :contentReference[oaicite:0]{index=0}
- **RS (Reed–Solomon) parity (multi-shard recovery)** prototype :contentReference[oaicite:1]{index=1}
- **RDP-style double-failure recovery** prototype (parallel variant) :contentReference[oaicite:2]{index=2}

The code is designed for **performance experimentation**, **fault-tolerance research**, and **integration studies** for resilient LLM serving systems.

---
## Run

```bash
conda activate GhostServe

cd /benchmark/XOR/
python main.py --num_data_shards 8 --num_parity_shards 1
```
## Implementations Included

### 1) XOR Parity Prototype (Single Failure)
- Packs FP16 K/V into bytes and computes **1 XOR parity shard**
- Can recover **exactly one missing shard**
- Includes exactness checks and attention-level validation :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

### 2) Reed–Solomon (RS) Prototype (Multiple Failures)
- Uses GF(2^16) arithmetic with lookup tables (`gf_exp`, `gf_log`)
- Encodes `num_parity_shards` parity rows for configurable fault tolerance
- Reconstructs multiple missing **data shards** (bounded by parity count) :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

### 3) RDP Prototype (Double Failure, Full-KV Path)
- Packs full KV and computes **RDP parity (P, Q)**
- Supports recovery of up to **two missing data rows**
- Returns full K/V tensors and recovery timing stats :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}

---
## System Requirements (Single GPU)

### Recommended Hardware (Single-GPU Setup)
- **1 NVIDIA GPU** (CUDA-capable)
  - Recommended: RTX 3090 / RTX 4090 / A100 / H100 (or similar)
- **GPU memory**: at least **12 GB VRAM** recommended  
  - More VRAM may be needed for larger `seq`, `heads`, `head_dim`, or shard counts
- CPU with at least **4+ cores**
- System RAM: **16 GB minimum** (**32 GB recommended**)

### Notes for Single-GPU Execution
- These prototypes can run on a **single GPU** for encoding/recovery correctness and performance experiments.
- “Missing shard” behavior is **simulated in software** by erasing selected shard buffers (not by physically failing a GPU).
- Single-GPU mode is ideal for:
  - kernel debugging
  - correctness validation
  - pack/unpack verification
  - timing microbenchmarks
 ---

## Benchmark Structure

```text
.
├── main.py      # Python driver / benchmark / correctness checks
├── kernel.cu    # CUDA extension kernels (XOR / RS / RDP variant depending on repo version)
