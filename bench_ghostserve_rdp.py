#!/usr/bin/env python3
"""
bench_ghostserve_rdp.py — Fault-tolerant TP inference demo on SGLang using Row-Diagonal Parity (RDP).

Backend setting (paper / recommended):
  • SGLang v0.5.1 as backend; GhostServe integrated as a plug-in module
  • FlashInfer v0.3.1 attention backend (PyTorch 2.8 + CUDA 12.6)

This script launches SGLang and injects a `sitecustomize.py` plugin that:
  • Hooks FlashInferAttnBackend.forward_extend (prefill/extend path)
  • Extracts the prefill-delta KV for the current step using batch.slot_mapping
  • Performs TP-wide gather of delta KV to a rotating dst rank
  • Packs [K|V] fp16 bit-patterns into uint16 rows on GPU
  • Computes RDP parity rows P (row XOR) and Q (diagonal XOR w/ star)
  • Copies parity to CPU

Example:
  python bench_ghostserve_rdp.py --model meta-llama/Llama-3.1-8B-Instruct --tp 4 --chunk-size 2048 \
    --batch-size 1 --kv-cache-dtype auto --download-dir /work/nvme/bfgy/llm-models \
    --gather-kv true --input-tokens 32768

Env toggles (defaults set by this script):
  SGLANG_GATHER_KV_TO_GPU0=1         enable hook
  SGLANG_SNAPSHOT_WEIGHTS=0/1        (optional) snapshot weights on init
  SGLANG_SIMULATE_GPU01_FAIL=0/1     (optional) simulate a failure demo
  RDP_STAR_BASE=0                    RDP diagonal star offset
"""

import argparse, os, sys, time, tempfile, subprocess, statistics, threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import requests
from openai import OpenAI


# ---------------------- KV sizing helpers ----------------------

@dataclass
class ModelSpec:
    layers: int
    n_heads: int
    n_kv_heads: int
    hidden_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads


FALLBACK_SPECS: Dict[str, ModelSpec] = {
    "meta-llama/Llama-3.1-8B-Instruct":  ModelSpec(layers=32, n_heads=32, n_kv_heads=8,  hidden_size=4096),
    "meta-llama/Llama-3.1-70B-Instruct": ModelSpec(layers=80, n_heads=64, n_kv_heads=8,  hidden_size=8192),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": ModelSpec(layers=64, n_heads=40, n_kv_heads=8, hidden_size=5120),
    "openai/gpt-oss-120b":               ModelSpec(layers=36, n_heads=64, n_kv_heads=8,  hidden_size=4096),
}


def load_model_spec(model_id: str) -> ModelSpec:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return ModelSpec(
            layers=int(getattr(cfg, "num_hidden_layers")),
            n_heads=int(getattr(cfg, "num_attention_heads")),
            n_kv_heads=int(getattr(cfg, "num_key_value_heads", int(getattr(cfg, "num_attention_heads")))),
            hidden_size=int(getattr(cfg, "hidden_size")),
        )
    except Exception as e:
        print(f"[warn] AutoConfig load failed for '{model_id}': {e}")
        if model_id in FALLBACK_SPECS:
            print("[info] Using fallback spec.")
            return FALLBACK_SPECS[model_id]
        raise


def kv_bytes(n_layers: int, n_kv_heads: int, head_dim: int, tokens: int, bytes_per_elem: int = 2) -> int:
    # K and V
    return n_layers * tokens * (2 * n_kv_heads * head_dim) * bytes_per_elem


def mib(x: int) -> float:
    return x / (1024 ** 2)


def gib(x: int) -> float:
    return x / (1024 ** 3)


# ---------------------- CUDA extension: pack rows + RDP parity ----------------------

CUDA_RDP_SRC = r'''
#include <stdint.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Pack one row: [K|V] from fp16 into uint16 bit pattern (linear view)
__global__ void pack_row_from_block_kernel(
    const half* __restrict__ k_blk,
    const half* __restrict__ v_blk,
    uint16_t* __restrict__ shards,  // [(nd+2) x shard_size]
    long long words,                // numel(K) for this row
    int row,
    int shard_size)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= words) return;
    uint16_t* row_ptr = shards + (long long)row * shard_size;
    row_ptr[idx]         = __half_as_ushort(k_blk[idx]);
    row_ptr[words + idx] = __half_as_ushort(v_blk[idx]);
}

// RDP parity: P is row-wise XOR; Q is diagonal XOR with one "star" per row
__global__ void compute_rdp_parity_kernel(
    const uint16_t* __restrict__ data_rows, // [nd x S]
    uint16_t* __restrict__ out_rows,        // [(nd+2) x S], P at nd, Q at nd+1
    int nd, int S, int star_base)
{
    // Row parity P
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < S; c += blockDim.x * gridDim.x) {
        uint16_t acc = 0;
        #pragma unroll 4
        for (int r = 0; r < nd; ++r) {
            acc ^= data_rows[r * S + c];
        }
        out_rows[nd * S + c] = acc;
    }
    __syncthreads();

    // Diagonal parity Q (skip one star cell per row)
    for (int off = blockIdx.x * blockDim.x + threadIdx.x; off < S; off += blockDim.x * gridDim.x) {
        uint16_t acc = 0;
        #pragma unroll 4
        for (int r = 0; r < nd; ++r) {
            int c = (int)(((long long)off - r) % S);
            if (c < 0) c += S;
            int star_c = (star_base + r) % S;
            if (c == star_c) continue;
            acc ^= data_rows[r * S + c];
        }
        out_rows[(nd + 1) * S + off] = acc;
    }
}

void pack_row_from_block_launcher(
    torch::Tensor k_blk, torch::Tensor v_blk, torch::Tensor shards, int row)
{
    TORCH_CHECK(k_blk.is_cuda() && v_blk.is_cuda() && shards.is_cuda(), "CUDA tensors required.");
    TORCH_CHECK(k_blk.dtype() == torch::kHalf && v_blk.dtype() == torch::kHalf, "KV must be fp16.");
    TORCH_CHECK(shards.dtype() == torch::kUInt16, "shards must be uint16.");
    TORCH_CHECK(k_blk.is_contiguous() && v_blk.is_contiguous(), "KV must be contiguous.");
    TORCH_CHECK(shards.dim() == 2, "shards must be [rows, S].");

    long long words = k_blk.numel();
    int64_t S = shards.size(1);
    TORCH_CHECK(S >= 2 * words, "shard_size must be >= 2 * words.");

    int threads = 256;
    int blocks  = (int)((words + threads - 1) / threads);

    auto stream = at::cuda::getCurrentCUDAStream();
    pack_row_from_block_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(k_blk.data_ptr<torch::Half>()),
        reinterpret_cast<const half*>(v_blk.data_ptr<torch::Half>()),
        shards.data_ptr<uint16_t>(),
        words, row, (int)S
    );
}

static inline int calc_blocks(int work, int threads = 256) {
    return std::max(1, (work + threads - 1) / threads);
}

void compute_rdp_parity_launcher(
    torch::Tensor rows, // [(nd+2), S] (first nd rows are data)
    int nd, int S, int star_base)
{
    TORCH_CHECK(rows.is_cuda(), "rows must be CUDA");
    TORCH_CHECK(rows.dtype() == torch::kUInt16, "rows must be uint16");
    TORCH_CHECK(rows.dim() == 2 && rows.size(0) >= nd+2 && rows.size(1) >= S, "bad shapes");

    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    int blocks  = calc_blocks(S, threads);

    const uint16_t* data_ptr = rows.data_ptr<uint16_t>();
    uint16_t* out_ptr = rows.data_ptr<uint16_t>();

    compute_rdp_parity_kernel<<<blocks, threads, 0, stream>>>(
        data_ptr, out_ptr, nd, S, star_base
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_row_from_block", &pack_row_from_block_launcher,
          py::arg("k_blk"), py::arg("v_blk"), py::arg("shards"), py::arg("row"));
    m.def("compute_rdp_parity", &compute_rdp_parity_launcher,
          py::arg("rows"), py::arg("nd"), py::arg("S"), py::arg("star_base"));
}
'''


# ---------------------- sitecustomize plugin ----------------------

SITECUSTOMIZE_TEMPLATE = r'''
import os, sys, threading, time, types, tempfile, functools

ENV_FLAG = "SGLANG_GATHER_KV_TO_GPU0"
if os.getenv(ENV_FLAG, "0") != "1":
    raise SystemExit

LOG_PREFIX = "[ft-rdp]"
def _log(msg):
    try:
        print(f"{LOG_PREFIX} {msg}", flush=True)
    except Exception:
        pass

# Toggles
_SNAPSHOT_WEIGHTS = int(os.getenv("SGLANG_SNAPSHOT_WEIGHTS", "0"))
_SIM_FAIL_GPU01   = int(os.getenv("SGLANG_SIMULATE_GPU01_FAIL", "0"))
_STAR_BASE        = int(os.getenv("RDP_STAR_BASE", "0"))

# Metrics export (JSONL)
_METRICS_FILE = os.getenv("FT_METRICS_FILE")
_PROG_START_MONO = float(os.getenv("FT_PROG_START_MONO", "nan"))
_FIRST_CHUNK_WALL_MS = None

def _append_metrics(entry: dict):
    if not _METRICS_FILE:
        return
    try:
        import json
        with open(_METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

# State
_RDP_MOD = None
_PATCHED_FLASHINFER = False
_CHUNK_COUNTER = 0  # global chunk counter (broadcast from rank0)

def _try_imports():
    try:
        import torch
        import torch.distributed as dist
        return torch, dist
    except Exception as e:
        _log(f"torch import failed: {e}")
        return None, None

def _pg_ready(dist):
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False

def _ensure_rdp_mod():
    global _RDP_MOD
    if _RDP_MOD is not None:
        return _RDP_MOD
    import torch
    from torch.utils.cpp_extension import load
    src_dir = tempfile.mkdtemp(prefix="rdp_kv_ops_")
    cu_path = os.path.join(src_dir, "rdp_kv_ops.cu")
    open(cu_path, "w", encoding="utf-8").write("""{{CUDA_RDP_SRC}}""")
    _log(f"Compiling RDP CUDA extension at {cu_path} ...")
    _RDP_MOD = load(
        name="rdp_kv_ops",
        sources=[cu_path],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    _log("RDP CUDA extension loaded.")
    return _RDP_MOD

def _next_chunk_id(torch, dist, device):
    """
    Rank0 increments a counter; broadcast to all ranks.
    Returns (chunk_id:int, dst_rank:int) where dst = (chunk_id - 1) % world.
    """
    global _CHUNK_COUNTER
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        _CHUNK_COUNTER += 1
    t = torch.tensor([_CHUNK_COUNTER], device=device, dtype=torch.int64)
    dist.broadcast(t, src=0)
    cid = int(t.item())
    dst = (cid - 1) % int(world)
    return cid, dst

def _distributed_gather_kv(local_k, local_v, dst_rank):
    """
    Gather flattened local_k and local_v to dst_rank.
    Returns (k_all, v_all, elems_per_rank) on dst_rank, else None.
    """
    torch, dist = _try_imports()
    if torch is None or not _pg_ready(dist):
        return None

    rank = dist.get_rank()
    world = dist.get_world_size()

    kf = local_k.contiguous().view(-1)
    vf = local_v.contiguous().view(-1)
    elems = int(kf.numel())
    if elems == 0:
        return None

    if rank == dst_rank:
        k_all = torch.empty(elems * world, dtype=kf.dtype, device=kf.device)
        v_all = torch.empty(elems * world, dtype=vf.dtype, device=vf.device)
        list_k = [k_all.narrow(0, i * elems, elems) for i in range(world)]
        list_v = [v_all.narrow(0, i * elems, elems) for i in range(world)]
        dist.gather(kf, gather_list=list_k, dst=dst_rank)
        dist.gather(vf, gather_list=list_v, dst=dst_rank)
        return (k_all, v_all, elems)
    else:
        dist.gather(kf, gather_list=None, dst=dst_rank)
        dist.gather(vf, gather_list=None, dst=dst_rank)
        return None

def _compute_parity_to_cpu(K_all, V_all, elems_per_rank, chunk_id, dst_rank):
    import torch, time as _time
    mod = _ensure_rdp_mod()
    world = int(K_all.numel() // elems_per_rank)
    S = 2 * elems_per_rank  # [K|V] words per row

    # rows on dst GPU: [world + 2, S] uint16
    rows = torch.empty((world + 2, S), dtype=torch.uint16, device=K_all.device)

    # pack each rank row: [K_r | V_r]
    for r in range(world):
        k = K_all.narrow(0, r * elems_per_rank, elems_per_rank).to(torch.float16)
        v = V_all.narrow(0, r * elems_per_rank, elems_per_rank).to(torch.float16)
        mod.pack_row_from_block(
            k.view(1, 1, elems_per_rank, 1),
            v.view(1, 1, elems_per_rank, 1),
            rows,
            r
        )

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    mod.compute_rdp_parity(rows, nd=world, S=S, star_base=_STAR_BASE)
    e1.record()
    torch.cuda.synchronize()
    rdp_encode_ms = float(e0.elapsed_time(e1))

    # Copy P & Q to CPU
    t0 = _time.perf_counter()
    P_cpu = rows[world].to("cpu", non_blocking=True).contiguous()
    Q_cpu = rows[world + 1].to("cpu", non_blocking=True).contiguous()
    torch.cuda.synchronize()
    gpu_to_cpu_ms = (_time.perf_counter() - t0) * 1000.0

    # Probe CPU->GPU time (optional)
    t1 = _time.perf_counter()
    _ = P_cpu.to(rows.device, non_blocking=True)
    _ = Q_cpu.to(rows.device, non_blocking=True)
    torch.cuda.synchronize()
    cpu_to_gpu_ms = (_time.perf_counter() - t1) * 1000.0

    global _FIRST_CHUNK_WALL_MS
    record = {
        "chunk_id": int(chunk_id),
        "dst_rank": int(dst_rank),
        "first_chunk_wall_ms": None if _FIRST_CHUNK_WALL_MS is None else float(_FIRST_CHUNK_WALL_MS),
        "timings": {
            "rdp_encode_ms": rdp_encode_ms,
            "rdp_decode_ms": None,
            "gpu_to_cpu_ms": gpu_to_cpu_ms,
            "cpu_to_gpu_ms": cpu_to_gpu_ms,
            "parity_bytes": int((P_cpu.numel() + Q_cpu.numel()) * 2),
        },
    }
    _append_metrics(record)

    mb = (P_cpu.numel() + Q_cpu.numel()) * 2 / (1024 ** 2)
    _log(f"chunk={chunk_id} dst={dst_rank} encode={rdp_encode_ms:.3f}ms D2H={gpu_to_cpu_ms:.3f}ms H2D={cpu_to_gpu_ms:.3f}ms parity={mb:.3f}MiB")

# -------------------- FlashInfer hook --------------------

def _patch_flashinfer_backend(model_runner):
    """
    Hook FlashInferAttnBackend.forward_extend to compute RDP parity on the *prefill delta* (slot_mapping).
    Robust to signature changes by accepting *args, **kwargs.
    """
    global _PATCHED_FLASHINFER
    if _PATCHED_FLASHINFER:
        return True

    torch, dist = _try_imports()
    if torch is None:
        return False

    backend = getattr(model_runner, "attn_backend", None) or getattr(model_runner, "attention_backend", None)
    if backend is None:
        return False

    if "FlashInfer" not in backend.__class__.__name__:
        _log(f"SKIP: backend is {backend.__class__.__name__}, not FlashInfer")
        return False

    if not hasattr(backend, "forward_extend"):
        _log("SKIP: FlashInfer backend missing forward_extend")
        return False

    orig_forward_extend = backend.forward_extend

    @functools.wraps(orig_forward_extend)
    def hooked_forward_extend(*args, **kwargs):
        # IMPORTANT: forward all args/kwargs to preserve SGLang signature compatibility (e.g., save_kv_cache=...)
        res = orig_forward_extend(*args, **kwargs)

        try:
            torch, dist = _try_imports()
            if torch is None or not _pg_ready(dist):
                return res

            # Find "batch" object
            batch = kwargs.get("batch", None)
            if batch is None and len(args) >= 1:
                batch = args[0]
            if batch is None:
                return res

            # KV pool buffer (FlashInfer layout often: [Max_Tokens, 2, Num_KV_Heads, Head_Dim])
            kv_pool = getattr(backend, "kv_buffer", None) or getattr(backend, "kv_pool", None) or getattr(backend, "kv_cache", None)
            if kv_pool is None:
                return res

            slots = getattr(batch, "slot_mapping", None)
            if slots is None:
                return res
            if not isinstance(slots, torch.Tensor):
                slots = torch.as_tensor(slots, device=kv_pool.device)

            active = (slots != -1)
            if int(active.sum().item()) == 0:
                return res
            active_slots = slots[active].to(dtype=torch.long)

            # Delta KV for this extend step
            local_k = kv_pool[active_slots, 0].contiguous()
            local_v = kv_pool[active_slots, 1].contiguous()

            # Chunk counter + rotating dst = (chunk_id - 1) % world
            chunk_id, dst_rank = _next_chunk_id(torch, dist, device=local_k.device)

            # record time-to-first-chunk once
            global _FIRST_CHUNK_WALL_MS
            if _FIRST_CHUNK_WALL_MS is None and _PROG_START_MONO == _PROG_START_MONO:
                _FIRST_CHUNK_WALL_MS = (time.perf_counter() - _PROG_START_MONO) * 1000.0

            gathered = _distributed_gather_kv(local_k, local_v, dst_rank=dst_rank)
            if gathered is not None:
                k_all, v_all, elems = gathered
                _compute_parity_to_cpu(k_all, v_all, elems, chunk_id, dst_rank)

        except Exception as e:
            _log(f"FlashInfer hook error: {e}")

        return res

    backend.forward_extend = hooked_forward_extend
    _PATCHED_FLASHINFER = True
    _log("Successfully hooked FlashInferAttnBackend.forward_extend")
    return True

# -------------------- Patch discovery --------------------

_TARGET_MODULE_PREFIX = "sglang.srt"

def _patch_modelrunner_init(mod):
    MR = getattr(mod, "ModelRunner", None)
    if MR is None:
        return False
    if getattr(MR, "_ft_rdp_patched_init", False):
        return True

    orig_init = getattr(MR, "__init__", None)
    if not callable(orig_init):
        return False

    def wrapped_init(self, *a, **kw):
        out = orig_init(self, *a, **kw)
        try:
            _patch_flashinfer_backend(self)
        except Exception as e:
            _log(f"FlashInfer patch in __init__ failed: {e}")
        return out

    setattr(MR, "__init__", wrapped_init)
    setattr(MR, "_ft_rdp_patched_init", True)
    _log("patched ModelRunner.__init__ to hook FlashInfer backend")
    return True

def _watch_and_patch():
    deadline = time.time() + 600.0
    while time.time() < deadline:
        did = False
        # Snapshot keys to avoid: RuntimeError: dictionary changed size during iteration
        names = list(sys.modules.keys())
        for name in names:
            mod = sys.modules.get(name, None)
            if not isinstance(mod, types.ModuleType):
                continue
            if not name.startswith(_TARGET_MODULE_PREFIX):
                continue
            try:
                if _patch_modelrunner_init(mod):
                    did = True
            except Exception as e:
                _log(f"patch failed in {name}: {e}")
        time.sleep(1.0 if did else 0.2)
    _log("watcher timeout; if no hook logs appeared, adjust hook targets.")

threading.Thread(target=_watch_and_patch, daemon=True).start()
'''


def _write_sitecustomize(tmpdir: str) -> str:
    path = os.path.join(tmpdir, "sitecustomize.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SITECUSTOMIZE_TEMPLATE.replace("{{CUDA_RDP_SRC}}", CUDA_RDP_SRC))
    return path


# ---------------------- server orchestration ----------------------

def launch_sglang_server(
    model_id: str,
    tp: int,
    chunk_size: int,
    port: int,
    host: str,
    gpus: str,
    python_bin: str,
    extra_args: Optional[List[str]],
    kv_cache_dtype: str,
    download_dir: Optional[str],
    gather_kv: bool,
    prog_start_mono: float,
) -> Tuple[subprocess.Popen, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")

    # Prefer NVLink/NVSwitch intra-node paths when present
    env.setdefault("NCCL_P2P_LEVEL", "NVL")
    # If you rely on NCCL-IB, remove/override this
    env.setdefault("NCCL_IB_DISABLE", "1")

    env["SGLANG_GATHER_KV_TO_GPU0"] = "1" if gather_kv else "0"
    env.setdefault("SGLANG_SNAPSHOT_WEIGHTS", "0")
    env.setdefault("SGLANG_SIMULATE_GPU01_FAIL", "0")
    env.setdefault("RDP_STAR_BASE", "0")

    tmpdir = tempfile.mkdtemp(prefix="kvhook_rdp_")
    sc_path = _write_sitecustomize(tmpdir)
    metrics_file = os.path.join(tmpdir, "ft_metrics.jsonl")

    env["PYTHONPATH"] = f"{tmpdir}:{env.get('PYTHONPATH','')}"
    env["FT_METRICS_FILE"] = metrics_file
    env["FT_PROG_START_MONO"] = f"{prog_start_mono:.9f}"

    print(f"[info] Injected sitecustomize at: {sc_path}")
    print(f"[info] Server metrics file: {metrics_file}")

    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model", model_id,
        "--tensor-parallel-size", str(tp),
        "--chunked-prefill-size", str(chunk_size),
        "--enable-mixed-chunk",
        "--dtype", "bfloat16",
        "--kv-cache-dtype", kv_cache_dtype,
        "--host", host, "--port", str(port),
        "--trust-remote-code",
    ]
    if download_dir:
        cmd += ["--download-dir", download_dir]
    if extra_args:
        cmd.extend(extra_args)

    print("[info] Launching SGLang server:\n       " + " ".join(cmd))
    print(f"[info] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pump(p):
        try:
            for line in iter(p.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write("[sglang] " + line)
        except Exception:
            pass

    threading.Thread(target=_pump, args=(proc,), daemon=True).start()
    return proc, metrics_file


def wait_until_ready(base_url: str, timeout_s: int = 900) -> None:
    t0 = time.time()
    last_error = None
    print(f"[info] Waiting for server ready at {base_url} ...")
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url}/models", timeout=5)
            if r.status_code == 200:
                print("[info] Server is ready.")
                return
            last_error = f"HTTP {r.status_code} {r.text[:120]}"
        except Exception as e:
            last_error = str(e)
        time.sleep(2)
    raise RuntimeError(f"SGLang server did not become ready in {timeout_s}s. Last error: {last_error}")


def terminate_process(proc: Optional[subprocess.Popen], grace_s: float = 10.0):
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass


# ---------------------- simple TTFT client ----------------------

def _ttft_one(base_url: str, api_key: str, prompt: str, max_new_tokens: int,
              start_gate: threading.Barrier, idx: int, out: List[float]):
    client = OpenAI(base_url=base_url, api_key=api_key)
    start_gate.wait()
    t0 = time.perf_counter()
    first_ms = None
    stream = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.0,
        max_tokens=max_new_tokens,
    )
    for event in stream:
        if getattr(event, "choices", None):
            delta = event.choices[0].delta
            if (delta and (delta.content or delta.role)) and first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000.0
                if idx == 0 and getattr(delta, "content", None):
                    print(delta.content, end="", flush=True)
    if idx == 0:
        print()
    out[idx] = first_ms if first_ms is not None else float("nan")


def measure_ttft_batch(base_url: str, api_key: str, prompt: str,
                       batch_size: int, max_new_tokens: int = 64,
                       input_tokens: Optional[int] = None,
                       model_id: Optional[str] = None) -> List[float]:
    if input_tokens is not None:
        assert model_id is not None, "model_id is required when input_tokens is set"
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        base = "hello "
        base_ids = tok.encode(base, add_special_tokens=False)
        rep = max(1, (input_tokens // max(1, len(base_ids))) + 1)
        text = base * rep
        ids = tok.encode(text, add_special_tokens=False)[:input_tokens]
        prompt = tok.decode(ids)

    out: List[float] = [float("nan")] * batch_size
    gate = threading.Barrier(parties=batch_size)
    threads = []
    for i in range(batch_size):
        t = threading.Thread(target=_ttft_one, args=(base_url, api_key, prompt, max_new_tokens, gate, i, out))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return out


# ---------------------- main ----------------------

def main():
    parser = argparse.ArgumentParser(description="GhostServe: FlashInfer delta-KV hook + RDP parity + JSONL metrics (SGLang).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--prompt-tokens", type=int, default=3000, help="Used only for KV sizing math.")
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8_e5m2", "fp8_e4m3"])
    parser.add_argument("--kv-bytes-per-elem", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--prompt", default="You are a concise assistant. Explain chunked prefill for TP inference in 2 sentences. Then give a one-line use case.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY", "EMPTY"))
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--download-dir", default="/work/nvme/bfgy/llm-models")
    parser.add_argument("--gather-kv", default="true", choices=["true", "false"])
    parser.add_argument("--input-tokens", type=int, default=None, help="If set, generate a synthetic prompt with ~this many tokens.")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, help="Pass-through args for sglang.launch_server after this flag")
    args = parser.parse_args()

    if args.kv_bytes_per_elem is None:
        args.kv_bytes_per_elem = 1 if args.kv_cache_dtype.startswith("fp8") else 2

    spec = load_model_spec(args.model)
    head_dim = spec.head_dim

    first_chunk_bytes_single = kv_bytes(spec.layers, spec.n_kv_heads, head_dim, args.chunk_size, args.kv_bytes_per_elem)
    full_prompt_bytes_single = kv_bytes(spec.layers, spec.n_kv_heads, head_dim, args.prompt_tokens, args.kv_bytes_per_elem)

    print("\n========== KV Cache Sizing ==========")
    print(f"Model: {args.model}")
    print(f"Layers={spec.layers}, heads={spec.n_heads}, kv_heads={spec.n_kv_heads}, head_dim={head_dim}")
    print(f"KV cache dtype: {args.kv_cache_dtype}  (bytes/elem={args.kv_bytes_per_elem})")
    print(f"First chunk tokens: {args.chunk_size}")
    print(f"Total prompt tokens: {args.prompt_tokens}   # used for sizing only")
    print(f"Batch size: {args.batch_size}")
    print(f"Per-request KV (first chunk): {mib(first_chunk_bytes_single):.2f} MiB")
    print(f"Per-request KV (prompt total): {gib(full_prompt_bytes_single):.3f} GiB")
    print("====================================\n")

    base_url = (
        f"http://127.0.0.1:{args.port}/v1"
        if args.host in ("0.0.0.0", "127.0.0.1", "localhost")
        else f"http://{args.host}:{args.port}/v1"
    )

    prog_start_mono = time.perf_counter()

    proc = None
    metrics_file = None
    try:
        if not args.no_launch:
            proc, metrics_file = launch_sglang_server(
                model_id=args.model,
                tp=args.tp,
                chunk_size=args.chunk_size,
                port=args.port,
                host=args.host,
                gpus=args.gpus,
                python_bin=args.server_python,
                extra_args=args.server_args,
                kv_cache_dtype=args.kv_cache_dtype,
                download_dir=args.download_dir,
                gather_kv=(args.gather_kv == "true"),
                prog_start_mono=prog_start_mono,
            )

        wait_until_ready(base_url, timeout_s=900)

        print("\n========== Inference (streamed batch) ==========")
        ttft = measure_ttft_batch(
            base_url=base_url,
            api_key=args.api_key,
            prompt=args.prompt,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            input_tokens=args.input_tokens,
            model_id=args.model,
        )
        clean = [x for x in ttft if x == x]
        print("[result] ttft per request (ms):", ", ".join(f"{x:.2f}" for x in ttft))
        if clean:
            p95 = statistics.quantiles(clean, n=20)[18] if len(clean) >= 20 else max(clean)
            print(f"[result] ttft stats (ms): min={min(clean):.2f} median={statistics.median(clean):.2f} mean={statistics.mean(clean):.2f} p95={p95:.2f}")
        else:
            print("[warn] No valid ttft values captured.")
    finally:
        if proc is not None:
            print("[info] Shutting down SGLang server...")
            terminate_process(proc)

    # Best-effort: print last JSONL line
    try:
        if metrics_file and os.path.exists(metrics_file):
            with open(metrics_file, "rb") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    pos = f.tell() - 1
                    while pos > 0:
                        f.seek(pos)
                        if f.read(1) == b"\n":
                            break
                        pos -= 1
                    line = f.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        print("\n========== Last FT Metrics (JSONL) ==========")
                        print(line)
                        print("===========================================\n")
    except Exception as e:
        print(f"[warn] Could not parse server metrics: {e}")


if __name__ == "__main__":
    main()
