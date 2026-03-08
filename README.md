# Artifact Evaluation for GhostServe - MLSys 26

---
## Installation

```bash
conda create -n GhostServe python=3.10

conda activate GhostServe

git clone https://github.com/project-ghostserve/26mlsys-AE-GhostServe.git && cd 26mlsys-AE-GhostServe

pip install -r requirements.txt

pip install "flashinfer-python==0.3.1"

export HF_TOKEN="YOUR_TOKEN"

python bench_ghostserve_rdp.py \
--model meta-llama/Llama-3.1-8B-Instruct \
--tp 8 \
--chunk-size 2048 \
--batch-size 16 \
--kv-cache-dtype auto \
--download-dir /work/nvme/bfgy/llm-models \
--gather-kv true --input-tokens 32768
```