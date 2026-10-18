#!/bin/bash
#SBATCH -J canary-fetch
#SBATCH -p ice-cpu
#SBATCH -N1 -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
# Download model weights once so GPU jobs don't spend their time on it.
source pace/env.sh
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["answerdotai/ModernBERT-base", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
          "sentence-transformers/all-MiniLM-L6-v2"]:
    print(m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"]), flush=True)
PY
