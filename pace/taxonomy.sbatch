#!/bin/bash
#SBATCH -J canary-taxonomy
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
# label (vLLM venv) -> cluster (canary venv) -> name (vLLM venv)
# The vLLM venv is shared with another project; pandas/pyarrow for it live in
# ~/scratch/canary-vllm-extras (pip install --target, numpy removed so vLLM keeps its own).
source pace/env.sh
VLLM_VENV="${VLLM_VENV:-$HOME/scratch/kvreuse-vllm-venv312}"
S="${SCAFFOLD:-swe}"
vllm_py() { PYTHONPATH="$HOME/scratch/canary-vllm-extras:$PROJECT_DIR/src" "$VLLM_VENV/bin/python" -m canary.cli "$@"; }
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -f "$CANARY_RESULTS/$S/taxonomy/labels.parquet" ] || vllm_py taxonomy label "$S" ${LIMIT:+--limit $LIMIT}
canary taxonomy cluster "$S" ${K:+--k $K}
vllm_py taxonomy name "$S"
