#!/bin/bash
#SBATCH -J canary-taxonomy2
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH -o logs/%x-%j.out
# Second taxonomy pass over existing labels: LLM proposes categories, then assigns each run.
source pace/env.sh
VLLM_VENV="${VLLM_VENV:-$HOME/scratch/kvreuse-vllm-venv312}"
S="${SCAFFOLD:-swe}"
PATH="$VLLM_VENV/bin:$PATH" PYTHONPATH="$HOME/scratch/canary-vllm-extras:$PROJECT_DIR/src" \
    "$VLLM_VENV/bin/python" -m canary.cli taxonomy propose_assign "$S"
