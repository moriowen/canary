#!/bin/bash
#SBATCH -J canary-gpu
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
# Any GPU command:  sbatch --export=ALL,CMD="canary seq swe" pace/gpu.sbatch
source pace/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
eval "$CMD"
