#!/bin/bash
#SBATCH -J canary-tabular
#SBATCH -p ice-cpu
#SBATCH -N1 -c 24
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
# In-domain tabular models + evaluation for one scaffold, then (for swe) the transfer run.
#   sbatch --export=ALL,SCAFFOLD=swe pace/tabular.sbatch
source pace/env.sh
S="${SCAFFOLD:-swe}"
canary tabular "$S"
canary evaluate "$S"
