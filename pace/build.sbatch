#!/bin/bash
#SBATCH -J canary-build
#SBATCH -p ice-cpu
#SBATCH -N1 -c 24
#SBATCH --mem=160G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
# Download and parse one scaffold:  sbatch --export=ALL,SCAFFOLD=swe pace/build.sbatch
source pace/env.sh
canary build "${SCAFFOLD:-swe}" --workers 24
