#!/bin/bash
#SBATCH -J canary-transfer
#SBATCH -p ice-cpu
#SBATCH -N1 -c 24
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
# Cross-scaffold: portable features, train on one scaffold and score on the other,
# against an in-domain model trained on the same portable features.
source pace/env.sh
canary tabular swe --eval-on oh && canary evaluate swe_to_oh --data oh
canary tabular oh --eval-on swe && canary evaluate oh_to_swe --data swe
canary tabular oh --portable && canary evaluate oh_portable --data oh
canary tabular swe --portable && canary evaluate swe_portable --data swe
# Percentile-aligned features: same-k percentile within each scaffold's own train split.
canary tabular swe --eval-on oh --align && canary evaluate swe_to_oh_aligned --data oh
canary tabular oh --eval-on swe --align && canary evaluate oh_to_swe_aligned --data swe
canary tabular oh --portable --align && canary evaluate oh_portable_aligned --data oh
