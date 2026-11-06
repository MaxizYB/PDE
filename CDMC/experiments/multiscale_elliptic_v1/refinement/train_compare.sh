#!/usr/bin/env bash
set -euo pipefail
family=${1:?smooth or rough}
method=${2:?DCNO or M2NO}
py=/home/caohuilin/.conda/envs/gtmsmw/bin/python
exp=CDMC/experiments/multiscale_elliptic_v1
run=$exp/results/extension_20261103
data=baselines/data/multiscale_elliptic_v1/generated/suite129_${family}_20261103
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$run/$family"
"$py" -u "$exp/refinement/baseline_train.py" --method "$method" \
  --data "$data" --out "$run/$family/$method" --epochs 150
"$py" -u "$exp/refinement/study.py" evaluate --data "$data" \
  --no-dir "$run/$family/$method" \
  --checkpoint "$exp/results/refinement_20261103/balanced128_train/model.pt" \
  --matrix-bound --count 32 --out "$run/$family/${method}_corrected"
