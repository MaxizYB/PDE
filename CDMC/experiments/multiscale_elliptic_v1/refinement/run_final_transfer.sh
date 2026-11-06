#!/usr/bin/env bash
set -euo pipefail
family=${1:?smooth or rough}
py=/home/caohuilin/.conda/envs/gtmsmw/bin/python
exp=CDMC/experiments/multiscale_elliptic_v1
run=$exp/results/refinement_20261103
data=baselines/data/multiscale_elliptic_v1/generated/suite129_${family}_20261103
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for base in fno_h1 mgno; do
  if [ -e "$run/$family/frozen128_$base" ]; then
    echo "Existing transfer run: choose a new output before repeating." >&2
    exit 2
  fi
  "$py" -u "$exp/refinement/study.py" evaluate --data "$data" --no-dir "$run/$family/$base" \
    --checkpoint "$run/balanced128_train/model.pt" --out "$run/$family/frozen128_$base" \
    --count 32 --matrix-bound > "$run/$family/frozen128_$base.log" 2>&1
done
