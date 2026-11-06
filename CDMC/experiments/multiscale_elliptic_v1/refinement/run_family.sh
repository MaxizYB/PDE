#!/usr/bin/env bash
# A single coefficient family, identical solver/model/loss settings for all.
set -euo pipefail
family=${1:?smooth or rough}
py=/home/caohuilin/.conda/envs/gtmsmw/bin/python
exp=CDMC/experiments/multiscale_elliptic_v1
data=baselines/data/multiscale_elliptic_v1/generated/suite129_${family}_20261103
run_root=${2:-$exp/results/refinement_20261103}
run=$run_root/$family
if [ -f "$run/fno_h1/no.pt" ]; then
  echo "Existing trained run: choose a new run_root as the second argument." >&2
  exit 2
fi
mkdir -p "$run"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
if [ ! -f "$data/manifest.json" ]; then
  "$py" -u baselines/data/multiscale_elliptic_v1/generate_suite.py --family "$family" --out "$data" > "$run/generate.log" 2>&1
fi
"$py" -u "$exp/run.py" train-no --data "$data" --out "$run/fno_h1" --no-loss h1 --epochs 150 > "$run/fno.log" 2>&1
"$py" -u "$exp/refinement/study.py" cache-no --data "$data" --no-dir "$run/fno_h1" --out "$run/cache" > "$run/cache.log" 2>&1
"$py" -u "$exp/refinement/study.py" train --data "$data" --no-dir "$run/fno_h1" --out "$run/dictionary64" --epochs 30 > "$run/train.log" 2>&1
"$py" -u "$exp/refinement/study.py" evaluate --data "$data" --no-dir "$run/fno_h1" --checkpoint "$run/dictionary64/model.pt" --out "$run/test32" --count 32 > "$run/evaluate.log" 2>&1
"$py" -u "$exp/run.py" numerics --data "$data" --no-dir "$run/fno_h1" --out "$run/numerics" --amg-method rs --count 16 --cycles 1 2 4 > "$run/numerics_rs.log" 2>&1
"$py" -u "$exp/run.py" numerics --data "$data" --no-dir "$run/fno_h1" --out "$run/numerics" --amg-method sa --count 16 --cycles 1 2 4 > "$run/numerics_sa.log" 2>&1
