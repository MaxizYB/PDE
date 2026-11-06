#!/usr/bin/env bash
# Execute from the Graph workspace root. No system Python and no environment edits.
set -euo pipefail
PY=/home/caohuilin/.conda/envs/gtmsmw/bin/python
RUN=CDMC/experiments/multiscale_elliptic_v1
DATA=baselines/data/multiscale_elliptic_v1/generated/n257_s20261023
OUT=$RUN/results/n257_s20261023
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT"
if [[ ! -f "$DATA/manifest.json" ]]; then
    echo 'Data generation is not complete; launch only after its manifest exists.'
    exit 1
fi
echo 'Stage 1: standard and derivative-trained FNO, same data and seed'
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/run.py" train-no --data "$DATA" --out "$OUT/fno_l2" --no-loss l2 --epochs 150 > "$OUT/fno_l2.log" 2>&1 &
PID_L2=$!
CUDA_VISIBLE_DEVICES=2 "$PY" -u "$RUN/run.py" train-no --data "$DATA" --out "$OUT/fno_h1" --no-loss h1 --epochs 150 > "$OUT/fno_h1.log" 2>&1 &
PID_H1=$!
wait "$PID_L2"
wait "$PID_H1"
echo 'Stage 2: matched and generic CDMC, same graph, budget and seed'
# Serialize cache creation before launching two readers.
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/run.py" numerics --data "$DATA" --no-dir "$OUT/fno_l2" --out "$OUT/evaluation_l2" --count 16 --cycles 1 2 4 --repeats 3 > "$OUT/numerics_l2.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/run.py" train-cdmc --data "$DATA" --no-dir "$OUT/fno_l2" --out "$OUT/actual" --task actual > "$OUT/actual.log" 2>&1 &
PID_ACTUAL=$!
CUDA_VISIBLE_DEVICES=2 "$PY" -u "$RUN/run.py" train-cdmc --data "$DATA" --no-dir "$OUT/fno_l2" --out "$OUT/generic" --task generic > "$OUT/generic.log" 2>&1 &
PID_GENERIC=$!
wait "$PID_ACTUAL"
wait "$PID_GENERIC"
echo 'Stage 3: held-out paired evaluation; total timing reruns the full predictor'
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/run.py" evaluate --data "$DATA" --no-dir "$OUT/fno_l2" --out "$OUT/evaluation_l2" --corrector-dirs "$OUT/actual" "$OUT/generic" --count 32 --cycles 1 2 4 --diagnose-probes > "$OUT/evaluation_l2.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/run.py" report --data "$DATA" --no-dir "$OUT/fno_l2" --out "$OUT/evaluation_l2" > "$OUT/report_l2.log" 2>&1
echo 'Stage 4: stronger NO; frozen-corrector transfer is explicitly separate'
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/run.py" evaluate --data "$DATA" --no-dir "$OUT/fno_h1" --out "$OUT/evaluation_h1_transfer" --corrector-dirs "$OUT/actual" "$OUT/generic" --count 16 --cycles 1 2 > "$OUT/evaluation_h1.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/run.py" report --data "$DATA" --no-dir "$OUT/fno_h1" --out "$OUT/evaluation_h1_transfer" --cycles 1 2 > "$OUT/report_h1.log" 2>&1
echo 'Stage 5: matched corrector for derivative-supervised NO, not just transfer'
CUDA_VISIBLE_DEVICES=2 "$PY" -u "$RUN/run.py" train-cdmc --data "$DATA" --no-dir "$OUT/fno_h1" --out "$OUT/h1_actual" --task actual > "$OUT/h1_actual.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/run.py" evaluate --data "$DATA" --no-dir "$OUT/fno_h1" --out "$OUT/evaluation_h1_matched" --corrector-dirs "$OUT/h1_actual" --count 32 --cycles 1 2 4 > "$OUT/evaluation_h1_matched.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/run.py" report --data "$DATA" --no-dir "$OUT/fno_h1" --out "$OUT/evaluation_h1_matched" > "$OUT/report_h1_matched.log" 2>&1
echo 'Stage 6: stronger numerical controls and train-calibrated scalar audit'
for method in rs asa; do
    "$PY" -u "$RUN/run.py" numerics --data "$DATA" --no-dir "$OUT/fno_l2" --out "$OUT/evaluation_l2" --count 16 --cycles 1 2 4 --repeats 3 --amg-method "$method" --device cpu > "$OUT/numerics_${method}.log" 2>&1
done
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN/audit.py" --data "$DATA" --no-dir "$OUT/fno_l2" --corrector-dir "$OUT/actual" --evaluation "$OUT/evaluation_l2/evaluation.json" --out "$OUT/stability_audit.json" > "$OUT/stability_audit.log" 2>&1
"$PY" -u "$RUN/run.py" report --data "$DATA" --no-dir "$OUT/fno_l2" --out "$OUT/evaluation_l2" --device cpu > "$OUT/report_l2.log" 2>&1
echo 'Finished: single-seed exploratory study, not a repeated-seed final benchmark.'
