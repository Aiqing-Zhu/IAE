#!/bin/bash
# Example 1, all paper runs: 4 encoders x 3 normalizations x 5 seeds = 60 trainings.
# Results are appended to results/ex1_runs.csv; aggregate them with aggregate_ex1.py.
#
#   bash run_ex1.sh                 # 60 runs on the GPUs listed in $GPUS
#   GPUS="0 1" PER_GPU=1 bash run_ex1.sh
#   ITERS=2000 bash run_ex1.sh      # short smoke version
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
ITERS="${ITERS:-120000}"
read -r -a GPUS <<< "${GPUS:-0}"
PER_GPU="${PER_GPU:-1}"
NW=$(( ${#GPUS[@]} * PER_GPU ))

mkdir -p results
OUT=results/ex1_runs.csv
LOG=results/ex1_runs.log
: > "$OUT"; : > "$LOG"

configs=()
for enc in mfe2d mfe1d iae_cosine iae_legendre; do
  for norm in none perdim lowfreq; do
    for seed in 0 1 2 3 4; do
      configs+=("$enc $norm $seed")
    done
  done
done

worker() {                                  # worker w takes configs w, w+NW, w+2NW, ...
  local w=$1 gpu=${GPUS[$(( $1 / PER_GPU ))]} i=$1
  while [ $i -lt ${#configs[@]} ]; do
    set -- ${configs[$i]}
    CUDA_VISIBLE_DEVICES=$gpu "$PY" train_ex1.py "$1" "$2" "$3" "$ITERS" 2>>"$LOG" \
      | grep '^RESULT,' >> "$OUT"
    i=$(( i + NW ))
  done
}

echo "running ${#configs[@]} configs on GPUs ${GPUS[*]} ($NW workers, $ITERS iters each)"
for w in $(seq 0 $((NW - 1))); do worker "$w" & done
wait
echo "done: $(grep -c '^RESULT,' "$OUT")/${#configs[@]} runs -> $OUT"
"$PY" aggregate_ex1.py "$OUT"
