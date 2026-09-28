#!/bin/bash
# Example 2, all paper runs: 4 configurations x 5 seeds = 20 trainings, then the
# stratified table and the rollout figure.
#
#   python train_evolution.py encode     # once, before this script
#   bash run_ex2.sh
#   GPUS="0 1" ITERS=40000 bash run_ex2.sh
set -eu
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
ITERS="${ITERS:-40000}"
read -r -a GPUS <<< "${GPUS:-0}"

i=0
for cfg in "code 24" "code 16" "code 12" "linear 24"; do
  set -- $cfg
  for seed in 0 1 2 3 4; do
    gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}; i=$((i + 1))
    echo "$1 $2 $seed $gpu"
  done
done | xargs -P "${#GPUS[@]}" -L1 bash -c '
  echo "[gpu$3] train_evolution '"$ITERS"' $0 $1 seed $2"
  CUDA_VISIBLE_DEVICES=$3 '"$PY"' train_evolution.py '"$ITERS"' $0 $1 $2 > /dev/null'

echo "--- Table 2 ---"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" eval_evolution.py table
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" eval_evolution.py figure
