#!/bin/bash
# Example 3, all paper runs: 5 seeds, then Table 3 and the gallery figure.
#
#   python train_operator.py encode      # once, before this script
#   bash run_ex3.sh
#   GPUS="0 1" ITERS=30000 bash run_ex3.sh
set -eu
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
ITERS="${ITERS:-30000}"
read -r -a GPUS <<< "${GPUS:-0}"

for seed in 0 1 2 3 4; do
  echo "$seed ${GPUS[$(( seed % ${#GPUS[@]} ))]}"
done | xargs -P "${#GPUS[@]}" -L1 bash -c '
  echo "[gpu$1] train_operator '"$ITERS"' seed $0"
  CUDA_VISIBLE_DEVICES=$1 '"$PY"' train_operator.py '"$ITERS"' $0 > /dev/null'

echo "--- Table 3 ---"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" eval_operator.py
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" plot_gallery.py
