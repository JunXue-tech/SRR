#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
GPU_IDS=${GPU_IDS:-0}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -ra DEVICES <<< "$GPU_IDS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#DEVICES[@]}" train.py \
  --mode distill --output-dir "${RUN_DIR:-runs/distill}" \
  --epochs 50 --patience 10 --lr 1e-6 --weight-decay 0.01 \
  --batch-size "${BATCH_SIZE:-4}" --global-batch "${GLOBAL_BATCH:-60}" \
  --rawboost-algo 4 --augmentation-probability 0.9 --kd-weight 0.3 --temperature 2.0 "$@"
