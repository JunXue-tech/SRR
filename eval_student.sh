#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
GPU_IDS=${GPU_IDS:-0}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -ra DEVICES <<< "$GPU_IDS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#DEVICES[@]}" eval_student.py "$@"
