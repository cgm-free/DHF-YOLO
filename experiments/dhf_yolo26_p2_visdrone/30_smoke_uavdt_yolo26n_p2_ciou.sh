#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/yolo26/bin/python}"
MODEL="${MODEL:-ultralytics/cfg/models/26/yolo26-p2.yaml}"
DATA="${DATA:-ultralytics/cfg/datasets/UAVDT-repo.yaml}"
PROJECT="${PROJECT:-runs/uavdt_ablation}"
NAME="${NAME:-30_uavdt_smoke_yolo26n_p2_ciou}"
LOG_DIR="${LOG_DIR:-runs/uavdt_ablation_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="$LOG_DIR/${NAME}_${STAMP}.out.log"
STDERR_LOG="$LOG_DIR/${NAME}_${STAMP}.err.log"
exec > >(tee -a "$STDOUT_LOG") 2> >(tee -a "$STDERR_LOG" >&2)
"$PYTHON_BIN" -c "from ultralytics.cfg import entrypoint; entrypoint()" detect train \
  model="$MODEL" \
  data="$DATA" \
  epochs=1 \
  imgsz=640 \
  batch=8 \
  device=0 \
  workers=8 \
  project="$PROJECT" \
  name="$NAME" \
  fraction=0.05 \
  box_loss=ciou
