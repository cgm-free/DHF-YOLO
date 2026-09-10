#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUTF8=1

if command -v python >/dev/null 2>&1; then
  DEFAULT_PYTHON="python"
elif [ -x "$HOME/anaconda3/envs/yolo26/bin/python" ]; then
  DEFAULT_PYTHON="$HOME/anaconda3/envs/yolo26/bin/python"
elif [ -x /root/miniconda3/bin/python ]; then
  DEFAULT_PYTHON="/root/miniconda3/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  DEFAULT_PYTHON="python3"
else
  echo "No Python executable found. Set PYTHON_BIN=/path/to/python and rerun." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
MODEL="${MODEL:-ultralytics/cfg/models/26/yolo26.yaml}"
DATA="${DATA:-ultralytics/cfg/datasets/VisDrone2019-DET-repo.yaml}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
PROJECT="${PROJECT:-runs/visdrone_ablation}"
NAME="${NAME:-14_yolo26n_no_p2_ciou_lab4090_e100}"
FRACTION="${FRACTION:-1.0}"
LOG_DIR="${LOG_DIR:-runs/visdrone_ablation_logs}"

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="$LOG_DIR/${NAME}_${STAMP}.out.log"
STDERR_LOG="$LOG_DIR/${NAME}_${STAMP}.err.log"

echo "Logging stdout to $STDOUT_LOG"
echo "Logging stderr to $STDERR_LOG"
exec > >(tee -a "$STDOUT_LOG") 2> >(tee -a "$STDERR_LOG" >&2)

"$PYTHON_BIN" -c "from ultralytics.cfg import entrypoint; entrypoint()" detect train \
  model="$MODEL" \
  data="$DATA" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device="$DEVICE" \
  workers="$WORKERS" \
  project="$PROJECT" \
  name="$NAME" \
  fraction="$FRACTION" \
  box_loss=ciou
