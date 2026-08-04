#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/meeting_20260804/direct_editing_wikibio_gpt2xl.yaml}"
RUN_ID="${RUN_ID:-wikibio_direct_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"

dry_run=0
for argument in "$@"; do
  [[ "$argument" == "--dry-run" ]] && dry_run=1
done
if ((dry_run == 0)); then
  PROJECT_ROOT="$PROJECT_ROOT" bash "$SCRIPT_DIR/fetch_knowedit_wikibio.sh"
fi

PROJECT_ROOT="$PROJECT_ROOT" \
PYTHON="$PYTHON" \
CONFIG_FILE="$CONFIG_FILE" \
RUN_ID="$RUN_ID" \
RESULT_ROOT="$RESULT_ROOT" \
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}" \
NNODES="${NNODES:-1}" \
NODE_RANK="${NODE_RANK:-${PADDLE_TRAINER_ID:-${SLURM_NODEID:-0}}}" \
LAUNCH_TOKEN="${LAUNCH_TOKEN:-}" \
  bash "$SCRIPT_DIR/run_meeting_editing_8gpu.sh" \
    --phase confirm \
    --dataset wikibio \
    --recipe plain,ssr_only \
    --mapping projective \
    "$@"
