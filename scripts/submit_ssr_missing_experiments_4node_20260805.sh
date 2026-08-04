#!/usr/bin/env bash
# Complete only the missing/invalid meeting experiments on four 8-GPU nodes.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-/usr/local/bin/python}"
RUN_ID="${SSR_MISSING_RUN_ID:-}"
RESULT_ROOT="${SSR_MISSING_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
GPU_COUNT="${GPU_COUNT:-8}"
EXPECTED_NNODES="${EXPECTED_NNODES:-4}"
SOURCE_R2="${SSR_COMPLETED_SOURCE_R2:-$PROJECT_ROOT/results/ssr_meeting_extension_4node_20260804_r2}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-600}"
DRY_RUN="${SSR_MISSING_DRY_RUN:-0}"

[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_MISSING_RUN_ID to a fresh identifier.\n' >&2
  exit 2
}
[[ "$EXPECTED_NNODES" == "4" && "$GPU_COUNT" == "8" ]] || {
  printf 'This protocol requires four nodes with eight GPUs each.\n' >&2
  exit 2
}
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || exit 2

resolve_global_rank() {
  local key value resolved=""
  for key in GLOBAL_NODE_RANK NODE_RANK PADDLE_TRAINER_ID SLURM_NODEID GROUP_RANK; do
    value="${!key:-}"
    [[ -n "$value" ]] || continue
    [[ "$value" =~ ^[0-9]+$ ]] || return 2
    if [[ -n "$resolved" && "$resolved" != "$value" ]]; then
      printf 'Conflicting node ranks: %s and %s=%s.\n' "$resolved" "$key" "$value" >&2
      return 2
    fi
    resolved="$value"
  done
  if [[ -n "$resolved" ]]; then
    printf '%s\n' "$resolved"
    return
  fi
  local host="${HOSTNAME:-$(hostname)}"
  if [[ "$host" =~ master-([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  elif [[ "$host" =~ worker-([0-9]+) ]]; then
    printf '%s\n' "$((BASH_REMATCH[1] + 1))"
  else
    printf 'Set GLOBAL_NODE_RANK explicitly to 0, 1, 2, or 3.\n' >&2
    return 2
  fi
}

GLOBAL_NODE_RANK="$(resolve_global_rank)"
((GLOBAL_NODE_RANK >= 0 && GLOBAL_NODE_RANK < 4)) || exit 2
if [[ "$DRY_RUN" == "1" ]]; then
  case "$GLOBAL_NODE_RANK" in
    0)
      "$PYTHON" "$PROJECT_ROOT/scripts/run_meeting_segmentation_ablation.py" \
        --result-root "$RESULT_ROOT/cub_segmentation" --manifest-only
      ;;
    1)
      "$PYTHON" "$PROJECT_ROOT/scripts/run_multidataset_segmentation.py" \
        --result-root "$RESULT_ROOT/multidataset_segmentation" --manifest-only
      ;;
    2)
      env OUTPUT_ROOT="$RESULT_ROOT/four_dataset_adapter" \
        DATASETS="flowers102 food101 oxfordiiitpet dtd" PHASE=development \
        DRY_RUN=1 PYTHON="$PYTHON" \
        bash "$PROJECT_ROOT/scripts/run_multidataset_adapter_ssr_node.sh"
      ;;
    3)
      env OUTPUT_ROOT="$RESULT_ROOT/vit_lora" \
        DATASETS="flowers102 oxfordiiitpet" DRY_RUN=1 PYTHON="$PYTHON" \
        bash "$PROJECT_ROOT/scripts/run_vit_lora_confirmation_8gpu.sh"
      ;;
  esac
  exit 0
fi
physical_gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count+0}')"
[[ "$physical_gpu_count" == "$GPU_COUNT" ]] || {
  printf 'Expected eight GPUs, found %s on %s.\n' "$physical_gpu_count" "$(hostname)" >&2
  exit 2
}

git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
IDENTITY_FILE="$RESULT_ROOT/.missing_4node_identity"
STATUS_DIR="$RESULT_ROOT/status"
LOG_DIR="$RESULT_ROOT/logs"
if ((GLOBAL_NODE_RANK == 0)); then
  [[ ! -e "$RESULT_ROOT" ]] || {
    printf 'Fresh result root required: %s already exists.\n' "$RESULT_ROOT" >&2
    exit 2
  }
  mkdir -p "$STATUS_DIR" "$LOG_DIR"
  {
    printf 'protocol=missing_experiments_high_utilization_v1\n'
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'git_commit=%s\n' "$git_commit"
    printf 'completed_source_r2=%s\n' "$SOURCE_R2"
  } > "$IDENTITY_FILE.tmp.$$"
  mv "$IDENTITY_FILE.tmp.$$" "$IDENTITY_FILE"
else
  deadline=$((SECONDS + INIT_WAIT_SEC))
  while [[ ! -s "$IDENTITY_FILE" ]]; do
    ((SECONDS < deadline)) || exit 2
    sleep 2
  done
fi
mkdir -p "$STATUS_DIR" "$LOG_DIR"
grep -Fqx "git_commit=$git_commit" "$IDENTITY_FILE" || exit 2

atomic_status() {
  local path="$1"; shift
  local temporary="$path.tmp.$(hostname).$$"
  {
    printf 'git_commit=%s\n' "$git_commit"
    printf 'host=%s\n' "$(hostname)"
    printf '%s\n' "$@"
  } > "$temporary"
  mv "$temporary" "$path"
}

role=""
terminal_status=0
record_failure() {
  local code=$?
  if ((code != 0 && terminal_status == 0)); then
    atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" \
      "role=$role" "exit_code=$code" "finished_at=$(date -Iseconds)"
  fi
  exit "$code"
}
trap record_failure EXIT

GPU_LIST=(0 1 2 3 4 5 6 7)
case "$GLOBAL_NODE_RANK" in
  0)
    role=cub_segmentation_2x2
    "$PYTHON" "$PROJECT_ROOT/scripts/run_meeting_segmentation_ablation.py" \
      --result-root "$RESULT_ROOT/cub_segmentation" \
      --phase all \
      --gpus "${GPU_LIST[@]}" \
      --jobs-per-gpu "${CUB_JOBS_PER_GPU:-2}" \
      --workers "${CUB_DATA_WORKERS:-4}" \
      --seg-batch-size "${CUB_SEG_BATCH_SIZE:-48}" \
      --python "$PYTHON" \
      > "$LOG_DIR/node_0_cub_segmentation.log" 2>&1
    ;;
  1)
    role=multidataset_segmentation
    "$PYTHON" "$PROJECT_ROOT/scripts/run_multidataset_segmentation.py" \
      --result-root "$RESULT_ROOT/multidataset_segmentation" \
      --data-root "$PROJECT_ROOT/data" \
      --phase all \
      --gpus "${GPU_LIST[@]}" \
      --jobs-per-gpu "${SEG_JOBS_PER_GPU:-2}" \
      --workers "${SEG_DATA_WORKERS:-4}" \
      --batch-size "${SEG_BATCH_SIZE:-64}" \
      --feature-batch-size "${SEG_FEATURE_BATCH_SIZE:-192}" \
      --prepare-data \
      --python "$PYTHON" \
      > "$LOG_DIR/node_1_multidataset_segmentation.log" 2>&1
    ;;
  2)
    role=four_dataset_adapter
    adapter_root="$RESULT_ROOT/four_dataset_adapter"
    env \
      OUTPUT_ROOT="$adapter_root" \
      ADAPTER_DATA_ROOT="$PROJECT_ROOT/data/torchvision" \
      ADAPTER_CACHE_ROOT="$PROJECT_ROOT/data/torchvision/feature_cache" \
      DATASETS="flowers102 food101 oxfordiiitpet dtd" \
      GPU_LIST="0 1 2 3 4 5 6 7" \
      JOBS_PER_GPU="${ADAPTER_JOBS_PER_GPU:-2}" \
      PHASE=development \
      SELECT_LOCK=1 \
      PYTHON="$PYTHON" \
      bash "$PROJECT_ROOT/scripts/run_multidataset_adapter_ssr_node.sh" \
      > "$LOG_DIR/node_2_adapter_development.log" 2>&1
    env \
      OUTPUT_ROOT="$adapter_root" \
      ADAPTER_DATA_ROOT="$PROJECT_ROOT/data/torchvision" \
      ADAPTER_CACHE_ROOT="$PROJECT_ROOT/data/torchvision/feature_cache" \
      DATASETS="flowers102 food101 oxfordiiitpet dtd" \
      GPU_LIST="0 1 2 3 4 5 6 7" \
      JOBS_PER_GPU="${ADAPTER_JOBS_PER_GPU:-2}" \
      PHASE=confirmation \
      LOCK_FILE="$adapter_root/development_summary/selection_lock.json" \
      PYTHON="$PYTHON" \
      bash "$PROJECT_ROOT/scripts/run_multidataset_adapter_ssr_node.sh" \
      > "$LOG_DIR/node_2_adapter_confirmation.log" 2>&1
    ;;
  3)
    role=raw_image_vit_lora
    env \
      OUTPUT_ROOT="$RESULT_ROOT/vit_lora" \
      REUSE_RESULTS_ROOT= \
      VIT_LORA_DATA_ROOT="$PROJECT_ROOT/data" \
      VIT_LORA_PRETRAINED_CHECKPOINT="$PROJECT_ROOT/data/pretrained/vit_tiny_patch16_224_augreg_in21k_ft_in1k.safetensors" \
      DATASETS="flowers102 oxfordiiitpet" \
      SEEDS="5201 5203 5205 5207 5209" \
      GPU_LIST="0 1 2 3 4 5 6 7" \
      JOBS_PER_GPU="${VIT_JOBS_PER_GPU:-2}" \
      PYTHON="$PYTHON" \
      bash "$PROJECT_ROOT/scripts/run_vit_lora_confirmation_8gpu.sh" \
      > "$LOG_DIR/node_3_vit_lora.log" 2>&1
    ;;
esac

atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
  "role=$role" "exit_code=0" "finished_at=$(date -Iseconds)"
terminal_status=1
trap - EXIT
printf 'Completed node %s role=%s result_root=%s\n' \
  "$GLOBAL_NODE_RANK" "$role" "$RESULT_ROOT"
