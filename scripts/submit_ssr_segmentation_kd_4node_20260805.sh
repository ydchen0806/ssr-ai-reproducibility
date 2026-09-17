#!/usr/bin/env bash
# Four-node screen-lock-confirm study of KD versus KD+SSR segmentation.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-/usr/local/bin/python}"
RUN_ID="${SSR_SEG_RUN_ID:-}"
RESULT_ROOT="${SSR_SEG_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-900}"
LOCK_WAIT_SEC="${LOCK_WAIT_SEC:-14400}"
FINISH_WAIT_SEC="${FINISH_WAIT_SEC:-28800}"

[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_SEG_RUN_ID to a fresh identifier.\n' >&2
  exit 2
}
[[ "$EXPECTED_NNODES" == "4" && "$GPU_COUNT" == "8" ]] || {
  printf 'This protocol requires four nodes with eight GPUs each.\n' >&2
  exit 2
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

physical_gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count+0}')"
[[ "$physical_gpu_count" == "$GPU_COUNT" ]] || {
  printf 'Expected eight GPUs, found %s on %s.\n' "$physical_gpu_count" "$(hostname)" >&2
  exit 2
}

git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
IDENTITY_FILE="$RESULT_ROOT/.segmentation_kd_4node_identity"
STATUS_DIR="$RESULT_ROOT/status"
LOG_DIR="$RESULT_ROOT/logs"
if ((GLOBAL_NODE_RANK == 0)); then
  [[ ! -e "$RESULT_ROOT" ]] || {
    printf 'Fresh result root required: %s already exists.\n' "$RESULT_ROOT" >&2
    exit 2
  }
  mkdir -p "$STATUS_DIR" "$LOG_DIR"
  {
    printf 'protocol=segmentation_kd_screen_lock_confirm_v2\n'
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'git_commit=%s\n' "$git_commit"
  } > "$IDENTITY_FILE.tmp.$$"
  mv "$IDENTITY_FILE.tmp.$$" "$IDENTITY_FILE"
else
  deadline=$((SECONDS + INIT_WAIT_SEC))
  while [[ ! -s "$IDENTITY_FILE" ]]; do
    ((SECONDS < deadline)) || {
      printf 'Timed out waiting for run identity.\n' >&2
      exit 2
    }
    sleep 2
  done
fi
mkdir -p "$STATUS_DIR" "$LOG_DIR"
grep -Fqx "git_commit=$git_commit" "$IDENTITY_FILE" || {
  printf 'Git identity differs across nodes.\n' >&2
  exit 2
}

LOCAL_STAGE="$(mktemp -d "/dev/shm/ssr_seg_${RUN_ID}_${GLOBAL_NODE_RANK}.XXXXXX")"
cleanup_stage() {
  rm -rf -- "$LOCAL_STAGE"
}
trap cleanup_stage EXIT

stage_cache() {
  local source="$1"
  local destination="$LOCAL_STAGE/$(basename "$source")"
  cp "$source" "$destination.tmp"
  mv "$destination.tmp" "$destination"
  [[ "$(sha256sum "$source" | awk '{print $1}')" == "$(sha256sum "$destination" | awk '{print $1}')" ]] || {
    printf 'Local cache checksum mismatch: %s\n' "$source" >&2
    exit 2
  }
  if [[ -f "$source.manifest.json" ]]; then
    cp "$source.manifest.json" "$destination.manifest.json"
  fi
  printf '%s\n' "$destination"
}

wait_for_file() {
  local path="$1" timeout="$2" label="$3"
  local deadline=$((SECONDS + timeout))
  while [[ ! -s "$path" ]]; do
    ((SECONDS < deadline)) || {
      printf 'Timed out waiting for %s: %s\n' "$label" "$path" >&2
      exit 2
    }
    sleep 10
  done
}

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
trap 'record_failure; cleanup_stage' INT TERM ERR

GPU_LIST=(0 1 2 3 4 5 6 7)
CUB_ROOT="$RESULT_ROOT/cub200_masks"
LOCK_FILE="$CUB_ROOT/SELECTION_LOCK.json"
case "$GLOBAL_NODE_RANK" in
  0)
    role=cub_screen_and_confirmation_shard_0
    cub_cache="$(stage_cache "$PROJECT_ROOT/data/cub200/cub200_seg_resnet18_dense_192.pt")"
    "$PYTHON" "$PROJECT_ROOT/scripts/run_meeting_segmentation_ablation.py" \
      --result-root "$CUB_ROOT" \
      --phase screen \
      --confirmation-seeds \
        9201 9203 9205 9207 9209 9211 9213 9215 9217 9219 \
        9221 9223 9225 9227 9229 9231 9233 9235 9237 9239 \
      --confirmation-methods kd biocs_kd \
      --gpus "${GPU_LIST[@]}" \
      --jobs-per-gpu 1 \
      --workers 0 \
      --seg-batch-size 24 \
      --data-root "$PROJECT_ROOT/data/cub200" \
      --segmentation-cache "$cub_cache" \
      --python "$PYTHON" \
      > "$LOG_DIR/node_0_cub_screen.log" 2>&1
    "$PYTHON" "$PROJECT_ROOT/scripts/run_cub_segmentation_kd_shard.py" \
      --result-root "$CUB_ROOT" \
      --selection-lock "$LOCK_FILE" \
      --data-root "$PROJECT_ROOT/data/cub200" \
      --segmentation-cache "$cub_cache" \
      --shard-index 0 --shard-count 2 \
      --gpus "${GPU_LIST[@]}" --jobs-per-gpu 1 --workers 0 \
      --python "$PYTHON" \
      > "$LOG_DIR/node_0_cub_confirm.log" 2>&1
    atomic_status "$STATUS_DIR/node_0.shard_done" \
      "role=$role" "finished_at=$(date -Iseconds)"
    wait_for_file "$STATUS_DIR/node_1.done" "$FINISH_WAIT_SEC" "CUB shard 1"
    "$PYTHON" "$PROJECT_ROOT/scripts/run_cub_segmentation_kd_shard.py" \
      --result-root "$CUB_ROOT" \
      --selection-lock "$LOCK_FILE" \
      --data-root "$PROJECT_ROOT/data/cub200" \
      --segmentation-cache "$cub_cache" \
      --summary-only \
      > "$LOG_DIR/node_0_cub_summary.log" 2>&1
    ;;
  1)
    role=cub_confirmation_shard_1
    wait_for_file "$LOCK_FILE" "$LOCK_WAIT_SEC" "CUB selection lock"
    cub_cache="$(stage_cache "$PROJECT_ROOT/data/cub200/cub200_seg_resnet18_dense_192.pt")"
    "$PYTHON" "$PROJECT_ROOT/scripts/run_cub_segmentation_kd_shard.py" \
      --result-root "$CUB_ROOT" \
      --selection-lock "$LOCK_FILE" \
      --data-root "$PROJECT_ROOT/data/cub200" \
      --segmentation-cache "$cub_cache" \
      --shard-index 1 --shard-count 2 \
      --gpus "${GPU_LIST[@]}" --jobs-per-gpu 1 --workers 0 \
      --python "$PYTHON" \
      > "$LOG_DIR/node_1_cub_confirm.log" 2>&1
    ;;
  2)
    role=pet_locked_transfer
    wait_for_file "$LOCK_FILE" "$LOCK_WAIT_SEC" "CUB selection lock"
    pet_cache="$(stage_cache "$PROJECT_ROOT/data/oxford_iiit_pet/resnet18_layer3_masks_160_provenance_v2.pt")"
    "$PYTHON" "$PROJECT_ROOT/scripts/run_locked_multidataset_segmentation_kd.py" \
      --dataset oxford_iiit_pet \
      --dataset-root "$PROJECT_ROOT/data/oxford_iiit_pet" \
      --feature-cache "$pet_cache" \
      --selection-lock "$LOCK_FILE" \
      --result-root "$RESULT_ROOT/oxford_iiit_pet_masks" \
      --seeds 9401 9403 9405 9407 9409 9411 9413 9415 9417 9419 9421 9423 9425 9427 9429 9431 9433 9435 9437 9439 \
      --gpus "${GPU_LIST[@]}" --jobs-per-gpu 1 --workers 0 \
      --lambda-kd 2.0 --python "$PYTHON" \
      > "$LOG_DIR/node_2_pet_transfer.log" 2>&1
    ;;
  3)
    role=flowers_locked_transfer
    wait_for_file "$LOCK_FILE" "$LOCK_WAIT_SEC" "CUB selection lock"
    flowers_cache="$(stage_cache "$PROJECT_ROOT/data/flowers-102/resnet18_layer3_masks_160_provenance_v2.pt")"
    "$PYTHON" "$PROJECT_ROOT/scripts/run_locked_multidataset_segmentation_kd.py" \
      --dataset oxford_flowers102 \
      --dataset-root "$PROJECT_ROOT/data/flowers-102" \
      --feature-cache "$flowers_cache" \
      --selection-lock "$LOCK_FILE" \
      --result-root "$RESULT_ROOT/oxford_flowers102_masks" \
      --seeds 9601 9603 9605 9607 9609 9611 9613 9615 9617 9619 9621 9623 9625 9627 9629 9631 9633 9635 9637 9639 \
      --gpus "${GPU_LIST[@]}" --jobs-per-gpu 1 --workers 0 \
      --lambda-kd 2.0 --python "$PYTHON" \
      > "$LOG_DIR/node_3_flowers_transfer.log" 2>&1
    ;;
esac

atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
  "role=$role" "exit_code=0" "finished_at=$(date -Iseconds)"

if ((GLOBAL_NODE_RANK == 0)); then
  for rank in 1 2 3; do
    wait_for_file "$STATUS_DIR/node_${rank}.done" "$FINISH_WAIT_SEC" "node $rank completion"
  done
  for summary in \
    "$CUB_ROOT/KD_SSR_CONFIRMATION_SUMMARY.json" \
    "$RESULT_ROOT/oxford_iiit_pet_masks/CONFIRMATION_SUMMARY.json" \
    "$RESULT_ROOT/oxford_flowers102_masks/CONFIRMATION_SUMMARY.json"; do
    [[ -s "$summary" ]] || {
      printf 'Missing required summary: %s\n' "$summary" >&2
      exit 2
    }
  done
  atomic_status "$RESULT_ROOT/COMPLETED" \
    "protocol=segmentation_kd_screen_lock_confirm_v2" \
    "nodes=4" "finished_at=$(date -Iseconds)"
fi

terminal_status=1
trap - INT TERM ERR
printf 'Completed node %s role=%s result_root=%s\n' \
  "$GLOBAL_NODE_RANK" "$role" "$RESULT_ROOT"
