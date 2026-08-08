#!/usr/bin/env bash
# Run once on each node of one four-node, eight-GPU-per-node allocation.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
RUN_ID="${SSR_SEG_RUN_ID:-}"
RESULT_ROOT="${SSR_SEG_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
DRY_RUN="${SSR_SEG_DRY_RUN:-0}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-900}"
LOCK_WAIT_SEC="${LOCK_WAIT_SEC:-14400}"
FINISH_WAIT_SEC="${FINISH_WAIT_SEC:-28800}"
# The cached 192px dense decoder historically left substantial GPU headroom.
# Two worker processes per GPU with a conservative 24-sample batch is the
# default; set JOBS_PER_GPU=1 or lower SEG_BATCH_SIZE if a host has less VRAM.
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
SEG_BATCH_SIZE="${SEG_BATCH_SIZE:-24}"

CONFIRMATION_SEEDS=(
  9201 9203 9205 9207 9209 9211 9213 9215 9217 9219
  9221 9223 9225 9227 9229 9231 9233 9235 9237 9239
)
FACTORIAL_METHODS=(baseline kd biocs biocs_kd)

[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_SEG_RUN_ID to a fresh identifier.\n' >&2
  exit 2
}
[[ "$EXPECTED_NNODES" == "4" && "$GPU_COUNT" == "8" ]] || {
  printf 'This protocol requires four nodes with eight GPUs each.\n' >&2
  exit 2
}
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || {
  printf 'SSR_SEG_DRY_RUN must be 0 or 1.\n' >&2
  exit 2
}
for seconds in "$INIT_WAIT_SEC" "$LOCK_WAIT_SEC" "$FINISH_WAIT_SEC" \
  "$JOBS_PER_GPU" "$SEG_BATCH_SIZE"; do
  [[ "$seconds" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Wait durations, JOBS_PER_GPU, and SEG_BATCH_SIZE must be positive integers: %s\n' \
      "$seconds" >&2
    exit 2
  }
done

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
((GLOBAL_NODE_RANK >= 0 && GLOBAL_NODE_RANK < EXPECTED_NNODES)) || exit 2

git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
if [[ "$DRY_RUN" == "0" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/check_experiment_worktree.py" \
    --project-root "$PROJECT_ROOT"
  physical_gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count+0}')"
  [[ "$physical_gpu_count" == "$GPU_COUNT" ]] || {
    printf 'Expected eight GPUs, found %s on %s.\n' "$physical_gpu_count" "$(hostname)" >&2
    exit 2
  }
fi

IDENTITY_FILE="$RESULT_ROOT/.segmentation_factorial_4node_identity"
STATUS_DIR="$RESULT_ROOT/status"
LOG_DIR="$RESULT_ROOT/logs"
CUB_ROOT="$RESULT_ROOT/cub200_masks"
LOCK_FILE="$CUB_ROOT/SELECTION_LOCK.json"

if ((GLOBAL_NODE_RANK == 0)); then
  [[ ! -e "$RESULT_ROOT" ]] || {
    printf 'Fresh result root required: %s already exists.\n' "$RESULT_ROOT" >&2
    exit 2
  }
  mkdir -p "$STATUS_DIR" "$LOG_DIR"
  {
    printf 'protocol=segmentation_factorial_2x2_4node_v1\n'
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'git_commit=%s\n' "$git_commit"
    printf 'confirmation_seed_count=%s\n' "${#CONFIRMATION_SEEDS[@]}"
    printf 'jobs_per_gpu=%s\n' "$JOBS_PER_GPU"
    printf 'seg_batch_size=%s\n' "$SEG_BATCH_SIZE"
  } > "$IDENTITY_FILE.tmp.$$"
  mv "$IDENTITY_FILE.tmp.$$" "$IDENTITY_FILE"
else
  deadline=$((SECONDS + INIT_WAIT_SEC))
  while [[ ! -s "$IDENTITY_FILE" ]]; do
    ((SECONDS < deadline)) || {
      printf 'Timed out waiting for run identity.\n' >&2
      exit 2
    }
    sleep 1
  done
fi
mkdir -p "$STATUS_DIR" "$LOG_DIR"
for expected_line in \
  'protocol=segmentation_factorial_2x2_4node_v1' \
  "run_id=$RUN_ID" \
  "git_commit=$git_commit" \
  'confirmation_seed_count=20' \
  "jobs_per_gpu=$JOBS_PER_GPU" \
  "seg_batch_size=$SEG_BATCH_SIZE"; do
  grep -Fqx "$expected_line" "$IDENTITY_FILE" || {
    printf 'Run identity differs across nodes: %s\n' "$expected_line" >&2
    exit 2
  }
done

atomic_status() {
  local path="$1"
  shift
  local temporary="$path.tmp.$(hostname).$$"
  {
    printf 'git_commit=%s\n' "$git_commit"
    printf 'host=%s\n' "$(hostname)"
    printf '%s\n' "$@"
  } > "$temporary"
  mv "$temporary" "$path"
}

wait_for_file() {
  local path="$1" timeout="$2" label="$3"
  local deadline=$((SECONDS + timeout))
  while [[ ! -s "$path" ]]; do
    ((SECONDS < deadline)) || {
      printf 'Timed out waiting for %s: %s\n' "$label" "$path" >&2
      exit 2
    }
    sleep 2
  done
}

LOCAL_STAGE=""
cleanup() {
  if [[ -n "$LOCAL_STAGE" && -d "$LOCAL_STAGE" ]]; then
    rm -rf -- "$LOCAL_STAGE"
  fi
}

role=""
terminal_status=0
on_exit() {
  local code=$?
  if ((code != 0 && terminal_status == 0)); then
    atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" \
      "role=$role" "exit_code=$code" "finished_at=$(date -Iseconds)"
  fi
  cleanup
  trap - EXIT
  exit "$code"
}
trap on_exit EXIT

if [[ "$DRY_RUN" == "1" ]]; then
  role=matrix_plan
  if ((GLOBAL_NODE_RANK == 0)); then
    "$PYTHON" "$PROJECT_ROOT/scripts/run_meeting_segmentation_ablation.py" \
      --result-root "$CUB_ROOT" \
      --phase screen \
      --protocol factorial_2x2 \
      --confirmation-seeds "${CONFIRMATION_SEEDS[@]}" \
      --confirmation-methods "${FACTORIAL_METHODS[@]}" \
      --manifest-only \
      > "$LOG_DIR/node_0_plan.log" 2>&1
  else
    wait_for_file "$CUB_ROOT/MANIFEST.json" "$INIT_WAIT_SEC" "factorial manifest"
  fi
  atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
    "role=$role" "exit_code=0" "finished_at=$(date -Iseconds)"
  if ((GLOBAL_NODE_RANK == 0)); then
    for rank in 1 2 3; do
      wait_for_file "$STATUS_DIR/node_${rank}.done" "$FINISH_WAIT_SEC" "node $rank plan"
    done
    "$PYTHON" - "$CUB_ROOT" "$RESULT_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

cub_root = Path(sys.argv[1])
result_root = Path(sys.argv[2])
manifest = json.loads((cub_root / "MANIFEST.json").read_text(encoding="utf-8"))
expected = {
    "protocol": "factorial_2x2",
    "development": 42,
    "confirmation": 80,
    "total": 122,
}
observed = {
    "protocol": manifest.get("protocol"),
    "development": manifest.get("jobs", {}).get("development"),
    "confirmation": manifest.get("jobs", {}).get("confirmation"),
    "total": manifest.get("jobs", {}).get("total"),
}
if observed != expected:
    raise SystemExit(f"Factorial plan mismatch: expected={expected}, observed={observed}")
payload = {"status": "PASS", **observed, "nodes": 4}
path = result_root / "MATRIX_PLAN_VALIDATION.json"
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
    terminal_status=1
  else
    wait_for_file "$RESULT_ROOT/MATRIX_PLAN_VALIDATION.json" "$FINISH_WAIT_SEC" "plan validation"
    terminal_status=1
  fi
  printf 'Validated factorial segmentation plan on node %s.\n' "$GLOBAL_NODE_RANK"
  exit 0
fi

LOCAL_STAGE="$(mktemp -d "/dev/shm/ssr_factorial_${RUN_ID}_${GLOBAL_NODE_RANK}.XXXXXX")"
stage_cache() {
  local source="$1"
  local destination="$LOCAL_STAGE/$(basename "$source")"
  cp "$source" "$destination.tmp"
  mv "$destination.tmp" "$destination"
  [[ "$(sha256sum "$source" | awk '{print $1}')" == "$(sha256sum "$destination" | awk '{print $1}')" ]] || {
    printf 'Local cache checksum mismatch: %s\n' "$source" >&2
    exit 2
  }
  printf '%s\n' "$destination"
}

GPU_LIST=(0 1 2 3 4 5 6 7)
cub_cache="$(stage_cache "$PROJECT_ROOT/data/cub200/cub200_seg_resnet18_dense_192.pt")"
if ((GLOBAL_NODE_RANK == 0)); then
  role=cub_factorial_screen_and_shard_0
  "$PYTHON" "$PROJECT_ROOT/scripts/run_meeting_segmentation_ablation.py" \
    --result-root "$CUB_ROOT" \
    --phase screen \
    --protocol factorial_2x2 \
    --confirmation-seeds "${CONFIRMATION_SEEDS[@]}" \
    --confirmation-methods "${FACTORIAL_METHODS[@]}" \
    --gpus "${GPU_LIST[@]}" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --workers 0 \
    --seg-batch-size "$SEG_BATCH_SIZE" \
    --data-root "$PROJECT_ROOT/data/cub200" \
    --segmentation-cache "$cub_cache" \
    --python "$PYTHON" \
    > "$LOG_DIR/node_0_cub_screen.log" 2>&1
else
  role="cub_factorial_confirmation_shard_${GLOBAL_NODE_RANK}"
  wait_for_file "$LOCK_FILE" "$LOCK_WAIT_SEC" "CUB factorial selection lock"
fi

"$PYTHON" "$PROJECT_ROOT/scripts/run_cub_segmentation_factorial_shard.py" \
  --result-root "$CUB_ROOT" \
  --selection-lock "$LOCK_FILE" \
  --data-root "$PROJECT_ROOT/data/cub200" \
  --segmentation-cache "$cub_cache" \
  --shard-index "$GLOBAL_NODE_RANK" --shard-count 4 \
  --gpus "${GPU_LIST[@]}" --jobs-per-gpu "$JOBS_PER_GPU" --workers 0 \
  --seg-batch-size "$SEG_BATCH_SIZE" --python "$PYTHON" \
  > "$LOG_DIR/node_${GLOBAL_NODE_RANK}_cub_confirm.log" 2>&1
atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
  "role=$role" "exit_code=0" "finished_at=$(date -Iseconds)"

if ((GLOBAL_NODE_RANK == 0)); then
  for rank in 1 2 3; do
    wait_for_file "$STATUS_DIR/node_${rank}.done" "$FINISH_WAIT_SEC" "CUB shard $rank"
  done
  # This reads the completed 80-record cohort; --summary-only never calls run_jobs.
  "$PYTHON" "$PROJECT_ROOT/scripts/run_cub_segmentation_factorial_shard.py" \
    --result-root "$CUB_ROOT" \
    --selection-lock "$LOCK_FILE" \
    --data-root "$PROJECT_ROOT/data/cub200" \
    --segmentation-cache "$cub_cache" \
    --shard-index 0 --shard-count 4 \
    --summary-only --python "$PYTHON" \
    > "$LOG_DIR/node_0_cub_summary.log" 2>&1
  "$PYTHON" - "$CUB_ROOT" "$RESULT_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

cub_root = Path(sys.argv[1])
result_root = Path(sys.argv[2])
manifest = json.loads((cub_root / "MANIFEST.json").read_text(encoding="utf-8"))
lock = json.loads((cub_root / "SELECTION_LOCK.json").read_text(encoding="utf-8"))
summary = json.loads((cub_root / "CONFIRMATION_SUMMARY.json").read_text(encoding="utf-8"))
development = list((cub_root / "development").glob("**/result_record.json"))
confirmation = list((cub_root / "confirmation").glob("**/result_record.json"))
required_contrasts = {
    "task_to_task_ssr",
    "kd_to_kd_ssr",
    "task_to_kd",
    "task_ssr_to_kd_ssr",
    "task_to_full_recipe",
}
if manifest.get("jobs") != {"development": 42, "confirmation": 80, "total": 122}:
    raise SystemExit(f"Unexpected factorial manifest: {manifest.get('jobs')}")
if lock.get("protocol") != "factorial_2x2" or len(lock.get("screen_result_records", [])) != 42:
    raise SystemExit("Selection lock does not bind the complete factorial screen")
if len(development) != 42 or len(confirmation) != 80:
    raise SystemExit(
        f"Incomplete factorial records: development={len(development)}, confirmation={len(confirmation)}"
    )
if set(summary.get("contrasts", {})) != required_contrasts:
    raise SystemExit("Confirmation summary is missing a prespecified factorial contrast")
if "factorial_interaction" not in summary:
    raise SystemExit("Confirmation summary is missing the factorial interaction")
payload = {
    "status": "PASS",
    "development_records": len(development),
    "confirmation_records": len(confirmation),
    "selection_lock_sha256": lock["lock_sha256"],
    "contrasts": sorted(required_contrasts),
}
path = result_root / "MATRIX_RESULTS_VALIDATION.json"
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
  atomic_status "$RESULT_ROOT/COMPLETED" \
    "protocol=segmentation_factorial_2x2_4node_v1" \
    "nodes=4" "finished_at=$(date -Iseconds)"
  terminal_status=1
else
  wait_for_file "$RESULT_ROOT/MATRIX_RESULTS_VALIDATION.json" "$FINISH_WAIT_SEC" "matrix result validation"
  terminal_status=1
fi

printf 'Completed factorial segmentation node %s role=%s result_root=%s\n' \
  "$GLOBAL_NODE_RANK" "$role" "$RESULT_ROOT"
