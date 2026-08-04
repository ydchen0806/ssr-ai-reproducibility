#!/usr/bin/env bash
# Run once on every node in one allocation of exactly four eight-GPU nodes.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
RUN_ID="${SSR_MEETING_RUN_ID:-}"
EXPECTED_NNODES="${EXPECTED_NNODES:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
DRY_RUN="${SSR_MEETING_DRY_RUN:-0}"
LAUNCH_TOKEN="${SSR_MEETING_LAUNCH_TOKEN:-$RUN_ID}"
MEETING_RESULT_ROOT="${MEETING_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
REUSE_RESULTS_ROOTS="${REUSE_RESULTS_ROOTS:-$PROJECT_ROOT/results/meeting_20260803/imported_decisive_editing}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/meeting_20260803/locked_editing_gpt2xl.yaml}"
CLAIM_WAIT_SEC="${CLAIM_WAIT_SEC:-1200}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-300}"
FINAL_WAIT_SEC="${FINAL_WAIT_SEC:-86400}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_MEETING_RUN_ID to a fresh identifier using letters, digits, dot, underscore, or dash.\n' >&2
  exit 2
}
[[ -n "$LAUNCH_TOKEN" && "$LAUNCH_TOKEN" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'SSR_MEETING_LAUNCH_TOKEN may contain only letters, digits, dot, underscore, or dash.\n' >&2
  exit 2
}
[[ "$EXPECTED_NNODES" == "4" ]] || {
  printf 'This wrapper requires EXPECTED_NNODES=4; got %s.\n' "$EXPECTED_NNODES" >&2
  exit 2
}
[[ "$GPU_COUNT" == "8" ]] || {
  printf 'This wrapper requires GPU_COUNT=8; got %s.\n' "$GPU_COUNT" >&2
  exit 2
}
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || {
  printf 'SSR_MEETING_DRY_RUN must be 0 or 1.\n' >&2
  exit 2
}
for seconds in "$CLAIM_WAIT_SEC" "$INIT_WAIT_SEC" "$FINAL_WAIT_SEC"; do
  [[ "$seconds" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Wait durations must be positive integers; got %s.\n' "$seconds" >&2
    exit 2
  }
done

resolve_global_rank() {
  local key value resolved=""
  for key in GLOBAL_NODE_RANK NODE_RANK PADDLE_TRAINER_ID SLURM_NODEID GROUP_RANK; do
    value="${!key:-}"
    [[ -n "$value" ]] || continue
    [[ "$value" =~ ^[0-9]+$ ]] || {
      printf '%s must be a non-negative integer; got %q.\n' "$key" "$value" >&2
      return 2
    }
    if [[ -n "$resolved" && "$resolved" != "$value" ]]; then
      printf 'Conflicting node-rank variables: resolved %s before %s=%s.\n' \
        "$resolved" "$key" "$value" >&2
      return 2
    fi
    resolved="$value"
  done
  if [[ -n "$resolved" ]]; then
    printf '%s\n' "$resolved"
    return 0
  fi

  local host="${HOSTNAME:-$(hostname)}"
  if [[ "$host" =~ master-([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$host" =~ worker-([0-9]+) ]]; then
    printf '%s\n' "$((BASH_REMATCH[1] + 1))"
    return 0
  fi
  printf 'Cannot resolve GLOBAL_NODE_RANK; set it explicitly to 0, 1, 2, or 3.\n' >&2
  return 2
}

GLOBAL_NODE_RANK="$(resolve_global_rank)" || exit $?
((GLOBAL_NODE_RANK >= 0 && GLOBAL_NODE_RANK < EXPECTED_NNODES)) || {
  printf 'GLOBAL_NODE_RANK=%s is outside 0..3.\n' "$GLOBAL_NODE_RANK" >&2
  exit 2
}

for topology_key in PADDLE_TRAINERS_NUM NNODES SLURM_NNODES PET_NNODES; do
  topology_value="${!topology_key:-}"
  [[ -n "$topology_value" ]] || continue
  [[ "$topology_value" =~ ^[0-9]+$ && "$topology_value" == "$EXPECTED_NNODES" ]] || {
    printf '%s=%s conflicts with the required four-node allocation.\n' \
      "$topology_key" "$topology_value" >&2
    exit 2
  }
done
if [[ -n "${PADDLE_TRAINER_ENDPOINTS:-}" ]]; then
  endpoint_count="$(awk -F, '{print NF}' <<<"$PADDLE_TRAINER_ENDPOINTS")"
  [[ "$endpoint_count" == "$EXPECTED_NNODES" ]] || {
    printf 'PADDLE_TRAINER_ENDPOINTS describes %s nodes; expected 4.\n' "$endpoint_count" >&2
    exit 2
  }
fi

git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/check_experiment_worktree.py" \
    --project-root "$PROJECT_ROOT"
  physical_gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count+0}')"
  [[ "$physical_gpu_count" == "$GPU_COUNT" ]] || {
    printf 'Expected %s physical GPUs on %s, found %s.\n' \
      "$GPU_COUNT" "$(hostname)" "$physical_gpu_count" >&2
    exit 2
  }
fi

IDENTITY_FILE="$MEETING_RESULT_ROOT/.meeting_revision_4node_identity"
if ((GLOBAL_NODE_RANK == 0)); then
  if [[ -e "$MEETING_RESULT_ROOT" ]]; then
    printf 'Fresh launch refused because MEETING_RESULT_ROOT already exists: %s\n' \
      "$MEETING_RESULT_ROOT" >&2
    exit 2
  fi
  mkdir -p "$MEETING_RESULT_ROOT"
  identity_tmp="$IDENTITY_FILE.tmp.$$"
  {
    printf 'protocol=meeting_revision_4node_v1\n'
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'launch_token=%s\n' "$LAUNCH_TOKEN"
    printf 'git_commit=%s\n' "$git_commit"
    printf 'expected_nnodes=%s\n' "$EXPECTED_NNODES"
    printf 'gpu_count=%s\n' "$GPU_COUNT"
  } > "$identity_tmp"
  mv "$identity_tmp" "$IDENTITY_FILE"
else
  init_deadline=$((SECONDS + INIT_WAIT_SEC))
  while [[ ! -s "$IDENTITY_FILE" ]]; do
    ((SECONDS < init_deadline)) || {
      printf 'Timed out waiting for node 0 to initialize %s.\n' "$IDENTITY_FILE" >&2
      exit 2
    }
    sleep 1
  done
fi

for expected_line in \
  'protocol=meeting_revision_4node_v1' \
  "run_id=$RUN_ID" \
  "launch_token=$LAUNCH_TOKEN" \
  "git_commit=$git_commit" \
  "expected_nnodes=$EXPECTED_NNODES" \
  "gpu_count=$GPU_COUNT"; do
  grep -Fqx "$expected_line" "$IDENTITY_FILE" || {
    printf 'Run identity mismatch in %s: missing %s\n' \
      "$IDENTITY_FILE" "$expected_line" >&2
    exit 2
  }
done

STATUS_DIR="$MEETING_RESULT_ROOT/launcher_status"
LOG_DIR="$MEETING_RESULT_ROOT/logs"
EDITING_ROOT="$MEETING_RESULT_ROOT/editing"
MATCHED_KD_ROOT="$MEETING_RESULT_ROOT/matched_kd_cl"
SEGMENTATION_ROOT="$MEETING_RESULT_ROOT/cub_segmentation"
mkdir -p "$STATUS_DIR" "$LOG_DIR"

atomic_status() {
  local path="$1"
  shift
  local temporary="$path.tmp.$(hostname).$$"
  {
    printf 'launch_token=%s\n' "$LAUNCH_TOKEN"
    printf 'git_commit=%s\n' "$git_commit"
    printf '%s\n' "$@"
  } > "$temporary"
  mv "$temporary" "$path"
}

status_matches() {
  local path="$1"
  [[ -s "$path" ]] && grep -Fqx "launch_token=$LAUNCH_TOKEN" "$path"
}

for suffix in done failed claimed; do
  previous="$STATUS_DIR/node_${GLOBAL_NODE_RANK}.${suffix}"
  if [[ -e "$previous" ]]; then
    mv "$previous" "$previous.previous.$(date '+%Y%m%d_%H%M%S').$$"
  fi
done

terminal_status=0
claimed=0
record_failure_on_exit() {
  local exit_code=$?
  if ((exit_code != 0 && claimed == 1 && terminal_status == 0)); then
    atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" \
      "exit_code=$exit_code" "finished_at=$(timestamp)"
  fi
  trap - EXIT
  exit "$exit_code"
}
trap record_failure_on_exit EXIT

atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.claimed" \
  "rank=$GLOBAL_NODE_RANK" "host=$(hostname)" "started_at=$(timestamp)"
claimed=1

claim_deadline=$((SECONDS + CLAIM_WAIT_SEC))
for rank in 0 1 2 3; do
  claim="$STATUS_DIR/node_${rank}.claimed"
  while ! status_matches "$claim"; do
    ((SECONDS < claim_deadline)) || {
      printf 'Timed out waiting for global node %s to claim this launch.\n' "$rank" >&2
      exit 2
    }
    sleep 2
  done
done

GPU_LIST="$(seq -s ' ' 0 $((GPU_COUNT - 1)))"
case "$GLOBAL_NODE_RANK" in
  0|1)
    workload=editing
    child=(
      env
      "NODE_RANK=$GLOBAL_NODE_RANK"
      "NNODES=2"
      "LAUNCH_TOKEN=${LAUNCH_TOKEN}.editing"
      "RUN_ID=${RUN_ID}_editing"
      "RESULT_ROOT=$EDITING_ROOT"
      "REUSE_RESULTS_ROOTS=$REUSE_RESULTS_ROOTS"
      "GPU_LIST=$GPU_LIST"
      "CONFIG_FILE=$CONFIG_FILE"
      "PYTHON=$PYTHON"
      "FINAL_WAIT_SEC=$FINAL_WAIT_SEC"
      "MODEL_NAME=${MODEL_NAME:-}"
      bash "$PROJECT_ROOT/scripts/run_meeting_editing_8gpu.sh"
    )
    [[ "$DRY_RUN" != "1" ]] || child+=(--dry-run)
    ;;
  2)
    workload=matched_kd_cl
    child=(
      env
      "OUTPUT_ROOT=$MATCHED_KD_ROOT"
      "GPU_LIST=$GPU_LIST"
      "PYTHON=$PYTHON"
      "DRY_RUN=$DRY_RUN"
      bash "$PROJECT_ROOT/scripts/run_meeting_matched_kd_cl.sh"
    )
    ;;
  3)
    workload=cub_segmentation
    child=(
      "$PYTHON" "$PROJECT_ROOT/scripts/run_meeting_segmentation_ablation.py"
      --result-root "$SEGMENTATION_ROOT"
      --phase all
      --gpus
    )
    read -r -a gpu_arguments <<< "$GPU_LIST"
    child+=("${gpu_arguments[@]}" --python "$PYTHON")
    [[ "$DRY_RUN" != "1" ]] || child+=(--manifest-only)
    ;;
esac

log_file="$LOG_DIR/node_${GLOBAL_NODE_RANK}_${workload}.log"
run_child() {
  if ((GLOBAL_NODE_RANK != 0)); then
    "${child[@]}" > "$log_file" 2>&1
    return
  fi

  "${child[@]}" > "$log_file" 2>&1 &
  local child_pid=$!
  local local_editing_done="$EDITING_ROOT/launcher_status/node_0.done"
  while kill -0 "$child_pid" 2>/dev/null; do
    if status_matches "$STATUS_DIR/node_1.failed" \
      && [[ -s "$local_editing_done" ]] \
      && grep -Fqx "launch_token=${LAUNCH_TOKEN}.editing" "$local_editing_done"; then
      printf 'Editing peer node 1 failed; stopping node 0 barrier wait.\n' >> "$log_file"
      kill "$child_pid" 2>/dev/null || true
      wait "$child_pid" 2>/dev/null || true
      return 70
    fi
    sleep 2
  done
  wait "$child_pid"
}

set +e
run_child
exit_code=$?
set -e
if ((exit_code == 0)); then
  atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
    "role=$workload" "exit_code=0" "finished_at=$(timestamp)" "log=$log_file"
  terminal_status=1
else
  atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" \
    "role=$workload" "exit_code=$exit_code" "finished_at=$(timestamp)" "log=$log_file"
  terminal_status=1
fi

global_deadline=$((SECONDS + FINAL_WAIT_SEC))
while :; do
  pending=0
  failed=0
  for rank in 0 1 2 3; do
    if status_matches "$STATUS_DIR/node_${rank}.failed"; then
      failed=1
    elif ! status_matches "$STATUS_DIR/node_${rank}.done"; then
      pending=1
    fi
  done
  ((failed == 0)) || exit 1
  ((pending != 0)) || break
  ((SECONDS < global_deadline)) || {
    printf 'Timed out waiting for all four meeting workloads.\n' >&2
    exit 1
  }
  sleep 10
done

finalize() {
  if [[ "$DRY_RUN" == "1" ]]; then
    "$PYTHON" - "$MEETING_RESULT_ROOT" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])

def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

editing = read_tsv(root / "editing" / "planned_runs.tsv")
editing_keys = {
    (row["phase"], row["dataset"], row["recipe"], row["mapping"], row["seed"])
    for row in editing
}
matched_kd = read_tsv(root / "matched_kd_cl" / "planned_runs.tsv")
matched_keys = {(row["dataset"], row["method"], row["seed"]) for row in matched_kd}
segmentation = json.loads((root / "cub_segmentation" / "MANIFEST.json").read_text())

observed = {
    "editing_cells": len(editing),
    "editing_unique_cells": len(editing_keys),
    "matched_kd_jobs": len(matched_kd),
    "matched_kd_unique_jobs": len(matched_keys),
    "segmentation_development_jobs": segmentation["jobs"]["development"],
    "segmentation_confirmation_jobs": segmentation["jobs"]["confirmation"],
}
expected = {
    "editing_cells": 240,
    "editing_unique_cells": 240,
    "matched_kd_jobs": 80,
    "matched_kd_unique_jobs": 80,
    "segmentation_development_jobs": 21,
    "segmentation_confirmation_jobs": 40,
}
if observed != expected:
    raise SystemExit(f"Dry-run matrix mismatch: expected={expected}, observed={observed}")

report = {"status": "PASS", **observed}
path = root / "DRY_RUN_VALIDATION.json"
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
print(json.dumps(report, indent=2))
PY
  else
    "$PYTHON" "$PROJECT_ROOT/scripts/build_result_manifest.py" \
      --root "$MEETING_RESULT_ROOT" \
      --output "$MEETING_RESULT_ROOT/result_manifest.yaml"
    "$PYTHON" "$PROJECT_ROOT/scripts/aggregate_meeting_revision.py" \
      --manifest "$PROJECT_ROOT/configs/meeting_20260803/manifest.yaml" \
      --results-root "$MEETING_RESULT_ROOT" \
      --output "$MEETING_RESULT_ROOT/aggregate"
  fi
}

FINAL_DONE="$STATUS_DIR/final.done"
FINAL_FAILED="$STATUS_DIR/final.failed"
if ((GLOBAL_NODE_RANK == 0)); then
  set +e
  finalize > "$LOG_DIR/finalize.log" 2>&1
  finalize_exit=$?
  set -e
  if ((finalize_exit == 0)); then
    atomic_status "$FINAL_DONE" \
      "exit_code=0" "finished_at=$(timestamp)" "log=$LOG_DIR/finalize.log"
  else
    atomic_status "$FINAL_FAILED" \
      "exit_code=$finalize_exit" "finished_at=$(timestamp)" "log=$LOG_DIR/finalize.log"
    exit "$finalize_exit"
  fi
else
  final_deadline=$((SECONDS + FINAL_WAIT_SEC))
  while ! status_matches "$FINAL_DONE" && ! status_matches "$FINAL_FAILED"; do
    ((SECONDS < final_deadline)) || {
      printf 'Timed out waiting for node 0 finalization.\n' >&2
      exit 1
    }
    sleep 5
  done
  status_matches "$FINAL_FAILED" && exit 1
fi

printf 'Meeting revision role %s complete: %s\n' "$workload" "$MEETING_RESULT_ROOT"
