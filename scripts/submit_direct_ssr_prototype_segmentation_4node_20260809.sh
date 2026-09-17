#!/usr/bin/env bash
# Run once on every node in one four-node, eight-GPU-per-node allocation.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
RUN_ID="${SSR_DIRECT_SEG_RUN_ID:-}"
RESULT_ROOT="${SSR_DIRECT_SEG_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
SEG_WORKERS="${SEG_WORKERS:-0}"
MAX_WALLTIME_SEC="${MAX_WALLTIME_SEC:-28800}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-900}"
FINAL_WAIT_SEC="${FINAL_WAIT_SEC:-50400}"
DRY_RUN="${SSR_DIRECT_SEG_DRY_RUN:-0}"

[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_DIRECT_SEG_RUN_ID to a fresh identifier.\n' >&2
  exit 2
}
[[ "$EXPECTED_NNODES" == "4" && "$GPU_COUNT" == "8" ]] || {
  printf 'This protocol requires four nodes with eight GPUs each.\n' >&2
  exit 2
}
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ && "$SEG_WORKERS" =~ ^[0-9]+$ ]] || {
  printf 'JOBS_PER_GPU must be positive and SEG_WORKERS non-negative.\n' >&2
  exit 2
}
for seconds in "$MAX_WALLTIME_SEC" "$INIT_WAIT_SEC" "$FINAL_WAIT_SEC"; do
  [[ "$seconds" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Wall-time and wait values must be positive integers.\n' >&2
    exit 2
  }
done
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || {
  printf 'SSR_DIRECT_SEG_DRY_RUN must be 0 or 1.\n' >&2
  exit 2
}

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
    printf 'Expected eight GPUs, found %s on %s.\n' \
      "$physical_gpu_count" "$(hostname)" >&2
    exit 2
  }
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

IDENTITY_FILE="$RESULT_ROOT/.direct_ssr_segmentation_identity"
STATUS_DIR="$RESULT_ROOT/launcher_status"
LOG_DIR="$RESULT_ROOT/launcher_logs"
mkdir -p "$RESULT_ROOT"
if ((GLOBAL_NODE_RANK == 0)) && [[ ! -s "$IDENTITY_FILE" ]]; then
  mkdir -p "$STATUS_DIR" "$LOG_DIR"
  temporary="$IDENTITY_FILE.tmp.$(hostname).$$"
  {
    printf 'protocol=direct_ssr_prototype_segmentation_v1\n'
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'git_commit=%s\n' "$git_commit"
    printf 'nodes=4\n'
    printf 'gpus_per_node=8\n'
    printf 'jobs_per_gpu=%s\n' "$JOBS_PER_GPU"
  } > "$temporary"
  mv "$temporary" "$IDENTITY_FILE"
fi

deadline=$((SECONDS + INIT_WAIT_SEC))
while [[ ! -s "$IDENTITY_FILE" ]]; do
  ((SECONDS < deadline)) || {
    printf 'Timed out waiting for the shared experiment identity.\n' >&2
    exit 2
  }
  sleep 1
done
mkdir -p "$STATUS_DIR" "$LOG_DIR"
for expected_line in \
  'protocol=direct_ssr_prototype_segmentation_v1' \
  "run_id=$RUN_ID" \
  "git_commit=$git_commit" \
  'nodes=4' \
  'gpus_per_node=8' \
  "jobs_per_gpu=$JOBS_PER_GPU"; do
  grep -Fqx "$expected_line" "$IDENTITY_FILE" || {
    printf 'Run identity mismatch: %s\n' "$expected_line" >&2
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

for prior in \
  "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
  "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed"; do
  [[ ! -e "$prior" ]] || mv "$prior" "$prior.previous.$(date +%Y%m%dT%H%M%S).$$"
done
INTERNAL_LAUNCHER_FAILURE="$RESULT_ROOT/.barriers/launcher/node_${GLOBAL_NODE_RANK}.failed.json"
rm -f -- "$INTERNAL_LAUNCHER_FAILURE"

terminal_status=0
on_exit() {
  local code=$?
  if ((code != 0 && terminal_status == 0)); then
    atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" \
      "exit_code=$code" "finished_at=$(date -Iseconds)"
    mkdir -p "$(dirname "$INTERNAL_LAUNCHER_FAILURE")"
    local temporary="$INTERNAL_LAUNCHER_FAILURE.tmp.$(hostname).$$"
    printf '{"node_rank":%s,"status":"failed","exit_code":%s}\n' \
      "$GLOBAL_NODE_RANK" "$code" > "$temporary"
    mv "$temporary" "$INTERNAL_LAUNCHER_FAILURE"
  fi
  trap - EXIT
  exit "$code"
}
trap on_exit EXIT

GPU_ARGS=(0 1 2 3 4 5 6 7)
if [[ "$DRY_RUN" == "1" ]]; then
  if ((GLOBAL_NODE_RANK == 0)); then
    "$PYTHON" "$PROJECT_ROOT/scripts/run_direct_ssr_prototype_segmentation_4node.py" \
      --result-root "$RESULT_ROOT" \
      --node-rank 0 --nodes 4 \
      --gpus "${GPU_ARGS[@]}" \
      --jobs-per-gpu "$JOBS_PER_GPU" \
      --workers "$SEG_WORKERS" \
      --python "$PYTHON" \
      --manifest-only > "$LOG_DIR/node_0_plan.log" 2>&1
  fi
else
  timeout --signal=TERM --kill-after=300 "$MAX_WALLTIME_SEC" \
    "$PYTHON" "$PROJECT_ROOT/scripts/run_direct_ssr_prototype_segmentation_4node.py" \
      --result-root "$RESULT_ROOT" \
      --node-rank "$GLOBAL_NODE_RANK" --nodes 4 \
      --gpus "${GPU_ARGS[@]}" \
      --jobs-per-gpu "$JOBS_PER_GPU" \
      --workers "$SEG_WORKERS" \
      --python "$PYTHON" \
      --barrier-timeout "$FINAL_WAIT_SEC" \
      > "$LOG_DIR/node_${GLOBAL_NODE_RANK}.log" 2>&1
  [[ -s "$RESULT_ROOT/COMPLETED.json" ]]
fi

atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
  'exit_code=0' "finished_at=$(date -Iseconds)"
terminal_status=1

if ((GLOBAL_NODE_RANK == 0)); then
  deadline=$((SECONDS + FINAL_WAIT_SEC))
  for rank in 1 2 3; do
    while [[ ! -s "$STATUS_DIR/node_${rank}.done" ]]; do
      if [[ -s "$STATUS_DIR/node_${rank}.failed" ]]; then
        printf 'Node %s failed:\n' "$rank" >&2
        sed -n '1,20p' "$STATUS_DIR/node_${rank}.failed" >&2
        exit 2
      fi
      ((SECONDS < deadline)) || {
        printf 'Timed out waiting for node %s.\n' "$rank" >&2
        exit 2
      }
      sleep 1
    done
  done
  if [[ "$DRY_RUN" == "1" ]]; then
    "$PYTHON" - "$RESULT_ROOT/MANIFEST.json" "$RESULT_ROOT/MATRIX_PLAN_VALIDATION.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "screen_pairs": 96,
    "screen_controls": 6,
    "screen_treatments": 96,
    "screen_runs": 102,
    "refine_pairs": 56,
    "refine_controls": 14,
    "refine_treatments": 56,
    "refine_runs": 70,
    "confirmation_pairs": 80,
    "confirmation_controls": 80,
    "confirmation_treatments": 80,
    "confirmation_runs": 160,
    "total_pairs": 232,
    "total_runs": 332,
}
if manifest.get("protocol") != "direct_ssr_prototype_segmentation_v1":
    raise SystemExit("protocol mismatch")
if manifest.get("jobs") != expected:
    raise SystemExit(f"job-budget mismatch: {manifest.get('jobs')}")
if len(manifest.get("candidate_matrix", [])) != 16:
    raise SystemExit("candidate matrix must contain 16 entries")
payload = {"status": "PASS", "protocol": manifest["protocol"], **expected, "nodes": 4}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  fi
fi

printf 'Completed node_rank=%s result_root=%s\n' "$GLOBAL_NODE_RANK" "$RESULT_ROOT"
