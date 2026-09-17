#!/usr/bin/env bash
# Run once on every node of one four-node, eight-GPU-per-node allocation.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
RUN_ID="${SSR_DIRECT_RUN_ID:-}"
RESULT_ROOT="${SSR_DIRECT_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
MAX_WALLTIME_SEC="${MAX_WALLTIME_SEC:-28800}"
FINAL_WAIT_SEC="${FINAL_WAIT_SEC:-32400}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-900}"
DRY_RUN="${SSR_DIRECT_DRY_RUN:-0}"

[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_DIRECT_RUN_ID to a run identifier.\n' >&2
  exit 2
}
[[ "$EXPECTED_NNODES" == "4" && "$GPU_COUNT" == "8" ]] || {
  printf 'This launcher requires four nodes with eight GPUs each.\n' >&2
  exit 2
}
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || exit 2
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

IDENTITY_FILE="$RESULT_ROOT/.direct_attribution_identity"
STATUS_DIR="$RESULT_ROOT/status"
LOG_DIR="$RESULT_ROOT/logs"
mkdir -p "$RESULT_ROOT"
if ((GLOBAL_NODE_RANK == 0)) && [[ ! -s "$IDENTITY_FILE" ]]; then
  mkdir -p "$STATUS_DIR" "$LOG_DIR"
  temporary="$IDENTITY_FILE.tmp.$(hostname).$$"
  {
    printf 'protocol=direct_ssr_attribution_4node_v1\n'
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
for expected in \
  'protocol=direct_ssr_attribution_4node_v1' \
  "run_id=$RUN_ID" \
  "git_commit=$git_commit" \
  'nodes=4' \
  'gpus_per_node=8' \
  "jobs_per_gpu=$JOBS_PER_GPU"; do
  grep -Fqx "$expected" "$IDENTITY_FILE" || {
    printf 'Run identity mismatch: %s\n' "$expected" >&2
    exit 2
  }
done

for stale in \
  "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
  "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed"; do
  if [[ -e "$stale" ]]; then
    mv "$stale" "$stale.previous.$(date +%Y%m%dT%H%M%S).$$"
  fi
done

atomic_status() {
  local target="$1" role="$2" code="$3"
  local temporary="$target.tmp.$(hostname).$$"
  {
    printf 'git_commit=%s\n' "$git_commit"
    printf 'host=%s\n' "$(hostname)"
    printf 'role=%s\n' "$role"
    printf 'exit_code=%s\n' "$code"
    printf 'finished_at=%s\n' "$(date -Iseconds)"
  } > "$temporary"
  mv "$temporary" "$target"
}

role="unassigned"
completed=0
on_exit() {
  local code=$?
  if ((code != 0 && completed == 0)); then
    atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" "$role" "$code"
  fi
  trap - EXIT
  exit "$code"
}
trap on_exit EXIT

run_bounded() {
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=300 "$MAX_WALLTIME_SEC" "$@"
  elif [[ "$DRY_RUN" == "1" ]]; then
    "$@"
  else
    printf 'GNU timeout is required for a live launch.\n' >&2
    return 2
  fi
}

GPU_ARGS=(0 1 2 3 4 5 6 7)
if ((GLOBAL_NODE_RANK < 3)); then
  role="prototype_segmentation_rank_${GLOBAL_NODE_RANK}_of_3"
  segmentation_command=(
    "$PYTHON"
    "$PROJECT_ROOT/scripts/run_direct_ssr_prototype_segmentation_4node.py"
    --result-root "$RESULT_ROOT/segmentation"
    --node-rank "$GLOBAL_NODE_RANK"
    --nodes 3
    --gpus "${GPU_ARGS[@]}"
    --jobs-per-gpu "$JOBS_PER_GPU"
    --workers 0
    --barrier-timeout "$FINAL_WAIT_SEC"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    segmentation_command+=(--manifest-only)
  fi
  run_bounded "${segmentation_command[@]}" \
    > "$LOG_DIR/node_${GLOBAL_NODE_RANK}_segmentation.log" 2>&1
  if [[ "$DRY_RUN" == "0" && "$GLOBAL_NODE_RANK" == "0" ]]; then
    [[ -s "$RESULT_ROOT/segmentation/CONFIRMATION_SUMMARY.json" ]]
  fi
else
  role="pet_vit_lora_task_to_task_ssr_srlc"
  pet_command=(
    "$PYTHON" "$PROJECT_ROOT/scripts/run_vit_lora_pet_srlc.py"
    --result-root "$RESULT_ROOT/pet_vit_lora"
    --data-root "$PROJECT_ROOT/data"
    --pretrained-checkpoint
    "$PROJECT_ROOT/data/pretrained/vit_tiny_patch16_224_augreg_in21k_ft_in1k.safetensors"
    --gpus "${GPU_ARGS[@]}"
    --jobs-per-gpu "$JOBS_PER_GPU"
    --workers 2
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    pet_command+=(--dry-run)
  fi
  run_bounded "${pet_command[@]}" \
    > "$LOG_DIR/node_3_pet_lora.log" 2>&1
  if [[ "$DRY_RUN" == "0" ]]; then
    [[ -s "$RESULT_ROOT/pet_vit_lora/CONFIRMATION_SUMMARY.json" ]]
  fi
fi

atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" "$role" 0
completed=1

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
      sleep 5
    done
  done
  "$PYTHON" - "$RESULT_ROOT" "$DRY_RUN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dry_run = sys.argv[2] == "1"
if dry_run:
    segmentation_log = (root / "logs" / "node_0_segmentation.log").read_text()
    pet_log = (root / "logs" / "node_3_pet_lora.log").read_text()
    segmentation = json.loads(segmentation_log[segmentation_log.index("{"):])
    pet = json.loads(pet_log[pet_log.index("{"):])
    if segmentation["jobs"]["total_runs"] != 332:
        raise SystemExit("Unexpected segmentation workload")
    if pet["maximum_total_jobs"] != 240:
        raise SystemExit("Unexpected Pet workload")
    payload = {
        "status": "PLAN_PASS",
        "segmentation_runs": 332,
        "pet_maximum_jobs": 240,
        "nodes": {"segmentation": 3, "pet_vit_lora": 1},
    }
else:
    segmentation = json.loads(
        (root / "segmentation" / "CONFIRMATION_SUMMARY.json").read_text()
    )
    pet = json.loads(
        (root / "pet_vit_lora" / "CONFIRMATION_SUMMARY.json").read_text()
    )
    payload = {
        "status": "COMPLETE",
        "execution_complete_not_scientific_pass": True,
        "segmentation_primary_gates": {
            name: values["primary_gate"]
            for name, values in segmentation["datasets"].items()
        },
        "pet_status": pet["status"],
        "segmentation_summary": str(
            root / "segmentation" / "CONFIRMATION_SUMMARY.json"
        ),
        "pet_summary": str(root / "pet_vit_lora" / "CONFIRMATION_SUMMARY.json"),
    }
(root / "DIRECT_ATTRIBUTION_SUMMARY.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY
fi

printf 'Direct SSR attribution node %s complete: %s\n' \
  "$GLOBAL_NODE_RANK" "$RESULT_ROOT"
