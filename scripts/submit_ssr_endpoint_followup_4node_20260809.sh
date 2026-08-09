#!/usr/bin/env bash
# Run once on every node of one four-node, eight-GPU-per-node allocation.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
RUN_ID="${SSR_ENDPOINT_RUN_ID:-}"
RESULT_ROOT="${SSR_ENDPOINT_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
MAX_WALLTIME_SEC="${MAX_WALLTIME_SEC:-28800}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-900}"
FINISH_WAIT_SEC="${FINISH_WAIT_SEC:-32400}"
DRY_RUN="${SSR_ENDPOINT_DRY_RUN:-0}"

[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_ENDPOINT_RUN_ID to a run identifier.\n' >&2
  exit 2
}
[[ "$EXPECTED_NNODES" == "4" && "$GPU_COUNT" == "8" ]] || {
  printf 'This protocol requires four nodes with eight GPUs on each node.\n' >&2
  exit 2
}
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
  printf 'JOBS_PER_GPU must be a positive integer.\n' >&2
  exit 2
}
for seconds in "$MAX_WALLTIME_SEC" "$INIT_WAIT_SEC" "$FINISH_WAIT_SEC"; do
  [[ "$seconds" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Wall-time and wait values must be positive integers.\n' >&2
    exit 2
  }
done
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || {
  printf 'SSR_ENDPOINT_DRY_RUN must be 0 or 1.\n' >&2
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

IDENTITY_FILE="$RESULT_ROOT/.endpoint_followup_identity"
STATUS_DIR="$RESULT_ROOT/status"
LOG_DIR="$RESULT_ROOT/logs"
mkdir -p "$RESULT_ROOT"
if ((GLOBAL_NODE_RANK == 0)) && [[ ! -s "$IDENTITY_FILE" ]]; then
  mkdir -p "$STATUS_DIR" "$LOG_DIR"
  temporary="$IDENTITY_FILE.tmp.$(hostname).$$"
  {
    printf 'protocol=ssr_endpoint_followup_4node_v1\n'
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
for previous_status in \
  "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
  "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed"; do
  if [[ -e "$previous_status" ]]; then
    mv "$previous_status" \
      "$previous_status.previous.$(date +%Y%m%dT%H%M%S).$$"
  fi
done
for expected_line in \
  'protocol=ssr_endpoint_followup_4node_v1' \
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

role="unassigned"
terminal_status=0
on_exit() {
  local code=$?
  if ((code != 0 && terminal_status == 0)); then
    atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" \
      "role=$role" "exit_code=$code" "finished_at=$(date -Iseconds)"
  fi
  trap - EXIT
  exit "$code"
}
trap on_exit EXIT

GPU_ARGS=(0 1 2 3 4 5 6 7)
if [[ "$DRY_RUN" == "1" ]]; then
  case "$GLOBAL_NODE_RANK" in
    0)
      role=flowers102_vit_lora_plan
      "$PYTHON" "$PROJECT_ROOT/scripts/run_vit_lora_flowers_srlc.py" \
        --result-root "$RESULT_ROOT/plans/flowers102_vit_lora" \
        --gpus "${GPU_ARGS[@]}" --jobs-per-gpu "$JOBS_PER_GPU" --dry-run \
        > "$LOG_DIR/node_0_plan.log" 2>&1
      ;;
    1) dataset=cub200 ;;
    2) dataset=oxford_iiit_pet ;;
    3) dataset=oxford_flowers102 ;;
  esac
  if ((GLOBAL_NODE_RANK > 0)); then
    role="${dataset}_segmentation_plan"
    case "$dataset" in
      cub200)
        dataset_root="$PROJECT_ROOT/data/cub200"
        cache="$dataset_root/cub200_seg_resnet18_dense_192.pt"
        ;;
      oxford_iiit_pet)
        dataset_root="$PROJECT_ROOT/data/oxford_iiit_pet"
        cache="$dataset_root/resnet18_layer3_masks_160_provenance_v2.pt"
        ;;
      oxford_flowers102)
        dataset_root="$PROJECT_ROOT/data/flowers-102"
        cache="$dataset_root/resnet18_layer3_masks_160_provenance_v2.pt"
        ;;
    esac
    "$PYTHON" "$PROJECT_ROOT/scripts/run_segmentation_nested_dataset.py" \
      --dataset "$dataset" --dataset-root "$dataset_root" \
      --feature-cache "$cache" \
      --result-root "$RESULT_ROOT/plans/$dataset" \
      --gpus "${GPU_ARGS[@]}" --jobs-per-gpu "$JOBS_PER_GPU" \
      --manifest-only > "$LOG_DIR/node_${GLOBAL_NODE_RANK}_plan.log" 2>&1
  fi
else
  case "$GLOBAL_NODE_RANK" in
    0)
      role=flowers102_vit_lora_srlc
      timeout --signal=TERM --kill-after=300 "$MAX_WALLTIME_SEC" \
        "$PYTHON" "$PROJECT_ROOT/scripts/run_vit_lora_flowers_srlc.py" \
        --result-root "$RESULT_ROOT/flowers102_vit_lora" \
        --data-root "$PROJECT_ROOT/data" \
        --pretrained-checkpoint \
          "$PROJECT_ROOT/data/pretrained/vit_tiny_patch16_224_augreg_in21k_ft_in1k.safetensors" \
        --gpus "${GPU_ARGS[@]}" --jobs-per-gpu "$JOBS_PER_GPU" --workers 2 \
        > "$LOG_DIR/node_0_flowers_lora.log" 2>&1
      [[ -s "$RESULT_ROOT/flowers102_vit_lora/CONFIRMATION_SUMMARY.json" ]]
      ;;
    1) dataset=cub200 ;;
    2) dataset=oxford_iiit_pet ;;
    3) dataset=oxford_flowers102 ;;
  esac
  if ((GLOBAL_NODE_RANK > 0)); then
    role="${dataset}_segmentation_srlc"
    SSR_SEG_DATASET="$dataset" \
    SSR_SEG_RUN_ID="$RUN_ID" \
    SSR_SEG_RESULT_ROOT="$RESULT_ROOT/segmentation/$dataset" \
    SSR_SEG_RESUME=1 \
    GPU_COUNT="$GPU_COUNT" \
    JOBS_PER_GPU="$JOBS_PER_GPU" \
    MAX_WALLTIME_SEC="$MAX_WALLTIME_SEC" \
      bash "$PROJECT_ROOT/scripts/run_segmentation_nested_dataset_8gpu_20260809.sh" \
      > "$LOG_DIR/node_${GLOBAL_NODE_RANK}_${dataset}.log" 2>&1
    [[ -s "$RESULT_ROOT/segmentation/$dataset/CONFIRMATION_SUMMARY.json" ]]
  fi
fi

atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
  "role=$role" "exit_code=0" "finished_at=$(date -Iseconds)"
terminal_status=1

if ((GLOBAL_NODE_RANK == 0)); then
  deadline=$((SECONDS + FINISH_WAIT_SEC))
  for rank in 1 2 3; do
    while [[ ! -s "$STATUS_DIR/node_${rank}.done" ]]; do
      if [[ -s "$STATUS_DIR/node_${rank}.failed" ]]; then
        printf 'Node %s failed:\n' "$rank" >&2
        cat "$STATUS_DIR/node_${rank}.failed" >&2
        exit 2
      fi
      ((SECONDS < deadline)) || {
        printf 'Timed out waiting for node %s.\n' "$rank" >&2
        exit 2
      }
      sleep 5
    done
  done
  if [[ "$DRY_RUN" == "1" ]]; then
    "$PYTHON" - "$RESULT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
plan_text = (root / "logs" / "node_0_plan.log").read_text()
json_start = plan_text.find("{")
if json_start < 0:
    raise SystemExit("LoRA plan did not emit JSON")
lora = json.loads(plan_text[json_start:])
expected_lora = {
    "candidates": 22,
    "screen_jobs": 75,
    "maximum_refine_jobs": 35,
    "confirmation_jobs": 20,
}
for key, value in expected_lora.items():
    if lora.get(key) != value:
        raise SystemExit(f"Unexpected LoRA plan {key}: {lora.get(key)} != {value}")
expected = {
    "cub200": 436,
    "oxford_iiit_pet": 436,
    "oxford_flowers102": 436,
}
observed = {}
for dataset, total in expected.items():
    manifest = json.loads((root / "plans" / dataset / "MANIFEST.json").read_text())
    observed[dataset] = manifest["jobs"]["total_maximum"]
    if observed[dataset] != total:
        raise SystemExit(f"Unexpected {dataset} plan: {observed[dataset]} != {total}")
payload = {
    "status": "PASS",
    "nodes": 4,
    "flowers102_vit_lora_jobs": {
        key: lora[key] for key in expected_lora if key != "candidates"
    },
    "segmentation_jobs": observed,
}
(root / "PLAN_VALIDATION.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
  else
    "$PYTHON" - "$RESULT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = {
    "flowers102_vit_lora": root / "flowers102_vit_lora" / "CONFIRMATION_SUMMARY.json",
    "cub200_segmentation": root / "segmentation" / "cub200" / "CONFIRMATION_SUMMARY.json",
    "pet_segmentation": root / "segmentation" / "oxford_iiit_pet" / "CONFIRMATION_SUMMARY.json",
    "flowers102_segmentation": root / "segmentation" / "oxford_flowers102" / "CONFIRMATION_SUMMARY.json",
}
payload = {
    "status": "complete",
    "summaries": {name: json.loads(path.read_text()) for name, path in paths.items()},
}
(root / "ENDPOINT_FOLLOWUP_SUMMARY.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
PY
  fi
fi

printf 'Node %s completed role=%s result_root=%s\n' \
  "$GLOBAL_NODE_RANK" "$role" "$RESULT_ROOT"
