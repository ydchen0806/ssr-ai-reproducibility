#!/usr/bin/env bash
# Run the same command once on each node of one four-node, eight-GPU allocation.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
RUN_ID="${SSR_ADAPTER_RUN_ID:-}"
LAUNCH_TOKEN="${SSR_ADAPTER_LAUNCH_TOKEN:-$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
DRY_RUN="${SSR_ADAPTER_DRY_RUN:-0}"
RESULT_ROOT="${ADAPTER_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
DATA_ROOT="${ADAPTER_DATA_ROOT:-$PROJECT_ROOT/data/torchvision}"
CACHE_ROOT="${ADAPTER_CACHE_ROOT:-$DATA_ROOT/feature_cache}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-600}"
FINAL_WAIT_SEC="${FINAL_WAIT_SEC:-86400}"

[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_ADAPTER_RUN_ID to a fresh identifier.\n' >&2
  exit 2
}
[[ "$EXPECTED_NNODES" == "4" && "$GPU_COUNT" == "8" ]] || {
  printf 'This protocol requires EXPECTED_NNODES=4 and GPU_COUNT=8.\n' >&2
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
case "$GLOBAL_NODE_RANK" in
  0) DATASET=flowers102 ;;
  1) DATASET=food101 ;;
  2) DATASET=oxfordiiitpet ;;
  3) DATASET=dtd ;;
esac
DATASETS=(flowers102 food101 oxfordiiitpet dtd)
DEVELOPMENT_SEEDS=(3101 3103 3105)
DEVELOPMENT_SCALES=(0.03 0.1 0.3 1.0)
CONFIRMATION_SEEDS=(4101 4103 4105 4107 4109)
RANK=16

if [[ "$DRY_RUN" == "1" ]]; then
  "$PYTHON" - "$GLOBAL_NODE_RANK" "$DATASET" <<'PY'
import json, sys
print(json.dumps({
    "node_rank": int(sys.argv[1]),
    "dataset": sys.argv[2],
    "development_pairs": 12,
    "confirmation_pairs": 5,
    "methods_per_pair": ["task", "task_ssr"],
    "matched_budget": True,
}, sort_keys=True))
PY
  exit 0
fi

physical_gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count+0}')"
[[ "$physical_gpu_count" == "$GPU_COUNT" ]] || {
  printf 'Expected 8 GPUs, found %s on %s.\n' "$physical_gpu_count" "$(hostname)" >&2
  exit 2
}

git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
IDENTITY_FILE="$RESULT_ROOT/.adapter_4node_identity"
STATUS_DIR="$RESULT_ROOT/status"
LOG_DIR="$RESULT_ROOT/logs"

if ((GLOBAL_NODE_RANK == 0)); then
  [[ ! -e "$RESULT_ROOT" ]] || {
    printf 'Fresh result root required: %s already exists.\n' "$RESULT_ROOT" >&2
    exit 2
  }
  mkdir -p "$STATUS_DIR" "$LOG_DIR"
  {
    printf 'protocol=multidataset_adapter_task_vs_ssr_v1\n'
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'launch_token=%s\n' "$LAUNCH_TOKEN"
    printf 'git_commit=%s\n' "$git_commit"
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
grep -Fqx "launch_token=$LAUNCH_TOKEN" "$IDENTITY_FILE" || exit 2
grep -Fqx "git_commit=$git_commit" "$IDENTITY_FILE" || exit 2

atomic_status() {
  local path="$1" value="$2" temporary="$1.tmp.$(hostname).$$"
  printf 'launch_token=%s\n%s\n' "$LAUNCH_TOKEN" "$value" > "$temporary"
  mv "$temporary" "$path"
}

wait_statuses() {
  local suffix="$1" deadline=$((SECONDS + FINAL_WAIT_SEC)) rank path
  for rank in 0 1 2 3; do
    path="$STATUS_DIR/node_${rank}.${suffix}"
    while ! { [[ -s "$path" ]] && grep -Fqx "launch_token=$LAUNCH_TOKEN" "$path"; }; do
      if [[ -s "$STATUS_DIR/node_${rank}.failed" ]]; then
        printf 'Node %s reported failure.\n' "$rank" >&2
        return 1
      fi
      ((SECONDS < deadline)) || return 1
      sleep 5
    done
  done
}

record_failure() {
  local code=$?
  if ((code != 0)); then
    atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" "exit_code=$code"
  fi
  exit "$code"
}
trap record_failure EXIT

atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.claimed" "dataset=$DATASET"
wait_statuses claimed

CACHE_FILE="$CACHE_ROOT/${DATASET}_resnet18.pt"
mkdir -p "$CACHE_ROOT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  "$PROJECT_ROOT/experiments/multidataset_adapter_ssr.py" \
  --dataset "$DATASET" \
  --data-root "$DATA_ROOT" \
  --feature-cache "$CACHE_FILE" \
  --output-dir "$RESULT_ROOT/prepare/$DATASET" \
  --device cuda \
  --download \
  --prepare-only \
  > "$LOG_DIR/node_${GLOBAL_NODE_RANK}_${DATASET}_prepare.log" 2>&1
atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.prepared" "dataset=$DATASET"

run_grid() {
  local phase="$1" seeds_name="$2" scales_name="$3"
  local -n seeds_ref="$seeds_name"
  local -n scales_ref="$scales_name"
  local active=0 gpu=0 seed scale output log pid
  local -a pids=()
  for scale in "${scales_ref[@]}"; do
    for seed in "${seeds_ref[@]}"; do
      output="$RESULT_ROOT/$phase/$DATASET/scale_${scale}/rank_${RANK}/seed_${seed}"
      log="$LOG_DIR/${phase}_${DATASET}_scale_${scale}_seed_${seed}.log"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
        "$PROJECT_ROOT/experiments/multidataset_adapter_ssr.py" \
        --dataset "$DATASET" \
        --data-root "$DATA_ROOT" \
        --feature-cache "$CACHE_FILE" \
        --output-dir "$output" \
        --device cuda \
        --seed "$seed" \
        --rank "$RANK" \
        --ssr-scale "$scale" \
        > "$log" 2>&1 &
      pids+=("$!")
      active=$((active + 1))
      gpu=$(((gpu + 1) % GPU_COUNT))
      if ((active == GPU_COUNT)); then
        for pid in "${pids[@]}"; do wait "$pid"; done
        pids=()
        active=0
      fi
    done
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

run_grid development DEVELOPMENT_SEEDS DEVELOPMENT_SCALES
atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.development_done" "dataset=$DATASET"
wait_statuses development_done

LOCK_DIR="$RESULT_ROOT/development_summary"
LOCK_FILE="$LOCK_DIR/selection_lock.json"
if ((GLOBAL_NODE_RANK == 0)); then
  "$PYTHON" "$PROJECT_ROOT/scripts/summarize_multidataset_adapter_ssr.py" \
    --input-root "$RESULT_ROOT/development" \
    --output-dir "$LOCK_DIR" \
    --expected-datasets "${DATASETS[@]}" \
    --expected-seeds "${DEVELOPMENT_SEEDS[@]}" \
    --select-lock \
    > "$LOG_DIR/development_aggregate.log" 2>&1
fi
deadline=$((SECONDS + FINAL_WAIT_SEC))
while [[ ! -s "$LOCK_FILE" ]]; do
  ((SECONDS < deadline)) || exit 1
  sleep 5
done
LOCKED_SCALE="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["ssr_scale"])' "$LOCK_FILE")"
LOCKED_RANK="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["rank"])' "$LOCK_FILE")"
[[ "$LOCKED_RANK" == "$RANK" ]] || exit 1
CONFIRMATION_SCALES=("$LOCKED_SCALE")

run_grid confirmation CONFIRMATION_SEEDS CONFIRMATION_SCALES
atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.confirmation_done" "dataset=$DATASET"
wait_statuses confirmation_done

if ((GLOBAL_NODE_RANK == 0)); then
  "$PYTHON" "$PROJECT_ROOT/scripts/summarize_multidataset_adapter_ssr.py" \
    --input-root "$RESULT_ROOT/confirmation" \
    --output-dir "$RESULT_ROOT/confirmation_summary" \
    --expected-datasets "${DATASETS[@]}" \
    --expected-seeds "${CONFIRMATION_SEEDS[@]}" \
    > "$LOG_DIR/confirmation_aggregate.log" 2>&1
  atomic_status "$STATUS_DIR/aggregate.done" "locked_scale=$LOCKED_SCALE"
fi
deadline=$((SECONDS + FINAL_WAIT_SEC))
while [[ ! -s "$STATUS_DIR/aggregate.done" ]]; do
  ((SECONDS < deadline)) || exit 1
  sleep 5
done

trap - EXIT
printf 'Completed dataset=%s; locked SSR scale=%s; results=%s\n' \
  "$DATASET" "$LOCKED_SCALE" "$RESULT_ROOT"
