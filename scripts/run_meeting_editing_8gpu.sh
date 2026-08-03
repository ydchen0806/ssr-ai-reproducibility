#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
RUN_ID="${RUN_ID:-meeting_editing_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/results/$RUN_ID}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${PADDLE_TRAINER_ID:-${SLURM_NODEID:-0}}}"
read -r -a gpus <<< "$GPU_LIST"
if ((${#gpus[@]} == 0)); then
  printf 'GPU_LIST must contain at least one GPU index.\n' >&2
  exit 2
fi
if [[ ! "$NNODES" =~ ^[1-9][0-9]*$ || ! "$NODE_RANK" =~ ^[0-9]+$ ]] \
  || ((NODE_RANK >= NNODES)); then
  printf 'Require 0 <= NODE_RANK < NNODES; got %s/%s.\n' "$NODE_RANK" "$NNODES" >&2
  exit 2
fi

pids=()
global_shard_count=$((${#gpus[@]} * NNODES))
for index in "${!gpus[@]}"; do
  global_shard_index=$((NODE_RANK * ${#gpus[@]} + index))
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" \
  SHARD_INDEX="$global_shard_index" \
  SHARD_COUNT="$global_shard_count" \
  RUN_ID="$RUN_ID" \
  RESULT_ROOT="$RESULT_ROOT" \
    bash "$SCRIPT_DIR/run_meeting_editing_ablation.sh" "$@" &
  pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
  wait "$pid" || exit_code=1
done
((exit_code == 0)) || exit "$exit_code"

mkdir -p "$RESULT_ROOT/launcher_status"
printf 'finished_at=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" \
  > "$RESULT_ROOT/launcher_status/node_${NODE_RANK}.done"
if ((NNODES > 1 && NODE_RANK != 0)); then
  printf 'Editing node %s/%s completed: %s\n' "$NODE_RANK" "$NNODES" "$RESULT_ROOT"
  exit 0
fi
if ((NNODES > 1)); then
  deadline=$((SECONDS + ${FINAL_WAIT_SEC:-43200}))
  for rank in $(seq 0 $((NNODES - 1))); do
    while [[ ! -s "$RESULT_ROOT/launcher_status/node_${rank}.done" ]]; do
      ((SECONDS < deadline)) || { printf 'Timed out waiting for node %s.\n' "$rank" >&2; exit 1; }
      sleep 30
    done
  done
fi

combined="$RESULT_ROOT/planned_runs.tsv"
first=1
for plan in "$RESULT_ROOT"/planned_runs_shard_*_of_*.tsv; do
  if ((first)); then
    sed -n '1p' "$plan" > "$combined"
    first=0
  fi
  sed -n '2,$p' "$plan" >> "$combined"
done
printf 'All editing shards completed: %s\n' "$RESULT_ROOT"
