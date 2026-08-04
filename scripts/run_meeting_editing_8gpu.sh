#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
RUN_ID="${RUN_ID:-meeting_editing_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/results/$RUN_ID}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/meeting_20260803/locked_editing_gpt2xl.yaml}"
PYTHON="${PYTHON:-python3}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${PADDLE_TRAINER_ID:-${SLURM_NODEID:-0}}}"
LAUNCH_TOKEN="${LAUNCH_TOKEN:-}"
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
if ((NNODES > 1)) && [[ -z "$LAUNCH_TOKEN" ]]; then
  printf 'LAUNCH_TOKEN is required when NNODES > 1 and must be identical on every node.\n' >&2
  exit 2
fi
if [[ -z "$LAUNCH_TOKEN" ]]; then
  LAUNCH_TOKEN="single_${RUN_ID}_${NODE_RANK}"
fi
mkdir -p "$RESULT_ROOT/launcher_status"
node_done="$RESULT_ROOT/launcher_status/node_${NODE_RANK}.done"
if [[ -s "$node_done" ]]; then
  mv "$node_done" "$node_done.previous.$(date '+%Y%m%d_%H%M%S')"
fi

dry_run=0
for argument in "$@"; do
  [[ "$argument" == "--dry-run" ]] && dry_run=1
done
model_path="${MODEL_NAME:-$($PYTHON - "$CONFIG_FILE" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["model"]["default_cluster_path"])
PY
)}"
if ((dry_run == 0)); then
  "$PYTHON" "$SCRIPT_DIR/verify_locked_editing_assets.py" \
    --config "$CONFIG_FILE" \
    --model-path "$model_path" \
    --output "$RESULT_ROOT/LOCKED_ASSET_VERIFICATION.node_${NODE_RANK}.json"
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
  CONFIG_FILE="$CONFIG_FILE" \
  PYTHON="$PYTHON" \
  MODEL_NAME="$model_path" \
    bash "$SCRIPT_DIR/run_meeting_editing_ablation.sh" "$@" &
  pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
  wait "$pid" || exit_code=1
done
((exit_code == 0)) || exit "$exit_code"

{
  printf 'launch_token=%s\n' "$LAUNCH_TOKEN"
  printf 'finished_at=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "$node_done"
if ((NNODES > 1 && NODE_RANK != 0)); then
  printf 'Editing node %s/%s completed: %s\n' "$NODE_RANK" "$NNODES" "$RESULT_ROOT"
  exit 0
fi
if ((NNODES > 1)); then
  deadline=$((SECONDS + ${FINAL_WAIT_SEC:-43200}))
  for rank in $(seq 0 $((NNODES - 1))); do
    peer_done="$RESULT_ROOT/launcher_status/node_${rank}.done"
    while [[ ! -s "$peer_done" ]] \
      || ! grep -Fqx "launch_token=$LAUNCH_TOKEN" "$peer_done"; do
      ((SECONDS < deadline)) || { printf 'Timed out waiting for node %s.\n' "$rank" >&2; exit 1; }
      sleep 30
    done
  done
fi

combined="$RESULT_ROOT/planned_runs.tsv"
if ((global_shard_count > 1)); then
  combined_tmp="$combined.tmp.$$"
  first=1
  for plan in "$RESULT_ROOT"/planned_runs_shard_*_of_*.tsv; do
    [[ -f "$plan" ]] || {
      printf 'Missing editing shard plans in %s.\n' "$RESULT_ROOT" >&2
      exit 1
    }
    if ((first)); then
      sed -n '1p' "$plan" > "$combined_tmp"
      first=0
    fi
    sed -n '2,$p' "$plan" >> "$combined_tmp"
  done
  mv "$combined_tmp" "$combined"
elif [[ ! -s "$combined" ]]; then
  printf 'Missing editing plan: %s\n' "$combined" >&2
  exit 1
fi
if ((dry_run == 0)); then
  "$PYTHON" "$SCRIPT_DIR/summarize_direct_editing.py" \
    --plan "$combined" \
    --output-dir "$RESULT_ROOT/paired_summary"
fi
printf 'All editing shards completed: %s\n' "$RESULT_ROOT"
