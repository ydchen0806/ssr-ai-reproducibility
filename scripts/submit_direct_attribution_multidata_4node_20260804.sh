#!/usr/bin/env bash
# Run the complete meeting-extension matrix in one four-node allocation.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-/usr/local/bin/python}"
RUN_ID="${SSR_DIRECT_RUN_ID:-}"
LAUNCH_TOKEN="${SSR_DIRECT_LAUNCH_TOKEN:-$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
DRY_RUN="${SSR_DIRECT_DRY_RUN:-0}"
RESULT_ROOT="${DIRECT_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
DATA_ROOT="${DIRECT_DATA_ROOT:-$PROJECT_ROOT/data}"
VIT_CHECKPOINT="${VIT_LORA_PRETRAINED_CHECKPOINT:-$DATA_ROOT/pretrained/vit_tiny_patch16_224_augreg_in21k_ft_in1k.safetensors}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-600}"
FINAL_WAIT_SEC="${FINAL_WAIT_SEC:-86400}"
RECOVERY_SOURCE_ROOT="${SSR_DIRECT_RECOVERY_SOURCE_ROOT:-}"
RECOVERY_SOURCE_COMMIT="${SSR_DIRECT_RECOVERY_SOURCE_COMMIT:-}"

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_DIRECT_RUN_ID to a fresh identifier.\n' >&2
  exit 2
}
[[ -n "$LAUNCH_TOKEN" && "$LAUNCH_TOKEN" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
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
for topology_key in PADDLE_TRAINERS_NUM NNODES SLURM_NNODES PET_NNODES; do
  topology_value="${!topology_key:-}"
  [[ -z "$topology_value" || "$topology_value" == "4" ]] || {
    printf '%s=%s conflicts with the four-node protocol.\n' "$topology_key" "$topology_value" >&2
    exit 2
  }
done

git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
if [[ -n "$RECOVERY_SOURCE_ROOT" ]]; then
  [[ "$RECOVERY_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    printf 'Recovery requires an explicit 40-character SSR_DIRECT_RECOVERY_SOURCE_COMMIT.\n' >&2
    exit 2
  }
  RECOVERY_SOURCE_ROOT="$(cd "$RECOVERY_SOURCE_ROOT" && pwd)"
  [[ "$RECOVERY_SOURCE_ROOT" != "$RESULT_ROOT" ]] || {
    printf 'Recovery source and destination result roots must differ.\n' >&2
    exit 2
  }
  source_identity="$RECOVERY_SOURCE_ROOT/.direct_attribution_4node_identity"
  [[ -s "$source_identity" ]] || {
    printf 'Recovery source lacks an identity file: %s\n' "$source_identity" >&2
    exit 2
  }
  grep -Fqx 'protocol=meeting_extension_multidata_4node_v2' "$source_identity" || {
    printf 'Recovery source uses an incompatible protocol.\n' >&2
    exit 2
  }
  observed_source_commit="$(awk -F= '$1=="git_commit" {print $2}' "$source_identity")"
  [[ "$observed_source_commit" == "$RECOVERY_SOURCE_COMMIT" ]] || {
    printf 'Recovery source commit mismatch: expected %s, observed %s.\n' \
      "$RECOVERY_SOURCE_COMMIT" "$observed_source_commit" >&2
    exit 2
  }
  source_recovery_root="$(awk -F= '$1=="recovery_source_root" {print substr($0, index($0, "=") + 1)}' "$source_identity")"
  [[ -z "$source_recovery_root" ]] || {
    printf 'Chained recovery sources are not supported; use the original locked run.\n' >&2
    exit 2
  }
  git -C "$PROJECT_ROOT" merge-base --is-ancestor "$RECOVERY_SOURCE_COMMIT" "$git_commit" || {
    printf 'Recovery source commit %s is not an ancestor of %s.\n' \
      "$RECOVERY_SOURCE_COMMIT" "$git_commit" >&2
    exit 2
  }
elif [[ -n "$RECOVERY_SOURCE_COMMIT" ]]; then
  printf 'SSR_DIRECT_RECOVERY_SOURCE_COMMIT requires SSR_DIRECT_RECOVERY_SOURCE_ROOT.\n' >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/check_experiment_worktree.py" \
    --project-root "$PROJECT_ROOT"
  physical_gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count+0}')"
  [[ "$physical_gpu_count" == "$GPU_COUNT" ]] || {
    printf 'Expected 8 GPUs, found %s on %s.\n' "$physical_gpu_count" "$(hostname)" >&2
    exit 2
  }
fi

IDENTITY_FILE="$RESULT_ROOT/.direct_attribution_4node_identity"
STATUS_DIR="$RESULT_ROOT/launcher_status"
LOG_DIR="$RESULT_ROOT/logs"
if ((GLOBAL_NODE_RANK == 0)); then
  [[ ! -e "$RESULT_ROOT" ]] || {
    printf 'Fresh result root required: %s already exists.\n' "$RESULT_ROOT" >&2
    exit 2
  }
  mkdir -p "$STATUS_DIR" "$LOG_DIR"
  {
    printf 'protocol=meeting_extension_multidata_4node_v2\n'
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'launch_token=%s\n' "$LAUNCH_TOKEN"
    printf 'git_commit=%s\n' "$git_commit"
    printf 'recovery_source_root=%s\n' "$RECOVERY_SOURCE_ROOT"
    printf 'recovery_source_commit=%s\n' "$RECOVERY_SOURCE_COMMIT"
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
for expected in \
  'protocol=meeting_extension_multidata_4node_v2' \
  "run_id=$RUN_ID" \
  "launch_token=$LAUNCH_TOKEN" \
  "git_commit=$git_commit" \
  "recovery_source_root=$RECOVERY_SOURCE_ROOT" \
  "recovery_source_commit=$RECOVERY_SOURCE_COMMIT"; do
  grep -Fqx "$expected" "$IDENTITY_FILE" || exit 2
done

atomic_status() {
  local path="$1"; shift
  local temporary="$path.tmp.$(hostname).$$"
  {
    printf 'launch_token=%s\n' "$LAUNCH_TOKEN"
    printf 'git_commit=%s\n' "$git_commit"
    printf '%s\n' "$@"
  } > "$temporary"
  mv "$temporary" "$path"
}
status_matches() {
  [[ -s "$1" ]] && grep -Fqx "launch_token=$LAUNCH_TOKEN" "$1"
}

terminal_status=0
claimed=0
record_failure() {
  local code=$?
  if ((code != 0 && claimed == 1 && terminal_status == 0)); then
    atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" \
      "exit_code=$code" "finished_at=$(timestamp)"
  fi
  trap - EXIT
  exit "$code"
}
trap record_failure EXIT

atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.claimed" \
  "rank=$GLOBAL_NODE_RANK" "host=$(hostname)" "started_at=$(timestamp)"
claimed=1
deadline=$((SECONDS + INIT_WAIT_SEC))
for rank in 0 1 2 3; do
  while ! status_matches "$STATUS_DIR/node_${rank}.claimed"; do
    ((SECONDS < deadline)) || exit 2
    sleep 2
  done
done

# Materialize and validate shared inputs once before any consumer starts. This
# prevents node 2 from reading Pet while node 3 is still creating it.
ASSET_READY="$STATUS_DIR/assets.ready"
if ((GLOBAL_NODE_RANK == 0)); then
  if [[ "$DRY_RUN" != "1" ]]; then
    asset_log="$LOG_DIR/prepare_shared_assets.log"
    {
      wikibio_test="$PROJECT_ROOT/dataset/knowedit/benchmark/WikiBio/wikibio-test-all.json"
      if [[ ! -s "$wikibio_test" ]]; then
        PROJECT_ROOT="$PROJECT_ROOT" bash "$PROJECT_ROOT/scripts/fetch_knowedit_wikibio.sh"
      fi
      pet_manifest="$DATA_ROOT/oxford_iiit_pet/DATASET_MANIFEST.json"
      flowers_manifest="$DATA_ROOT/oxford_flowers102/DATASET_MANIFEST.json"
      if [[ ! -s "$pet_manifest" || ! -s "$flowers_manifest" ]]; then
        "$PYTHON" "$PROJECT_ROOT/scripts/prepare_multidataset_segmentation.py" \
          --data-root "$DATA_ROOT" \
          --datasets oxford_iiit_pet oxford_flowers102
      fi
      "$PYTHON" "$PROJECT_ROOT/scripts/prepare_vit_lora_assets.py" \
        --configs \
          "$PROJECT_ROOT/configs/meeting_20260804/vit_lora_flowers102.yaml" \
          "$PROJECT_ROOT/configs/meeting_20260804/vit_lora_oxfordiiitpet.yaml" \
        --data-root "$DATA_ROOT" \
        --pretrained-checkpoint "$VIT_CHECKPOINT"
    } > "$asset_log" 2>&1
  fi
  atomic_status "$ASSET_READY" "status=ready" "finished_at=$(timestamp)"
else
  deadline=$((SECONDS + INIT_WAIT_SEC))
  while ! status_matches "$ASSET_READY"; do
    status_matches "$STATUS_DIR/node_0.failed" && exit 1
    ((SECONDS < deadline)) || exit 2
    sleep 2
  done
fi

GPU_LIST="$(seq -s ' ' 0 7)"
run_editing_suite() {
  local config="$1" child_root="$2" child_token="$3" datasets="$4"
  local recipes="$5" mappings="$6"
  local reuse_roots=""
  if [[ -n "$RECOVERY_SOURCE_ROOT" ]]; then
    reuse_roots="$RECOVERY_SOURCE_ROOT/$(basename "$child_root")/confirm"
  fi
  local -a command=(
    env
    "NODE_RANK=$GLOBAL_NODE_RANK"
    "NNODES=2"
    "LAUNCH_TOKEN=$child_token"
    "RUN_ID=${RUN_ID}_${child_token##*.}"
    "RESULT_ROOT=$child_root"
    "REUSE_RESULTS_ROOTS=$reuse_roots"
    "ALLOWED_EXISTING_GIT_COMMITS=$RECOVERY_SOURCE_COMMIT"
    "ALLOW_LEGACY_EXISTING_GIT=0"
    "GPU_LIST=$GPU_LIST"
    "CONFIG_FILE=$config"
    "PYTHON=$PYTHON"
    "FINAL_WAIT_SEC=$FINAL_WAIT_SEC"
    bash "$PROJECT_ROOT/scripts/run_meeting_editing_8gpu.sh"
    --phase confirm
    --dataset "$datasets"
    --recipe "$recipes"
    --mapping "$mappings"
  )
  [[ "$DRY_RUN" != "1" ]] || command+=(--dry-run)
  "${command[@]}"
}

case "$GLOBAL_NODE_RANK" in
  0|1)
    role=knowledge_editing_factorial
    run_role() {
      run_editing_suite \
        "$PROJECT_ROOT/configs/meeting_20260803/locked_editing_gpt2xl.yaml" \
        "$RESULT_ROOT/ke_factorial" "${LAUNCH_TOKEN}.ke_factorial" \
        "zsre,cf,recent" \
        "plain,anchor,spectral,stabilized,ssr_only,full" \
        "projective,cosine"
      run_editing_suite \
        "$PROJECT_ROOT/configs/meeting_20260804/direct_editing_wikibio_gpt2xl.yaml" \
        "$RESULT_ROOT/ke_wikibio" "${LAUNCH_TOKEN}.ke_wikibio" \
        "wikibio" "plain,ssr_only" "projective"
    }
    ;;
  2)
    role=vit_lora_and_matched_kd
    run_role() {
      env \
        "OUTPUT_ROOT=$RESULT_ROOT/vit_lora" \
        "REUSE_RESULTS_ROOT=${RECOVERY_SOURCE_ROOT:+$RECOVERY_SOURCE_ROOT/vit_lora}" \
        "ALLOWED_EXISTING_GIT_COMMITS=$RECOVERY_SOURCE_COMMIT" \
        "VIT_LORA_DATA_ROOT=$DATA_ROOT" \
        "VIT_LORA_PRETRAINED_CHECKPOINT=$VIT_CHECKPOINT" \
        "DATASETS=flowers102 oxfordiiitpet" \
        "GPU_LIST=$GPU_LIST" \
        "PYTHON=$PYTHON" \
        "DRY_RUN=$DRY_RUN" \
        bash "$PROJECT_ROOT/scripts/run_vit_lora_confirmation_8gpu.sh" || return $?
      env \
        "OUTPUT_ROOT=$RESULT_ROOT/matched_kd_cl" \
        "DATASETS=split_cifar100 split_tiny_imagenet" \
        "METHODS=kd kd_ewc kd_mas kd_si kd_center kd_protodecor kd_spectral kd_ssr" \
        "SEEDS=3101 3103 3105 3107 3109" \
        "GPU_LIST=$GPU_LIST" \
        "PYTHON=$PYTHON" \
        "DRY_RUN=$DRY_RUN" \
        bash "$PROJECT_ROOT/scripts/run_meeting_matched_kd_cl.sh"
    }
    ;;
  3)
    role=segmentation_direct
    run_role() {
      local -a cub=(
        "$PYTHON" "$PROJECT_ROOT/scripts/run_meeting_segmentation_ablation.py"
        --result-root "$RESULT_ROOT/cub_segmentation"
        --phase all --gpus 0 1 2 3 4 5 6 7 --python "$PYTHON"
      )
      [[ "$DRY_RUN" != "1" ]] || cub+=(--manifest-only)
      "${cub[@]}"
      local -a extra=(
        "$PYTHON" "$PROJECT_ROOT/scripts/run_multidataset_segmentation.py"
        --result-root "$RESULT_ROOT/multidataset_segmentation"
        --data-root "$DATA_ROOT"
        --phase all --gpus 0 1 2 3 4 5 6 7 --python "$PYTHON"
      )
      if [[ "$DRY_RUN" == "1" ]]; then
        extra+=(--manifest-only)
      else
        extra+=(--prepare-data)
      fi
      "${extra[@]}"
    }
    ;;
esac

log_file="$LOG_DIR/node_${GLOBAL_NODE_RANK}_${role}.log"
set +e
run_role > "$log_file" 2>&1
exit_code=$?
set -e
if ((exit_code == 0)); then
  atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.done" \
    "role=$role" "exit_code=0" "finished_at=$(timestamp)" "log=$log_file"
else
  atomic_status "$STATUS_DIR/node_${GLOBAL_NODE_RANK}.failed" \
    "role=$role" "exit_code=$exit_code" "finished_at=$(timestamp)" "log=$log_file"
fi
terminal_status=1

deadline=$((SECONDS + FINAL_WAIT_SEC))
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
  ((SECONDS < deadline)) || exit 1
  sleep 10
done

FINAL_DONE="$STATUS_DIR/final.done"
FINAL_FAILED="$STATUS_DIR/final.failed"
if ((GLOBAL_NODE_RANK == 0)); then
  mode=results
  [[ "$DRY_RUN" != "1" ]] || mode=plan
  set +e
  "$PYTHON" "$PROJECT_ROOT/scripts/validate_direct_attribution_matrix.py" \
    --root "$RESULT_ROOT" --mode "$mode" > "$LOG_DIR/finalize.log" 2>&1
  final_code=$?
  set -e
  if ((final_code == 0)); then
    atomic_status "$FINAL_DONE" "exit_code=0" "finished_at=$(timestamp)"
  else
    atomic_status "$FINAL_FAILED" "exit_code=$final_code" "finished_at=$(timestamp)"
    exit "$final_code"
  fi
else
  deadline=$((SECONDS + FINAL_WAIT_SEC))
  while ! status_matches "$FINAL_DONE" && ! status_matches "$FINAL_FAILED"; do
    ((SECONDS < deadline)) || exit 1
    sleep 5
  done
  status_matches "$FINAL_FAILED" && exit 1
fi

trap - EXIT
printf 'Meeting-extension role %s complete: %s\n' "$role" "$RESULT_ROOT"
