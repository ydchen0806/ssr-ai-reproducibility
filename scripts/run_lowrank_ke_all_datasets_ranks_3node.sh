#!/usr/bin/env bash
# Three-node frozen transfer of a ZsRE-selected LoRA+SSR rank sweep.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ASSET_ROOT="${ASSET_ROOT:-/unify/ydchen/unidit/ssr_teacher_assets_20260818}"
RESULTS_BASE="${RESULTS_BASE:-/unify/ydchen/unidit/bioreg_meeting_20260803_final/results}"
PYTHON="${PYTHON:-$ASSET_ROOT/envs/editing-py311-system-torch28/bin/python}"
RUN_ID="${SSR_V50_RUN_ID:-}"
LAUNCH_TOKEN="${SSR_V50_LAUNCH_TOKEN:-$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-3}"
GPU_COUNT_REQUEST="${GPU_COUNT:-auto}"
WAIT_SEC="${SSR_V50_STAGE_WAIT_SEC:-345600}"
DRY_RUN="${SSR_V50_DRY_RUN:-0}"

MODEL_PATH="$ASSET_ROOT/models/Qwen2.5-7B-Instruct"
MODEL_LABEL="Qwen2.5-7B-Instruct"
EASYEDIT_ROOT="$ASSET_ROOT/sources/EasyEdit_ssr_v28_rank_canary_20260826"
HF_CACHE_ROOT="$ASSET_ROOT/hf_cache"
RUNNER="$PROJECT_ROOT/scripts/run_lowrank_ke_easyedit.py"
MANIFEST_BUILDER="$PROJECT_ROOT/scripts/make_lowrank_ke_stream_manifests.py"
SUMMARIZER="$PROJECT_ROOT/scripts/summarize_lowrank_ke_dataset_rank.py"
FROZEN_CONFIG="$PROJECT_ROOT/configs/lowrank_ke/qwen25_all_dataset_rank_transfer_100edit.json"
STATUS_ROOT="$RESULTS_BASE/${RUN_ID}_launcher_status"
RESULT_ROOT="$RESULTS_BASE/${RUN_ID}_lowrank_lora_dataset_ranks"
STREAM_ROOT="$STATUS_ROOT/streams"

DATASET_LABELS=(WikiRecent WikiCounterFact)
DATASET_PATHS=(
  /unify/ydchen/unidit/bioreg_meeting_20260803_final/dataset/knowedit/benchmark/wiki_recent/recent_test.json
  /unify/ydchen/unidit/bioreg_meeting_20260803_final/dataset/knowedit/benchmark/wiki_counterfact/test_cf.json
)
DATASET_SLUGS=(wikirecent wikicounterfact)
WIKIRECENT_SEEDS=(17631 17632 17633 17634 17635 17636 17637 17638 17639 17640)
WIKICOUNTERFACT_SEEDS=(17731 17732 17733 17734 17735 17736 17737 17738 17739 17740)
ADAPTER_RANKS=(8 16 32 64)
CHECKPOINTS=(1 2 5 10 25 50 100)
LORA_LEARNING_RATE=0.005
LORA_STEPS=60

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
atomic_write() {
  local path="$1"; shift
  local temporary="${path}.tmp.$(hostname).$$"
  mkdir -p "$(dirname "$path")"
  printf '%s\n' "$@" > "$temporary"
  mv "$temporary" "$path"
}
die() {
  local message="$*"
  printf 'SSR-v50 FAILED: %s\n' "$message" >&2
  if [[ -n "${RANK:-}" && -d "${STATUS_ROOT:-}" ]]; then
    atomic_write "$STATUS_ROOT/node_${RANK}.failed" \
      'status=failed' "reason=$message" "failed_at=$(timestamp)"
  fi
  exit 2
}
coefficient_for_rank() {
  case "$1" in
    8) printf '0.03\n' ;;
    16) printf '0.01\n' ;;
    32) printf '0.03\n' ;;
    64) printf '0.003\n' ;;
    *) die "Unsupported adapter rank: $1" ;;
  esac
}
seeds_for_dataset() {
  case "$1" in
    0) printf '%s\n' "${WIKIRECENT_SEEDS[*]}" ;;
    1) printf '%s\n' "${WIKICOUNTERFACT_SEEDS[*]}" ;;
    *) die "Unsupported dataset index: $1" ;;
  esac
}
resolve_rank() {
  local key value rank=""
  for key in GLOBAL_NODE_RANK NODE_RANK PADDLE_TRAINER_ID SLURM_NODEID GROUP_RANK; do
    value="${!key:-}"
    [[ -n "$value" ]] || continue
    [[ "$value" =~ ^[0-9]+$ ]] || die "Invalid $key=$value"
    if [[ -n "$rank" && "$rank" != "$value" ]]; then die 'Conflicting node ranks'; fi
    rank="$value"
  done
  if [[ -z "$rank" ]]; then
    local host="${HOSTNAME:-$(hostname)}"
    if [[ "$host" =~ master-([0-9]+) ]]; then rank="${BASH_REMATCH[1]}"
    elif [[ "$host" =~ worker-([0-9]+) ]]; then rank="$((BASH_REMATCH[1] + 1))"
    else die 'Unable to resolve node rank'; fi
  fi
  printf '%s\n' "$rank"
}
visible_gpu_count() {
  local observed
  observed="$(nvidia-smi -L 2>/dev/null | awk '/^GPU / {count++} END {print count+0}')"
  [[ "$observed" == '8' ]] || die "This protocol requires eight visible GPUs per node; found $observed"
  if [[ "$GPU_COUNT_REQUEST" != 'auto' && "$GPU_COUNT_REQUEST" != "$observed" ]]; then
    die "GPU_COUNT=$GPU_COUNT_REQUEST but rank $RANK exposes $observed GPUs"
  fi
  printf '%s\n' "$observed"
}
wait_for_file() {
  local path="$1" deadline=$((SECONDS + WAIT_SEC)) failed peer
  while [[ ! -s "$path" ]]; do
    for ((peer=0; peer<EXPECTED_NNODES; peer++)); do
      failed="$STATUS_ROOT/node_${peer}.failed"
      [[ ! -s "$failed" ]] || die "Peer failed: $failed"
    done
    ((SECONDS < deadline)) || die "Timed out waiting for $path"
    sleep 5
  done
}
preflight() {
  [[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die 'Set SSR_V50_RUN_ID'
  [[ -n "$LAUNCH_TOKEN" && "$LAUNCH_TOKEN" =~ ^[A-Za-z0-9._-]+$ ]] || die 'Invalid SSR_V50_LAUNCH_TOKEN'
  [[ "$EXPECTED_NNODES" == '3' ]] || die 'This campaign requires exactly three nodes'
  [[ "$GPU_COUNT_REQUEST" == 'auto' || "$GPU_COUNT_REQUEST" == '8' ]] || die 'GPU_COUNT must be auto or 8'
  [[ "$WAIT_SEC" =~ ^[1-9][0-9]*$ ]] || die 'SSR_V50_STAGE_WAIT_SEC must be positive'
  for path in "$RUNNER" "$MANIFEST_BUILDER" "$SUMMARIZER" "$FROZEN_CONFIG" \
    "$MODEL_PATH/config.json" "${DATASET_PATHS[@]}" \
    "$EASYEDIT_ROOT/hparams/LoRA/qwen2.5-7b.yaml"; do
    [[ -s "$path" ]] || die "Missing dependency: $path"
  done
  "$PYTHON" -m py_compile "$RUNNER" "$MANIFEST_BUILDER" "$SUMMARIZER"
}
run_one() {
  local gpu="$1" dataset_index="$2" adapter_rank="$3" seed="$4" coefficient="$5"
  local dataset_path="${DATASET_PATHS[$dataset_index]}"
  local dataset_slug="${DATASET_SLUGS[$dataset_index]}"
  local tag="lambda_${coefficient//./p}"
  local output="$RESULT_ROOT/$dataset_slug/rank_$adapter_rank/$tag/seed_$seed/results.json"
  local cache="$RESULT_ROOT/cache/$dataset_slug/rank_$adapter_rank/$tag/seed_$seed"
  local log="$STATUS_ROOT/logs/${dataset_slug}-rank-${adapter_rank}-${tag}-seed-${seed}.log"
  local manifest="$STREAM_ROOT/$dataset_slug/confirmation_seed_$seed.json"
  (
    export CUDA_VISIBLE_DEVICES="$gpu" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export HF_HOME="$HF_CACHE_ROOT" HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
    export SSR_REPO_DIR="$PROJECT_ROOT" EASYEDIT_DIR="$EASYEDIT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PYTHON" "$RUNNER" --method LoRA --model-path "$MODEL_PATH" --model-label "$MODEL_LABEL" \
      --hparams-root "$EASYEDIT_ROOT/hparams" --hparams-stem qwen2.5-7b --dataset "$dataset_path" \
      --stream-manifest "$manifest" --n-edits 100 --lora-rank "$adapter_rank" \
      --lora-lr "$LORA_LEARNING_RATE" --lora-steps "$LORA_STEPS" \
      --seed "$seed" --ssr-lambda "$coefficient" --history-checkpoints "${CHECKPOINTS[@]}" \
      --history-max-samples 128 --pre-edit-max-samples 32 \
      --method-cache-dir "$cache" --output "$output"
  ) > "$log" 2>&1
}

preflight
RANK="$(resolve_rank)"
((RANK >= 0 && RANK < EXPECTED_NNODES)) || die "Rank outside 0..2: $RANK"
trap 'code=$?; line=$LINENO; trap - ERR; if [[ -d "$STATUS_ROOT" ]]; then atomic_write "$STATUS_ROOT/node_${RANK}.failed" "status=failed" "exit_code=$code" "line=$line" "failed_at=$(timestamp)"; fi; exit "$code"' ERR

if [[ "$DRY_RUN" == '1' ]]; then
  temporary="$(mktemp -d /tmp/ssr_v50_all_dataset_ranks.XXXXXX)"
  trap 'find "$temporary" -type f -delete; find "$temporary" -depth -type d -empty -delete' EXIT
  for dataset_index in 0 1; do
    read -r -a seeds <<< "$(seeds_for_dataset "$dataset_index")"
    "$PYTHON" "$MANIFEST_BUILDER" --dataset "${DATASET_PATHS[$dataset_index]}" \
      --dataset-label "${DATASET_LABELS[$dataset_index]}" --output-dir "$temporary/${DATASET_SLUGS[$dataset_index]}" \
      --development-edits 100 --development-seeds 17601 \
      --confirmation-seeds "${seeds[@]}" --confirmation-edits 100
    for adapter_rank in "${ADAPTER_RANKS[@]}"; do
      coefficient="$(coefficient_for_rank "$adapter_rank")"
      for arm in 0 "$coefficient"; do
        SSR_REPO_DIR="$PROJECT_ROOT" EASYEDIT_DIR="$EASYEDIT_ROOT" \
          PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
          "$PYTHON" "$RUNNER" --method LoRA --model-path "$MODEL_PATH" --model-label "$MODEL_LABEL" \
            --hparams-root "$EASYEDIT_ROOT/hparams" --hparams-stem qwen2.5-7b \
            --dataset "${DATASET_PATHS[$dataset_index]}" \
            --stream-manifest "$temporary/${DATASET_SLUGS[$dataset_index]}/confirmation_seed_${seeds[0]}.json" \
            --n-edits 100 --lora-rank "$adapter_rank" --lora-lr "$LORA_LEARNING_RATE" \
            --lora-steps "$LORA_STEPS" --seed "${seeds[0]}" --ssr-lambda "$arm" \
            --history-checkpoints "${CHECKPOINTS[@]}" --history-max-samples 128 \
            --pre-edit-max-samples 32 --method-cache-dir "$temporary/cache/$dataset_index/$adapter_rank/$arm" \
            --output "$temporary/validated-${dataset_index}-${adapter_rank}-${arm//./p}.json" --validate-only
      done
    done
  done
  printf 'SSR-v50 all-dataset rank dry run OK: node-rank=%s\n' "$RANK"
  exit 0
fi

LOCAL_GPU_COUNT="$(visible_gpu_count)"
if [[ "$RANK" == '0' ]]; then
  [[ ! -e "$STATUS_ROOT" && ! -e "$RESULT_ROOT" ]] || die 'Fresh run roots are required'
  mkdir -p "$STATUS_ROOT/logs"
  CONFIG_SHA256="$(sha256sum "$FROZEN_CONFIG" | awk '{print $1}')"
  atomic_write "$STATUS_ROOT/identity" \
    'protocol=lowrank_lora_ssr_all_dataset_rank_transfer_v1' "run_id=$RUN_ID" \
    "launch_token=$LAUNCH_TOKEN" "code_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)" \
    "frozen_config=$FROZEN_CONFIG" "frozen_config_sha256=$CONFIG_SHA256" \
    'model=Qwen2.5-7B-Instruct' 'datasets=WikiRecent_WikiCounterFact' \
    'method=EasyEdit_LoRA' 'ranks=8_16_32_64' 'learning_rate=0.005' 'lora_steps=60' \
    'frozen_lambdas=rank8:0.03_rank16:0.01_rank32:0.03_rank64:0.003' \
    'n_edits=100' 'checkpoints=1_2_5_10_25_50_100' \
    'primary_endpoints=immediate_efficacy_and_pre_edit_output_consistency' \
    'confirmation=all_ten_new_predeclared_orders_per_dataset_and_rank_no_filtering' \
    "created_at=$(timestamp)"
  for dataset_index in 0 1; do
    read -r -a seeds <<< "$(seeds_for_dataset "$dataset_index")"
    "$PYTHON" "$MANIFEST_BUILDER" --dataset "${DATASET_PATHS[$dataset_index]}" \
      --dataset-label "${DATASET_LABELS[$dataset_index]}" \
      --output-dir "$STREAM_ROOT/${DATASET_SLUGS[$dataset_index]}" \
      --development-edits 100 --development-seeds 17601 \
      --confirmation-seeds "${seeds[@]}" --confirmation-edits 100 \
      > "$STATUS_ROOT/logs/streams-${DATASET_SLUGS[$dataset_index]}.log" 2>&1
  done
else
  wait_for_file "$STATUS_ROOT/identity"
fi
wait_for_file "$STREAM_ROOT/wikirecent/INDEX.json"
wait_for_file "$STREAM_ROOT/wikicounterfact/INDEX.json"

TASKS=()
for dataset_index in 0 1; do
  read -r -a seeds <<< "$(seeds_for_dataset "$dataset_index")"
  for adapter_rank in "${ADAPTER_RANKS[@]}"; do
    coefficient="$(coefficient_for_rank "$adapter_rank")"
    for seed in "${seeds[@]}"; do
      TASKS+=("$dataset_index|$adapter_rank|$seed|0" "$dataset_index|$adapter_rank|$seed|$coefficient")
    done
  done
done
LOCAL_TASKS=()
for index in "${!TASKS[@]}"; do
  if ((index % EXPECTED_NNODES == RANK)); then LOCAL_TASKS+=("${TASKS[$index]}"); fi
done
pids=()
for ((gpu=0; gpu<LOCAL_GPU_COUNT; gpu++)); do
  (
    for ((local_index=gpu; local_index<${#LOCAL_TASKS[@]}; local_index+=LOCAL_GPU_COUNT)); do
      IFS='|' read -r dataset_index adapter_rank seed coefficient <<< "${LOCAL_TASKS[$local_index]}"
      run_one "$gpu" "$dataset_index" "$adapter_rank" "$seed" "$coefficient"
    done
  ) &
  pids+=("$!")
done
failures=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failures=$((failures + 1)); fi; done
((failures == 0)) || die "$failures local GPU queues failed"
atomic_write "$STATUS_ROOT/node_${RANK}.confirmation.complete" 'status=complete' "at=$(timestamp)"

if [[ "$RANK" == '0' ]]; then
  wait_for_file "$STATUS_ROOT/node_1.confirmation.complete"
  wait_for_file "$STATUS_ROOT/node_2.confirmation.complete"
  for dataset_index in 0 1; do
    read -r -a seeds <<< "$(seeds_for_dataset "$dataset_index")"
    for adapter_rank in "${ADAPTER_RANKS[@]}"; do
      coefficient="$(coefficient_for_rank "$adapter_rank")"
      "$PYTHON" "$SUMMARIZER" \
        --result-root "$RESULT_ROOT/${DATASET_SLUGS[$dataset_index]}/rank_$adapter_rank" \
        --seeds "${seeds[@]}" --rank "$adapter_rank" \
        --learning-rate "$LORA_LEARNING_RATE" --lora-steps "$LORA_STEPS" \
        --ssr-lambda "$coefficient" --frozen-from "$FROZEN_CONFIG" \
        --dataset-label "${DATASET_LABELS[$dataset_index]}" \
        --output "$RESULT_ROOT/${DATASET_SLUGS[$dataset_index]}_rank_${adapter_rank}_summary.json" \
        --csv-output "$RESULT_ROOT/${DATASET_SLUGS[$dataset_index]}_rank_${adapter_rank}_curves.csv" \
        > "$STATUS_ROOT/logs/summarize-${DATASET_SLUGS[$dataset_index]}-rank-$adapter_rank.log" 2>&1
    done
  done
  atomic_write "$STATUS_ROOT/node_0.complete" 'status=complete' "at=$(timestamp)"
  atomic_write "$STATUS_ROOT/campaign.complete" 'status=complete' \
    "summaries=$RESULT_ROOT/{wikirecent,wikicounterfact}_rank_{8,16,32,64}_summary.json" \
    "completed_at=$(timestamp)"
else
  atomic_write "$STATUS_ROOT/node_${RANK}.complete" 'status=complete' "at=$(timestamp)"
  wait_for_file "$STATUS_ROOT/campaign.complete"
fi
printf 'SSR-v50 all-dataset rank confirmation complete: %s\n' "$STATUS_ROOT"
