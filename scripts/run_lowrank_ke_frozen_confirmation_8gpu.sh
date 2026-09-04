#!/usr/bin/env bash
# One-node, eight-GPU confirmation of one frozen low-rank LoRA+SSR tuple.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EASYEDIT_ROOT="${EASYEDIT_DIR:-$PROJECT_ROOT/external/EasyEdit}"
PYTHON="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-}"
LAUNCH_TOKEN="${LAUNCH_TOKEN:-$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-1}"
GPU_COUNT_REQUEST="${GPU_COUNT:-auto}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Qwen2.5-7B-Instruct snapshot}"
MODEL_LABEL="${MODEL_LABEL:-Qwen2.5-7B-Instruct}"
ZSRE="${DATASET_PATH:?Set DATASET_PATH to ZsRE-test-all.json}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$PROJECT_ROOT/.cache/huggingface}"
RUNNER="$PROJECT_ROOT/scripts/run_lowrank_ke_easyedit.py"
MANIFEST_BUILDER="$PROJECT_ROOT/scripts/make_lowrank_ke_stream_manifests.py"
SUMMARIZER="$PROJECT_ROOT/scripts/summarize_lowrank_ke_confirmation.py"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
STATUS_ROOT="$RESULTS_BASE/${RUN_ID}_launcher_status"
RESULT_ROOT="$RESULTS_BASE/${RUN_ID}_lowrank_lora_ssr"
STREAM_ROOT="$STATUS_ROOT/streams"

ADAPTER_RANK="${ADAPTER_RANK:?Set the rank frozen during development}"
LORA_LEARNING_RATE="${LORA_LEARNING_RATE:?Set the learning rate frozen during development}"
LORA_STEPS="${LORA_STEPS:-60}"
SSR_LAMBDA="${SSR_LAMBDA:?Set the SSR coefficient frozen during development}"
FROZEN_FROM="${FROZEN_FROM:?Set the immutable development-selection record}"
read -r -a CONFIRMATION_SEEDS <<< "${CONFIRMATION_SEEDS_TEXT:-16331 16332 16333 16334 16335 16336 16337 16338 16339 16340}"
CHECKPOINTS=(1 2 5 10 25 50 100)

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
  printf 'Frozen low-rank confirmation FAILED: %s\n' "$message" >&2
  if [[ -d "${STATUS_ROOT:-}" ]]; then
    atomic_write "$STATUS_ROOT/node_0.failed" \
      'status=failed' "reason=$message" "failed_at=$(timestamp)"
  fi
  exit 2
}
visible_gpu_count() {
  local observed
  observed="$(nvidia-smi -L 2>/dev/null | awk '/^GPU / {count++} END {print count+0}')"
  [[ "$observed" == "8" ]] || die "This protocol requires eight visible GPUs; found $observed"
  if [[ "$GPU_COUNT_REQUEST" != "auto" && "$GPU_COUNT_REQUEST" != "$observed" ]]; then
    die "GPU_COUNT=$GPU_COUNT_REQUEST but the node exposes $observed GPUs"
  fi
  printf '%s\n' "$observed"
}
preflight() {
  [[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die 'Set a valid RUN_ID'
  [[ -n "$LAUNCH_TOKEN" && "$LAUNCH_TOKEN" =~ ^[A-Za-z0-9._-]+$ ]] || die 'Invalid LAUNCH_TOKEN'
  [[ "$EXPECTED_NNODES" == "1" ]] || die 'This confirmation requires exactly one node'
  [[ "$GPU_COUNT_REQUEST" == "auto" || "$GPU_COUNT_REQUEST" == "8" ]] || die 'GPU_COUNT must be auto or 8'
  [[ "$ADAPTER_RANK" =~ ^[1-9][0-9]*$ ]] || die 'ADAPTER_RANK must be positive'
  [[ "$LORA_STEPS" =~ ^[1-9][0-9]*$ ]] || die 'LORA_STEPS must be positive'
  [[ "${#CONFIRMATION_SEEDS[@]}" == "10" ]] || die 'Exactly ten confirmation seeds are required'
  [[ -s "$FROZEN_FROM" ]] || die "Missing development-selection record: $FROZEN_FROM"
  for path in "$RUNNER" "$MANIFEST_BUILDER" "$SUMMARIZER" \
    "$MODEL_PATH/config.json" "$ZSRE" "$EASYEDIT_ROOT/hparams/LoRA/${HPARAMS_STEM:-qwen2.5-7b}.yaml"; do
    [[ -s "$path" ]] || die "Missing dependency: $path"
  done
  "$PYTHON" -m py_compile "$RUNNER" "$MANIFEST_BUILDER" "$SUMMARIZER"
}
run_one() {
  local gpu="$1" seed="$2" coefficient="$3"
  local tag output cache log manifest
  tag="lambda_${coefficient//./p}"
  output="$RESULT_ROOT/confirmation/rank_$ADAPTER_RANK/lr_${LORA_LEARNING_RATE//./p}/$tag/seed_$seed/results.json"
  cache="$RESULT_ROOT/cache/confirmation/rank_$ADAPTER_RANK/lr_${LORA_LEARNING_RATE//./p}/$tag/seed_$seed"
  log="$STATUS_ROOT/logs/confirmation-rank-${ADAPTER_RANK}-lr-${LORA_LEARNING_RATE//./p}-${tag}-seed-${seed}.log"
  manifest="$STREAM_ROOT/confirmation_seed_$seed.json"
  (
    export CUDA_VISIBLE_DEVICES="$gpu" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export HF_HOME="$HF_CACHE_ROOT" HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
    export SSR_REPO_DIR="$PROJECT_ROOT" EASYEDIT_DIR="$EASYEDIT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PYTHON" "$RUNNER" --method LoRA --model-path "$MODEL_PATH" --model-label "$MODEL_LABEL" \
      --hparams-root "$EASYEDIT_ROOT/hparams" --hparams-stem "${HPARAMS_STEM:-qwen2.5-7b}" --dataset "$ZSRE" \
      --stream-manifest "$manifest" --n-edits 100 --lora-rank "$ADAPTER_RANK" --seed "$seed" \
      --lora-lr "$LORA_LEARNING_RATE" --lora-steps "$LORA_STEPS" \
      --ssr-lambda "$coefficient" --history-checkpoints "${CHECKPOINTS[@]}" \
      --history-max-samples 128 --pre-edit-max-samples 32 \
      --method-cache-dir "$cache" --output "$output"
  ) > "$log" 2>&1
}

preflight
if [[ "$DRY_RUN" == "1" ]]; then
  temporary="$(mktemp -d /tmp/lowrank_ssr_frozen_confirmation.XXXXXX)"
  trap 'find "$temporary" -type f -delete; find "$temporary" -depth -type d -empty -delete' EXIT
  "$PYTHON" "$MANIFEST_BUILDER" --dataset "$ZSRE" --output-dir "$temporary/streams" \
    --development-edits 100 --development-seeds 16301 \
    --confirmation-seeds "${CONFIRMATION_SEEDS[@]}" --confirmation-edits 100
  for coefficient in 0 "$SSR_LAMBDA"; do
    SSR_REPO_DIR="$PROJECT_ROOT" EASYEDIT_DIR="$EASYEDIT_ROOT" PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON" "$RUNNER" --method LoRA --model-path "$MODEL_PATH" --model-label "$MODEL_LABEL" \
        --hparams-root "$EASYEDIT_ROOT/hparams" --hparams-stem "${HPARAMS_STEM:-qwen2.5-7b}" --dataset "$ZSRE" \
        --stream-manifest "$temporary/streams/confirmation_seed_${CONFIRMATION_SEEDS[0]}.json" \
        --n-edits 100 --lora-rank "$ADAPTER_RANK" --lora-lr "$LORA_LEARNING_RATE" \
        --lora-steps "$LORA_STEPS" --seed "${CONFIRMATION_SEEDS[0]}" --ssr-lambda "$coefficient" \
        --history-checkpoints "${CHECKPOINTS[@]}" --history-max-samples 128 --pre-edit-max-samples 32 \
        --method-cache-dir "$temporary/cache/$coefficient" --output "$temporary/validated_$coefficient.json" --validate-only
  done
  printf 'Frozen low-rank confirmation dry run OK\n'
  exit 0
fi

LOCAL_GPU_COUNT="$(visible_gpu_count)"
[[ ! -e "$STATUS_ROOT" && ! -e "$RESULT_ROOT" ]] || die 'Fresh run roots are required'
mkdir -p "$STATUS_ROOT/logs"
CODE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
atomic_write "$STATUS_ROOT/identity" \
  'protocol=lowrank_lora_ssr_v1_frozen_confirmation' "run_id=$RUN_ID" "launch_token=$LAUNCH_TOKEN" \
  "code_commit=$CODE_COMMIT" "model=$MODEL_LABEL" 'dataset=ZsRE' 'method=EasyEdit_LoRA' \
  "rank=$ADAPTER_RANK" "learning_rate=$LORA_LEARNING_RATE" "lora_steps=$LORA_STEPS" \
  "ssr_lambda=$SSR_LAMBDA" "frozen_from=$FROZEN_FROM" \
  'n_edits=100' 'checkpoints=1_2_5_10_25_50_100' \
  'primary_endpoints=immediate_efficacy_and_pre_edit_output_consistency_locality' \
  'confirmation=all_ten_new_predeclared_orders_no_filtering_two_primary_CIs_above_zero' \
  "created_at=$(timestamp)"
"$PYTHON" "$MANIFEST_BUILDER" --dataset "$ZSRE" --output-dir "$STREAM_ROOT" \
  --development-edits 100 --development-seeds 16301 \
  --confirmation-seeds "${CONFIRMATION_SEEDS[@]}" --confirmation-edits 100 \
  > "$STATUS_ROOT/logs/streams.log" 2>&1

tasks=()
for seed in "${CONFIRMATION_SEEDS[@]}"; do
  tasks+=("$seed|0" "$seed|$SSR_LAMBDA")
done
pids=()
for ((gpu=0; gpu<LOCAL_GPU_COUNT; gpu++)); do
  (
    for ((index=gpu; index<${#tasks[@]}; index+=LOCAL_GPU_COUNT)); do
      IFS='|' read -r seed coefficient <<< "${tasks[$index]}"
      run_one "$gpu" "$seed" "$coefficient"
    done
  ) &
  pids+=("$!")
done
failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failures=$((failures + 1)); fi
done
((failures == 0)) || die "$failures local GPU queues failed"

LR_TAG="lr_${LORA_LEARNING_RATE//./p}"
"$PYTHON" "$SUMMARIZER" \
  --result-root "$RESULT_ROOT/confirmation/rank_$ADAPTER_RANK/$LR_TAG" \
  --seeds "${CONFIRMATION_SEEDS[@]}" --rank "$ADAPTER_RANK" \
  --learning-rate "$LORA_LEARNING_RATE" --lora-steps "$LORA_STEPS" \
  --ssr-lambda "$SSR_LAMBDA" --frozen-from "$FROZEN_FROM" \
  --output "$RESULT_ROOT/confirmation_summary.json" \
  --csv-output "$RESULT_ROOT/confirmation_curves.csv" > "$STATUS_ROOT/logs/summarize.log" 2>&1
atomic_write "$STATUS_ROOT/node_0.complete" 'status=complete' "at=$(timestamp)"
atomic_write "$STATUS_ROOT/campaign.complete" 'status=complete' \
  "summary=$RESULT_ROOT/confirmation_summary.json" "completed_at=$(timestamp)"
printf 'Frozen low-rank SSR confirmation complete: %s\n' "$STATUS_ROOT"
