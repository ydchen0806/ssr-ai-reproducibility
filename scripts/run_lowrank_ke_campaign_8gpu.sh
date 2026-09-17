#!/usr/bin/env bash
# One-node, eight-GPU development screen and frozen confirmation for low-rank SSR.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EASYEDIT_ROOT="${EASYEDIT_DIR:-$PROJECT_ROOT/external/EasyEdit}"
PYTHON="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-}"
LAUNCH_TOKEN="${LAUNCH_TOKEN:-$RUN_ID}"
EXPECTED_NNODES="${EXPECTED_NNODES:-1}"
GPU_COUNT_REQUEST="${GPU_COUNT:-auto}"
WAIT_SEC="${STAGE_WAIT_SEC:-345600}"
DRY_RUN="${DRY_RUN:-0}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0}"
ANCHOR_STEPS="${ANCHOR_STEPS:-0}"
ANCHOR_LR="${ANCHOR_LR:-0.0001}"
ANCHOR_BATCH_SIZE="${ANCHOR_BATCH_SIZE:-4}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Qwen2.5-7B-Instruct snapshot}"
MODEL_LABEL="${MODEL_LABEL:-Qwen2.5-7B-Instruct}"
ZSRE="${DATASET_PATH:?Set DATASET_PATH to ZsRE-test-all.json}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$PROJECT_ROOT/.cache/huggingface}"
RUNNER="$PROJECT_ROOT/scripts/run_lowrank_ke_easyedit.py"
MANIFEST_BUILDER="$PROJECT_ROOT/scripts/make_lowrank_ke_stream_manifests.py"
SELECTOR="$PROJECT_ROOT/scripts/select_lowrank_ke_candidate.py"
SUMMARIZER="$PROJECT_ROOT/scripts/summarize_lowrank_ke_confirmation.py"
STATUS_ROOT="$PROJECT_ROOT/results/${RUN_ID}_launcher_status"
RESULT_ROOT="$PROJECT_ROOT/results/${RUN_ID}_lowrank_lora_ssr"
STREAM_ROOT="$STATUS_ROOT/streams"
read -r -a DEVELOPMENT_SEEDS <<< "${DEVELOPMENT_SEEDS_TEXT:-16101 16102 16103 16104}"
read -r -a CONFIRMATION_SEEDS <<< "${CONFIRMATION_SEEDS_TEXT:-16131 16132 16133 16134 16135 16136 16137 16138 16139 16140}"
read -r -a CHECKPOINTS <<< "${CHECKPOINTS_TEXT:-1 2 5 10 25 50 100}"
read -r -a ADAPTER_RANKS <<< "${ADAPTER_RANKS_TEXT:-8 40}"
read -r -a LORA_LEARNING_RATES <<< "${LORA_LEARNING_RATES_TEXT:-0.0025 0.0035 0.005}"
LORA_STEPS=60
read -r -a SSR_LAMBDAS <<< "${SSR_LAMBDAS_TEXT:-0.024 0.036 0.048 0.064}"
SELECTION="$RESULT_ROOT/development_selection.json"

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
  printf 'Low-rank SSR FAILED: %s\n' "$message" >&2
  if [[ -n "${RANK:-}" && -d "${STATUS_ROOT:-}" ]]; then
    atomic_write "$STATUS_ROOT/node_${RANK}.failed" \
      "status=failed" "reason=$message" "failed_at=$(timestamp)"
  fi
  exit 2
}
resolve_rank() {
  local key value rank=""
  for key in GLOBAL_NODE_RANK NODE_RANK PADDLE_TRAINER_ID SLURM_NODEID GROUP_RANK; do
    value="${!key:-}"
    [[ -n "$value" ]] || continue
    [[ "$value" =~ ^[0-9]+$ ]] || die "Invalid $key=$value"
    if [[ -n "$rank" && "$rank" != "$value" ]]; then die "Conflicting node ranks"; fi
    rank="$value"
  done
  if [[ -z "$rank" ]]; then
    local host="${HOSTNAME:-$(hostname)}"
    if [[ "$host" =~ master-([0-9]+) ]]; then rank="${BASH_REMATCH[1]}"
    elif [[ "$host" =~ worker-([0-9]+) ]]; then rank="$((BASH_REMATCH[1] + 1))"
    elif [[ "$EXPECTED_NNODES" == "1" ]]; then rank=0
    else die "Unable to resolve node rank"; fi
  fi
  printf '%s\n' "$rank"
}
visible_gpu_count() {
  local observed
  observed="$(nvidia-smi -L 2>/dev/null | awk '/^GPU / {count++} END {print count+0}')"
  [[ "$observed" == "8" ]] || die "This protocol requires eight visible GPUs per node; found $observed"
  if [[ "$GPU_COUNT_REQUEST" != "auto" && "$GPU_COUNT_REQUEST" != "$observed" ]]; then
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
  [[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "Set RUN_ID"
  [[ -n "$LAUNCH_TOKEN" && "$LAUNCH_TOKEN" =~ ^[A-Za-z0-9._-]+$ ]] || die "Invalid LAUNCH_TOKEN"
  [[ "$EXPECTED_NNODES" == "1" ]] || die "This campaign requires exactly one node"
  [[ "$GPU_COUNT_REQUEST" == "auto" || "$GPU_COUNT_REQUEST" == "8" ]] || die "GPU_COUNT must be auto or 8"
  [[ "$WAIT_SEC" =~ ^[1-9][0-9]*$ ]] || die "STAGE_WAIT_SEC must be positive"
  for path in "$RUNNER" "$MANIFEST_BUILDER" "$SELECTOR" "$SUMMARIZER" \
    "$MODEL_PATH/config.json" "$ZSRE" "$EASYEDIT_ROOT/hparams/LoRA/${HPARAMS_STEM:-qwen2.5-7b}.yaml"; do
    [[ -s "$path" ]] || die "Missing dependency: $path"
  done
  "$PYTHON" -m py_compile "$RUNNER" "$MANIFEST_BUILDER" "$SELECTOR" "$SUMMARIZER"
}
run_one() {
  local stage="$1" gpu="$2" adapter_rank="$3" learning_rate="$4" seed="$5" coefficient="$6"
  local lr_tag tag output cache log manifest
  lr_tag="lr_${learning_rate//./p}"
  tag="lambda_${coefficient//./p}"
  output="$RESULT_ROOT/$stage/rank_$adapter_rank/$lr_tag/$tag/seed_$seed/results.json"
  cache="$RESULT_ROOT/cache/$stage/rank_$adapter_rank/$lr_tag/$tag/seed_$seed"
  log="$STATUS_ROOT/logs/${stage}-rank-${adapter_rank}-${lr_tag}-${tag}-seed-${seed}.log"
  if [[ "$stage" == "development" ]]; then
    manifest="$STREAM_ROOT/development_seed_$seed.json"
  else
    manifest="$STREAM_ROOT/confirmation_seed_$seed.json"
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export HF_HOME="$HF_CACHE_ROOT" HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
    export SSR_REPO_DIR="$PROJECT_ROOT" EASYEDIT_DIR="$EASYEDIT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PYTHON" "$RUNNER" --method LoRA --model-path "$MODEL_PATH" --model-label "$MODEL_LABEL" \
      --hparams-root "$EASYEDIT_ROOT/hparams" --hparams-stem "${HPARAMS_STEM:-qwen2.5-7b}" --dataset "$ZSRE" \
      --stream-manifest "$manifest" --n-edits 100 --lora-rank "$adapter_rank" --seed "$seed" \
      --lora-lr "$learning_rate" --lora-steps "$LORA_STEPS" \
      --ssr-lambda "$coefficient" --history-checkpoints "${CHECKPOINTS[@]}" \
      --anchor-weight "$ANCHOR_WEIGHT" --anchor-steps "$ANCHOR_STEPS" \
      --anchor-lr "$ANCHOR_LR" --anchor-batch-size "$ANCHOR_BATCH_SIZE" \
      --history-max-samples 128 --pre-edit-max-samples 32 \
      --method-cache-dir "$cache" --output "$output"
  ) > "$log" 2>&1
}
run_task_list() {
  local stage="$1"; shift
  local tasks=("$@") pids=() failures=0 gpu local_index adapter_rank learning_rate seed coefficient
  for ((gpu=0; gpu<LOCAL_GPU_COUNT; gpu++)); do
    (
      for ((local_index=gpu; local_index<${#tasks[@]}; local_index+=LOCAL_GPU_COUNT)); do
        IFS='|' read -r adapter_rank learning_rate seed coefficient <<< "${tasks[$local_index]}"
        run_one "$stage" "$gpu" "$adapter_rank" "$learning_rate" "$seed" "$coefficient"
      done
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failures=$((failures + 1)); fi; done
  ((failures == 0)) || die "$failures local GPU queues failed during $stage"
}

preflight
RANK="$(resolve_rank)"
[[ "$RANK" == "0" ]] || die "Single-node campaign requires rank 0; found $RANK"
mark_unexpected_failure() {
  local exit_code="$1" line="$2"
  trap - ERR
  if [[ -d "$STATUS_ROOT" ]]; then
    atomic_write "$STATUS_ROOT/node_${RANK}.failed" \
      "status=failed" "exit_code=$exit_code" "line=$line" "failed_at=$(timestamp)"
  fi
  exit "$exit_code"
}
trap 'mark_unexpected_failure $? $LINENO' ERR

if [[ "$DRY_RUN" == "1" ]]; then
  temporary="$(mktemp -d /tmp/lowrank_ssr_effloc_lr.XXXXXX)"
  trap 'find "$temporary" -type f -delete; find "$temporary" -depth -type d -empty -delete' EXIT
  "$PYTHON" "$MANIFEST_BUILDER" --dataset "$ZSRE" --output-dir "$temporary/streams" \
    --development-edits 100 --development-seeds "${DEVELOPMENT_SEEDS[@]}" \
    --confirmation-seeds "${CONFIRMATION_SEEDS[@]}"
  for adapter_rank in "${ADAPTER_RANKS[@]}"; do
    for learning_rate in "${LORA_LEARNING_RATES[@]}"; do
      lr_tag="lr_${learning_rate//./p}"
      SSR_REPO_DIR="$PROJECT_ROOT" EASYEDIT_DIR="$EASYEDIT_ROOT" \
        PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" "$RUNNER" --method LoRA --model-path "$MODEL_PATH" --model-label "$MODEL_LABEL" \
          --hparams-root "$EASYEDIT_ROOT/hparams" --hparams-stem "${HPARAMS_STEM:-qwen2.5-7b}" --dataset "$ZSRE" \
          --stream-manifest "$temporary/streams/development_seed_${DEVELOPMENT_SEEDS[0]}.json" \
          --n-edits 100 --lora-rank "$adapter_rank" --lora-lr "$learning_rate" \
          --lora-steps "$LORA_STEPS" --seed "${DEVELOPMENT_SEEDS[0]}" \
          --ssr-lambda 0.036 --history-checkpoints "${CHECKPOINTS[@]}" \
          --anchor-weight "$ANCHOR_WEIGHT" --anchor-steps "$ANCHOR_STEPS" \
          --anchor-lr "$ANCHOR_LR" --anchor-batch-size "$ANCHOR_BATCH_SIZE" \
          --history-max-samples 128 --pre-edit-max-samples 32 \
          --method-cache-dir "$temporary/cache/rank_$adapter_rank/$lr_tag" \
          --output "$temporary/validated_rank_${adapter_rank}_${lr_tag}.json" --validate-only
      "$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="validated" and r["protocol"]=="lowrank_lora_ssr_v1" and int(r["lora"]["rank"])==int(sys.argv[2]) and abs(float(r["lora"]["learning_rate"])-float(sys.argv[3]))<1e-12' \
        "$temporary/validated_rank_${adapter_rank}_${lr_tag}.json" "$adapter_rank" "$learning_rate"
    done
  done
  printf 'Low-rank SSR efficacy-locality learning-rate dry run OK: node-rank=%s\n' "$RANK"
  exit 0
fi

LOCAL_GPU_COUNT="$(visible_gpu_count)"
if [[ "$RANK" == "0" ]]; then
  [[ ! -e "$STATUS_ROOT" && ! -e "$RESULT_ROOT" ]] || die "Fresh run roots required"
  mkdir -p "$STATUS_ROOT/logs"
  atomic_write "$STATUS_ROOT/identity" \
    'protocol=lowrank_lora_ssr_v1' "run_id=$RUN_ID" "launch_token=$LAUNCH_TOKEN" \
    "model=$MODEL_LABEL" 'dataset=ZsRE' 'method=EasyEdit_LoRA' \
    "development_ranks=${ADAPTER_RANKS[*]}" \
    "development_learning_rates=${LORA_LEARNING_RATES[*]}" \
    "lora_steps=$LORA_STEPS" "development_lambdas=${SSR_LAMBDAS[*]}" \
    "anchor_weight=$ANCHOR_WEIGHT" "anchor_steps=$ANCHOR_STEPS" \
    "anchor_lr=$ANCHOR_LR" "anchor_batch_size=$ANCHOR_BATCH_SIZE" \
    'n_edits=100' 'checkpoints=1_2_5_10_25_50_100' \
    'primary_endpoints=immediate_efficacy_and_pre_edit_output_consistency_locality' \
    'secondary_locality=immediate_locality_target_consistency' \
    'selection=four_disjoint_development_streams_then_one_frozen_configuration' \
    'confirmation=all_ten_new_predeclared_orders_no_filtering_two_primary_CIs_above_zero' \
    "created_at=$(timestamp)"
  "$PYTHON" "$MANIFEST_BUILDER" --dataset "$ZSRE" --output-dir "$STREAM_ROOT" \
    --development-edits 100 --development-seeds "${DEVELOPMENT_SEEDS[@]}" \
    --confirmation-seeds "${CONFIRMATION_SEEDS[@]}" > "$STATUS_ROOT/logs/streams.log" 2>&1
else
  wait_for_file "$STATUS_ROOT/identity"
fi
wait_for_file "$STREAM_ROOT/INDEX.json"

ALL_DEVELOPMENT_TASKS=()
for adapter_rank in "${ADAPTER_RANKS[@]}"; do
  for learning_rate in "${LORA_LEARNING_RATES[@]}"; do
    for seed in "${DEVELOPMENT_SEEDS[@]}"; do
      ALL_DEVELOPMENT_TASKS+=("$adapter_rank|$learning_rate|$seed|0")
      for coefficient in "${SSR_LAMBDAS[@]}"; do
        ALL_DEVELOPMENT_TASKS+=("$adapter_rank|$learning_rate|$seed|$coefficient")
      done
    done
  done
done
LOCAL_TASKS=()
for index in "${!ALL_DEVELOPMENT_TASKS[@]}"; do
  if ((index % EXPECTED_NNODES == RANK)); then LOCAL_TASKS+=("${ALL_DEVELOPMENT_TASKS[$index]}"); fi
done
run_task_list development "${LOCAL_TASKS[@]}"
atomic_write "$STATUS_ROOT/node_${RANK}.development.complete" "status=complete" "at=$(timestamp)"

"$PYTHON" "$SELECTOR" --result-root "$RESULT_ROOT/development" \
  --ranks "${ADAPTER_RANKS[@]}" --learning-rates "${LORA_LEARNING_RATES[@]}" \
  --lora-steps "$LORA_STEPS" --seeds "${DEVELOPMENT_SEEDS[@]}" \
  --lambdas "${SSR_LAMBDAS[@]}" --output "$SELECTION" > "$STATUS_ROOT/logs/select.log" 2>&1
wait_for_file "$SELECTION"
SELECTION_STATUS="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$SELECTION")"
if [[ "$SELECTION_STATUS" == "no_eligible_configuration" ]]; then
  atomic_write "$STATUS_ROOT/node_${RANK}.complete" "status=no_eligible_configuration" "at=$(timestamp)"
  atomic_write "$STATUS_ROOT/campaign.no_eligible_configuration" \
    "status=no_eligible_configuration" "selection=$SELECTION" "completed_at=$(timestamp)"
  atomic_write "$STATUS_ROOT/campaign.complete" \
    "status=no_eligible_configuration" "selection=$SELECTION" "completed_at=$(timestamp)"
  printf 'Low-rank SSR ended without an eligible development configuration: %s\n' "$STATUS_ROOT"
  exit 0
fi
[[ "$SELECTION_STATUS" == "selected" ]] || die "Unexpected selection status: $SELECTION_STATUS"
SELECTED_RANK="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["rank"])' "$SELECTION")"
SELECTED_LR="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["learning_rate"])' "$SELECTION")"
SELECTED_LAMBDA="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["lambda"])' "$SELECTION")"
atomic_write "$STATUS_ROOT/development.selected" \
  "rank=$SELECTED_RANK" "learning_rate=$SELECTED_LR" "lora_steps=$LORA_STEPS" \
  "lambda=$SELECTED_LAMBDA" "selection=$SELECTION" "at=$(timestamp)"

ALL_CONFIRMATION_TASKS=()
for seed in "${CONFIRMATION_SEEDS[@]}"; do
  ALL_CONFIRMATION_TASKS+=( \
    "$SELECTED_RANK|$SELECTED_LR|$seed|0" \
    "$SELECTED_RANK|$SELECTED_LR|$seed|$SELECTED_LAMBDA"
  )
done
LOCAL_TASKS=()
for index in "${!ALL_CONFIRMATION_TASKS[@]}"; do
  if ((index % EXPECTED_NNODES == RANK)); then LOCAL_TASKS+=("${ALL_CONFIRMATION_TASKS[$index]}"); fi
done
run_task_list confirmation "${LOCAL_TASKS[@]}"
atomic_write "$STATUS_ROOT/node_${RANK}.confirmation.complete" "status=complete" "at=$(timestamp)"

SELECTED_LR_TAG="lr_${SELECTED_LR//./p}"
"$PYTHON" "$SUMMARIZER" \
  --result-root "$RESULT_ROOT/confirmation/rank_$SELECTED_RANK/$SELECTED_LR_TAG" \
  --seeds "${CONFIRMATION_SEEDS[@]}" --rank "$SELECTED_RANK" \
  --learning-rate "$SELECTED_LR" --lora-steps "$LORA_STEPS" \
  --ssr-lambda "$SELECTED_LAMBDA" --frozen-from "$SELECTION" \
  --output "$RESULT_ROOT/confirmation_summary.json" \
  --csv-output "$RESULT_ROOT/confirmation_curves.csv" > "$STATUS_ROOT/logs/summarize.log" 2>&1
atomic_write "$STATUS_ROOT/node_0.complete" "status=complete" "at=$(timestamp)"
atomic_write "$STATUS_ROOT/campaign.complete" "status=complete" \
  "summary=$RESULT_ROOT/confirmation_summary.json" "completed_at=$(timestamp)"
printf 'Low-rank SSR efficacy-locality learning-rate campaign complete: %s\n' "$STATUS_ROOT"
