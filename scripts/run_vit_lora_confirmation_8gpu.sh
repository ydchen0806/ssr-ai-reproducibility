#!/usr/bin/env bash
# Raw-image ViT-LoRA confirmation: task loss versus task loss + SSR.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/vit_lora_confirmation}"
DATA_ROOT="${VIT_LORA_DATA_ROOT:-$PROJECT_ROOT/data}"
PRETRAINED_CHECKPOINT="${VIT_LORA_PRETRAINED_CHECKPOINT:-$PROJECT_ROOT/data/pretrained/vit_tiny_patch16_224_augreg_in21k_ft_in1k.safetensors}"
DATASETS_VALUE="${DATASETS:-flowers102 oxfordiiitpet}"
SEEDS_VALUE="${SEEDS:-5201 5203 5205 5207 5209}"
GPU_LIST_VALUE="${GPU_LIST:-0 1 2 3 4 5 6 7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
EPOCHS="${EPOCHS:-15}"
LAMBDA_SPATIAL="${LAMBDA_SPATIAL:-0.01}"
LAMBDA_ADAPTER="${LAMBDA_ADAPTER:-0.002}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_NETWORK_DOWNLOAD="${ALLOW_NETWORK_DOWNLOAD:-0}"
REUSE_RESULTS_ROOT="${REUSE_RESULTS_ROOT:-}"
ALLOWED_EXISTING_GIT_COMMITS="${ALLOWED_EXISTING_GIT_COMMITS:-}"
RESULT_VALIDATOR="$PROJECT_ROOT/scripts/validate_result_record.py"
git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

read -r -a DATASET_LIST <<< "$DATASETS_VALUE"
read -r -a SEED_LIST <<< "$SEEDS_VALUE"
read -r -a GPUS <<< "$GPU_LIST_VALUE"
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
  printf 'JOBS_PER_GPU must be a positive integer.\n' >&2
  exit 2
}
SLOTS=()
for gpu in "${GPUS[@]}"; do
  for ((slot = 0; slot < JOBS_PER_GPU; slot++)); do
    SLOTS+=("$gpu")
  done
done
ARMS=(task task_ssr)
for dataset in "${DATASET_LIST[@]}"; do
  case "$dataset" in
    flowers102|oxfordiiitpet) ;;
    *) printf 'Unsupported ViT-LoRA dataset: %s\n' "$dataset" >&2; exit 2 ;;
  esac
done
[[ ${#SEED_LIST[@]} -gt 0 && ${#GPUS[@]} -gt 0 ]] || exit 2

mkdir -p "$OUTPUT_ROOT"
PLAN="$OUTPUT_ROOT/planned_runs.tsv"
printf 'dataset\tarm\tseed\tconfig\toutput\n' > "$PLAN"
for dataset in "${DATASET_LIST[@]}"; do
  config="$PROJECT_ROOT/configs/meeting_20260804/vit_lora_${dataset}.yaml"
  [[ -s "$config" ]] || { printf 'Missing config: %s\n' "$config" >&2; exit 2; }
  for seed in "${SEED_LIST[@]}"; do
    for arm in "${ARMS[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$arm" "$seed" "$config" \
        "$OUTPUT_ROOT/$dataset/$arm/seed_$seed" >> "$PLAN"
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Dry-run: %s raw-image ViT-LoRA jobs (%s strict pairs).\n' \
    "$(( ${#DATASET_LIST[@]} * ${#ARMS[@]} * ${#SEED_LIST[@]} ))" \
    "$(( ${#DATASET_LIST[@]} * ${#SEED_LIST[@]} ))"
  exit 0
fi

"$PYTHON" "$PROJECT_ROOT/scripts/check_experiment_worktree.py" \
  --project-root "$PROJECT_ROOT"
[[ -s "$PRETRAINED_CHECKPOINT" ]] || {
  printf 'Missing offline ViT checkpoint: %s\n' "$PRETRAINED_CHECKPOINT" >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT/logs" "$DATA_ROOT"
configs=()
for dataset in "${DATASET_LIST[@]}"; do
  configs+=("$PROJECT_ROOT/configs/meeting_20260804/vit_lora_${dataset}.yaml")
done
prepare_command=(
  "$PYTHON" "$PROJECT_ROOT/scripts/prepare_vit_lora_assets.py"
  --configs "${configs[@]}"
  --data-root "$DATA_ROOT"
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT"
)
if [[ "$ALLOW_NETWORK_DOWNLOAD" == "1" ]]; then
  prepare_command+=(--allow-download)
fi
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${prepare_command[@]}" \
  > "$OUTPUT_ROOT/logs/prepare_assets.log" 2>&1

validate_record() {
  local record="$1" dataset="$2" arm="$3" seed="$4" allow_previous="$5"
  local recipe="plain" mapping="none"
  if [[ "$arm" == "task_ssr" ]]; then
    recipe="ssr_only"
    mapping="cosine"
  fi
  local -a command=(
    "$PYTHON" "$RESULT_VALIDATOR"
    --record "$record"
    --task-family classification
    --dataset "${dataset}_cl"
    --model vit_tiny_patch16_224
    --recipe "$recipe"
    --mapping "$mapping"
    --seed "$seed"
    --expected-git-commit "$git_commit"
    --config-file "$(dirname "$record")/config.yaml"
  )
  if [[ "$allow_previous" == "1" ]]; then
    local allowed_commit
    for allowed_commit in $ALLOWED_EXISTING_GIT_COMMITS; do
      command+=(--allowed-git-commit "$allowed_commit")
    done
  fi
  "${command[@]}"
}

worker() {
  local gpu="$1" worker_index="$2" worker_count="$3"
  local index=0 dataset arm seed config output log pair previous_pair="" assigned=0
  while IFS=$'\t' read -r dataset arm seed config output; do
    [[ "$dataset" != "dataset" ]] || continue
    pair="$dataset:$seed"
    if [[ "$pair" != "$previous_pair" ]]; then
      assigned=$((index % worker_count))
      index=$((index + 1))
      previous_pair="$pair"
    fi
    if ((assigned != worker_index)); then
      continue
    fi
    if [[ -s "$output/result_record.json" ]]; then
      validate_record "$output/result_record.json" "$dataset" "$arm" "$seed" 1 \
        >/dev/null || return 65
      printf 'SKIP %s/%s/seed_%s\n' "$dataset" "$arm" "$seed"
      continue
    fi
    if [[ -n "$REUSE_RESULTS_ROOT" ]]; then
      reuse_record="$REUSE_RESULTS_ROOT/$dataset/$arm/seed_$seed/result_record.json"
      if [[ -s "$reuse_record" ]]; then
        validate_record "$reuse_record" "$dataset" "$arm" "$seed" 1 \
          >/dev/null || return 65
        mkdir -p "$output"
        cp -a "$(dirname "$reuse_record")/." "$output/"
        printf 'source_record=%s\nsource_git_commit=%s\n' \
          "$reuse_record" "$ALLOWED_EXISTING_GIT_COMMITS" > "$output/REUSED_FROM"
        printf 'REUSE %s/%s/seed_%s from %s\n' \
          "$dataset" "$arm" "$seed" "$reuse_record"
        continue
      fi
    fi
    mkdir -p "$output"
    log="$OUTPUT_ROOT/logs/${dataset}_${arm}_seed_${seed}.log"
    common_overrides=(
      "dataset.data_root=$DATA_ROOT"
      "model.pretrained_checkpoint=$PRETRAINED_CHECKPOINT"
      "training.epochs=$EPOCHS"
    )
    if [[ "$arm" == "task" ]]; then
      arm_overrides=(
        "experiment_name=task"
        "method.recipe=plain"
        "method.lambda_spatial=0.0"
        "method.lambda_adapter=0.0"
        "method.lambda_spectral=0.0"
      )
    else
      arm_overrides=(
        "experiment_name=task_ssr"
        "method.recipe=ssr_only"
        "method.lambda_spatial=$LAMBDA_SPATIAL"
        "method.lambda_adapter=$LAMBDA_ADAPTER"
        "method.lambda_spectral=0.0"
      )
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$PROJECT_ROOT/main.py" \
      --config "$config" \
      --seed "$seed" \
      --device cuda \
      --output_dir "$OUTPUT_ROOT/$dataset" \
      --overrides "${common_overrides[@]}" "${arm_overrides[@]}" \
      > "$log" 2>&1
    [[ -s "$output/result_record.json" ]] || {
      printf 'Missing result for %s/%s/seed_%s; see %s\n' \
        "$dataset" "$arm" "$seed" "$log" >&2
      return 70
    }
    validate_record "$output/result_record.json" "$dataset" "$arm" "$seed" 0 \
      >/dev/null || return 65
  done < "$PLAN"
}

pids=()
for index in "${!SLOTS[@]}"; do
  worker "${SLOTS[$index]}" "$index" "${#SLOTS[@]}" &
  pids+=("$!")
done
exit_code=0
for pid in "${pids[@]}"; do wait "$pid" || exit_code=1; done
((exit_code == 0)) || exit "$exit_code"

"$PYTHON" "$PROJECT_ROOT/scripts/validate_vit_lora_pairs.py" \
  --results-root "$OUTPUT_ROOT" \
  --datasets "${DATASET_LIST[@]}" \
  --seeds "${SEED_LIST[@]}" \
  --output-dir "$OUTPUT_ROOT/paired_summary"
printf 'Raw-image ViT-LoRA confirmation complete: %s\n' "$OUTPUT_ROOT"
