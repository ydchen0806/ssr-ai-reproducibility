#!/usr/bin/env bash
# One-node worker for matched task versus task+SSR adapter pairs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a fresh node-specific directory}"
DATA_ROOT="${ADAPTER_DATA_ROOT:-$PROJECT_ROOT/data/torchvision}"
CACHE_ROOT="${ADAPTER_CACHE_ROOT:-$DATA_ROOT/feature_cache}"
DATASETS_VALUE="${DATASETS:-flowers102 food101 oxfordiiitpet dtd}"
GPU_LIST_VALUE="${GPU_LIST:-0 1 2 3 4 5 6 7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
PHASE="${PHASE:-development}"
DRY_RUN="${DRY_RUN:-0}"
RANK="${ADAPTER_RANK:-16}"

read -r -a DATASET_LIST <<< "$DATASETS_VALUE"
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
[[ ${#DATASET_LIST[@]} -gt 0 && ${#GPUS[@]} -gt 0 ]] || exit 2
[[ "$PHASE" == "development" || "$PHASE" == "confirmation" ]] || {
  printf 'PHASE must be development or confirmation.\n' >&2
  exit 2
}
for dataset in "${DATASET_LIST[@]}"; do
  case "$dataset" in
    cifar100|flowers102|food101|oxfordiiitpet|dtd) ;;
    *) printf 'Unsupported dataset: %s\n' "$dataset" >&2; exit 2 ;;
  esac
done

if [[ "$PHASE" == "development" ]]; then
  SEEDS_VALUE="${SEEDS:-3101 3103 3105}"
  SCALES_VALUE="${SSR_SCALES:-0.03 0.1 0.3 1.0}"
else
  SEEDS_VALUE="${SEEDS:-4101 4103 4105 4107 4109}"
  if [[ -n "${SSR_SCALES:-}" ]]; then
    SCALES_VALUE="$SSR_SCALES"
  else
    LOCK_FILE="${LOCK_FILE:?Set LOCK_FILE or SSR_SCALES for confirmation}"
    SCALES_VALUE="$("$PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["ssr_scale"])' \
      "$LOCK_FILE")"
  fi
fi
read -r -a SEEDS_LIST <<< "$SEEDS_VALUE"
read -r -a SCALES_LIST <<< "$SCALES_VALUE"

job_count=$((${#DATASET_LIST[@]} * ${#SEEDS_LIST[@]} * ${#SCALES_LIST[@]}))
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'phase=%s datasets=%s seeds=%s scales=%s paired_jobs=%s methods_per_pair=2\n' \
    "$PHASE" "$DATASETS_VALUE" "$SEEDS_VALUE" "$SCALES_VALUE" "$job_count"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" "$OUTPUT_ROOT/logs"
for dataset in "${DATASET_LIST[@]}"; do
  cache="$CACHE_ROOT/${dataset}_resnet18.pt"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" \
    "$PROJECT_ROOT/experiments/multidataset_adapter_ssr.py" \
    --dataset "$dataset" \
    --data-root "$DATA_ROOT" \
    --feature-cache "$cache" \
    --output-dir "$OUTPUT_ROOT/prepare/$dataset" \
    --device cuda \
    --download \
    --prepare-only \
    > "$OUTPUT_ROOT/logs/prepare_${dataset}.log" 2>&1
done

active=0
gpu_index=0
pids=()
for dataset in "${DATASET_LIST[@]}"; do
  cache="$CACHE_ROOT/${dataset}_resnet18.pt"
  for scale in "${SCALES_LIST[@]}"; do
    for seed in "${SEEDS_LIST[@]}"; do
      output="$OUTPUT_ROOT/$PHASE/$dataset/scale_${scale}/rank_${RANK}/seed_${seed}"
      [[ ! -e "$output/pair.json" ]] || {
        printf 'Refusing to overwrite completed pair: %s\n' "$output/pair.json" >&2
        exit 2
      }
      gpu="${SLOTS[$gpu_index]}"
      log="$OUTPUT_ROOT/logs/${PHASE}_${dataset}_scale_${scale}_seed_${seed}.log"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
        "$PROJECT_ROOT/experiments/multidataset_adapter_ssr.py" \
        --dataset "$dataset" \
        --data-root "$DATA_ROOT" \
        --feature-cache "$cache" \
        --output-dir "$output" \
        --device cuda \
        --seed "$seed" \
        --rank "$RANK" \
        --ssr-scale "$scale" \
        > "$log" 2>&1 &
      pids+=("$!")
      active=$((active + 1))
      gpu_index=$(((gpu_index + 1) % ${#SLOTS[@]}))
      if ((active == ${#SLOTS[@]})); then
        for pid in "${pids[@]}"; do wait "$pid"; done
        pids=()
        active=0
      fi
    done
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

summary_args=(
  "$PYTHON" "$PROJECT_ROOT/scripts/summarize_multidataset_adapter_ssr.py"
  --input-root "$OUTPUT_ROOT/$PHASE"
  --output-dir "$OUTPUT_ROOT/${PHASE}_summary"
  --expected-datasets "${DATASET_LIST[@]}"
  --expected-seeds "${SEEDS_LIST[@]}"
)
if [[ "$PHASE" == "development" && "${SELECT_LOCK:-1}" == "1" ]]; then
  summary_args+=(--select-lock)
fi
"${summary_args[@]}"

printf 'Completed %s paired jobs under %s\n' "$job_count" "$OUTPUT_ROOT/$PHASE"
