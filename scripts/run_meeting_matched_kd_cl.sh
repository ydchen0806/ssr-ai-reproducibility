#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/meeting_20260803/matched_kd_cl}"
SEEDS="${SEEDS:-3101 3103 3105 3107 3109}"
DATASETS="${DATASETS:-split_cifar100 split_tiny_imagenet}"
METHODS="${METHODS:-kd kd_ewc kd_mas kd_si kd_ssr}"
GPU_LIST="${GPU_LIST:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESULT_VALIDATOR="$PROJECT_ROOT/scripts/validate_result_record.py"

git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
if [[ "$DRY_RUN" != "1" && -n "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=no)" ]]; then
  printf 'Formal matched-KD runs require a clean tracked worktree.\n' >&2
  exit 2
fi

read -r -a gpu_array <<< "$GPU_LIST"
read -r -a seed_array <<< "$SEEDS"
read -r -a dataset_array <<< "$DATASETS"
read -r -a method_array <<< "$METHODS"
if ((${#gpu_array[@]} == 0)); then
  printf 'GPU_LIST must contain at least one GPU index.\n' >&2
  exit 2
fi

for dataset in "${dataset_array[@]}"; do
  case "$dataset" in
    split_cifar100|split_tiny_imagenet) ;;
    *) printf 'Unknown matched-KD dataset: %s\n' "$dataset" >&2; exit 2 ;;
  esac
done
for method in "${method_array[@]}"; do
  case "$method" in
    kd|kd_ewc|kd_mas|kd_si|kd_ssr) ;;
    *) printf 'Unknown matched-KD method: %s\n' "$method" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_ROOT"
task_file="$OUTPUT_ROOT/planned_runs.tsv"
printf 'dataset\tmethod\tseed\tconfig\toutput\n' > "$task_file"
for dataset in "${dataset_array[@]}"; do
  config_name="$dataset"
  if [[ "$dataset" == "split_tiny_imagenet" ]]; then
    config_name="tinyimagenet"
  fi
  config="$PROJECT_ROOT/configs/meeting_20260803/cl/${config_name}.yaml"
  [[ -s "$config" ]] || { printf 'Missing config: %s\n' "$config" >&2; exit 2; }
  for method in "${method_array[@]}"; do
    for seed in "${seed_array[@]}"; do
      [[ "$seed" =~ ^[0-9]+$ ]] || { printf 'Invalid seed: %s\n' "$seed" >&2; exit 2; }
      output="$OUTPUT_ROOT/$dataset/$method/seed_$seed"
      printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$method" "$seed" "$config" "$output" >> "$task_file"
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Dry-run: %s matched-KD jobs in %s\n' "$(( $(wc -l < "$task_file") - 1 ))" "$task_file"
  exit 0
fi

validate_record() {
  local record="$1"
  local dataset="$2"
  local method="$3"
  local seed="$4"
  local mapping="none"
  if [[ "$method" == "kd_ssr" ]]; then
    mapping="cosine"
  fi
  "$PYTHON" "$RESULT_VALIDATOR" \
    --record "$record" \
    --task-family classification \
    --dataset "$dataset" \
    --model resnet18 \
    --recipe "$method" \
    --mapping "$mapping" \
    --seed "$seed" \
    --expected-git-commit "$git_commit"
}

worker() {
  local gpu="$1"
  local worker_index="$2"
  local worker_count="$3"
  local row_index=0
  while IFS=$'\t' read -r dataset method seed config output; do
    [[ "$dataset" != "dataset" ]] || continue
    if ((row_index % worker_count != worker_index)); then
      row_index=$((row_index + 1))
      continue
    fi
    row_index=$((row_index + 1))
    if [[ -s "$output/result_record.json" ]]; then
      validate_record "$output/result_record.json" "$dataset" "$method" "$seed" \
        >/dev/null || {
          printf 'Conflicting existing result: %s\n' "$output/result_record.json" >&2
          return 65
        }
      printf 'SKIP %s/%s/seed_%s\n' "$dataset" "$method" "$seed"
      continue
    fi
    mkdir -p "$output"
    printf 'RUN gpu=%s %s/%s/seed_%s\n' "$gpu" "$dataset" "$method" "$seed"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$PROJECT_ROOT/main.py" \
      --config "$config" \
      --seed "$seed" \
      --device cuda \
      --output_dir "$OUTPUT_ROOT/$dataset" \
      --overrides "method.name=$method" "experiment_name=$method" \
      >"$output/launcher.log" 2>&1
    [[ -s "$output/result_record.json" ]] || {
      printf 'Missing result_record.json for %s\n' "$output" >&2
      return 70
    }
    validate_record "$output/result_record.json" "$dataset" "$method" "$seed" \
      >/dev/null || return 65
  done < "$task_file"
}

pids=()
for index in "${!gpu_array[@]}"; do
  worker "${gpu_array[$index]}" "$index" "${#gpu_array[@]}" &
  pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
  wait "$pid" || exit_code=1
done
((exit_code == 0)) || exit "$exit_code"

expected_methods="kd kd_ewc kd_mas kd_si kd_ssr"
observed_methods="$(printf '%s\n' "${method_array[@]}" | sort | tr '\n' ' ' | sed 's/ $//')"
sorted_expected="$(printf '%s\n' $expected_methods | sort | tr '\n' ' ' | sed 's/ $//')"
if [[ "$observed_methods" == "$sorted_expected" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/check_matched_kd_fairness.py" \
    --results-root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/fairness_report.json" \
    --expected-datasets "${dataset_array[@]}" \
    --expected-seeds "${seed_array[@]}"
fi
printf 'Matched-KD jobs completed: %s\n' "$OUTPUT_ROOT"
