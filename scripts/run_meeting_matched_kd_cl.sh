#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/meeting_20260803/matched_kd_cl}"
SEEDS="${SEEDS:-3101 3103 3105 3107 3109}"
DATASETS="${DATASETS:-split_cifar100 split_tiny_imagenet}"
METHODS="${METHODS:-kd kd_ewc kd_mas kd_si kd_center kd_protodecor kd_spectral kd_ssr}"
GPU_LIST="${GPU_LIST:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESULT_VALIDATOR="$PROJECT_ROOT/scripts/validate_result_record.py"

git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/check_experiment_worktree.py" \
    --project-root "$PROJECT_ROOT"
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
    kd|kd_ewc|kd_mas|kd_si|kd_center|kd_protodecor|kd_spectral|kd_ssr) ;;
    *) printf 'Unknown matched-KD method: %s\n' "$method" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_ROOT"
task_file="$OUTPUT_ROOT/planned_runs.tsv"
printf 'dataset\tmethod\tseed\tconfig\tteacher_root\toutput\n' > "$task_file"
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
      teacher_root="$OUTPUT_ROOT/$dataset/kd_teacher_trajectories/seed_$seed"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$method" "$seed" "$config" "$teacher_root" "$output" >> "$task_file"
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
  local teacher_root="$5"
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
    --teacher-protocol locked_kd_only_trajectory_v1 \
    --require-sha256-field teacher_trajectory_hash \
    --expected-git-commit "$git_commit"
  "$PYTHON" "$PROJECT_ROOT/scripts/validate_teacher_trajectory.py" \
    --root "$teacher_root" \
    --dataset "$dataset" \
    --model resnet18 \
    --seed "$seed" \
    --expected-checkpoints 10 \
    --result-record "$record" \
    >/dev/null
}

worker() {
  local gpu="$1"
  local worker_index="$2"
  local worker_count="$3"
  local phase="$4"
  local dataset_index seed_index group_index index
  while IFS=$'\t' read -r dataset method seed config teacher_root output; do
    [[ "$dataset" != "dataset" ]] || continue
    if [[ "$phase" == "teacher" && "$method" != "kd" ]]; then
      continue
    fi
    if [[ "$phase" == "treatment" && "$method" == "kd" ]]; then
      continue
    fi
    dataset_index=-1
    seed_index=-1
    for index in "${!dataset_array[@]}"; do
      [[ "${dataset_array[$index]}" != "$dataset" ]] || dataset_index="$index"
    done
    for index in "${!seed_array[@]}"; do
      [[ "${seed_array[$index]}" != "$seed" ]] || seed_index="$index"
    done
    ((dataset_index >= 0 && seed_index >= 0)) || return 2
    group_index=$((dataset_index * ${#seed_array[@]} + seed_index))
    if ((group_index % worker_count != worker_index)); then
      continue
    fi
    if [[ -s "$output/result_record.json" ]]; then
      validate_record "$output/result_record.json" "$dataset" "$method" "$seed" "$teacher_root" \
        >/dev/null || {
          printf 'Conflicting existing result: %s\n' "$output/result_record.json" >&2
          return 65
        }
      printf 'SKIP %s/%s/seed_%s\n' "$dataset" "$method" "$seed"
      continue
    fi
    mkdir -p "$output"
    printf 'RUN gpu=%s %s/%s/seed_%s\n' "$gpu" "$dataset" "$method" "$seed"
    log_file="$output/launcher_$(date '+%Y%m%d_%H%M%S').log"
    if [[ -s "$output/FAILED" ]]; then
      mv "$output/FAILED" "$output/FAILED.previous.$(date '+%Y%m%d_%H%M%S')"
    fi
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$PROJECT_ROOT/main.py" \
      --config "$config" \
      --seed "$seed" \
      --device cuda \
      --output_dir "$OUTPUT_ROOT/$dataset" \
      --overrides \
        "method.name=$method" \
        "method.teacher_mode=locked_kd_trajectory" \
        "method.teacher_checkpoint_root=$teacher_root" \
        "method.teacher_dataset=$dataset" \
        "method.teacher_model=resnet18" \
        "experiment_name=$method" \
      >"$log_file" 2>&1; then
      printf 'failed_at=%s\nlog=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$log_file" \
        > "$output/FAILED"
      return 70
    fi
    [[ -s "$output/result_record.json" ]] || {
      printf 'Missing result_record.json for %s\n' "$output" >&2
      printf 'missing_result_at=%s\nlog=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$log_file" \
        > "$output/FAILED"
      return 70
    }
    validate_record "$output/result_record.json" "$dataset" "$method" "$seed" "$teacher_root" \
      >/dev/null || return 65
  done < "$task_file"
}

run_phase() {
  local phase="$1"
  local -a pids=()
  for index in "${!gpu_array[@]}"; do
    worker "${gpu_array[$index]}" "$index" "${#gpu_array[@]}" "$phase" &
    pids+=("$!")
  done
  local exit_code=0
  for pid in "${pids[@]}"; do
    wait "$pid" || exit_code=1
  done
  ((exit_code == 0)) || return "$exit_code"
}

# Every treatment must consume the completed KD-only checkpoint trajectory for
# its dataset/seed, so teacher jobs are a hard barrier before treatment jobs.
run_phase teacher
run_phase treatment

expected_methods="kd kd_ewc kd_mas kd_si kd_center kd_protodecor kd_spectral kd_ssr"
observed_methods="$(printf '%s\n' "${method_array[@]}" | sort | tr '\n' ' ' | sed 's/ $//')"
sorted_expected="$(printf '%s\n' $expected_methods | sort | tr '\n' ' ' | sed 's/ $//')"
if [[ "$observed_methods" == "$sorted_expected" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/check_matched_kd_fairness.py" \
    --results-root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/fairness_report.json" \
    --expected-datasets "${dataset_array[@]}" \
    --expected-seeds "${seed_array[@]}"
  "$PYTHON" "$PROJECT_ROOT/scripts/summarize_matched_kd.py" \
    --plan "$task_file" \
    --output-dir "$OUTPUT_ROOT/paired_summary"
fi
printf 'Matched-KD jobs completed: %s\n' "$OUTPUT_ROOT"
