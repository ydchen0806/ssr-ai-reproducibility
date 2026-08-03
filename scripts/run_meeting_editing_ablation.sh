#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/meeting_20260803/locked_editing_gpt2xl.yaml}"
RESULT_VALIDATOR="$PROJECT_ROOT/scripts/validate_result_record.py"

PHASE="${PHASE:-confirm}"
DATASET_OVERRIDE="${DATASET:-${DATASETS:-}}"
RECIPE_OVERRIDE="${RECIPE:-${RECIPES:-}}"
MAPPING_OVERRIDE="${MAPPING:-${MAPPINGS:-}}"
SEED_OVERRIDE="${SEED:-${SEEDS:-}}"
FULL_MATRIX="${FULL_MATRIX:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-meeting_editing_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID}"
REUSE_RESULTS_ROOTS="${REUSE_RESULTS_ROOTS:-}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

usage() {
  printf '%s\n' \
    "Usage: $0 [--phase confirm|screen] [--dataset LIST] [--recipe LIST]" \
    "          [--mapping LIST] [--seed LIST] [--full-matrix] [--dry-run]" \
    "LIST may be comma- or space-separated. Environment overrides with the" \
    "same uppercase names are also supported."
}

while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --dataset|--datasets) DATASET_OVERRIDE="$2"; shift 2 ;;
    --recipe|--recipes) RECIPE_OVERRIDE="$2"; shift 2 ;;
    --mapping|--mappings) MAPPING_OVERRIDE="$2"; shift 2 ;;
    --seed|--seeds) SEED_OVERRIDE="$2"; shift 2 ;;
    --full-matrix) FULL_MATRIX=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
  printf 'Locked editing config not found: %s\n' "$CONFIG_FILE" >&2
  exit 2
fi
git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/check_experiment_worktree.py" \
    --project-root "$PROJECT_ROOT"
fi
if [[ "$PHASE" != "confirm" && "$PHASE" != "screen" ]]; then
  printf 'Unsupported phase %q; choose confirm or screen.\n' "$PHASE" >&2
  exit 2
fi
if [[ ! "$SHARD_INDEX" =~ ^[0-9]+$ || ! "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] \
  || ((SHARD_INDEX >= SHARD_COUNT)); then
  printf 'Require 0 <= SHARD_INDEX < SHARD_COUNT; got %s/%s.\n' \
    "$SHARD_INDEX" "$SHARD_COUNT" >&2
  exit 2
fi

yaml_value() {
  "$PYTHON" - "$CONFIG_FILE" "$1" <<'PY'
import sys
import yaml

path, dotted = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
for key in dotted.split("."):
    value = value[key]
if isinstance(value, bool):
    print("1" if value else "0")
elif isinstance(value, list):
    print(" ".join(str(item) for item in value))
else:
    print(value)
PY
}

split_words() {
  local raw="$1"
  raw="${raw//,/ }"
  read -r -a SPLIT_RESULT <<< "$raw"
}

default_datasets="$(yaml_value "phases.$PHASE.datasets")"
default_recipes="$(yaml_value objective.recipes)"
default_mappings="$(yaml_value kernel.confirmation_distance_mappings)"
default_seeds="$(yaml_value "phases.$PHASE.seeds")"

split_words "${DATASET_OVERRIDE:-$default_datasets}"
datasets=("${SPLIT_RESULT[@]}")
split_words "${RECIPE_OVERRIDE:-$default_recipes}"
recipes=("${SPLIT_RESULT[@]}")
split_words "${MAPPING_OVERRIDE:-$default_mappings}"
mappings=("${SPLIT_RESULT[@]}")
split_words "${SEED_OVERRIDE:-$default_seeds}"
seeds=("${SPLIT_RESULT[@]}")

if ((${#datasets[@]} == 0 || ${#recipes[@]} == 0 || ${#mappings[@]} == 0 || ${#seeds[@]} == 0)); then
  printf 'Dataset, recipe, mapping, and seed selections must all be non-empty.\n' >&2
  exit 2
fi

for dataset in "${datasets[@]}"; do
  case "$dataset" in zsre|cf|recent) ;; *) printf 'Unknown dataset: %s\n' "$dataset" >&2; exit 2 ;; esac
done
for recipe in "${recipes[@]}"; do
  case "$recipe" in
    plain|anchor|spectral|stabilized|ssr_only|ssr_anchor|ssr_spectral|full) ;;
    *) printf 'Unknown recipe: %s\n' "$recipe" >&2; exit 2 ;;
  esac
done
for mapping in "${mappings[@]}"; do
  case "$mapping" in cosine|projective) ;; *) printf 'Unknown mapping: %s\n' "$mapping" >&2; exit 2 ;; esac
done
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    printf 'Seed must be a non-negative integer: %s\n' "$seed" >&2
    exit 2
  fi
done

primary_mapping="$(yaml_value kernel.primary_distance_mapping)"
control_mapping="${mappings[0]}"
for mapping in "${mappings[@]}"; do
  if [[ "$mapping" == "$primary_mapping" ]]; then
    control_mapping="$mapping"
    break
  fi
done

n_edits="$(yaml_value "phases.$PHASE.n_edits")"
data_offset="$(yaml_value "phases.$PHASE.data_offset")"
evaluate_history="$(yaml_value "phases.$PHASE.evaluate_history")"
history_checkpoints="$(yaml_value "phases.$PHASE.history_checkpoints")"
history_max_samples="$(yaml_value "phases.$PHASE.history_max_samples")"

model_name="${MODEL_NAME:-$(yaml_value model.default_cluster_path)}"
canonical_model="$(yaml_value model.name)"
lambda_ssr="$(yaml_value objective.lambda_ssr)"
lambda_anchor="$(yaml_value objective.lambda_anchor)"
lambda_spectral="$(yaml_value objective.lambda_spectral)"
model_hash="$(yaml_value model.fingerprint_sha256)"
evaluator="$(yaml_value evaluation.name)"
evaluator_version="$(yaml_value evaluation.version)"
evaluation_protocol_hash="$(yaml_value evaluation.protocol_sha256)"
pairing_protocol_hash="$(yaml_value pairing.protocol_sha256)"

export BIOCS_KERNEL_FAMILY="$(yaml_value kernel.family)"
export BIOCS_A_EXC="$(yaml_value kernel.a_exc)"
export BIOCS_A_INH="$(yaml_value kernel.a_inh)"
export BIOCS_SIGMA_EXC="$(yaml_value kernel.sigma_exc)"
export BIOCS_SIGMA_INH="$(yaml_value kernel.sigma_inh)"
export BIOCS_TARGET="$(yaml_value kernel.target)"
export BIOCS_ROW_SELECTION="$(yaml_value kernel.row_selection)"
export BIOCS_MAX_SPATIAL_ROWS="$(yaml_value kernel.max_spatial_rows)"
export BIOCS_TARGET_LAYERS="$(yaml_value model.target_layers)"
export BIOCS_LR="$(yaml_value optimizer.lr)"
export BIOCS_NUM_STEPS="$(yaml_value optimizer.num_steps)"
export FT_LR="$BIOCS_LR"
export FT_NUM_STEPS="$BIOCS_NUM_STEPS"
export KE_DATA_OFFSET="$data_offset"
export KE_FORCE_TEXT_ONLY="$(yaml_value model.force_text_only)"
export KE_LOCAL_ONLY="$(yaml_value model.local_files_only)"
export KE_MODEL_FINGERPRINT="$model_hash"
export KE_PAIRING_PROTOCOL_HASH="$pairing_protocol_hash"
export KE_EXPECTED_LOCALITY_EVALUATOR="$evaluator"
export KE_EXPECTED_LOCALITY_EVALUATOR_VERSION="$evaluator_version"
export KE_EXPECTED_LOCALITY_PROTOCOL_HASH="$evaluation_protocol_hash"
export KE_REQUIRE_LEGACY_LOCALITY_COMPATIBLE="$(yaml_value evaluation.require_legacy_compatible_input)"

mkdir -p "$RESULT_ROOT"
plan_file="$RESULT_ROOT/planned_runs.tsv"
if ((SHARD_COUNT > 1)); then
  plan_file="$RESULT_ROOT/planned_runs_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.tsv"
fi
printf 'phase\tdataset\trecipe\tmapping\tseed\tdata_offset\tn_edits\taction\tsource\toutput\n' > "$plan_file"

is_mapping_dependent() {
  case "$1" in ssr_only|ssr_anchor|ssr_spectral|full) return 0 ;; *) return 1 ;; esac
}

validate_record() {
  local record="$1"
  local dataset="$2"
  local recipe="$3"
  local mapping="$4"
  local seed="$5"
  local allow_legacy="$6"
  local -a command=(
    "$PYTHON" "$RESULT_VALIDATOR"
    --record "$record"
    --task-family editing
    --dataset "$dataset"
    --model "$canonical_model"
    --recipe "$recipe"
    --mapping "$mapping"
    --seed "$seed"
    --data-offset "$data_offset"
    --n-edits "$n_edits"
    --evaluator "$evaluator"
    --evaluator-version "$evaluator_version"
    --evaluation-protocol-hash "$evaluation_protocol_hash"
    --model-hash "$model_hash"
    --pairing-protocol-hash "$pairing_protocol_hash"
    --expected-git-commit "$git_commit"
  )
  if [[ "$allow_legacy" == "1" ]]; then
    command+=(--allow-legacy-git)
  fi
  "${command[@]}"
}

run_one() {
  local dataset="$1"
  local recipe="$2"
  local mapping="$3"
  local mapping_tag="$4"
  local seed="$5"
  local output_dir="$RESULT_ROOT/$PHASE/$dataset/$recipe/$mapping_tag/seed_$seed"
  local method="biocs"
  local action="run"
  local reuse_source=""
  if [[ "$recipe" == "plain" ]]; then
    method="ft"
  fi
  if [[ -s "$output_dir/result_record.json" ]]; then
    validate_record \
      "$output_dir/result_record.json" "$dataset" "$recipe" "$mapping_tag" "$seed" 1 \
      >/dev/null || {
        printf 'Conflicting existing result: %s\n' "$output_dir/result_record.json" >&2
        return 65
      }
    action="skip"
    reuse_source="$output_dir/result_record.json"
  elif [[ -n "$REUSE_RESULTS_ROOTS" ]]; then
    local reuse_root reuse_candidate
    for reuse_root in $REUSE_RESULTS_ROOTS; do
      reuse_candidate="$reuse_root/$dataset/$recipe/$mapping_tag/seed_$seed/result_record.json"
      if [[ -s "$reuse_candidate" ]]; then
        validate_record \
          "$reuse_candidate" "$dataset" "$recipe" "$mapping_tag" "$seed" 1 \
          >/dev/null || {
            printf 'Conflicting reusable result: %s\n' "$reuse_candidate" >&2
            return 65
          }
        action="reuse"
        reuse_source="$reuse_candidate"
        break
      fi
    done
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$PHASE" "$dataset" "$recipe" "$mapping_tag" "$seed" \
    "$data_offset" "$n_edits" "$action" "$reuse_source" "$output_dir" >> "$plan_file"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  mkdir -p "$output_dir"
  if [[ "$action" == "reuse" ]]; then
    cp -a "$(dirname "$reuse_source")/." "$output_dir/"
    if [[ ! -s "$output_dir/results.json" ]]; then
      printf 'Reusable record lacks its per-edit results.json: %s\n' "$reuse_source" >&2
      return 66
    fi
    printf 'REUSED\t%s\n' "$reuse_source" > "$output_dir/COMPLETE"
    printf 'REUSE verified record: %s\n' "$reuse_source"
    return 0
  fi
  if [[ "$action" == "skip" ]]; then
    printf 'SKIP complete: %s\n' "$output_dir"
    return 0
  fi

  local attempt
  attempt="$(date +%Y%m%d_%H%M%S)"
  local log_file="$output_dir/run_$attempt.log"
  local status_file="$output_dir/status.tsv"
  printf '%s\tRUNNING\n' "$attempt" >> "$status_file"

  local -a command=(
    "$PYTHON" "$PROJECT_ROOT/scripts/run_llm_ke_easyedit.py"
    --method "$method"
    --dataset "$dataset"
    --n_edits "$n_edits"
    --model_name "$model_name"
    --output "$output_dir"
    --seed "$seed"
    --recipe "$recipe"
  )
  if [[ "$method" == "biocs" ]]; then
    command+=(
      --distance_mapping "$mapping"
      --lambda_ssr "$lambda_ssr"
      --lambda_anchor "$lambda_anchor"
      --lambda_spectral "$lambda_spectral"
    )
  fi
  if [[ "$evaluate_history" == "1" ]]; then
    command+=(--evaluate_history --history_max_samples "$history_max_samples")
    if [[ -n "$history_checkpoints" ]]; then
      split_words "$history_checkpoints"
      command+=(--history_checkpoints "${SPLIT_RESULT[@]}")
    fi
  fi

  export KE_SEED="$seed"
  export BIOCS_SAMPLER_SEED="$seed"
  printf 'RUN %s dataset=%s recipe=%s mapping=%s seed=%s\n' \
    "$PHASE" "$dataset" "$recipe" "$mapping_tag" "$seed"
  set +e
  "${command[@]}" 2>&1 | tee "$log_file"
  local exit_code=${PIPESTATUS[0]}
  set -e
  if ((exit_code != 0)); then
    printf '%s\tFAILED\t%s\t%s\n' "$attempt" "$exit_code" "$log_file" >> "$status_file"
    printf '%s\n' "$exit_code" > "$output_dir/FAILED.$attempt"
    printf 'FAILED (%s): %s\n' "$exit_code" "$output_dir" >&2
    return "$exit_code"
  fi
  if [[ ! -s "$output_dir/results.json" || ! -s "$output_dir/result_record.json" ]]; then
    printf '%s\tFAILED\tmissing_outputs\t%s\n' "$attempt" "$log_file" >> "$status_file"
    printf 'Run exited zero but required outputs are missing: %s\n' "$output_dir" >&2
    return 70
  fi
  validate_record \
    "$output_dir/result_record.json" "$dataset" "$recipe" "$mapping_tag" "$seed" 0 \
    >/dev/null || {
      printf '%s\tFAILED\tinvalid_result_identity\t%s\n' "$attempt" "$log_file" >> "$status_file"
      return 65
    }
  printf '%s\tSUCCESS\t0\t%s\n' "$attempt" "$log_file" >> "$status_file"
  printf '%s\n' "$attempt" > "$output_dir/COMPLETE"
}

matrix_index=0
for dataset in "${datasets[@]}"; do
  for recipe in "${recipes[@]}"; do
    if is_mapping_dependent "$recipe" || [[ "$FULL_MATRIX" == "1" ]]; then
      recipe_mappings=("${mappings[@]}")
    else
      recipe_mappings=("$control_mapping")
    fi
    for mapping in "${recipe_mappings[@]}"; do
      mapping_tag="$mapping"
      if ! is_mapping_dependent "$recipe" && [[ "$FULL_MATRIX" != "1" ]]; then
        mapping_tag="none"
      fi
      for seed in "${seeds[@]}"; do
        assigned_index="$matrix_index"
        matrix_index=$((matrix_index + 1))
        if ((assigned_index % SHARD_COUNT != SHARD_INDEX)); then
          continue
        fi
        run_one "$dataset" "$recipe" "$mapping" "$mapping_tag" "$seed"
      done
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  planned_count="$(( $(wc -l < "$plan_file") - 1 ))"
  run_count="$(awk -F '\t' 'NR>1 && $8=="run" {n++} END {print n+0}' "$plan_file")"
  reuse_count="$(awk -F '\t' 'NR>1 && $8=="reuse" {n++} END {print n+0}' "$plan_file")"
  printf 'Dry-run complete: %s total, %s reused, %s to run in %s\n' \
    "$planned_count" "$reuse_count" "$run_count" "$plan_file"
else
  printf 'Assigned shard %s/%s completed: %s\n' "$SHARD_INDEX" "$SHARD_COUNT" "$RESULT_ROOT"
fi
