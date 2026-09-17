#!/usr/bin/env bash
# Run one dataset on one eight-GPU node. A parent launcher may invoke this on
# separate nodes for cub200, oxford_iiit_pet, and oxford_flowers102.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
DATASET="${SSR_SEG_DATASET:-}"
RUN_ID="${SSR_SEG_RUN_ID:-}"
RESULT_ROOT="${SSR_SEG_RESULT_ROOT:-$PROJECT_ROOT/results/$RUN_ID/$DATASET}"
GPU_COUNT="${GPU_COUNT:-8}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
WORKERS="${SEG_WORKERS:-0}"
RESUME="${SSR_SEG_RESUME:-0}"
MAX_WALLTIME_SEC="${MAX_WALLTIME_SEC:-28800}"

[[ "$DATASET" =~ ^(cub200|oxford_iiit_pet|oxford_flowers102)$ ]] || {
  printf 'Set SSR_SEG_DATASET to cub200, oxford_iiit_pet, or oxford_flowers102.\n' >&2
  exit 2
}
[[ -n "$RUN_ID" && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Set SSR_SEG_RUN_ID to a fresh identifier.\n' >&2
  exit 2
}
[[ "$GPU_COUNT" == "8" ]] || {
  printf 'This runner requires one node with eight GPUs.\n' >&2
  exit 2
}
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ && "$WORKERS" =~ ^[0-9]+$ ]] || {
  printf 'JOBS_PER_GPU must be positive and SEG_WORKERS non-negative.\n' >&2
  exit 2
}
[[ "$MAX_WALLTIME_SEC" =~ ^[1-9][0-9]*$ ]] || {
  printf 'MAX_WALLTIME_SEC must be a positive integer.\n' >&2
  exit 2
}
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || {
  printf 'SSR_SEG_RESUME must be 0 or 1.\n' >&2
  exit 2
}

physical_gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count+0}')"
[[ "$physical_gpu_count" == "$GPU_COUNT" ]] || {
  printf 'Expected eight GPUs, found %s on %s.\n' "$physical_gpu_count" "$(hostname)" >&2
  exit 2
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "$DATASET" in
  cub200)
    dataset_root="$PROJECT_ROOT/data/cub200"
    cache_source="$dataset_root/cub200_seg_resnet18_dense_192.pt"
    batch_size=24
    ;;
  oxford_iiit_pet)
    dataset_root="$PROJECT_ROOT/data/oxford_iiit_pet"
    cache_source="$dataset_root/resnet18_layer3_masks_160_provenance_v2.pt"
    batch_size=32
    ;;
  oxford_flowers102)
    dataset_root="$PROJECT_ROOT/data/flowers-102"
    cache_source="$dataset_root/resnet18_layer3_masks_160_provenance_v2.pt"
    batch_size=32
    ;;
esac

[[ -d "$dataset_root" && -f "$cache_source" ]] || {
  printf 'Prepared dataset/cache is missing: %s %s\n' "$dataset_root" "$cache_source" >&2
  exit 2
}

local_stage="$(mktemp -d "/dev/shm/ssr_seg_nested_${DATASET}.XXXXXX")"
cleanup() {
  rm -rf -- "$local_stage"
}
trap cleanup EXIT INT TERM

cache_local="$local_stage/$(basename "$cache_source")"
cp "$cache_source" "$cache_local.tmp"
mv "$cache_local.tmp" "$cache_local"
if [[ -f "$cache_source.manifest.json" ]]; then
  cp "$cache_source.manifest.json" "$cache_local.manifest.json"
fi
source_hash="$(sha256sum "$cache_source" | awk '{print $1}')"
local_hash="$(sha256sum "$cache_local" | awk '{print $1}')"
[[ "$source_hash" == "$local_hash" ]] || {
  printf 'Staged cache checksum mismatch for %s.\n' "$DATASET" >&2
  exit 2
}

resume_args=()
if [[ "$RESUME" == "1" ]]; then
  resume_args+=(--resume)
fi

timeout --signal=TERM --kill-after=300 "$MAX_WALLTIME_SEC" \
  "$PYTHON" "$PROJECT_ROOT/scripts/run_segmentation_nested_dataset.py" \
  --dataset "$DATASET" \
  --dataset-root "$dataset_root" \
  --feature-cache "$cache_local" \
  --result-root "$RESULT_ROOT" \
  --gpus 0 1 2 3 4 5 6 7 \
  --jobs-per-gpu "$JOBS_PER_GPU" \
  --workers "$WORKERS" \
  --batch-size "$batch_size" \
  --python "$PYTHON" \
  "${resume_args[@]}"

[[ -s "$RESULT_ROOT/COMPLETED.json" ]] || {
  printf 'Segmentation runner exited without COMPLETED.json: %s\n' "$RESULT_ROOT" >&2
  exit 2
}
printf 'Completed dataset=%s result_root=%s\n' "$DATASET" "$RESULT_ROOT"
