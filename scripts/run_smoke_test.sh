#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-results/smoke}"
SEED="${SEED:-7}"

echo "[smoke] Running a 2-task, 1-epoch LwF baseline."
python main.py \
  --config configs/lwf_cifar10_v2.yaml \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output_dir "${OUT}" \
  --overrides training.epochs=1 training.batch_size=128 dataset.n_tasks=2 training.num_workers=2

echo "[smoke] Running a 2-task, 1-epoch SSR+KD check."
python main.py \
  --config configs/sota_sweeps/biocs_plus_c10_sota_lkd3_t3_ls005.yaml \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output_dir "${OUT}" \
  --overrides training.epochs=1 training.batch_size=128 dataset.n_tasks=2 training.num_workers=2 training.save_final_model=false

python scripts/collect_results.py --results "${OUT}"

