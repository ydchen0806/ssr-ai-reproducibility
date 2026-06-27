#!/usr/bin/env bash
set -euo pipefail

# Main artificial-network reproduction queue for the SSR manuscript.
# Usage:
#   DEVICE=cuda SEEDS="42 123 456" bash scripts/run_reproducibility_suite.sh

DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-results/reproduce_main}"
SEEDS="${SEEDS:-42 123 456}"

CONFIGS=(
  "configs/lwf_cifar10_v2.yaml"
  "configs/sota_sweeps/biocs_plus_c10_sota_lkd3_t3_ls005.yaml"
  "configs/lwf_cifar100.yaml"
  "configs/biocs_plus_lkd2p0_c100.yaml"
  "configs/lwf_tinyimagenet.yaml"
  "configs/sota_sweeps/biocs_plus_tin_sota_ep80.yaml"
  "configs/er_ace_5datasets.yaml"
  "configs/sota_sweeps/biocs_plus_5ds_sota_lkd10_t3_ls005.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  for seed in ${SEEDS}; do
    echo "[run] config=${cfg} seed=${seed}"
    python main.py \
      --config "${cfg}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --output_dir "${OUT}"
  done
done

python scripts/collect_results.py --results "${OUT}" --csv "${OUT}/summary_table.csv"

