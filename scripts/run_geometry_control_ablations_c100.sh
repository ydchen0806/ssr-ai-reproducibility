#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-42 123 456}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
KD_LAMBDA="${KD_LAMBDA:-2.0}"
CONFIGS="${CONFIGS:-configs/biocs_plus_kd_only_lkd2_c100.yaml configs/biocs_plus_lkd2p0_c100.yaml configs/geomctrl_cosorth_c100.yaml configs/geomctrl_proto_decor_c100.yaml configs/geomctrl_spectral_c100.yaml configs/geomctrl_center_loss_c100.yaml configs/geomctrl_supcon_c100.yaml configs/geomctrl_kd_weight_decay_c100.yaml}"

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    echo "==> $cfg seed=$seed"
    python main.py --config "$cfg" --seed "$seed" --device "$DEVICE" --output_dir "$OUTPUT_DIR" \
      --overrides method.lambda_distill="$KD_LAMBDA"
  done
done
