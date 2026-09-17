#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-42 123 456}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
CONFIGS="${CONFIGS:-configs/geomctrl_cosorth_c100.yaml configs/geomctrl_proto_decor_c100.yaml configs/geomctrl_spectral_c100.yaml configs/geomctrl_center_loss_c100.yaml configs/geomctrl_supcon_c100.yaml configs/geomctrl_kd_weight_decay_c100.yaml}"

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    echo "==> $cfg seed=$seed"
    python main.py --config "$cfg" --seed "$seed" --device "$DEVICE" --output_dir "$OUTPUT_DIR"
  done
done
