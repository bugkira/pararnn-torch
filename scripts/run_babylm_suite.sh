#!/usr/bin/env bash
# BabyLM-10M mixing ablation on the RTX 3060.
# Smoke 10 steps per arm, then one epoch each.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG="${CFG:-configs/train/babylm.yaml}"
LOGDIR="${LOGDIR:-results}"
mkdir -p "$LOGDIR" checkpoints/babylm

uv run --extra lm python scripts/prepare_babylm.py --config "$CFG"

for cell in dense diag_seq diag_fused; do
  echo "=== smoke ${cell} 10 steps ==="
  uv run --extra lm python scripts/train_babylm.py \
    --config "$CFG" --cell_type "$cell" --max_steps 10 \
    2>&1 | tee "$LOGDIR/babylm_smoke_${cell}.log"
done

for cell in dense diag_seq diag_fused; do
  echo "=== full ${cell} ==="
  uv run --extra lm python scripts/train_babylm.py \
    --config "$CFG" --cell_type "$cell" \
    2>&1 | tee "$LOGDIR/babylm_${cell}.log"
done

uv run --extra lm python scripts/train_babylm.py --config "$CFG" --summarize
echo "wrote results/babylm_ablation.md"
