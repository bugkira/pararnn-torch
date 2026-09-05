#!/usr/bin/env bash
# 10-seed timing sweep on the lab 2080 Ti (paper Table 2 family).
# Each seed: 10 warmup / 50 runs; aggregate median_ms → mean±std at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/multiseed
export CUDA_DEVICE_ORDER=PCI_BUS_ID
LOG=outputs/multiseed/run.log
exec > >(tee -a "$LOG") 2>&1

echo "=== multiseed start $(date -Is) ==="

for seed in $(seq 0 9); do
  echo "=== seed=${seed} picard $(date -Is) ==="
  BENCH_SEED="${seed}" uv run python scripts/bench_time.py \
    --config configs/bench/newton_slstm_picard_multiseed.yaml

  echo "=== seed=${seed} flashrnn $(date -Is) ==="
  BENCH_SEED="${seed}" uv run --extra flashrnn python scripts/bench_time.py \
    --config configs/bench/newton_slstm_flashrnn_multiseed.yaml

  echo "=== seed=${seed} fused gru/lstm $(date -Is) ==="
  BENCH_SEED="${seed}" uv run python scripts/bench_time.py \
    --config configs/bench/newton_fused_multiseed.yaml
done

echo "=== aggregating $(date -Is) ==="
uv run python scripts/aggregate_bench_seeds.py
echo "=== multiseed done $(date -Is) ==="
