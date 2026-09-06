# Scripts

Development benches, training entrypoints, and diagnostics. Present in the
source tree; absent from the wheel. First contact with the API:
[`examples/`](../examples/).

Configs live under [`configs/`](../configs/) (`train/`, `bench/`, `cells/`).

## Supported (paper-adjacent)

| Script | Config | What |
|---|---|---|
| [`slstm_vs_flashrnn.py`](slstm_vs_flashrnn.py) | [`configs/bench/newton_slstm_flashrnn.yaml`](../configs/bench/newton_slstm_flashrnn.yaml) | Diag-sLSTM Newton vs FlashRNN forward timing |
| [`bench_gru_head.py`](bench_gru_head.py) | CLI flags | `ParaGRU(mix='head')` fused vs eager vs sequential; `--d-head-grid` |
| [`bench_cfc.py`](bench_cfc.py) | CLI flags | `ParaCfC` fused / triton / eager vs sequential; residual vs K; `--smoke` |
| [`bench_hopfield.py`](bench_hopfield.py) | CLI flags | `ParaHopfield` dense `scan_dense` triton/eager vs sequential; `--d-h` / `--smoke` |
| [`bench_k_star.py`](bench_k_star.py) | CLI flags | Critical Newton depth K*(T) + H1/H2/H3/H0 fits; `--cell cfc\|hopfield\|titans\|rwkv7\|all` |
| [`bench_slstm_head.py`](bench_slstm_head.py) | CLI flags | `ParaSLSTM(mix='head')` fused/stream tiers vs seq; T asymptotics |
| [`bench_m2rnn.py`](bench_m2rnn.py) | CLI flags | `ParaM2RNN` fused / eager / sequential latency |
| [`bench_m2rnn_k_scale.py`](bench_m2rnn_k_scale.py) | CLI flags | Critical Newton depth K*(T) + asymptotics (`--init both`) |
| [`train_babylm.py`](train_babylm.py) | [`configs/train/babylm.yaml`](../configs/train/babylm.yaml) | BabyLM LM train with MLflow |
| [`bench_time.py`](bench_time.py) | [`configs/bench/newton_fused.yaml`](../configs/bench/newton_fused.yaml) | Fused vs eager Newton wall time |
| [`run_multiseed_benches.sh`](run_multiseed_benches.sh) | multiseed YAML under `configs/bench/` | Seed sweeps → `outputs/multiseed/` |

Extras: `uv sync --extra flashrnn` for FlashRNN arms; `uv sync --extra lm` for BabyLM data deps; `uv sync --extra train` / `--group dev` for MLflow.

README proof figures (from documented lab numbers):

```bash
uv run python scripts/plot_readme_assets.py
```

```bash
uv run python scripts/slstm_vs_flashrnn.py --config configs/bench/newton_slstm_flashrnn.yaml
uv run python scripts/train_babylm.py --config configs/train/babylm.yaml
```

## Training helpers

| Script | Role |
|---|---|
| [`prepare_babylm.py`](prepare_babylm.py) / [`babylm_data.py`](babylm_data.py) / [`babylm_model.py`](babylm_model.py) | Dataset + tiny LM around `ParaRNN` |
| [`distill_babylm_layer.py`](distill_babylm_layer.py) | Layer distill ([`configs/train/babylm_distill.yaml`](../configs/train/babylm_distill.yaml)) |
| [`run_babylm_suite.sh`](run_babylm_suite.sh) | Suite wrapper |

## Diagnostics (lab)

One-off probes: `diag_*.py`, `eval_*_diag.py`, `profile_hotpath.py`,
`compare_naive.py`, `bench_slstm_*.py`, `bench_packed_vjp.py`,
`bench_hybrid_pk.py`, `bench_newton_patch.py`, `seq_parallel_ranks.py`.
Useful when chasing a residual or scan bug. Lab-only; no public API promise.

## Utilities

| Path | Role |
|---|---|
| [`gpu.py`](gpu.py) | Device pick + logging; CI GPU identity |
| [`utils/mlflow_helper.py`](utils/mlflow_helper.py) | Run tags, lock hash, git commit |
| [`utils/cuda_timing.py`](utils/cuda_timing.py) | CUDA median timers |
| [`utils/flashrnn_glue.py`](utils/flashrnn_glue.py) | FlashRNN wrapper for benches |
| [`aggregate_bench_seeds.py`](aggregate_bench_seeds.py) | Multiseed CSV → summary |
| [`fetch_papers.sh`](fetch_papers.sh) | Local PDF cache under `docs/sources/` (gitignored) |
