# Layout

One trunk (`main`). Layers:

1. **Ops** — `src/pararnn/kernels/`: Triton fused Newton and scans, selected with `NewtonConfig(scan_backend=)`.
2. **Modules** — `cells/` is $f$ (`step`); `solvers/` is Alg. 1; `layers/ParaRNN` is the sequence `nn.Module` (Newton in `.train()`, sequential in `.eval()`).
3. **Integrations** — `examples/` (onboarding), `notebooks/` (Colab demo).

Maintainer benches, verification suites, and paper drafts live in the sibling
`pararnn-lab` tree.

```
pararnn-torch/
├── docs/                       # public API / adoption docs
├── src/pararnn/
│   ├── cells/
│   ├── layers/
│   ├── models/
│   ├── serve/
│   ├── vllm_plugin/
│   ├── solvers/
│   ├── kernels/
│   ├── distributed.py
│   ├── tensor_parallel.py
│   ├── speculative.py
│   ├── paged.py
│   ├── layout.py
│   └── weight_init.py
├── examples/
├── configs/
├── notebooks/
├── pyproject.toml              # package name pararnn-torch
└── README.md
```

## Current surface (0.17)

`NewtonConfig(scan_backend="auto")`; fused kernels prepend `h0`; `NewtonStats`
and residual early-stop (`NewtonDivergenceError` if max|F|>1 after K).
Compile-safe preset: fixed-K Autograd path, fused Alg. 1 as
`pararnn::newton_*_fused` custom ops. `ParaRNN` takes a cell or a list,
`return_hidden`, LSTM `output_hidden`. **ParaSLSTM** `mix='diag'` is the fused
4×4 path; `mix='head'` is Beck per-head dense `R` with factorized CUDA Newton
/ reverse / packed VJP (`d_head≤32` fused, `≤128` streamed-`R`; `eager`
dense-J oracle); `mix='dense'` is a `d_h<=8` test oracle. **ParaGRU**
`mix='diag'` is fused; `mix='head'` is Dreamer-style block-diagonal Cho-GRU
(CUDA factorized Newton; `scan_backend='eager'` dense-J oracle; LN for Dreamer
LN-GRU is outside the cell). **ParaM2RNN** is a matrix-state research cell
($`H\in\mathbb{R}^{K\times V}`$) with factorized Newton (SRAM `K,V≤64`, hybrid
tiled above) and packed VJP. Zoo also: CfC, Hopfield, RWKV-7, Titans, NLRU —
see Models in the README.

Examples on `main`: CausalLM smoke, continuous batch, Dyck-1, Z₂ parity,
NX-AI `sLSTMBlock` hybrid, DDP/FSDP2, linear-draft verify, T=1 `decode_step`.
Architecture notes: [`docs/distributed.md`](distributed.md).

## Naming

- PyPI / project: `pararnn-torch`. Import: `pararnn`.
- Cells follow the paper: `ParaGRU`, `ParaLSTM`, `ParaSLSTM`, plus research `ParaM2RNN`.
- Docs: lowercase kebab-case.
- Scan APIs are `scan_block2` / `scan_block4` for the `(B,T,2,2,d)` / `(B,T,4,4,d)` layouts; files are named by cell (`scan_lstm_block`, `scan_slstm_block`).
