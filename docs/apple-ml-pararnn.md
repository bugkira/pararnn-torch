# Official Apple repo (`apple/ml-pararnn`)

Local clone: `third_party/ml-pararnn` (gitignored). Re-fetch: see `third_party/README.md`.

## License (blocking)

Custom Apple license, **not MIT**. Personal, non-exclusive; keep their notice if redistributing unmodified; no Apple trademarks; AS IS. We reimplement from the paper into `src/` and do not vendor their sources.

## What the package actually is

A **cell + solver library**, not a Hugging Face LM. Four application modes:

| Mode | Role |
|---|---|
| `sequential` | Ground-truth unroll; inference |
| `parallel` | Newton + PCR in PyTorch (prototyping) |
| `parallel_CUDA` | Jacobian in PyTorch, PCR in CUDA |
| `parallel_FUSED` | Whole Newton routine in CUDA |

Our `NewtonConfig(scan_backend="fused")` is a **Triton** reimplementation of paper Alg. 1 (cell + J + scan; `W_x` still a PyTorch GEMM). It is not a port of their `csrc/fused_*.cu`.

CUDA is **required** to install their package (`pip install -e . --no-build-isolation`). Our v0 is PyTorch-only so it can run on CPU.

## Layout (their tree)

```
pararnn/
  rnn_cell/           # BaseRNNCell, impl mixins, sequential/parallel/cuda/fused
    gru_diag_mh.py    # ParaGRU: diagonal A, multi-head B
    lstm_cifg_diag_mh.py  # ParaLSTM: CIFG + peephole, state = [c, h], 2x2 blocks
  parallel_reduction/ # NewtonConfig, ParallelSolve
  csrc/               # fused GRU/LSTM + generic PCR kernels
  utils/              # inits, nonlinearities, timing
```

API: implement `recurrence_step`; flag Jacobian as dense / diag / block-diag. Autograd builds J for the PyTorch path.

## Defaults we will cite (not copy)

From their `NewtonConfig` and paper App. A/C:

- `max_its = 3` — paper App. A: residual to machine precision in 3–4 steps for ParaGRU/ParaLSTM at init and after 400M training. If a new cell does not, measure residual vs K before raising K (Gonzalez et al. 2024: worst-case K=L is useless).
- `abs_tol = rel_tol = 0` — they currently **do not** early-stop; the tolerance checks are commented out. We should log residual every iteration even if we also fix K=3.
- `omega_sor = 1` — vanilla Newton. `<1` is available for damping (Levenberg–Marquardt / ELK territory; Gonzalez et al.).
- Newton init is **not zero**: \(h_l^0 = f(0, x_l)\) for all l in parallel (paper eq. A.1). README draft claiming \(H^{(0)}=0\) is wrong.
- Cell inits: `a_init_fn="xlstm"`, `w_init_fn="xavier_uniform"`, `b_init_fn="bias_minus_linspace"`, `num_heads=2`. Paper §C.1 is the training-scale source; confirm against the paper before using these in a run.
- GRU: sigmoid update/reset, tanh candidate. LSTM: CIFG, sigmoid f/o, tanh cell and state (Greff et al. 2017 peephole+CIFG, paper eq. 3.1b).
- Backward is **one** reverse parallel reduction (paper eq. 2.6), not Newton. IFT adjoint is *our* later extra, not Apple's.

## Install friction (why this repo exists)

- No `transformers.PreTrainedModel`.
- CUDA compile on install; no CPU-only prototype path advertised.
- No MLflow / configs-as-files training stack.
- Hybrid Mamba predictor is not in this repo.

## Other implementations

- [BanaanKiamanesh/Para-Seq](https://github.com/BanaanKiamanesh/Para-Seq): DEER / quasi-DEER / ELK layers, not Apple's fused kernels. Useful for solver ablations, not a drop-in substitute.
