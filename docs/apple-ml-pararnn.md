# Official Apple repo (`apple/ml-pararnn`)

Public code: [github.com/apple/ml-pararnn](https://github.com/apple/ml-pararnn). This package reimplements [Danieli et al.](https://arxiv.org/abs/2510.21450) in PyTorch / Triton.

## License

Custom Apple license. Personal, non-exclusive; keep their notice if redistributing unmodified; no Apple trademarks; AS IS.

## Their package

A cell + solver library. Four application modes:

| Mode | Role |
|---|---|
| `sequential` | Ground-truth unroll; inference |
| `parallel` | Newton + PCR in PyTorch |
| `parallel_CUDA` | Jacobian in PyTorch, PCR in CUDA |
| `parallel_FUSED` | Whole Newton routine in CUDA |

Their install compiles CUDA (`pip install -e . --no-build-isolation`). Our fused path is Triton Alg. 1 (cell + J + scan; `W_x` is a PyTorch GEMM) for ParaGRU, ParaLSTM, and ParaSLSTM `mix='diag'`. Default `scan_backend` is `auto`.

## Layout

```
pararnn/
  rnn_cell/           # BaseRNNCell, impl mixins, sequential/parallel/cuda/fused
    gru_diag_mh.py    # ParaGRU: diagonal A, multi-head B
    lstm_cifg_diag_mh.py  # ParaLSTM: CIFG + peephole, state = [c, h]
  parallel_reduction/ # NewtonConfig, ParallelSolve
  csrc/               # fused GRU/LSTM + generic PCR kernels
  utils/              # inits, nonlinearities, timing
```

API: implement `recurrence_step`; flag Jacobian as dense / diag / block-diag. Autograd builds J for the PyTorch path.

## Defaults (paper App. A/C and their `NewtonConfig`)

- `max_its = 3` — App. A: residual to machine precision in 3–4 steps for ParaGRU/ParaLSTM at init and after 400M training. For a new cell, measure residual vs K before raising K (Gonzalez et al. 2024).
- `abs_tol = rel_tol = 0` — they currently run a fixed K; the tolerance checks are commented out.
- `omega_sor = 1` — vanilla Newton. `<1` damps (Levenberg–Marquardt / ELK; Gonzalez et al.).
- Newton init \(h_l^0 = f(0, x_l)\) for all l in parallel (paper eq. A.1).
- Cell inits: `a_init_fn="xlstm"`, `w_init_fn="xavier_uniform"`, `b_init_fn="bias_minus_linspace"`, `num_heads=2`. Paper §C.1 is the training-scale source.
- GRU: sigmoid update/reset, tanh candidate. LSTM: CIFG, sigmoid f/o, tanh cell and state (Greff et al. 2017 peephole+CIFG, paper eq. 3.1b).
- Backward is one reverse parallel reduction (paper eq. 2.6).

## Other implementations

- [BanaanKiamanesh/Para-Seq](https://github.com/BanaanKiamanesh/Para-Seq): DEER / quasi-DEER / ELK layers.
