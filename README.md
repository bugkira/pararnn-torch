# ParaRNN

Independent PyTorch implementation of **ParaRNN** (Danieli et al., ICLR 2026 Oral): parallel training of *nonlinear* RNNs by treating the hidden-state trajectory as a root-finding problem and solving it with Newton iterations plus an associative scan.

This is **not** Apple’s library. The algorithm comes from [arXiv:2510.21450](https://arxiv.org/abs/2510.21450). The code in `src/` is original. The official CUDA package is [`apple/ml-pararnn`](https://github.com/apple/ml-pararnn) (Apple license, not MIT).

## Status

Working today: a **sequence module** you can train.

- **`ParaRNN`**: `nn.Module`, batch-first `(B, T, d_in)`. `.train()` runs Newton+scan (Alg. 1); `.eval()` unrolls `step`. Optional Triton fused Newton on CUDA.
- Diagonal **ParaGRU** and CIFG peephole **ParaLSTM** (paper §3), plus **any** ``step(h, x)`` cell: Autograd Jacobian (DEER). Default is analytic J when the cell provides it. Cell contract: `d_h` + `step` (`pararnn.cells.protocol`).
- Sequential unroll (eager oracle) and **Newton + Blelloch scan** (Alg. 1, \(K=3\); 2×2 blocks are elementwise, not ``bmm``). Mixing cells: ``jac_structure="dense"``.
- Optional ``NewtonConfig(scan_backend="triton")`` for the **diag and 2×2** scans, or ``"fused"`` for **cell + J + scan** in Triton (CUDA float16/float32; algebra fp32; ParaGRU/LSTM). Eager is the default. ``W_x(x)`` stays a PyTorch GEMM. Not Apple's fused CUDA. On this 2080 Ti at \(T=2048\) (10/50 smoke, not App. B, not 665×): fused GRU **2.4 ms** vs naive ParaRNN 27 ms (**11×**) vs compiled sequential 812 ms (**339×**) vs naive RNN 959 ms (**400×**); LSTM **6.5 ms** vs 62 ms (**9.6×**) vs 865 ms (**134×**) vs 1338 ms (**207×**). fp16 fused (same shapes, paired run): GRU **1.80 vs 2.61 ms** fp32 (**1.45×**), **66 vs 123 MiB**; LSTM **5.45 vs 6.41 ms** (**1.18×**). Not bf16. Tables: [`docs/bottlenecks.md`](docs/bottlenecks.md#fused-newton).
- Backward through Newton is **eq. 2.6 reverse scan**, not autograd through the \(K\) iterates. ParaGRU/ParaLSTM pack the cell VJP in Triton on CUDA (``W_x`` still one GEMM). Custom cells VJP through Autograd on ``step``. IFT is still ours-later.
- Numerics tests: analytic Jacobians vs autograd; parallel vs sequential; Newton grads vs sequential BPTT; custom diag/dense cells; packed VJP vs Autograd; Triton diag/2×2 scans and fused Newton vs substitution / sequential; **fp16 fused vs sequential separately** (atol \(2\times10^{-3}\); fused rejects bf16); wrapper train/eval vs raw solvers.
- Toy copy smoke: `uv run python -m pararnn.train.toy` (MLflow experiment `toy-copy`). Not SlimPajama.

`torch.compile` is **not** in `src/`. The library default is eager. A bench can wrap `newton_apply` at the call site (`configs/bench/newton_compile.yaml`). On this 2080 Ti that is real after warmup (GRU short \(T\) tens of ×; LSTM \(T=2048\) about 2.6×), but Dynamo is **4–113 s per new \(T\)**. LSTM at LM length needs ~2–3k forwards to break even. Do not quote those × as if compile were free. Numbers: [`docs/bottlenecks.md`](docs/bottlenecks.md#compile-newton).

Not implemented: Hugging Face models, Mamba warm-start, IFT adjoint, pretrained checkpoints.

## Why nonlinear RNNs

Transformers train in parallel but pay \(O(T^2)\) compute and \(O(T)\) KV-cache at decode. Linear SSMs (Mamba, Mamba-2) scan in \(O(T\log T)\) by forcing \(h_t = A_t h_{t-1} + B_t x_t\). That linearity is a real expressivity limit (Merrill et al., 2024: SSMs stay in \(\mathsf{TC}^0\); they do not get RNN-style state tracking).

ParaRNN keeps a nonlinear cell \(h_t = f(h_{t-1}, x_t)\) and still parallelizes training: stack the constraints \(F(H)_t = h_t - f(h_{t-1}, x_t) = 0\) and Newton-solve the block-bidiagonal Jacobian with a scan. Inference is the usual \(O(1)\)-state unroll; the solver is training-only.

The paper’s **665×** figure is versus a *naive sequential* cell, not versus Mamba. Fused CUDA ParaGRU in the paper is about **2.6×** vs Mamba at \(L=2^9\) (§5.1). This repo is the eager PyTorch prototype; those kernel numbers are Apple’s. Our `torch.compile` numbers are also not that figure: Dynamo+CUDA graphs on this card, after a long compile, versus *our* eager Newton.

## Method (as implemented)

Paper 1-based \(l=1\ldots L\), \(h_0=0\). Code is 0-based batch-first; see [`src/pararnn/layout.py`](src/pararnn/layout.py) for the index map (eq. 2.3 vs 2.4 and Alg. 2 HTML disagree — we follow 2.1–2.3).

Newton initial guess is **not** the zero trajectory. Appendix A:

\[
h_l^{(0)} = f(0, x_l) \qquad \text{for all } l \text{ in parallel.}
\]

Then \(K=3\) Newton steps (App. A: residual to machine precision in 3–4 iterations for these cells). Recurrent diagonals \(a_\star\) use Xavier-Gaussian init and are clipped to \(0.5\) (App. C.1, LM recipe). Input maps \(B_\star\) are Kaiming-uniform; biases are zero.

Jacobians must be diagonal (GRU) or \(2\times 2\) block-diagonal per channel (LSTM). That is a modeling choice, not a free lunch: channels do not mix inside the cell (eq. 3.3); mixing is left to later layers.

LSTM state is \((c, h)\) as in the paper. `torch.nn.LSTM` stores the hidden tuple as `(h, c)` — inverted.

## Install

Python 3.10+, [uv](https://docs.astral.sh/uv/), CUDA PyTorch (cu128 index in `pyproject.toml`). Distribution name **`pararnn-torch`**, import **`pararnn`**.

```bash
git clone <this-repo>
cd ParaRNN
uv sync --group dev
uv run pytest -q
```

The PyTorch extra is the cu128 index in `pyproject.toml`, not a `requirements.txt`. On a machine with several GPUs, `pararnn.device.experiment_device()` selects by **name** (default `"2080 Ti"`), not `cuda:0`. Override with `PARARNN_DEVICE=cuda:1`. Turing cards have no bf16 tensor cores; tests run float32 and fp16 (not bf16).

## Quickstart

```python
import torch
from pararnn import ParaGRU, ParaRNN, NewtonConfig

device = torch.device("cpu")  # or pararnn.device.experiment_device()
cell = ParaGRU(d_in=32, d_h=64)
model = ParaRNN(cell, config=NewtonConfig(max_iters=3)).to(device)
x = torch.randn(4, 128, 32, device=device)

model.train()  # Newton + scan (Alg. 1)
y_train = model(x)

model.eval()   # sequential unroll of cell.step
y_eval = model(x)
# y_train ≈ y_eval  (see tests/numerics/)
```

`ParaLSTM` uses the same wrapper. State shape is `(batch, time, 2, d_h)`. Stacking (`num_layers>1`) is naive — no residual or LayerNorm in `ParaRNN`.

Solvers are still callable on a cell: `newton_apply(cell, x)`, `sequential_apply(cell, x)`. Fused/fp16 are `NewtonConfig(scan_backend="fused")` on CUDA (performance, not the API). Any `nn.Module` with `step(h, x)` and `d_h` works; mixing hidden channels needs `jac_structure="dense"`. Nonzero `h0` with fused falls back to eager (kernels prepend zeros).

## Trade-offs

- Fixed-size \(h_t\) will not match attention on needle-in-a-haystack over 100k tokens.
- Nonlinear recurrence is the theoretically motivated tool for automata / parity / nested state (Merrill et al., 2024). That is a hypothesis to test, not a benchmark result from this repo.
- Audio DSP (NAM, Wright-style LSTM amps) already runs sequential LSTM in real time. ParaRNN would help *training* long takes, not a free quality win.

## Roadmap

- [x] PyTorch ParaGRU / ParaLSTM + Newton scan, sequential agreement tests
- [x] Any ``step(h, x)`` via Autograd Jacobian; dense scan for mixing cells
- [x] Reverse-scan backward (eq. 2.6); ParaGRU/LSTM cell VJP packed in Triton on CUDA
- [x] Triton / CUDA parallel reduction — diag + 2×2 scans and fused Newton (`kernels/`, opt-in `scan_backend="triton"` / `"fused"`). Not Apple's kernels.
- [x] Sequence module `ParaRNN` (`.train()` Newton, `.eval()` sequential) + toy copy MLflow smoke
- [ ] Linear SSM (Mamba-2) predictor + one Newton corrector (this repo’s idea, not in the paper)
- [ ] IFT adjoint (Bai et al., DEQ 2019) — optional extra; training backward today is eq. 2.6
- [ ] Hugging Face `PreTrainedModel` once a cell+solver stack actually trains
- [ ] SlimPajama baselines at small scale
- [ ] DSP / syntactic-state evals

## References

- Danieli, Rodríguez, Sarabia, Suau, Zappella. *ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models*. ICLR 2026 (Oral). [arXiv:2510.21450](https://arxiv.org/abs/2510.21450). Code: [apple/ml-pararnn](https://github.com/apple/ml-pararnn).
- Danieli et al. *DeepPCR*. NeurIPS 2023. [arXiv:2309.16318](https://arxiv.org/abs/2309.16318).
- Lim et al. *DEER*. ICLR 2024. [arXiv:2309.12252](https://arxiv.org/abs/2309.12252).
- Gonzalez et al. *Towards Scalable and Stable Parallelization of Nonlinear RNNs*. [arXiv:2407.19115](https://arxiv.org/abs/2407.19115).
- Gu & Dao. *Mamba*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). Dao & Gu. *Mamba-2*. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060).
- Merrill et al. *The Illusion of State in State-Space Models*. [arXiv:2404.08819](https://arxiv.org/abs/2404.08819).
- Bai, Kolter, Koltun. *Deep Equilibrium Models*. NeurIPS 2019. [arXiv:1909.01377](https://arxiv.org/abs/1909.01377).

Annotated bibliography: [`docs/literature.md`](docs/literature.md). Notes on the Apple tree: [`docs/apple-ml-pararnn.md`](docs/apple-ml-pararnn.md). Eager-scan bottlenecks and ranked fixes: [`docs/bottlenecks.md`](docs/bottlenecks.md).

## License

Code in this repository (`src/`, `tests/`, `configs/`) is MIT — see [LICENSE](LICENSE).

Do not copy files from `third_party/ml-pararnn`; that tree is Apple’s license and is gitignored. Re-clone with [`third_party/README.md`](third_party/README.md).
