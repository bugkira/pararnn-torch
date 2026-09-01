# ParaRNN

Independent PyTorch implementation of **ParaRNN** (Danieli et al., ICLR 2026 Oral): parallel training of *nonlinear* RNNs by treating the hidden-state trajectory as a root-finding problem and solving it with Newton iterations plus an associative scan.

This is **not** Apple’s library. The algorithm comes from [arXiv:2510.21450](https://arxiv.org/abs/2510.21450). The code in `src/` is original. The official CUDA package is [`apple/ml-pararnn`](https://github.com/apple/ml-pararnn) (Apple license, not MIT).

## Status

Working today: a **sequence module** you can train.

- **`ParaRNN`**: `nn.Module`, default batch-first `(B, T, d_in)` (`batch_first=False` is `nn.LSTM` time-major at the wrapper). `solver='auto'` (default): `.train()` runs Newton+scan (Alg. 1); `.eval()` unrolls `step`. `solver='newton'` / `'sequential'` force that path. Optional Triton fused Newton on CUDA.
- Diagonal **ParaGRU** and CIFG peephole **ParaLSTM** (paper §3), plus **any** ``step(h, x)`` cell: Autograd Jacobian (DEER). Default is analytic J when the cell provides it. Cell contract: `d_h` + `step` (`pararnn.cells.protocol`).
- Sequential unroll (eager oracle) and **Newton + Blelloch scan** (Alg. 1, \(K=3\); 2×2 blocks are elementwise, not ``bmm``). Mixing cells: ``jac_structure="dense"``.
- Optional ``NewtonConfig(scan_backend="triton")`` for the **diag, 2×2, and 4×4** scans, or ``"fused"`` for **cell + J + scan** in Triton (CUDA float16/float32, and **bf16 on compute capability ≥ 8.0**; algebra fp32; **ParaGRU/LSTM and ParaSLSTM ``mix='diag'``**, not any ``f``; head/dense sLSTM stay on ``step``). Default is ``"auto"``: fused on CUDA for those cells when `is_fused_dtype_supported` (fp16/fp32 any CUDA; bf16 if SM 8.0+), else Triton scan + ``step``, else eager. Explicit ``fused`` + bf16 on SM 7.x raises (capability, not a GPU name). ``W_x(x)`` stays a PyTorch GEMM. Not Apple's fused CUDA. On this 2080 Ti at \(T=2048\) (10/50 smoke, not App. B, not 665×): fused GRU **2.4 ms** vs naive ParaRNN 27 ms (**11×**) vs compiled sequential 812 ms (**339×**) vs naive RNN 959 ms (**400×**); LSTM **6.5 ms** vs 62 ms (**9.6×**) vs 865 ms (**134×**) vs 1338 ms (**207×**). fp16 fused (same shapes, paired run): GRU **1.80 vs 2.61 ms** fp32 (**1.45×**), **66 vs 123 MiB**; LSTM **5.45 vs 6.41 ms** (**1.18×**). Tables: [`docs/bottlenecks.md`](docs/bottlenecks.md#fused-newton).
- Backward through Newton is **eq. 2.6 reverse scan**, not autograd through the \(K\) iterates. ParaGRU/ParaLSTM and ParaSLSTM ``mix='diag'`` pack the cell VJP in Triton on CUDA (``W_x`` still one GEMM). Head/dense sLSTM and custom cells VJP through Autograd on ``step``. IFT is still ours-later.
- Numerics tests: analytic Jacobians vs autograd; parallel vs sequential; Newton grads vs sequential BPTT; custom diag/dense cells; packed VJP vs Autograd; Triton diag/2×2/4×4 scans and fused Newton vs substitution / sequential; **fp16 fused vs sequential separately** (atol \(2\times10^{-3}\)); fused **bf16 vs sequential on SM 8.0+**, still raises on SM 7.x; wrapper train/eval vs raw solvers (`solver=` override).
- Toy copy / Dyck-1 / Z2-parity smokes live in `examples/` (`uv run python examples/toy_copy.py`, `uv run python examples/dyck_language.py`, `uv run python examples/parity.py --config configs/train/parity_t16.yaml`). Dyck-1 train vs FlashRNN: `uv run python examples/slstm_vs_flashrnn.py` (`uv sync --extra flashrnn`). Head mix (same 1×32 as Turing FlashRNN): `--config configs/train/dyck_vs_flashrnn_head.yaml`. Not SlimPajama. Sequence-parallel scan (two CUDA streams as virtual ranks): `uv run python scripts/seq_parallel_ranks.py` — see [`docs/seq-parallel-report.md`](docs/seq-parallel-report.md) and [`docs/tex/seq-parallel-pararnn.tex`](docs/tex/seq-parallel-pararnn.tex). **DDP works for FlashRNN too**; the note is about *time* span, not “FlashRNN cannot use many GPUs”.

`torch.compile` is **not** in `src/`. A bench can wrap `newton_apply` at the call site (`configs/bench/newton_compile.yaml`). On this 2080 Ti that is real after warmup (GRU short \(T\) tens of ×; LSTM \(T=2048\) about 2.6×), but Dynamo is **4–113 s per new \(T\)**. LSTM at LM length needs ~2–3k forwards to break even. Do not quote those × as if compile were free. Numbers: [`docs/bottlenecks.md`](docs/bottlenecks.md#compile-newton).

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

The PyTorch extra is the cu128 index in `pyproject.toml`, not a `requirements.txt`. Put the module on a device like any `nn.Module` (`.to(device)` or `device=` / `dtype=` on the cell, same as `nn.Linear`). Lab benches pin a GPU **by name** in `scripts/gpu.py`; restrict visibility with `CUDA_VISIBLE_DEVICES`. Fused/Triton bf16 is **compute capability ≥ 8.0** (`torch.cuda.get_device_capability` on the tensor device), not a card name in `src/`. SM 7.x tests run float32 and fp16.

## Quickstart

```python
import torch
from pararnn import ParaGRU, ParaRNN, NewtonConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cell = ParaGRU(d_in=32, d_h=64, device=device)
model = ParaRNN(cell, config=NewtonConfig(max_iters=3))
x = torch.randn(4, 128, 32, device=device)

model.train()  # solver='auto' → Newton + scan (Alg. 1)
y_train = model(x)

model.eval()   # sequential unroll of cell.step
y_eval = model(x)
# y_train ≈ y_eval  (see tests/numerics/)
# ParaRNN(..., solver="newton") still Newton in eval(); solver="sequential" unrolls in train().
```

`ParaLSTM` / `ParaSLSTM` default **output is the hidden slot** `(B, T, d_h)` (a `Linear` on top). Full paper state `(c, h)` / `(c, n, m, h)` is `output_hidden=False`. Kernels stay paper `[c, h]`; `hidden_layout="pytorch"` swaps LSTM `h0` and `return_hidden`'s last state only. Stacking (`num_layers>1`) is naive — no residual or LayerNorm in `ParaRNN`. `xLSTMBlock` is that missing piece for sLSTM: pre-norm + residual, `backend="newton"` or `"eager"`. `ParaSLSTM(..., mix="head", n_heads=...)` is the xLSTM mixing (eager Newton, not fused); library fused path is `mix="diag"`. mLSTM is not in this repo — xLSTM already scans it; they would take our sLSTM Newton, not a second matrix cell ([`docs/xlstm.md`](docs/xlstm.md)).

Solvers are still callable on a cell: `newton_apply(cell, x)`, `sequential_apply(cell, x)`. `NewtonConfig(scan_backend="fused")` is a handwritten kernel for ParaGRU/LSTM and ParaSLSTM `mix='diag'`, not a generic `f`. Nonzero `h0` is prepended in that kernel. `NewtonStats` reports residual and K used. After K steps, `max|F| > 1` raises `NewtonDivergenceError` (disable with `residual_fail=None`). ParaSLSTM Picard P is `{1, 3, 5}` from T; do not raise K if that fires. Head-mix train at \(T\le 64\) needs **P=3** (`configs/train/dyck_vs_flashrnn_head.yaml`).

## Trade-offs

- Data-parallel SGD works for FlashRNN and for Newton. Sequence-parallel *time* is the Newton scan monoid, not “FlashRNN cannot use many GPUs”. See [`docs/tex/seq-parallel-pararnn.tex`](docs/tex/seq-parallel-pararnn.tex).
- Fixed-size \(h_t\) will not match attention on needle-in-a-haystack over 100k tokens.
- Nonlinear recurrence is the theoretically motivated tool for automata / parity / nested state (Merrill et al., 2024). This repo has a **Z2 tagging smoke** (not A5, not L=100): stacked `xLSTMBlock` `mix='diag'` last-token 1.0 at T=16 and held-out T=32; a trained S4D-Real selective SSM stays at chance on the last token. Numbers: [`docs/para-slstm.md`](docs/para-slstm.md#quality-z2-tagging).
- Audio DSP (NAM, Wright-style LSTM amps) already runs sequential LSTM in real time. ParaRNN would help *training* long takes, not a free quality win.

## Roadmap

- [x] PyTorch ParaGRU / ParaLSTM + Newton scan, sequential agreement tests
- [x] Any ``step(h, x)`` via Autograd Jacobian; dense scan for mixing cells
- [x] Reverse-scan backward (eq. 2.6); ParaGRU/LSTM cell VJP packed in Triton on CUDA
- [x] Triton / CUDA parallel reduction — diag + 2×2 + 4×4 scans and fused Newton (`kernels/`, opt-in `scan_backend="triton"` / `"fused"`). Not Apple's kernels.
- [x] Sequence module `ParaRNN` (`.train()` Newton, `.eval()` sequential) + toy copy / Dyck-1 examples
- [x] `scan_backend="auto"` (tensor device, not a lab GPU helper in `forward`); fused `h0`; NewtonStats / early-stop; list-of-cells / `return_hidden` / default hidden-slot output; `solver` / `batch_first` / `hidden_layout`
- [x] Train-diag Picard adapt: auto-P retries \{1,3,5\} if \(\max|F|>10^{-3}\) ([`docs/next.md`](docs/next.md)). Explicit `picard_iters` does not. Not Eisenstat–Walker.
- [ ] Linear SSM (Mamba-2) as a **trained cell**. Solver hybrid parked: P=5 K=1 is 24.2→20.1 ms fwd at T=2048 vs library P=3 K=3 (`scripts/bench_hybrid_pk.py`).
- [ ] IFT adjoint (Bai et al., DEQ 2019) — optional extra; training backward today is eq. 2.6
- [x] Para-sLSTM `mix='diag'` (4×4 fused Newton, auto Picard P∈{1,3,5}, fail-loud). `mix='head'` is eager Newton (`scan_dense` per head), not fused. Train smoke: `configs/train/dyck_vs_flashrnn_head.yaml` (K=4). Not xLSTM-7B.
- [x] `xLSTMBlock`: LayerNorm + residual around ParaSLSTM (`backend="newton"|"eager"`). Not NX-AI. FlashRNN stays a bench. If xLSTM took this repo it would be the sLSTM train path, not a second mLSTM ([`docs/xlstm.md`](docs/xlstm.md)).
- [x] Z2 tagging smoke (`examples/parity.py`): `mix='diag'` Newton = sequential; last-token 1.0 vs S4D-Real SSM chance. Not A5, not paper L=100.
- [ ] Hugging Face `PreTrainedModel` once a cell+solver stack actually trains
- [ ] SlimPajama baselines at small scale
- [ ] A5 / deeper Dyck / DSP

## References

- Danieli, Rodríguez, Sarabia, Suau, Zappella. *ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models*. ICLR 2026 (Oral). [arXiv:2510.21450](https://arxiv.org/abs/2510.21450). Code: [apple/ml-pararnn](https://github.com/apple/ml-pararnn).
- Danieli et al. *DeepPCR*. NeurIPS 2023. [arXiv:2309.16318](https://arxiv.org/abs/2309.16318).
- Lim et al. *DEER*. ICLR 2024. [arXiv:2309.12252](https://arxiv.org/abs/2309.12252).
- Gonzalez et al. *Towards Scalable and Stable Parallelization of Nonlinear RNNs*. [arXiv:2407.19115](https://arxiv.org/abs/2407.19115).
- Gu & Dao. *Mamba*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). Dao & Gu. *Mamba-2*. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060).
- Merrill et al. *The Illusion of State in State-Space Models*. [arXiv:2404.08819](https://arxiv.org/abs/2404.08819).
- Bai, Kolter, Koltun. *Deep Equilibrium Models*. NeurIPS 2019. [arXiv:1909.01377](https://arxiv.org/abs/1909.01377).

Annotated bibliography: [`docs/literature.md`](docs/literature.md). Notes on the Apple tree: [`docs/apple-ml-pararnn.md`](docs/apple-ml-pararnn.md). Eager-scan bottlenecks and ranked fixes: [`docs/bottlenecks.md`](docs/bottlenecks.md). xLSTM taking our sLSTM Newton: [`docs/xlstm.md`](docs/xlstm.md).

## License

Code in this repository (`src/`, `tests/`, `configs/`) is MIT — see [LICENSE](LICENSE).

Do not copy files from `third_party/ml-pararnn`; that tree is Apple’s license and is gitignored. Re-clone with [`third_party/README.md`](third_party/README.md).
