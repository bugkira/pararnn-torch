# Sequence-parallel scan on one GPU

Virtual ranks = two CUDA streams on the lab **2080 Ti** (selected by name,
never by index). Not NCCL, not two physical cards, not a weak-scaling
result.

Theory (DDP works for FlashRNN; the gap is *time* span):
[`docs/tex/seq-parallel-pararnn.tex`](tex/seq-parallel-pararnn.tex)
(`pdflatex -output-directory=outputs/tex docs/tex/seq-parallel-pararnn.tex`).
Code: `pararnn.solvers.seq_parallel.scan_diag_two_ranks`.
Script: `uv run python scripts/seq_parallel_ranks.py`.
MLflow experiment `seq-parallel-scan`, run `two-stream-virtual-ranks`.

## What was measured

`scan_diag` (Blelloch, one stream) vs two-tile local scans + Jacobian-prefix
axpy on two streams vs `sequential_prefix_two_ranks` (left scan, then right
scan *from the carry* — FlashRNN-style dependence).

Shapes: \(B=8\), \(d=256\), \(T\in\{512,2048,8192\}\), float32, 10 warmup /
50 runs, min ms. Seed \(T\). Agreement vs `scan_diag`. GPU:
`NVIDIA GeForce RTX 2080 Ti`, ~10.4 GiB free of 10.57.

## Results

2080 Ti, 2026-09-01. Max abs error over all \(T\): \(9.5\times10^{-7}\).

| T | max \|two−ref\| | scan_diag min ms | two-stream min ms | seq-prefix min ms | two/ref |
|---|---|---|---|---|---|
| 512 | 4.8e-7 | **5.91** | 16.2 | 11.2 | 2.75 |
| 2048 | 4.8e-7 | **7.43** | 20.5 | 14.6 | 2.77 |
| 8192 | 4.8e-7 | **12.0** | 26.0 | 20.2 | 2.16 |

Two streams are **slower** than one Blelloch pass. That is the honest
one-GPU outcome:

- The two-rank path does a local scan of each half *plus* a prefix-product
  scan of the right tile, then an axpy. Work is strictly more than one
  `scan_diag`.
- Both streams share one set of SMs. Overlap does not turn a memory-bound
  scan into two cards.
- `seq_prefix` cannot start the right scan until \(\delta_{m-1}\) exists, so
  it is a longer *chain* than a single scan, but still less work than
  two-stream (no extra prefix kernel). Times sit between `scan_diag` and
  two-stream, as expected.

The test that *did* pass: two-stream **matches** Blelloch to \(<10^{-6}\).
The algorithm is correct. Wall-clock speedup needs **two devices** (or a
fused kernel that keeps \(J\)-prefix in the same pass).

## What this does not show

- FlashRNN cannot DDP. It can. See the TeX note, Prop. 1.
- Multi-node Newton. Halo is \(\delta_{m-1}\) of size \(O(Bd)\) for diag
  \(J\); head mix is \(O(B\cdot 4 d_{\mathrm{head}})\) per head after packing.
- Fused `mix='head'` kernels. Still eager `scan_dense`.
- Wall-clock win vs FlashRNN train. Head mix: `examples/slstm_vs_flashrnn.py`
  `--config configs/train/dyck_vs_flashrnn_head.yaml` (Newton **8.2×**
  slower min-step than `triton_fused`; P=3 K=4; not fused).

## Still open (this pass)

- NCCL / two physical GPUs. Halo would be \(\delta_{m-1}\) only.
- Fused `mix='head'` (128×128 J on Turing SRAM).
- Eq. 2.6 reverse scan sharded the same way (same monoid, \(J^\top\)).
- Weak scaling. One card cannot show it.

