# If xLSTM took this repo

NX-AI / Beck et al. already have **mLSTM** (it scans). FlashRNN keeps
**sLSTM** sequential. The thing they would take is Newton+scan for that
sLSTM half during training — including eq. 2.6 **packed VJP** (in for
`mix='diag'`). We do not reimplement mLSTM.

Not a fork of [`NX-AI/xlstm`](https://github.com/NX-AI/xlstm). Not weight
compatible. Not xLSTM-7B.

## Takeable surface

| They keep | They would call |
|---|---|
| mLSTM, conv, gated MLP, Figure 3 blocks, checkpoints | `ParaSLSTM` / `xLSTMBlock` `mix='diag'` |
| FlashRNN at decode | `.eval()` unrolls `cell.step` (same \(O(1)\) state) |

Forward: fused cell + 4×4 J + scan on CUDA (`scan_backend='auto'`).
Backward: reverse scan + packed cell VJP (Triton elementwise + one
`W_x` GEMM). Measured vs Autograd VJP: Dyck Newton bwd **2.0×**, isolated
VJP **3.2×** (`scripts/bench_packed_vjp.py`, 2080 Ti). CE unchanged.

```python
from pararnn import NewtonConfig, xLSTMBlock

block = xLSTMBlock(d_model, mix="diag", solver="newton", config=NewtonConfig())
# .train() → fused Newton + packed VJP; .eval() → sequential step
```

`mix='head'` is their FlashRNN mixing (eager Newton, Autograd VJP, K=4).
That is a different cell, not the packed path. Do not sell 4×4 fused as
head-mix. Dyck smoke: `configs/train/dyck_vs_flashrnn_head.yaml`.

State we use is Beck: `(B, T, 4, d_h) = (c, n, m, h)`, hidden slot 3
(`pararnn.layout`).

## Still not a drop-in for their package

- **No NX-AI parameter map.** Xavier/Kaiming here is ParaRNN App. C.1,
  not their forget-bias table.
- **No Figure 3 around the cell.** LN + residual only (`xLSTMBlock`).
  Conv / up-proj / gated MLP stay in their module.
- **FlashRNN `cuda_fused`** is not a backend switch here (needs SM 8.0+).
- **bf16 fused** is compute capability ≥ 8.0 (`is_fused_dtype_supported`), not a
  missing packed VJP. SM 7.x still trains fp16/fp32.

Until a PR *they* would merge, this is a library they could vendor, not
`pip install` into xLSTM-7B.

Newton cell measurements (Picard rungs, head vs FlashRNN) live in
lab notes (`docs/internal/para-slstm.md`), not in this clone.
