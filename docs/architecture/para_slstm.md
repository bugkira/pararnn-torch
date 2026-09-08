# ParaSLSTM

**Fidelity:** paper-faithful for `mix='diag'` (channelwise \(R\), 4-slot sLSTM-style state); `mix='head'` / `mix='dense'` documented as Beck-style / oracle variants.  
**Sources:** Danieli et al. ParaRNN + xLSTM / sLSTM lineage (Beck et al.); library `mix` modes.  
**Code:** `pararnn.cells.para_slstm.ParaSLSTM`

## State

- Four slots, layout `(..., 4, d_h)` — cell / normalizer / hidden / bridge as in the library layout constants (`SLSTM_*` in `pararnn.layout`).
- Readout uses the hidden slot.

## Recurrence (`mix='diag'`)

sLSTM-style gates with **channelwise** recurrent mix \(R \in \mathbb{R}^{4\times d_h}\): each gate \(g\) gets \(r_g \odot h\) added to the input pre-activation. Stabilized exponential / max-gated updates follow the library `_acts_from_pre` path (normalizer \(n+\varepsilon\), `torch.maximum` ties split \(0.5/0.5\)).

Full gate algebra is in `ParaSLSTM._acts_from_pre` / `step_with_jacobian`; the public claim for observers is: **same `step` for sequential and as \(f\) inside Newton**.

## Jacobian class

- `mix='diag'` → `block4` (`(..., 4, 4, d_h)`)
- `mix='head'` → per-head dense over \(4 d_{\mathrm{head}}\)
- `mix='dense'` → full dense oracle (`d_h` capped for tests)

## Parallel path

Fused Newton+scan for `mix='diag'`. Factorized Newton for `mix='head'` on CUDA. Picard warm-start is supported on this cell (`picard_iters`). App. A-style guess; typical train pin \(K{=}3\), Picard \(P{=}3\) on BabyLM-style stacks.

## Deviations

- `mix='head'`: Beck-style dense \(R\) inside each head (factorized \(J\)), separate from the diag fused path.
- `mix='dense'`: Jacobian oracle for small \(d_h\), not a production train default.

## Reproduce (sequential)

```python
import torch
from pararnn import ParaSLSTM, ParaRNN

cell = ParaSLSTM(64, 64, mix="diag")
m = ParaRNN(cell)
y = m(torch.randn(2, 128, 64))  # train: Newton; eval(): sequential
```

For a hand oracle, call `cell.step(state, x)` in a Python loop over \(T\) and compare to `newton_apply` / `ParaRNN` train forward at fixed \(K\).

## Agreement

[Numerics contract](../numerics-contract.md). Stacking / block API: [xlstm.md](../xlstm.md).
