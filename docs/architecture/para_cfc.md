# ParaCfC

**Fidelity:** research-variant  
**Sources:** Hasani et al., *Closed-form continuous-time neural networks*, Nat. Mach. Intell. 2022 / arXiv:2106.13898, eq. (10) and Fig. 4; reference cell `ncps.torch.CfCCell` (Apache-2.0).  
**Code:** `pararnn.cells.para_cfc.ParaCfC`

## State

- Single slot \(h_t \in \mathbb{R}^{d_h}\).
- `state_slots = 1`.
- Input layout: features in \(x[..., :-1]\), positive \(\Delta t\) in \(x[..., -1]\) (`d_in >= 2`). Clamped at \(10^{-4}\).

## Recurrence (library)

One `Linear` on features maps to \((f_{\mathrm{pre}}, c_x)\) with width \(2 d_h\). Diagonal recurrent vector \(u\in\mathbb{R}^{d_h}\) (App. C.1 clip, default `max_recurrent_norm=0.5`):

\[
\begin{aligned}
a_t &= \sigma\bigl(-\mathrm{softplus}(f_{\mathrm{pre}}(x_t))\,\Delta t_t\bigr), \\
n_t &= \tanh\bigl(c_x(x_t) + u \odot h_{t-1}\bigr), \\
h_t &= a_t \odot h_{t-1} + (1-a_t)\odot n_t.
\end{aligned}
\]

**`gate_mix='diag_h'`** (optional). Liquid rate also mixes previous state with a second diagonal vector \(v\) (same clip):

\[
a_t = \sigma\bigl(-\mathrm{softplus}(f_{\mathrm{pre}}(x_t) + v \odot h_{t-1})\,\Delta t_t\bigr).
\]

Jacobian stays channelwise diagonal
\(J = a + (1-a)\odot(1-n^{\odot 2})\odot u + (\partial a/\partial h)\odot(h-n)\).
Fused Triton covers `gate_mix='input'` only; `diag_h` uses triton/eager scan + Autograd VJP.

`project_wx` packs \((f_{\mathrm{pre}}, c_x, \Delta t)\) as shape `(..., 3 d_h)` for the fused path.

## Paper / ncps reference (for comparison)

Hasani et al. eq. (10) (vector form; backbone + heads \(f,g,h\)):

\[
\mathbf{x}(t)
=
\sigma\bigl(-f(\mathbf{x},\mathbf{I};\theta_f)\,t\bigr)\odot g(\mathbf{x},\mathbf{I};\theta_g)
+
\bigl[1-\sigma\bigl(-f(\mathbf{x},\mathbf{I};\theta_f)\,t\bigr)\bigr]\odot h(\mathbf{x},\mathbf{I};\theta_h).
\]

`ncps` **default** mode implements a related discrete step with separate timespan `ts`: backbone on \(\mathrm{concat}(I_t, h_{t-1})\), heads `ff1`/`ff2` (tanh), affine time gate \(\sigma(t_a\cdot\mathrm{ts}+t_b)\), then

\[
h_t = \mathrm{ff1}\odot(1-t_{\mathrm{interp}}) + t_{\mathrm{interp}}\odot\mathrm{ff2}.
\]

`ncps` **pure** mode tracks the exponential closed form (eq. 9 lineage). Mixed-memory wraps an LSTM around the cell.

## Jacobian class

`jac_structure='diag'`. Analytic channelwise Jacobian used by Newton:

\[
J_t = a_t + (1-a_t)\odot(1-n_t^{\odot 2})\odot u.
\]

## Parallel path

Newton + diagonal scan (Alg. 1). Default App. A guess \(h_t^{(0)}=f(0,x_t)\).  
`NewtonConfig(max_iters=None)` → `cfc_auto_newton_iters`: measured **H1** \(O(1)\) envelope, recipe `{1: 3}` (lab \(K^*=2\) through \(T=131072\), ceiling 3). Pin `max_iters=int` or `newton_iters_by_t` to override. Fused op: `pararnn::newton_cfc_fused`.

## Deviations

Intentional design for a **diag-Newton brick** with irregular \(\Delta t\):

- **Update target.** Library mixes previous state \(h_{t-1}\) with a candidate \(n_t\). Paper / ncps default mix two heads \(g\) and \(h\) (both functions of backbone features); previous \(h\) enters the ncps cell through the backbone input concat.
- **Time gate.** Library uses \(\sigma(-\mathrm{softplus}(f_{\mathrm{pre}})\,\Delta t)\) with \(\Delta t\) as a data channel. Paper writes \(\sigma(-f\,t)\) with absolute / sample time \(t\). ncps default uses \(\sigma(t_a\,\mathrm{ts}+t_b)\).
- **Heads.** Single linear on features plus diagonal \(u\). Paper Fig. 4 uses a shared backbone branching into \(f,g,h\); ncps mirrors that with `ff1`/`ff2`/`time_*`.
- **Scope.** This module is the recurrence only. NCP wirings and CfC-mmRNN live in outer stacks.
- **softplus.** Forces a positive liquid rate before multiplying \(\Delta t\).

Treat `ParaCfC` as a Liquid-style parallelizable cell. Bit-matching Hasani eq. (10) or `ncps.torch.CfC` needs the full backbone+heads stack and a matching time API.

## Reproduce (sequential)

```python
import torch
from pararnn import ParaCfC

cell = ParaCfC(d_in=9, d_h=16)  # 8 features + Δt
h = torch.zeros(2, 16)
feat = torch.randn(2, 8)
dt = 0.05 + torch.rand(2, 1)
h = cell.step(h, torch.cat((feat, dt), dim=-1))
```

## Agreement

```python
from pararnn import verify_agreement, ParaCfC, ParaRNN
import torch

m = ParaRNN(ParaCfC(9, 16))
feat = torch.randn(2, 64, 8)
dt = 0.05 + torch.rand(2, 64, 1)
print(verify_agreement(m, torch.cat((feat, dt), dim=-1)).to_dict())
```

Lab: `tests/numerics/test_cfc.py`. Systems smoke: lab `scripts/bench_cfc.py`. See [numerics contract](../numerics-contract.md).
