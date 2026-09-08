# ParaM2RNN

**Fidelity:** research-variant / paper-core  
**Sources:** Mishra et al., *M²RNN*, arXiv:2603.14360  
**Upstream:** Full Mishra block (conv + matrix RNN + readout)  
**Code:** `pararnn.cells.para_m2rnn.ParaM2RNN`

## Diff (upstream → Para-native)

| Component | Upstream (paper block) | Para-native | Kind | Rationale |
|-----------|------------------------|-------------|------|-----------|
| Cell | Matrix recurrence + surrounding block | Matrix recurrence only | para | Newton brick = core |
| Projections | Conv + SiLU \(q,k,v\) | Single `Linear` → \(k,v,f\) | para | Outer stack owns conv |
| Readout | \(H^\top q\), \(W_o\) | Outside cell | para | Same |
| \(W\) | Learned | Default **identity** init | numerics | Paper-friendly start |
| Pack | — | `k \| v \| f_logit` contiguous | para | `project` slices |

## Strict Spec Contract

```yaml
Cell: ParaM2RNN
Fidelity: research-variant / paper-core
Inputs:
  x_step: "[B, d_in]"
  x_train: "[B, T, d_in]"
State:
  layout_step: "[B, K, V]"
  layout_train: "[B, T, K, V]"
  d_h_alias: "K*V"   # flat width only; state stays matrix
Shapes:
  H: "[B, K, V]"           # rows=K, cols=V; right-multiply by W
  W: "[V, V]"
  k: "[B, K]"
  v: "[B, V]"
  f_logit: "[B]"
  f: "[B]" -> broadcast "[B, 1, 1]" over H
  Z: "[B, K, V]"           # tanh(H@W + k.unsqueeze(-1)*v.unsqueeze(-2))
  kv_outer: "[B, K, V]"    # k[..., :, None] * v[..., None, :]
Broadcast:
  - "f.unsqueeze(-1).unsqueeze(-1) before f*H and (1-f)*Z"
  - "H @ W: (B,K,V) @ (V,V) -> (B,K,V)"
Parameters:
  W_x: "Linear(d_in, K+V+1, bias=True)"
  W_x_slices: "[0:K]=k, [K:K+V]=v, [K+V]=f_logit"
  W: "Parameter(V, V)"
  init:
    W: identity
    W_x.weight: xavier_uniform
    W_x.bias: zeros
Jacobian:
  structure: m2rnn
  storage: "factorized map on K×V; never materialize (KV)²"
  formula: "J[Δ]=f*Δ+(1-f)*(1-Z**2)⊙(Δ@W); Δ same shape as H"
Newton:
  residual: "F = f(H_prev,x) - H"   # pred - states
  linear_recurrence: "delta_t = J_t[delta_{t-1}] + F_t"
  solve: "factorized inclusive walk in T via m2rnn_jvp (NOT Sylvester/CG/linalg.solve)"
  solver: newton_m2rnn_factorized
  picard_optional: "m2rnn_frozen_w_scan (W=0 ax+b warm-start)"
Agreement:
  note: "prefer small K,V; product flat-RNN τ in numerics_contract.md"
  FP32_reference: { atol: 1.0e-4, rtol: 0.0 }
Weight_bridge: distinct
```

## Shapes & broadcasting

| Symbol | Shape | Notes |
|--------|-------|-------|
| \(H,H_t,\Delta,Z,F,\delta\) | `(B, K, V)` train also `(B, T, K, V)` | **Rows \(K\), cols \(V\)** |
| \(W\) | `(V, V)` | Right-multiply: `H @ W` |
| \(k\) | `(B, K)` | |
| \(v\) | `(B, V)` | |
| \(f\) | `(B,)` → `(B, 1, 1)` | Scalar per batch (and per \(t\) in train) |
| \(k v^\top\) | `(B, K, V)` | `k.unsqueeze(-1) * v.unsqueeze(-2)` |

Layout is **not** `(B, V, K)`. Row-major flatten of \(H\) (for dense-oracle compare) matches PyTorch `reshape(B, K*V)`.

## Recurrence

\[
H_t = f_t\, H_{t-1} + (1-f_t)\,\tanh(H_{t-1} W + k_t v_t^\top),\quad f=\sigma(f_{\mathrm{logit}}).
\]

## Jacobian class

**Dispatch:** `jac_structure='m2rnn'` → `newton_m2rnn_factorized`.  
**Storage:** structured map on matrices \(K\times V\); no dense \((KV)^2\).  
**Operator** (\(Z=\tanh(HW+kv^\top)\); \(k,v,f\) input-only):

\[
J[\Delta]
=
f\,\Delta
+
(1-f)\,(1-Z^{\odot 2})\odot(\Delta W).
\]

Same shapes as \(H\). Cost \(O(KV^2)\) per matvec. Adjoint:
\(J^\top[\mu]=f\mu+[(1-f)(1-Z^{\odot2})\odot\mu]W^\top\).

**Apply vs solve:** \(J[\Delta]\) is a **matvec**. The Newton step never solves
the Sylvester-looking equation \(f\Delta+(1-f)(1-Z^{\odot2})\odot(\Delta W)=R\)
in isolation (no Schur / Neumann / Richardson / CG). It only evaluates
\(J[\cdot]\) inside the time recurrence below.

## Parallel path

1. Residual \(F_t = f(H_{t-1},x_t)-H_t\) with \(F\) shape `(B, T, K, V)`.
2. Linear recurrence \(\delta_t = J_t[\delta_{t-1}] + F_t\) (factorized inclusive
   walk in \(T\): `m2rnn_jvp`).
3. \(H\leftarrow H+\omega\delta\).

Optional Picard: frozen \(W{=}0\) associative warm-start (`m2rnn_frozen_w_scan`).
Measure \(K\) budgets per \((K,V)\). Hub: [Newton + scan](../../core/newton_scan.md).

## Reproduce (sequential)

```python
import torch
from torch import nn

d_in, K, V = 32, 8, 8
W_x = nn.Linear(d_in, K + V + 1)
W = torch.eye(V)  # (V, V)

def step(H, x):
    # H: (B, K, V); x: (B, d_in)
    raw = W_x(x)                              # (B, K+V+1)
    k, v = raw[..., :K], raw[..., K : K + V]  # (B,K), (B,V)
    f = torch.sigmoid(raw[..., -1])           # (B,)
    f = f.unsqueeze(-1).unsqueeze(-1)         # (B, 1, 1)
    Z = torch.tanh(H @ W + k.unsqueeze(-1) * v.unsqueeze(-2))  # (B,K,V)
    return f * H + (1.0 - f) * Z

H = torch.zeros(2, K, V)
H = step(H, torch.randn(2, d_in))
```

## Agreement check

Matrix state needs a slightly higher \(K\) pin than the GRU/LSTM default of 3
(measured: \(K{=}5\) passes fp32 `atol=1e-4` on this shape).

```python
import torch
from pararnn import ParaM2RNN, ParaRNN, NewtonConfig, verify_agreement

cfg = NewtonConfig(max_iters=5)
m = ParaRNN(ParaM2RNN(d_in=32, k_dim=8, v_dim=8), config=cfg)
x = torch.randn(2, 32, 32)  # (B, T, d_in); state is (B, T, K, V)
res = verify_agreement(m, x, atol=1e-4, config=cfg)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

([numerics contract](../../core/numerics_contract.md)).
