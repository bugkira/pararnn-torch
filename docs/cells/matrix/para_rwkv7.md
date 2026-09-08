# ParaRWKV7

**Fidelity:** paper-core  
**Sources:** Peng et al., *RWKV-7 “Goose”*, arXiv:2503.14456 §3  
**Upstream:** RWKV-7 matrix-state delta rule  
**Code:** `pararnn.cells.para_rwkv7.ParaRWKV7`

## Diff (upstream → Para-native)

| Component | Upstream | Para-native | Kind | Rationale |
|-----------|----------|-------------|------|-----------|
| Transition | \(S\leftarrow S G + v^\top k\) | Same factorized form | author | Linear in \(S\) |
| \(G\) | \(\mathrm{diag}(w)-\hat\kappa^\top(a\odot\hat\kappa)\) | Same DPLR | author | Never materialize in `step` |
| Stack | Full RWKV-7 block | Cell = monoid transition + readout | para | Outer owns time-mix / channel-mix |
| Parallel | Associative \((G,U)\) scan | `scan_apply` / `rwkv7_associative_scan` | numerics | Classic monoid scan |
| Newton | N/A (linear) | `jac_structure='rwkv7'` redirects | numerics | No nonlinear fixed point in \(S\) |
| Init | Vendor | Kaiming `W_x`, zero bias | numerics | Mid-sigmoid gates at start |

## Strict Spec Contract

```yaml
Cell: ParaRWKV7
Fidelity: paper-core
Inputs:
  x_step: "[B, d_in]"; x_train: "[B, T, d_in]"
State:
  layout_step: "[B, n_heads, d_head, d_head]"
  layout_train: "[B, T, n_heads, d_head, d_head]"
Shapes:
  S: "[B, H, d, d]"           # H=n_heads, d=d_head
  w,a,kappa,v,k,r: "[B, H, d]"
  G_dense: "[B, H, d, d]"     # monoid path only
  U: "[B, H, d, d]"           # outer(v,k)
  W_x.out: "[B, 6*H*d]"
Broadcast:
  - "S * w.unsqueeze(-2): scale columns of S"
  - "kappa L2-normalized on last dim, eps=1e-6"
Parameters:
  W_x: "Linear(d_in, 6*n_heads*d_head, bias=True)"
  W_x_pack: "reshape (..., 6, n_heads, d_head) -> (w,a,κ,v,k,r)"
  gates:
    w: "sigmoid(raw[0])"
    a: "sigmoid(raw[1])"
    kappa: "normalize(raw[2], dim=-1, eps=1e-6)"
    v,k,r: "raw[3], raw[4], raw[5]"
  init:
    W_x.weight: kaiming_uniform
    W_x.bias: zeros
Jacobian:
  structure: rwkv7
  storage: "linear monoid; no Newton J"
  formula: "S' = S@diag(w) - outer(S κ̂, a⊙κ̂) + outer(v,k)"
Newton:
  residual: "N/A — map affine in S"
  linear_recurrence: "S_t = S_{t-1} G_t + U_t"
  solve: "associative (G,U) monoid scan OR sequential factorized step; Newton bypass (K*=0)"
  default_K_pin: 0
  fused_op: "kernels.rwkv7_scan"
Agreement:
  FP32: { atol: 1.0e-4, rtol: 0.0 }
Weight_bridge: distinct
```

## Shapes & broadcasting

| Symbol | Shape | Notes |
|--------|-------|-------|
| \(S\) | `(B, H, d, d)` | matrix state per head |
| \(w,a,\hat\kappa,v,k,r\) | `(B, H, d)` | from pack index 0..5 |
| \(G\) | `(B, H, d, d)` | built only for parallel monoid |
| \(U=v^\top k\) | `(B, H, d, d)` | `v[...,:,None]*k[...,None,:]` |

## Recurrence

Per head, matrix state \(S\in\mathbb{R}^{d\times d}\):

\[
S_t = S_{t-1} G_t + v_t^\top k_t,\quad
G_t = \mathrm{diag}(w_t) - \hat\kappa_t^\top (a_t \odot \hat\kappa_t).
\]

Factorized `step` (no dense \(G\)):

\[
S \leftarrow S\odot w_{\mathrm{col}} - (S\hat\kappa)\,(a\odot\hat\kappa)^\top + v^\top k.
\]

Readout: \(y = \mathrm{flatten}(S\, r)\) with receptance \(r\) from \(x\).

## Jacobian class

**Dispatch:** `jac_structure='rwkv7'`.  
**Storage:** linear affine map in \(S\); solver uses associative scan / sequential.  
**Operator:** Newton is redirected — there is no nonlinear fixed-point residual in \(S\).  
All of \(w,a,\kappa,v,k,r\) are **input-only**.

## Parallel path

**No nonlinear Newton.** Affine map in \(S\):
\(S_t = S_{t-1} G_t + U_t\), \(U=v^\top k\).
**Solve:** associative compose
\((G_l,U_l)\circ(G_r,U_r)=(G_l G_r,\, U_l G_r+U_r)\) via
`rwkv7_associative_scan`, or sequential `rwkv7_apply_factorized` (no dense \(G\)).
`cell.scan_apply(x)`. [Newton + scan](../../core/newton_scan.md) (bypass note).

## Reproduce (sequential)

```python
import torch
from torch import nn
import torch.nn.functional as F

d_in, n_heads, d_head = 32, 1, 16
W_x = nn.Linear(d_in, 6 * n_heads * d_head)

def gates(x):
    # x: (B, d_in) -> each gate (B, H, d)
    raw = W_x(x).reshape(*x.shape[:-1], 6, n_heads, d_head)
    w = torch.sigmoid(raw[..., 0, :, :])
    a = torch.sigmoid(raw[..., 1, :, :])
    kappa = F.normalize(raw[..., 2, :, :], dim=-1, eps=1e-6)
    v, k, r = raw[..., 3, :, :], raw[..., 4, :, :], raw[..., 5, :, :]
    return w, a, kappa, v, k, r

def step(S, x):
    # S: (B, H, d, d)
    w, a, kappa, v, k, _r = gates(x)
    sw = S * w.unsqueeze(-2)                              # scale columns
    sk = torch.matmul(S, kappa.unsqueeze(-1)).squeeze(-1) # (B,H,d)
    rank1 = sk.unsqueeze(-1) * (a * kappa).unsqueeze(-2)
    write = v.unsqueeze(-1) * k.unsqueeze(-2)
    return sw - rank1 + write

S = torch.zeros(2, n_heads, d_head, d_head)
S = step(S, torch.randn(2, d_in))
```

## Agreement check

```python
import torch
from pararnn import ParaRWKV7, ParaRNN, verify_agreement

m = ParaRNN(ParaRWKV7(d_in=32, n_heads=1, d_head=16))
x = torch.randn(2, 64, 32)
res = verify_agreement(m, x, atol=1e-4)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

FP32 `atol=1e-4` ([numerics contract](../../core/numerics_contract.md)).
