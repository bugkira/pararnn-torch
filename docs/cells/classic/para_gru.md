# ParaGRU

**Fidelity:** paper-faithful (`mix='diag'`); `mix='head'` documented  
**Sources:** Danieli et al., arXiv:2510.21450, eq. 3.1a, 3.3; Cho et al. 2014  
**Upstream:** Cho GRU / `torch.nn.GRU` (full recurrent matrices)  
**Code:** `pararnn.cells.para_gru.ParaGRU`

## Diff (upstream → Para-native)

| Component | Upstream | Para-native | Kind | Rationale |
|-----------|----------|-------------|------|-----------|
| Gates | \(z,r,n\) Cho | Same | author | 1∶1 gate math |
| Recurrent | Dense \(W_h\) | Diag \(a_z,a_r,a_n\) (or per-head \(A_*\)) | para | Diag / head Newton |
| Pack | Often separate / `(r,z,n)` variants | **`(z, r, n)`** | para | Matches `zx,rx,nx = wx.chunk(3)` |
| LN | Dreamer LN-GRU wraps LN | LN outside cell | para | Cell = gates only |
| Clamp | — | `max_recurrent_norm=0.5` (diag) | numerics | App. C.1 |

## Strict Spec Contract

```yaml
Cell: ParaGRU
Fidelity: paper-faithful  # mix=diag
Inputs:
  x_step: "[B, d_in]"; x_train: "[B, T, d_in]"
State:
  layout_step: "[B, d_h]"; layout_train: "[B, T, d_h]"
Shapes:
  h,z,r,n,a_*: "[B, d_h]"          # mix=diag; all channelwise
  W_x.out: "[B, 3*d_h]"
  jac_diag: "[B, T, d_h]"          # diagonal entries
  A_*: "[H, d_head, d_head]"       # mix=head; y = h @ A
  jac_head: "[B, T, H, d_head, d_head]"  # [..., out, in]
Broadcast:
  - "a_* * h and gate ⊙ are elementwise on d_h"
  - "mix=head: reshape h -> (B,H,d_head); matmul per head"
Parameters:
  W_x: "Linear(d_in, 3*d_h, bias=True)"
  W_x_chunk: "[0]=z_x, [1]=r_x, [2]=n_x"
  mix: diag                 # or head
  init:
    W_x.weight: kaiming_uniform
    W_x.bias: zeros
    a_z,a_r,a_n: xavier_gaussian_vec   # mix=diag
    A_*: orthogonal_gain_0.25          # mix=head
Constants:
  max_recurrent_norm: 0.5   # diag default; often null for head
Jacobian:
  structure: diag           # or head
  storage: "(..., d_h) | (..., H, d_head, d_head)"
  formula: "diag: J=(1-z)+(n-h)*z'*a_z+z*n'*a_n*(r+h*r'*a_r); full gate ∂"
Newton:
  residual: "F = f(h_prev,x) - h"
  linear_recurrence: "delta_t = j_t ⊙ delta_{t-1} + F_t"
  solve: "associative scan of (j,F) diag monoid (scan_diag / fused newton_gru)"
  default_K_pin: 3
  fused_op: "kernels.newton_gru / Alg. 1"
Agreement:
  FP32: { atol: 1.0e-4, rtol: 0.0 }
  BF16: { atol: 1.0e-2, rtol: 0.0 }
Weight_bridge: distinct
```

## Shapes & broadcasting

| Symbol | `mix='diag'` | `mix='head'` |
|--------|--------------|--------------|
| \(h,z,r,n\) | `(B, d_h)` | per-head `(B, H, d_head)` then concat |
| \(a_z,a_r,a_n\) | `(d_h,)` | — |
| \(A_z,A_r,A_n\) | — | `(H, d_head, d_head)`, `y = h @ A` |
| \(j\) / \(J\) | `(B, T, d_h)` | `(B, T, H, d_head, d_head)` `[out,in]` |

## Recurrence

**`mix='diag'`:**

\[
\begin{aligned}
z_t &= \sigma(a_z \odot h_{t-1} + W_z x_t + b_z), \\
r_t &= \sigma(a_r \odot h_{t-1} + W_r x_t + b_r), \\
n_t &= \tanh(a_n \odot (r_t \odot h_{t-1}) + W_n x_t + b_n), \\
h_t &= (1-z_t)\odot h_{t-1} + z_t \odot n_t.
\end{aligned}
\]

**`mix='head'`:** same gates with per-head dense \(A_*\):  
\(z=\sigma(h A_z + z_x)\), \(r=\sigma(h A_r + r_x)\), \(n=\tanh((h\odot r)A_n + n_x)\),  
\(h\leftarrow (1-z)\odot h + z\odot n\) (head layout), then concat heads.

## Jacobian class

**Dispatch:** `jac_structure='diag'` or `'head'` from `mix`.  
**Storage:** diag → `(..., d_h)`; head → `(..., H, d_head, d_head)` with `[..., out, in]`.  
**Operator (`diag`, primes \(z'=z(1-z)\), \(n'=1-n^2\)):**

\[
J = (1-z) + (n-h)\,z'\,a_z + z\,n'\,a_n\bigl(r + h\,r'\,a_r\bigr).
\]

Full gate derivatives (Danieli eq. 3.2a). Head path: same algebra with matvecs
\(v@A_*\) (`gru_head_jvp`).

## Parallel path

Residual \(F=f-H\). Linear step \(\delta_t = j_t\odot\delta_{t-1}+F_t\).
**Solve:** associative scan of the diag monoid (`scan_diag` / fused
`kernels.newton_gru`) — elementwise multiply/add, **not** `1/j` inversion of a
spatial system. Head: factorized \(J\delta\) walk (`newton_gru_head_factorized`)
or packed `scan_dense`. App. A guess \(h^{(0)}=f(0,x)\). Typical \(K{=}3\).
[Newton + scan](../../core/newton_scan.md).

## Reproduce (sequential)

```python
import torch
from torch import nn

d_in, d_h = 32, 32
W_x = nn.Linear(d_in, 3 * d_h)
a_z = torch.randn(d_h) * 0.1   # (d_h,)
a_r = torch.randn(d_h) * 0.1
a_n = torch.randn(d_h) * 0.1
cap = 0.5
a_z, a_r, a_n = (t.clamp(-cap, cap) for t in (a_z, a_r, a_n))

def step(h, x):
    # h: (B, d_h); x: (B, d_in)
    zx, rx, nx = W_x(x).chunk(3, dim=-1)  # each (B, d_h)
    z = torch.sigmoid(a_z * h + zx)
    r = torch.sigmoid(a_r * h + rx)
    n = torch.tanh(a_n * (r * h) + nx)
    return (1.0 - z) * h + z * n

h = torch.zeros(2, d_h)
h = step(h, torch.randn(2, d_in))
```

## Agreement check

```python
import torch
from pararnn import ParaGRU, ParaRNN, verify_agreement

m = ParaRNN(ParaGRU(32, 32))
x = torch.randn(2, 64, 32)
res = verify_agreement(m, x, atol=1e-4)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

([numerics contract](../../core/numerics_contract.md)).
