# ParaSLSTM

**Fidelity:** paper-faithful for `mix='diag'`; `mix='head'` / `'dense'` documented  
**Sources:** Danieli et al. + xLSTM / sLSTM lineage (Beck et al.)  
**Upstream:** xLSTM sLSTM-style 4-slot cell  
**Code:** `pararnn.cells.para_slstm.ParaSLSTM`

## Diff (upstream → Para-native)

| Component | Upstream | Para-native | Kind | Rationale |
|-----------|----------|-------------|------|-----------|
| State | 4-slot sLSTM | `c,n,m,h` = slots `0..3` (`pararnn.layout`) | para | Fixed indices for Newton |
| Recurrent | Dense / head \(R\) | Diag \(R\in\mathbb{R}^{4\times d_h}\) default | para | `block4` fused path |
| Pack | Vendor order | **`(i, f, z, o)`** = `z_i,z_f,z_z,z_o` | para | chunk of pre-activations |
| Normalizer | \(n+\varepsilon\) | `eps=1e-6` | numerics | Explicit default |
| Ties | `maximum` | Subgrad `0.5/0.5` | numerics | Documented split |
| Bias | Varies | `W_x.bias=0` | numerics | App. C.1 style |
| Clamp | — | `max_recurrent_norm=0.5` | numerics | App. C.1 |

## Strict Spec Contract

```yaml
Cell: ParaSLSTM
Fidelity: paper-faithful  # mix=diag
Inputs:
  x_step: "[B, d_in]"; x_train: "[B, T, d_in]"
State:
  layout_step: "[B, 4, d_h]"; layout_train: "[B, T, 4, d_h]"
  slots: { CELL: 0, NORMALIZER: 1, STABILIZER: 2, HIDDEN: 3 }
Shapes:
  c,n,m,h,i,f,z,o: "[B, d_h]"
  R: "[4, d_h]"                 # mix=diag; gates i,f,z,o
  pre: "[B, 4*d_h]"
  jac: "[B, T, 4, 4, d_h]"      # [..., out, in, d]
  residual: "[B, T, 4, d_h]"
Broadcast:
  - "R * h.unsqueeze(-2) -> (B,4,d_h) then reshape (B,4*d_h)"
  - "4×4 couples slots per channel; channels independent"
Parameters:
  W_x: "Linear(d_in, 4*d_h, bias=True)"
  W_x_chunk: "[0]=z_i, [1]=z_f, [2]=z_z, [3]=z_o"  # after + R h
  R: "Parameter(4, d_h)"   # mix=diag; gate order i,f,z,o
  init:
    W_x.weight: kaiming_uniform
    W_x.bias: zeros
    R: orthogonal_gain_0.25
Constants:
  eps: 1.0e-6
  max_recurrent_norm: 0.5
  maximum_tie_split: 0.5/0.5
Jacobian:
  structure: block4         # mix=diag; head|dense otherwise
  storage: "(..., 4, 4, d_h)"
  formula: "see Jacobian class; max-tie alpha=0.5"
Newton:
  residual: "F = f(state_prev,x) - state"
  linear_recurrence: "delta_t = J_t delta_{t-1} + F_t  # per-channel 4×4"
  solve: "associative scan_block4 (± Thomas–PCR tiles); factorized head walk if mix=head"
  default_K_pin: 3
  picard_optional: true
  fused_op: "kernels.newton_slstm"
Agreement:
  FP32: { atol: 1.0e-4, rtol: 0.0 }
  BF16: { atol: 1.0e-2, rtol: 0.0 }
Weight_bridge: distinct
```

## Shapes & broadcasting

| Symbol | Shape | Notes |
|--------|-------|-------|
| state | `(B, 4, d_h)` | slots \((c,n,m,h)\) |
| \(R\) | `(4, d_h)` | diag mix; `pre = W_x(x) + (R*h).reshape(..., 4*d_h)` |
| \(J\) | `(B, T, 4, 4, d_h)` | `[..., out, in, d]` |
| \(\varepsilon\) | scalar | `1e-6` in denom |

## Recurrence

Pre-activations (pack after recurrent mix on \(h\)):

\[
\mathrm{pre} = W_x x + R\odot_{\mathrm{gate}} h,\quad
(z_i,z_f,z_z,z_o)=\mathrm{chunk}_4(\mathrm{pre}).
\]

Stabilized exp gates (\(m\) = stabilizer slot):

\[
\begin{aligned}
m_t &= \max(z_f + m_{t-1},\, z_i), \\
i_t &= \exp(z_i - m_t),\quad
f_t = \exp(z_f + m_{t-1} - m_t), \\
z_t &= \tanh(z_z),\quad
o_t = \sigma(z_o), \\
n_t &= f_t\, n_{t-1} + i_t, \\
c_t &= f_t\, c_{t-1} + i_t\, z_t, \\
h_t &= o_t\, c_t / (n_t + \varepsilon).
\end{aligned}
\]

At ties of \(\max\), subgradient splits \(\alpha=0.5\) on the left branch
(\(z_f+m\)), \(\beta=1-\alpha\) on \(z_i\).

## Jacobian class

**Dispatch:** `block4` (`mix='diag'`); `head` / `dense` for other mixes.  
**Storage:** `(..., 4, 4, d_h)` with slots \((c,n,m,h)\).  
**Operator (`diag`):** let \(r_g=\partial\mathrm{pre}_g/\partial h\) (the vector \(R_g\)),
\(\mathrm{d}m/\mathrm{d}h=\alpha r_f+\beta r_i\),
\(\partial i/\partial m=-i\alpha\), \(\partial f/\partial m=f\beta\),
\(\partial i/\partial h=i(r_i-\mathrm{d}m/\mathrm{d}h)\), etc.
Nonzero blocks (acts \(c,n\) are **previous** slots; \(d=n_t+\varepsilon\)):

| | \(c\) | \(n\) | \(m\) | \(h\) |
|---|---|---|---|---|
| \(c'\) | \(f\) | 0 | \(f_m c + i_m z\) | \(f_h c + i_h z + i\,z_h\) |
| \(n'\) | 0 | \(f\) | \(f_m n + i_m\) | \(f_h n + i_h\) |
| \(m'\) | 0 | 0 | \(\alpha\) | \(\mathrm{d}m/\mathrm{d}h\) |
| \(h'\) | \(o/d\cdot f\) | \(-o c'/d^2\cdot f\) | chain via \(c',n'\) | readout + \(o'\) |

Head/dense: same skeleton with matvecs / material \((4d)\times(4d)\).
Picard warm-start freezes gates for init only; Newton still uses full \(J\).

## Parallel path

Residual \(F=f-H\). \(\delta_t=J_t\delta_{t-1}+F_t\) with per-channel \(4\times4\).
**Solve:** associative `scan_block4` / fused `kernels.newton_slstm` (± Thomas–PCR
tiles). Head: factorized JVP walk. Dense: packed `scan_dense`. Picard warm-start
optional. Typical \(K{=}3\). [Newton + scan](../../core/newton_scan.md).

## Reproduce (sequential)

```python
import torch
from torch import nn

d_in, d_h, eps = 64, 64, 1e-6
W_x = nn.Linear(d_in, 4 * d_h)
R = torch.randn(4, d_h) * 0.1  # (4, d_h); gate order i,f,z,o
cap = 0.5
R = R.clamp(-cap, cap)

def step(state, x):
    # state: (B, 4, d_h); x: (B, d_in)
    c, n, m, h = (state[..., i, :] for i in range(4))  # each (B, d_h)
    pre = W_x(x) + (R * h.unsqueeze(-2)).reshape(*h.shape[:-1], 4 * d_h)
    z_i, z_f, z_z, z_o = pre.chunk(4, dim=-1)
    left = z_f + m
    m_new = torch.maximum(left, z_i)
    i_t = torch.exp(z_i - m_new)
    f_t = torch.exp(z_f + m - m_new)
    z = torch.tanh(z_z)
    n_new = f_t * n + i_t
    c_new = f_t * c + i_t * z
    o = torch.sigmoid(z_o)
    h_new = o * (c_new / (n_new + eps))
    return torch.stack((c_new, n_new, m_new, h_new), dim=-2)

state = torch.zeros(2, 4, d_h)
state = step(state, torch.randn(2, d_in))
```

## Agreement check

```python
import torch
from pararnn import ParaSLSTM, ParaRNN, verify_agreement

m = ParaRNN(ParaSLSTM(64, 64, mix="diag"))
x = torch.randn(2, 64, 64)
res = verify_agreement(m, x, atol=1e-4)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

FP32 `atol=1e-4` ([numerics contract](../../core/numerics_contract.md)).
Stacking: [xLSTM notes](../../audit/xlstm_notes.md).
