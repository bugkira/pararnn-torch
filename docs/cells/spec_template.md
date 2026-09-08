# Architecture spec template

Copy to `docs/cells/<family>/para_<name>.md` where `<family>` is one of
`classic` | `normalized` | `matrix` | `continuous`. Keep Triton / SRAM /
autotune off this page.

**Page order is intentional.** Diff and the machine-readable contract come first
so humans and LLMs see packing / init / shapes / τ before display math. Every
change lives in the Diff table with an explicit **Kind**.

**DoD:** a reader with only this page + [numerics contract](../core/numerics_contract.md)
+ [Newton + scan](../core/newton_scan.md) can (1) separate author / Para-native /
numerics, (2) reimplement sequential `step` with **exact tensor shapes and
broadcasts**, (3) know the **linear-solve primitive** used inside each Newton
iteration (scan of \(\delta_t=J_t[\delta_{t-1}]+F_t\), not a guessed Sylvester /
CG / `linalg.solve`), (4) match library outputs within YAML τ.

```markdown
# ParaX

**Fidelity:** paper-faithful | paper-core | research-variant | unverified  
**Sources:** Author et al., venue/year, eq. / §  
**Upstream:** `torch.nn.…` / `ncps.…` / … (or “none — ParaRNN-native”)  
**Code:** `pararnn.cells.para_x.ParaX`

## Diff (upstream → Para-native)

| Component | Upstream (paper / library) | Para-native | Kind | Rationale |
|-----------|----------------------------|-------------|------|-----------|
| State | … | … | author | 1∶1 |
| Recurrence | … | … | para | e.g. diag \(R\) for fused scan |
| Pack / gates | `(…)` | `(…)` chunk indices | para | fused `W_x` layout |
| Init / ε | … | … | numerics | App. C.1 / measured |

**Kind** (exactly one of): `author` | `para` | `numerics` — see [fidelity](fidelity.md).

## Strict Spec Contract

```yaml
Cell: ParaX
Fidelity: …
Inputs:
  x: "[B, T, d_in]"   # or [..., d_in] for step
State:
  layout: "[B, T, …]" # full train layout + step layout
Shapes:               # mandatory — every symbol in Recurrence / J
  h: "[..., d_h]"
  W_x.out: "[..., 3*d_h]"
  …: …
Broadcast:
  - "…"               # e.g. f: [...,1,1] over H [...,K,V]
Parameters:
  W_x: "Linear(d_in, 3*d_h, bias=True)"
  W_x_chunk: "[0]=f, [1]=z, [2]=o"
  recurrent: "…"
  init: { … }
Constants:
  max_recurrent_norm: 0.5
Jacobian:
  structure: diag | block2 | block4 | head | m2rnn | dense | rwkv7
  storage: "…"
  formula: "…"
Newton:
  residual: "F = f(H_prev,x) - H"   # library convention
  linear_recurrence: "delta_t = J_t[delta_{t-1}] + F_t"
  solve: "associative scan of (J,F) monoid | factorized JVP walk | …"
  default_K_pin: 3
  fused_op: "…"
Agreement:
  FP32: { atol: 1.0e-4, rtol: 0.0 }
Weight_bridge: distinct | partial | identical
```

Shared τ: [numerics contract](../core/numerics_contract.md).  
Shared Newton algebra: [Newton + scan](../core/newton_scan.md).

## Shapes & broadcasting

Table: symbol → dtype role → shape at `step` and at train `(B,T,…)`.
Explicit broadcast rules (unsqueeze dims). PyTorch `@` operand order.

## Recurrence

Display equations + the same shapes inline or via the table above.

## Jacobian class

1. Dispatch tag  2. Storage shape  3. Operator \(J\) / \(J[\Delta]\)  
4. Gate note (input-only vs \(h\)-dependent)  
5. How \(J\) is **applied** (elementwise / 2×2 matvec / factor JVP) — the scan
   consumes applications of \(J\), it does not invent a separate Sylvester solver.

## Parallel path

Alg. 1: guess, residual \(F=f-H\), linear recurrence, **named scan / factor
walk**, \(K\) / \(K^*(T)\), fused op. Link [jacobian_classes](../core/jacobian_classes.md)
for the structure → primitive map.

## Reproduce (sequential)

20–40 lines with **literal shapes in comments** (e.g. `# H: (B,K,V)`, `# f: (B,1,1)`).
No `ParaX.step` in the body.

## Agreement check

Ready-to-run paste (assert on ``AgreementReport.ok`` / ``max_abs``):

```python
import torch
from pararnn import ParaX, ParaRNN, verify_agreement

m = ParaRNN(ParaX(...))
x = torch.randn(2, 64, d_in)  # shapes match the cell YAML
res = verify_agreement(m, x, atol=1e-4)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

Shared τ: [numerics contract](../core/numerics_contract.md).
```
