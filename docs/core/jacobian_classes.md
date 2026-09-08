# Jacobian classification

Newton+scan cost and kernel path follow the **structure of**
\(J_t = \partial f/\partial h_{t-1}\) (or \(J[\Delta]\) on matrix state).
Each cell declares a **dispatch tag** `jac_structure`; the operator and
**shapes** live on the cell page.

Shared Newton algebra (all nonlinear cells): residual \(F=f-H\), then
\(\delta_t=J_t[\delta_{t-1}]+F_t\) via associative scan or factorized JVP walk.
See [Newton + scan](newton_scan.md).

| Tag | Storage (per time) | \(J\) apply | Solve primitive | Typical cells |
|-----|--------------------|-------------|-----------------|---------------|
| `diag` | \(j\in\mathbb{R}^{d_h}\) | \(\delta\leftarrow j\odot\delta\) | Associative scan of \((j,r)\) | ParaGRU diag, ParaNLRU, ParaCfC, ParaTitans |
| `block2` | \(J\in\mathbb{R}^{2\times2\times d_h}\) `[out,in,d]` | Per-channel \(2\times2\) matvec | Scan with closed-form \(2\times2\) mul | ParaLSTM |
| `block4` | \(J\in\mathbb{R}^{4\times4\times d_h}\) `[out,in,d]` | Per-channel \(4\times4\) matvec | Scan (± Thomas–PCR tiles) | ParaSLSTM `diag` |
| `head` | Per-head dense or factor | Head matvecs / JVP | Factorized \(J\delta\) walk **or** packed `scan_dense` | ParaGRU / ParaSLSTM `head` |
| `m2rnn` | Map on \(K\times V\) | \(J[\Delta]=f\Delta+(1-f)(1-Z^{\odot2})\odot(\Delta W)\) | Factorized inclusive walk in \(T\) (no Sylvester) | ParaM2RNN |
| `dense` | \(J\in\mathbb{R}^{d\times d}\) `[out,in]` | `bmm` / `@` | Associative scan of dense monoid | ParaHopfield; sLSTM `dense` |
| `rwkv7` | Linear monoid \((G,U)\) | \(S\leftarrow SG+U\) | Associative \((G,U)\) scan; Newton bypass | ParaRWKV7 |

## Rules of thumb

- Prefer **diag** / small blocks for long \(T\) and fused kernels.
- `mix='head'` keeps Dreamer-style width with factorized \(J\).
- Dense \(J\) is an oracle or short-width specialty — see [OOM cookbook](../systems/oom_cookbook.md).
- Input-only gates drop \(\partial\mathrm{gate}/\partial h\) from \(J\); cell pages say so.
- If a formula looks like a Sylvester equation in \(\Delta\), check the cell page:
  the library still scans \(\delta_t=J_t[\delta_{t-1}]+F_t\) with a **matvec**
  \(J[\cdot]\), it does not call a Sylvester routine.

Per-cell Diff, shapes, YAML: [Cell catalog](../cells/index.md) ·
[Audit matrix](../cells/audit_matrix.md).
