# Parallel recurrence (Newton + scan)

ParaRNN (Danieli et al., arXiv:2510.21450) trains a nonlinear RNN by solving the
fixed-point system \(F(H)=0\) over the full sequence with **Newton** and a
**parallel scan** on the linearized recurrence (paper Alg. 1).

## What the library does

1. Cell \(f\): \(h_t = f(h_{t-1}, x_t)\) with an analytic Jacobian structure.
2. Newton guess: default App. A \(h_t^{(0)} = f(0, x_t)\) (sLSTM: zero-hidden / Picard).
3. \(K\) Newton iterations: each builds a linear recurrence and scans it.
4. Train path: `ParaRNN` / `newton_apply`. Eval / decode: sequential `step`.

## Linear algebra (one Newton iteration)

Library residual (sign matters for coding):

\[
F_t = f(H_{t-1}, x_t) - H_t
\qquad\text{(code: ``pred - states``)}.
\]

Update solves the **time recurrence** (not a single dense block matrix):

\[
\delta_t = J_t[\delta_{t-1}] + F_t,\quad \delta_{<0}=0,\quad
H \leftarrow H + \omega\,\delta.
\]

**Solve primitive** = associative scan of the affine monoid
\((J_r,r_r)\oplus(J_l,r_l)=(J_r J_l,\, J_r r_l + r_r)\), or an equivalent
**factorized JVP walk** \(\delta_t = J_t[\delta_{t-1}]+F_t\) when \(J\) is never
materialized (`m2rnn`, factorized `head`).

There is **no** Sylvester / discrete-Lyapunov solver, Neumann series,
Richardson, CG, or `torch.linalg.solve` on a big \((Td)\times(Td)\) system in
`src/pararnn/`. Structure only changes how \(J[\cdot]\) and \(J_r J_l\) are
evaluated (elementwise, \(2\times2\), \(4\times4\), dense `bmm`, factor map on
\(K\times V\)).

## Jacobian → scan class

See [Jacobian classification](jacobian_classes.md).

## Contracts

- Agreement (Newton ≡ sequential): [Numerics contract](numerics_contract.md)
- Long-\(T\) backward tiles: [Backward scan](backward_scan.md)
- Contiguity / packing: [Shapes & layout](../getting_started/shapes_layout.md)

Cell-level Diff + YAML + shapes: [Cell catalog](../cells/index.md).
