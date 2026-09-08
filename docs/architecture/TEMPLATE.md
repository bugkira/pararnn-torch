# Architecture spec template

Copy to `docs/architecture/para_<name>.md` and fill every section. Keep Triton /
SRAM / autotune out of this page; link systems notes separately if needed.

```markdown
# ParaX

**Fidelity:** paper-faithful | paper-core | research-variant | unverified  
**Sources:** Author et al., venue/year, eq. / §  
**Code:** `pararnn.cells.para_x.ParaX`

## State

- Slots / shapes
- Which slot is the “hidden” readout

## Recurrence

Display equations for \(h_t = f(h_{t-1}, x_t)\) (and gates).

## Jacobian class

`diag` | `block2` | `block4` | `head` | `m2rnn` | …

## Parallel path

Newton + scan on \(F(H)=0\) (ParaRNN Alg. 1). Defaults: guess, \(K\), clip.

## Deviations

Bullet list of intentional differences from the cited paper. Empty list if none.

## Reproduce (sequential)

20–40 lines of PyTorch or clear pseudocode for one `step`.

## Agreement

Expected τ / how to call `verify_agreement`.
```
