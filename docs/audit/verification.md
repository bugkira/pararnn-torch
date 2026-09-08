# Verification suite

Product claim: with recommended \(K^*(T)\), Newton+scan matches sequential
`step` within agreement τ.

## Canonical page

Full echelons (runtime residual, lab \(K^*\), opt-in oracle, docs):
**[Numerics & agreement contract](../core/numerics_contract.md)**.

Defaults: FP32 `atol=1e-4`, FP16/BF16 `atol=1e-2` (`default_agreement_atol`).

## Quick check

```python
from pararnn import ParaGRU, ParaRNN, verify_agreement
import torch

m = ParaRNN(ParaGRU(32, 32))
print(verify_agreement(m, torch.randn(2, 64, 32)).to_dict())
```

Per-cell τ and pack order: YAML on each [cell page](../cells/index.md).

## Upstream task oracles

Weight bridges are usually **architecturally distinct**. Compare stacks on the
same task (separate trainings), e.g. lab PhysioNet ncps vs ParaCfC — see each
cell’s Diff **Weight bridge** row and [audit matrix](../cells/audit_matrix.md).
