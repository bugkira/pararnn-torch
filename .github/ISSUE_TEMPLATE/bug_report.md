---
name: Bug report
about: Numerics mismatch, crash, or install failure
labels: bug
---

**Environment**

Paste the output of:

```bash
python -c "import torch, pararnn; import importlib.util as u; t=(__import__('triton').__version__ if u.find_spec('triton') else 'n/a'); print('pararnn', getattr(pararnn,'__version__', '?'), '| torch', torch.__version__, '| cuda', torch.version.cuda, '| triton', t, '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Also fill in:
- OS:
- Python:
- install path (`pip` / `uv` / editable git SHA):

**What happened**
(short description)

**Minimal reproduction**
```python
# paste a short script
from pararnn import verify_agreement
# report = verify_agreement(cell_or_model, x)
# print(report.to_dict())
```

**Expected**
(e.g. `verify_agreement(...).ok` with fp32 atol 1e-4; or a clean
`NewtonDivergenceError` when residual ≥ 1. Residual gate and agreement τ
are separate — see `docs/core/numerics_contract.md`)

**Logs**
(`newton_residual` / `report.to_dict()` / peak MiB from
`docs/oom-cookbook.md` smoke / traceback)
