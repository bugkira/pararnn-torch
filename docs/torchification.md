# Torchification (archive, v0.2)

Shipped in `6ac89c3`. This file is **not** the live backlog — see
[`bottlenecks.md`](bottlenecks.md) Next and the README roadmap.

It was the plan to turn the solver prototype into `ParaRNN` (`nn.Module`,
train = Newton, eval = sequential, toy copy). Do not treat the drafts below as
current API (fused `h0` is in the kernel; `scan_backend` default is `auto`).


Turn the current **solver prototype** into a drop-in **PyTorch module**: something you put in `nn.Sequential`, call as `model(x)`, train with AdamW, and install with `uv add`.

This is **not** the Next list in [`bottlenecks.md`](bottlenecks.md) (K=2, Mamba warm-start, IFT). Those stay research extras. Hugging Face / SlimPajama are later still.

Baseline: commit `668122c`. Public API today is functional (`newton_apply(cell, x)`), plus lab `experiment_device()`. Cells are `nn.Module` but they are *cells* (`step`), not sequence layers.

## Out of scope

- Apple kernels, 665×, fig. 2/5.
- Hybrid PC, IFT adjoint, HF `PreTrainedModel`, pretrained checkpoints.
- Changing Newton math. Train still uses Alg. 1 + eq. 2.6; decode still unrolls `step`.
- Baking `torch.compile` into `src/`.

## Done when we can say this

> `uv add pararnn-torch` (import `pararnn`). `ParaRNN(ParaGRU(...))` trains in parallel and decodes sequentially. Custom cells follow a documented protocol. One toy training run shows the loss drop.

## 1. Sequence layer

**Status: done.** `ParaRNN` in [`src/pararnn/layers/para_rnn.py`](../src/pararnn/layers/para_rnn.py). Batch-first. `self.training` → Newton; `eval()` → sequential. No residual/LayerNorm. `h0` wired through Newton (fused + nonzero `h0` → eager + log). Tests: [`tests/numerics/test_layer.py`](../tests/numerics/test_layer.py).

**Why.** Nobody trains `newton_apply` by hand. `nn.GRU` is `forward(x) → y`. We need the same shape of object.

**API (draft).**

```python
class ParaRNN(nn.Module):
    def __init__(
        self,
        cell: nn.Module,
        *,
        num_layers: int = 1,
        config: NewtonConfig | None = None,
    ) -> None: ...

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        # self.training → newton_apply per layer
        # eval()       → sequential_apply per layer
        ...
```

- `x`: `(batch, time, d_in)` — keep batch-first (we already disagree with `nn.LSTM`'s default).
- One cell class, cloned/stacked `num_layers`. Mixing ParaGRU with a custom cell is layer-0 only until we have a list-of-cells ctor.
- Residual / LayerNorm: **not** in v1 of this wrapper. Paper LM is a DCLM backbone with the cell swapped in; that is a *model*, not the module. Document “stacking is naive; residual is the caller’s job.”
- `h0` for decode and for teacher-forced train. Default zeros as today.
- Do **not** put `experiment_device` in `forward`. `.to(device)` like every other module.

**Tests.** Wrapper train vs raw `newton_apply` (same weights). Wrapper eval vs `sequential_apply`. `train()`/`eval()` actually switch solvers. Grad through wrapper matches cell-level BPTT tests we already have.

## 2. Cell protocol

**Status: done.** [`src/pararnn/cells/protocol.py`](../src/pararnn/cells/protocol.py) (`RNNCell` + `check_cell`). Tests: [`tests/unit/test_protocol.py`](../tests/unit/test_protocol.py).

**Why.** “Any `step(h, x)`” is true in the solver and tribal in the docs.

**Contract** (write it once, next to the wrapper; test it).

| Field | Rule |
|---|---|
| `d_h` | int, last dim of the state |
| `state_slots` | `1` → state `(…, d_h)`; `2` → `(…, 2, d_h)` (LSTM layout: 0=c, 1=h) |
| `step(h_prev, x)` | leading dims broadcast; last dims `d_h` / `d_in` |
| `wx=` | optional; only if the signature has it (sequential already inspects) |
| `step_with_jacobian` | optional fast path; else Autograd |

Document the Jacobian choice in the same place: default ones-JVP is exact **iff** `f` is channelwise in `h`; mixing channels needs `jac_structure="dense"` (`O(d_h^3)` scan). Fused Newton remains ParaGRU/ParaLSTM only.

A `@runtime_checkable` `Protocol` is enough; do not invent a base class people must inherit.

## 3. Training smoke

**Status: done.** [`configs/train/toy.yaml`](../configs/train/toy.yaml) + [`src/pararnn/train/toy.py`](../src/pararnn/train/toy.py). Copy tokens=targets; MLflow `toy-copy`. Not SlimPajama.

**Why.** Numerics tests prove Newton ≈ sequential. They do not prove “an optimizer can move this module.”

**Minimum run** (local, MLflow mandatory because it logs metrics):

- Task: character copy or tiny next-token on a synthetic alphabet. Not SlimPajama.
- Loop: `ParaRNN` + linear head → CE → AdamW → loss curve, 50–200 steps.
- Config YAML (`configs/train/toy.yaml`): lr, wd, `K`, dtype, seq len — each line cited (toy scale: not paper Table 3; say so).
- CPU or 2080 Ti by name. Seed, device name, dtype in the run.
- Success: train loss strictly down vs step 0; Newton residual still logged; no silent fused fallback.

This is the gate for the sentence “you can train with it.” It is not an LM result.

## 4. Packaging and public API

**Status: done.** Distribution `pararnn-torch` 0.2.0, import `pararnn`. `__all__`: `ParaRNN`, `ParaGRU`, `ParaLSTM`, `NewtonConfig`, `newton_apply`, `sequential_apply`. `experiment_device` / `sequential_apply_compiled` stay on submodules. No PyPI upload.

**Why.** `pyproject.toml` name is `pararnn`; [`STRUCTURE.md`](STRUCTURE.md) says PyPI `pararnn-torch`. Quickstart starts with `experiment_device()`.

- Decide and freeze: **distribution** `pararnn-torch`, **import** `pararnn` (Apple occupies the name conceptually; we do not claim their package).
- `readme` + classifiers + `requires-python` already 3.10. Point the PyTorch extra at the cu128 index in the README, not at a pip-only `requirements.txt`.
- `__all__` for humans: `ParaRNN`, `ParaGRU`, `ParaLSTM`, `NewtonConfig`, `newton_apply`, `sequential_apply`. Move `experiment_device` / `sequential_apply_compiled` to `pararnn.device` / bench notes — available, not the front door.
- Version stays `0.1.0` until the wrapper + toy train exist; then `0.2.0` is “library-shaped.”

No PyPI upload is required to *say* we made a module. A clean `uv add --editable .` story is.

## 5. Docs that must match the code

- README Status: lead with the wrapper + train/eval split; keep fused/fp16 numbers as *performance*, not as the API.
- Roadmap: check “sequence module”; leave HF/Mamba/IFT unchecked.
- This file: tick the sections as they land. Do not tick packaging because a README snippet exists.

## Order

1. Protocol + tests (cheap, unblocks custom cells in the wrapper).
2. `ParaRNN` wrapper + tests (the actual module).
3. Toy train YAML + MLflow smoke.
4. `__all__` / naming / README. Last, so the snippet is true.

## Honesty

After this we can say we shipped a **PyTorch library for parallel RNN training**. We still cannot say we matched Apple fused CUDA, trained 7B, or beat Mamba. Those are different tickets.
