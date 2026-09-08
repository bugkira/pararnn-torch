# Adoption: drop-in recurrent trunk

Short path from install to a working stack. Cell APIs:
[`docs/cells.md`](cells.md), [`docs/xlstm.md`](xlstm.md).

## Replace an Attention block

`ParaSLSTMBlock` is a pre-norm residual trunk:

`RMSNorm → ParaRNN(ParaSLSTM) → residual → RMSNorm → SwiGLU → residual`

```python
import torch
from pararnn import NewtonConfig, ParaSLSTMBlock

d_model = 256
block = ParaSLSTMBlock(
    d_model,
    mlp_ratio=4.0,
    config=NewtonConfig(max_iters=3),
)
x = torch.randn(2, 128, d_model)
y = block(x)  # (2, 128, 256)

# Stack like transformer layers (no GQA inside this package):
layers = torch.nn.ModuleList([ParaSLSTMBlock(d_model) for _ in range(8)])
```

In torchtitan / Megatron / Llama-Factory style code, swap the attention module
for this block (or alternate Attention every `k` layers outside ParaRNN). The
library owns the recurrent Newton solve; attention kernels stay in your stack.

## Full CausalLM

```python
from pararnn import ParaSLSTMConfig, ParaSLSTMForCausalLM

cfg = ParaSLSTMConfig(
    vocab_size=32000,
    hidden_size=512,
    num_hidden_layers=8,
    mlp_ratio=4.0,
)
model = ParaSLSTMForCausalLM(cfg)

logits, loss = model(input_ids, labels=input_ids)  # train
tokens = model.generate(prompt_ids, max_new_tokens=32)  # eval decode
model.save_pretrained("./ckpt")  # config.json + model.safetensors
model = ParaSLSTMForCausalLM.from_pretrained("./ckpt")
```

Smoke: [`examples/causal_lm_smoke.py`](https://github.com/bugkira/pararnn-torch/blob/main/examples/causal_lm_smoke.py).
Packed continuous batch: [`examples/continuous_batch.py`](https://github.com/bugkira/pararnn-torch/blob/main/examples/continuous_batch.py).
Serve: [`inference.md`](inference.md) (carry / `decode_step`) ·
[`vllm.md`](vllm.md) (plugin).

## Dreamer / RSSM recurrent slot

World-model imagination is usually `h_t = GRU(h_{t-1}, concat(z, a))`. Use
`ParaGRU(mix='head')` inside `ParaRNN` for that slot; keep encoder / prior /
actor in your RL code.

```python
from pararnn import NewtonConfig, ParaGRU, ParaRNN

# Cho gates; Dreamer LayerNorm stays outside the cell.
rssm_h = ParaRNN(
    ParaGRU(d_in=z_dim + a_dim, d_h=512, mix="head", n_heads=8),
    config=NewtonConfig(max_iters=3),
)
# .train() → parallel Newton over the imagination horizon
# .eval()  → sequential step (T=1 CUDA: decode_step)
```

Smoke: [`examples/rssm_recurrent.py`](https://github.com/bugkira/pararnn-torch/blob/main/examples/rssm_recurrent.py).

## Griffin / RecurrentGemma recurrent slot

For a nonlinear replacement of linear RG-LRU (input-only gate, diagonal mix,
`tanh` inside the step), use `ParaNLRU`:

```python
from pararnn import ParaNLRU, ParaRNN

core = ParaRNN(ParaNLRU(d_model, d_model))
```

This shares the diag-Jacobian class with fused ParaGRU. Griffin's
$`a^{c r_t}`$ parameterization needs its own weight map; treat `ParaNLRU` as
a nonlinear slot with the same Newton path, with separate checkpoints.

## Liquid / irregular-Δt slot

For a Liquid-**style** closed-form step with irregular sampling intervals,
use `ParaCfC` (fidelity **research-variant** vs Hasani / ncps — see
[architecture/para_cfc.md](architecture/para_cfc.md)). Features live in
`x[..., :-1]`; Δt is `x[..., -1]` (`d_in >= 2`):

```python
from pararnn import ParaCfC, ParaRNN

# d_in = feature_dim + 1
core = ParaRNN(ParaCfC(d_model + 1, d_model))
feat = ...  # (B, T, d_model)
dt = ...  # (B, T, 1), positive
y = core(torch.cat((feat, dt), dim=-1))
```

Gate $`a=\exp(-\mathrm{softplus}(f)\,\Delta t)`$ and diagonal mix $u$ keep
the Newton Jacobian channelwise diagonal (fused Alg. 1 on CUDA).

## Modern Hopfield / attractor slot

For a recurrent one-step Modern-Hopfield update with input-conditioned
pattern matrices, use `ParaHopfield`. Softmax couples channels, so Newton
uses a dense Jacobian and `scan_dense` (keep d_h ≤ 32; tests use 8):

```python
from pararnn import NewtonConfig, ParaHopfield, ParaRNN

core = ParaRNN(
    ParaHopfield(d_model, 8),
    config=NewtonConfig(max_iters=None, jac_structure="dense"),
)
y = core(x)  # (B, T, d_model) → (B, T, 8)
```

Default $`\beta=1/\sqrt{d_h}`$. `max_iters=None` selects the measured
`K*(T)` envelope (`hopfield_auto_newton_iters`); pin an `int` or pass
`newton_iters_by_t={64: 2, 1024: 3, …}` to override. Fused cell+scan and
packed VJP are parked; eq. 2.6 uses Autograd on `step`.

## RWKV-7 Goose / matrix-state delta slot

For the linear RWKV-7 Goose transition (Peng et al. arXiv:2503.14456), use
`ParaRWKV7`. State is per-head $`S\in\mathbb{R}^{d_{\mathrm{head}}\times d_{\mathrm{head}}}`$;
gates are input-only, so the map is an affine monoid in $S$:

```python
from pararnn import ParaRWKV7, newton_apply, sequential_apply

cell = ParaRWKV7(d_in=d_model, n_heads=4, d_head=16)
s = sequential_apply(cell, x)  # (B, T, H, D, D)
y = cell.scan_apply(x)  # (B, T, n_heads*d_head) = flatten(S @ r)
# newton_apply runs the linear (G,U) scan (iters=0).
s2 = newton_apply(cell, x)
```

This is a factorized matrix-state delta brick (same class as the M²RNN
linear warm-start). Full RWKV-7 token-mix and Wind CUDA live in the RWKV
stack.

## Titans / shallow neural memory slot

For a vector memory with one surprise-GD associative step plus a diagonal
nonlinear polish (Behrouz et al. arXiv:2501.00663 flavor), use
`ParaTitans`. Deep multi-layer MLP memory is parked:

```python
from pararnn import ParaTitans, ParaRNN

core = ParaRNN(ParaTitans(d_model, d_model))
```

The Newton Jacobian is channelwise diagonal (fused Alg. 1 on CUDA).

## Scope

Local `save_pretrained` / `from_pretrained` (`config.json` + `model.safetensors`).
Hugging Face `AutoModel` registration is a separate integration workstream.
Compose Attention / GQA in your trainer next to `ParaSLSTMBlock` for a hybrid
stack. RSSM / Dreamer keep encoder, prior, and actor in the RL codebase; this
library supplies the recurrent `h_t` slot.
