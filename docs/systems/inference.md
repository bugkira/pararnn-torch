# Inference contract: decode, generate, recurrent state

Serve path: a **recurrent carry** updated one token at a time. Train still
uses Newton+scan on full sequences (`.train()`). Decode is `.eval()` /
`T=1` / `decode_step`.

## Cheat sheet

| Want | Use |
|---|---|
| Full CausalLM loop | `ParaSLSTMForCausalLM.generate(...)` |
| Manual T=1 step (CUDA graph friendly) | `decode_wx` + `decode_step(..., out=)` |
| Multi-request GPU slots | `PagedStatePool` / `attach_pool` / `forward_continuous` |
| vLLM engine | plugin via `MambaBase` / `mamba_type=MAMBA1` — see [`vllm.md`](vllm.md) |

## Carry size

| | Attention KV | ParaRNN carry |
|---|---|---|
| Size vs generated length | grows with `T` | fixed `(slots, d_h)` per layer |
| Per-step op | attend over cache | `h ← f(h, x_t)` via `decode_step` |
| Prefill | fill K/V for prompt | sequential/Newton once; keep last state |
| Engine metadata | PagedAttention / FlashAttn | **Mamba1** pages + `state_indices_*` |

The vLLM worker supplies **Mamba1** / attention-free state metadata
(`state_indices_*`, mamba pages). Attention backends use their own page
layout and metadata.

## `decode_step` (library)

One Triton launch for the recurrent algebra after `W_x` (cuBLAS GEMM).
Supports diag `ParaGRU` / `ParaLSTM` / `ParaSLSTM(mix='diag')` on CUDA
(`can_decode_step`). Head-mix and CPU use eager `cell.step` through the same
API.

**CUDA-graph friendly when you pin buffers:**

- `decode_wx(cell, x, out=wx_buf)` — GEMM into a preallocated buffer
- `decode_step(cell, state, wx=wx_buf, out=state_out)` — step into `out=`
- App. C.1 gate clip runs **in-kernel** so a captured graph keeps a stable
  allocation set for that clip
- `block_table=(B,)` indexes rows in a pool-shaped `state` `(C, …)`

```python
import torch
from pararnn import ParaSLSTM, decode_step, decode_wx, sequential_apply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cell = ParaSLSTM(64, 64, mix="diag").to(device).eval()

# Prefill → carry = last time step (slots, d_h) for sLSTM
prompt = torch.randn(1, 16, 64, device=device)
carry = sequential_apply(cell, prompt)[:, -1].contiguous()

wx_buf = torch.empty(1, cell.W_x.out_features, device=device, dtype=prompt.dtype)
out_buf = torch.empty_like(carry)

for _ in range(8):  # decode loop, T=1 each time
    x_t = torch.randn(1, 64, device=device)
    decode_wx(cell, x_t, out=wx_buf)
    decode_step(cell, carry, wx=wx_buf, out=out_buf)
    carry = out_buf  # or keep writing into the same out_buf next step
```

Bench + CUDAGraph: [`examples/decode_step.py`](https://github.com/bugkira/pararnn-torch/blob/main/examples/decode_step.py).
CausalLM smoke: [`examples/causal_lm_smoke.py`](https://github.com/bugkira/pararnn-torch/blob/main/examples/causal_lm_smoke.py).
Minimal carry loop: [`examples/recurrent_state_loop.py`](https://github.com/bugkira/pararnn-torch/blob/main/examples/recurrent_state_loop.py).

## `generate()` (CausalLM)

`ParaSLSTMForCausalLM.generate` prefills the prompt, then appends tokens
with O(1) recurrent steps (`decode_step` when available). With
`attach_pool`, batch decode uses paged slots; otherwise per-layer carries
live on device.

```python
from pararnn import ParaSLSTMConfig, ParaSLSTMForCausalLM

cfg = ParaSLSTMConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2)
model = ParaSLSTMForCausalLM(cfg).eval()
ids = torch.randint(0, 256, (1, 8))
out = model.generate(ids, max_new_tokens=16)  # (1, 24)
```

## Paged pool (library continuous batch)

`PagedStatePool` / `BlockStackPool`: fixed GPU slots per request. Prefill
may use packed `cu_seqlens`; decode indexes `block_table` / `slot_ids`. See
[`vllm.md`](vllm.md) offline path and `examples/continuous_batch.py`.

## Related

- [`vllm.md`](vllm.md) — plugin: `MambaBase` / `mamba_type=MAMBA1`
- [`shapes_layout.md`](../getting_started/shapes_layout.md) — `T=1` / packing
- [`xlstm_notes.md`](../audit/xlstm_notes.md) — cell / block
- [`compile_amp.md`](compile_amp.md) — compile around serve loops
