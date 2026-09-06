# vLLM plugin

Out-of-tree **`ParaSLSTMForCausalLM`** for vLLM’s registry and v1 worker.

## Boundaries (read this first)

| Requirement | Value |
|---|---|
| Model kind | Attention-free recurrent (`IsAttentionFree`, `HasInnerState`) |
| Layer base | `ParaSLSTMRecurrentLayer` → **`MambaBase`**, `mamba_type=MAMBA1` |
| Temporal page | `(C, 4, d_h)` — sLSTM slots |
| Engine metadata | `Mamba1AttentionMetadata` (`state_indices_*`, mamba pages) |
| `config.json` | `"architectures": ["ParaSLSTMForCausalLM"]` |

Attention backends (GQA, FlashAttention, PagedAttention) use a different
page layout and metadata path. Wire this architecture on the Mamba1 /
attention-free worker path.

Library-side inference contract (carry, `decode_step`, `generate`):
[`inference.md`](inference.md).

## What ships

| Piece | Role |
|---|---|
| `pararnn.vllm_plugin.register` | `vllm.general_plugins` → `ModelRegistry` |
| `ParaSLSTMRecurrentLayer` | `MambaBase`, `mamba_type=MAMBA1`; temporal page `(C, 4, d_h)` |
| `ParaSLSTMDecoderLayer` | RMSNorm → mixer → SwiGLU (same trunk as `ParaSLSTMBlock`) |
| `VLLMParaSLSTMForCausalLM` | Embed + layers + norm + LM head; `get_mamba_state_*` / copy funcs |
| `pararnn.serve.BlockStackPool` | Offline continuous batch (also used via `slot_ids` kwargs) |

```bash
uv sync --extra vllm
uv run python -c "from importlib.metadata import entry_points; print(list(entry_points(group='vllm.general_plugins')))"
```

## Engine path

1. Worker allocates mamba pages from `get_mamba_state_shape_from_config` →
   `(conv=(1,), temporal=(4, d_h))`.
2. `bind_kv_cache` on each mixer unpacks pages into `kv_cache` (vLLM naming;
   contents are recurrent slots).
3. Each step, `Mamba1AttentionMetadata` supplies `state_indices_*` and
   `query_start_loc_*`.
4. Decode: `decode_step(..., block_table=indices)`. Prefill: packed
   sequential / Newton scan, last state scattered into the temporal page.

## Offline / library path

```python
model = ParaSLSTMForCausalLM(cfg).eval()
pool = model.attach_pool(8)
ids = pool.allocate(2)
model.forward_continuous(packed, ids, cu_seqlens=cu, solver="sequential")
```

`examples/continuous_batch.py`. The vLLM class also accepts `slot_ids` /
`cu_seqlens` on `forward` and routes through the mirrored library weights.

## Serve checklist

| Step | Status |
|---|---|
| Plugin entry + architecture name | done |
| `IsAttentionFree` / `HasInnerState` / prefix-caching marker | done |
| `MambaBase` layers + MAMBA1 metadata → our kernels | done |
| State shape / dtype / copy funcs | done |
| End-to-end `LLM.generate` on a Hub checkpoint | bring weights + `uv sync --extra vllm`; smoke on your GPU |

Prefix-caching **align/all** modes and CUDA-graph capture follow Mamba’s
worker; we consume the same indices. Speculative decoding slots are unread
(take column 0 of `state_indices_tensor_d`).

## Related

- [`inference.md`](inference.md) — `decode_step` / `generate` / carry size
- [`xlstm.md`](xlstm.md) — cell / block API
- [`distributed.md`](distributed.md) — paging notes
- `src/pararnn/vllm_plugin/`, `src/pararnn/serve/`, `src/pararnn/kernels/decode.py`
