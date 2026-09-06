# Shapes, layout, and packed `cu_seqlens`

Peer libraries drown in `CUDA illegal memory access` after `.transpose()`,
slicing, or ragged packs. This page is the ParaRNN contract: what we copy,
what we fuse, and what we refuse loudly.

## Layout (contiguity)

Fused / Triton Newton paths **copy** inputs to a contiguous layout before
pointer arithmetic (`tests/numerics/test_noncontiguous.py`). Strided
`(B, T, d)` views from `[:, ::2]`, feature skips, or permute round-trips
must still match sequential within the numerics band.

Practical rule: prefer contiguous batch-first `(B, T, d_in)` when you can.
If you pass a view, expect an extra copy on the fused path — correct
results, slightly more traffic.

`W_x` and some packed VJP helpers still see caller strides in places; the
non-contig suite covers the train forward+backward surface for diag
GRU/LSTM/sLSTM.

## Packed sequences (`cu_seqlens`)

Layout: `x` with batch `1` and length `N`, plus `cu_seqlens` of length
`S+1` (`int32`/`int64`). Segment heads use `J=0` on the scan. Validated by
`validate_cu_seqlens`.

### Support matrix (Newton `scan_backend`)

| Cell | `auto` + pack | explicit `fused` + pack | Notes |
|---|---|---|---|
| `ParaGRU(mix='diag')` | fused when CUDA dtype OK | fused in-kernel | Best packing story today |
| `ParaGRU(mix='head')` | → **eager** + `UserWarning` | **`TypeError`** | Factorized fused is rectangular-batch only |
| `ParaSLSTM(mix='head')` | → **eager** + `UserWarning` | **`TypeError`** | Same as head GRU |
| `ParaLSTM` / `ParaSLSTM(mix='diag')` | fused when eligible | fused → Triton scan fallback at segment heads | See newton package docstring |
| Dense / Hopfield / others | eager / triton per cell | follow `_resolve_backend` | Prefer pad to rectangular if unsure |

There is **no silent** `fused` → `eager` remap when the user pins
`scan_backend='fused'`. That path raises.

```python
from pararnn import NewtonConfig, ParaRNN

# Ragged pack: use auto (or eager). Do not pin fused on head-mix cells.
cfg = NewtonConfig(scan_backend="auto", max_iters=3)
# x: (1, N, d_in), cu_seqlens: (S+1,)
y = model(x, cu_seqlens=cu_seqlens)
```

Pad to a rectangular batch when you need head-fused speed on variable lengths.

## Decode `T=1`

`.eval()` / `solver='sequential'` on CUDA with `T=1` uses Triton
`decode_step` when available. Keep the single-token layout contiguous;
see [`docs/vllm.md`](vllm.md) / continuous batch notes for serve packing.

## Smoke (paste into a bug)

```python
import torch
from pararnn import NewtonConfig, ParaGRU, newton_apply, sequential_apply

device = torch.device("cuda")
cell = ParaGRU(8, 16).to(device)
# Strided time view
base = torch.randn(2, 64, 8, device=device)
x = base[:, ::2, :].detach().requires_grad_(True)
assert not x.is_contiguous()
cfg = NewtonConfig(max_iters=3, scan_backend="auto")
y_n = newton_apply(cell, x, cfg)
y_s = sequential_apply(cell, x)
print((y_n - y_s).abs().max().item())
```

Packed / head / fused failures: include `scan_backend`, `mix`, and whether
`cu_seqlens` was set.

## Related

- [`backward-scan-cap.md`](backward-scan-cap.md) — long-T tile pads
- [`oom-cookbook.md`](oom-cookbook.md) — VRAM geometry
- [`numerics-contract.md`](numerics-contract.md) — agreement vs residual
- [`compile-amp.md`](compile-amp.md) — `verify_first_step` skips packed `cu_seqlens`
