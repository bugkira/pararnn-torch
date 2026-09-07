"""One-shot builder for notebooks/paraslstm_demo.ipynb. Not imported by the package."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update(
    {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
)
cells: list = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(
    r"""
# ParaSLSTM demo (`pararnn-torch`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bugkira/pararnn-torch/blob/main/notebooks/paraslstm_demo.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-bugkira%2Fpararnn--torch-blue)](https://github.com/bugkira/pararnn-torch)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22302586.svg)](https://doi.org/10.5281/zenodo.22302586)

Parallel **training** of nonlinear sLSTM via Newton + associative scan
([Danieli et al., ICLR 2026](https://arxiv.org/abs/2510.21450)), with a tropical
warm-start for exponentially gated cells
([ParaSLSTM](https://doi.org/10.5281/zenodo.22558086)).

**Run All** (~30–90 s on a Colab T4 / lab GPU): install → forward → trust → latency → decode.

Use **Runtime → Change runtime type → GPU**. Demo tensors stay in `float32` so free-tier T4
(SM 7.5) works; `NewtonConfig(scan_backend="auto")` picks fused Triton when the GPU supports it.

Full \(\mathbb{Z}_2\) parity and BabyLM runs live in the repo:
[`examples/parity.py`](https://github.com/bugkira/pararnn-torch/blob/main/examples/parity.py),
[`scripts/README.md`](https://github.com/bugkira/pararnn-torch/blob/main/scripts/README.md).
"""
)

md(
    """
## 0. Setup

**Colab:** next cell installs `pararnn-torch` from PyPI (public).

**Local:** skip the install cell if you already `uv sync`'d this checkout.
"""
)

code(
    """
# Install: Colab only. Local editable installs already provide `pararnn`.
# Pin numpy<2.3 so Colab's preinstalled numba stays happy (pip otherwise
# pulls a newer numpy via torch and prints a red resolver conflict).
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    get_ipython().run_line_magic(
        "pip",
        'install -q "pararnn-torch>=0.17.4" "numpy>=2.2.6,<2.3"',
    )
else:
    print("Local / non-Colab: expecting an existing pararnn install (uv sync).")
"""
)

code(
    """
import logging
import time

import torch

from pararnn import (
    NewtonConfig,
    ParaRNN,
    ParaSLSTM,
    can_decode_step,
    decode_step,
    newton_apply,
    sequential_apply,
)

# Keep step metrics readable; NewtonDivergenceError still raises at ERROR.
logging.getLogger("pararnn.solvers.newton").setLevel(logging.ERROR)

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
# K=3: Danieli et al. App. A. P auto: 1 if T≤64, 3 if T≤2048, else 5.
cfg = NewtonConfig(max_iters=3, scan_backend="auto")

print(f"pararnn OK | torch {torch.__version__} | device={device}")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(device)
    print(
        f"GPU: {torch.cuda.get_device_name(device)}  "
        f"CC {props.major}.{props.minor}  dtype={dtype}"
    )
"""
)

md(
    """
## 1. Drop-in

`ParaRNN` in `.train()` runs Newton + scan; in `.eval()` it runs the sequential `step` unroll.
"""
)

code(
    """
cell = ParaSLSTM(64, 64, mix="diag", device=device, dtype=dtype)
model = ParaRNN(cell, config=cfg).to(device=device, dtype=dtype)
x = torch.randn(4, 256, 64, device=device, dtype=dtype)
"""
)

code(
    """
model.train()
y = model(x)
y.sum().backward()
y.shape, next(model.parameters()).grad is not None
"""
)

code(
    """
model.eval()
with torch.no_grad():
    y_eval = model(x)
y_eval.shape
"""
)

md(
    r"""
## 2. Trust

Sequential unroll vs Newton (\(K{=}3\), tropical warm-start). Absolute error should stay small.
"""
)

code(
    """
lengths = [128, 512, 1024, 2048] if device.type == "cuda" else [64, 128, 256]
dim = 128 if device.type == "cuda" else 64
trust_cell = ParaSLSTM(dim, dim, mix="diag", device=device, dtype=dtype).eval()

print(f"{'T':>6}  {'max|h_seq - h_newton|':>22}")
print("-" * 32)
with torch.no_grad():
    for T in lengths:
        xt = torch.randn(2, T, dim, device=device, dtype=dtype)
        err = (sequential_apply(trust_cell, xt) - newton_apply(trust_cell, xt, config=cfg)).abs().amax()
        print(f"{T:6d}  {float(err):22.4e}")
"""
)

md(
    """
## 3. Latency (CUDA fused)

Wall-clock win needs **Linux + NVIDIA + fused Triton**. On CPU / Mac,
`scan_backend="auto"` uses **eager** Newton: \(K\) full scans vs one sequential
pass, so Newton is *slower by design* there — use Trust above for numerics,
not this table.

On CUDA: same cell, sequential `step` unroll vs Newton (`scan_backend="fused"`).
Median ms after warmup. Expect Newton ≪ sequential once \(T\) is a few hundred+.
"""
)

code(
    """
def _sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median_ms(fn, *, warmup=3, runs=9):
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


if device.type != "cuda":
    print(
        f"Skip latency: device={device}. "
        "Runtime → Change runtime type → GPU, then re-run this cell. "
        "CPU eager Newton is for agreement checks, not speedups."
    )
else:
    # Fused path only — matches README / lab benches.
    bench_cfg = NewtonConfig(max_iters=3, scan_backend="fused")
    T_list = [256, 1024, 2048, 4096]
    bench_dim, bench_B = 128, 8
    bench_cell = ParaSLSTM(
        bench_dim, bench_dim, mix="diag", device=device, dtype=dtype
    ).eval()

    print(f"{'T':>6}  {'sequential_ms':>14}  {'newton_ms':>10}  {'speedup':>8}")
    print("-" * 46)
    with torch.no_grad():
        for T in T_list:
            xb = torch.randn(bench_B, T, bench_dim, device=device, dtype=dtype)
            seq_ms = median_ms(lambda xb=xb: sequential_apply(bench_cell, xb))
            newt_ms = median_ms(
                lambda xb=xb: newton_apply(bench_cell, xb, config=bench_cfg)
            )
            speed = seq_ms / max(newt_ms, 1e-9)
            print(f"{T:6d}  {seq_ms:14.2f}  {newt_ms:10.2f}  {speed:7.1f}×")
"""
)

md(
    """
## 4. Decode (T=1)

On CUDA, `decode_step` runs the recurrent step in one Triton kernel when the cell supports it.
"""
)

code(
    """
dec_cell = ParaSLSTM(64, 64, mix="diag", device=device, dtype=dtype).eval()
with torch.no_grad():
    carry = sequential_apply(
        dec_cell, torch.randn(4, 8, 64, device=device, dtype=dtype)
    )[:, -1]
    x1 = torch.randn(4, 64, device=device, dtype=dtype)
    wx = dec_cell.W_x(x1)
"""
)

code(
    """
# can_decode_step(cell, ref) uses ref's device/dtype.
if can_decode_step(dec_cell, carry):
    with torch.no_grad():
        got = decode_step(dec_cell, carry, wx=wx)
        ref = dec_cell.step(carry, x1, wx=wx)
    float((got - ref).abs().amax()), tuple(got.shape)
else:
    with torch.no_grad():
        ref = dec_cell.step(carry, x1, wx=wx)
    ("eager-only", tuple(ref.shape))
"""
)

md(
    r"""
## 5. Next steps

- \(\mathbb{Z}_2\) prefix tagging (ParaSLSTM vs S4D-Real): [`examples/parity.py`](https://github.com/bugkira/pararnn-torch/blob/main/examples/parity.py)
- Benches / BabyLM: [`scripts/README.md`](https://github.com/bugkira/pararnn-torch/blob/main/scripts/README.md)

```bibtex
@misc{sereda2026paraslstm,
  author       = {Sereda, Daniil},
  title        = {{ParaSLSTM}: Work-Efficient Parallel Training of Nonlinear {sLSTM} via Tropical Warm-Starts},
  month        = sep,
  year         = 2026,
  note         = {Version 2},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22558086},
  url          = {https://doi.org/10.5281/zenodo.22558086}
}

@inproceedings{danieli2026pararnn,
  title        = {{ParaRNN}: Unlocking Parallel Training of Nonlinear {RNNs} for Large Language Models},
  author       = {Danieli, Federico and Rodr{\'i}guez, Pau and Sarabia, Miguel and Suau, Xavier and Zappella, Luca},
  booktitle    = {International Conference on Learning Representations},
  year         = {2026},
  note         = {Oral. arXiv:2510.21450},
  url          = {https://arxiv.org/abs/2510.21450}
}
```
"""
)

nb.cells = cells
out = Path(__file__).resolve().parent / "paraslstm_demo.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
