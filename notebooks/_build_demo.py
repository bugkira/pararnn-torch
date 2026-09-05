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
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22302587.svg)](https://doi.org/10.5281/zenodo.22302587)

Parallel **training** of nonlinear sLSTM via Newton iterations and an associative scan (Danieli et al., ICLR 2026), with a tropical warm-start for exponentially gated cells (ParaSLSTM preprint).

**Run All** (~1–3 min on a Colab T4 / lab GPU): drop-in API → sequential↔Newton agreement → latency vs \(T\) → short \(\mathbb{Z}_2\) parity vs a linear SSM → T=1 decode step.

Use **Runtime → Change runtime type → GPU**. Demo tensors stay in `float32` so free-tier T4 (SM 7.5) works; `NewtonConfig(scan_backend="auto")` picks fused Triton when the GPU supports it.
"""
)

md(
    """
## 0. Setup

**Colab (private repo):** add a secret `GITHUB_TOKEN` with `repo` read access
(Runtime → Secrets), then run the next cell.

**Local:** skip the install cell if you already `uv sync`'d this checkout.
"""
)

code(
    """
# Install: Colab only. Local editable installs already provide `pararnn`.
import os
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import userdata

    token = os.environ.get("GITHUB_TOKEN") or userdata.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Set Colab secret GITHUB_TOKEN (repo read) before installing "
            "from the private GitHub source."
        )
    url = (
        f"git+https://x-access-token:{token}@github.com/"
        "bugkira/pararnn-torch.git"
    )
    # %pip keeps the Colab kernel in sync
    get_ipython().run_line_magic("pip", f'install -q "pararnn-torch @ {url}" matplotlib')
else:
    print("Local / non-Colab: expecting an existing pararnn install (uv sync).")
"""
)

code(
    """
import math
import time
import logging

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pararnn import (
    NewtonConfig,
    ParaRNN,
    ParaSLSTM,
    can_decode_step,
    decode_step,
    newton_apply,
    sequential_apply,
)
from pararnn.solvers.scan import scan_diag

# Library logs WARNING ``newton_residual_high`` when max|F| > 1e-3 after K
# (still under residual_fail). Demo stdout stays on step metrics; divergence
# still raises NewtonDivergenceError at ERROR.
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
        f"CC {props.major}.{props.minor}  "
        f"dtype={dtype}  scan_backend={cfg.scan_backend!r}"
    )
else:
    print("CUDA unavailable: trust + decode still run; timing plot uses CPU events.")
"""
)

md(
    """
## 1. Drop-in quickstart

`ParaRNN` in `.train()` runs Newton + scan; in `.eval()` it runs the sequential `step` unroll.
"""
)

code(
    """
cell = ParaSLSTM(64, 64, mix="diag", device=device, dtype=dtype)
model = ParaRNN(cell, config=cfg).to(device=device, dtype=dtype)
x = torch.randn(4, 256, 64, device=device, dtype=dtype)

model.train()
y = model(x)
y.sum().backward()
print(
    f"train  y={tuple(y.shape)}  "
    f"grad_ok={next(model.parameters()).grad is not None}"
)

model.eval()
with torch.no_grad():
    y_eval = model(x)
print(f"eval   y={tuple(y_eval.shape)}")
"""
)

md(
    r"""
## 2. Trust: sequential unroll vs Newton

With the library \(K{=}3\) and the sLSTM tropical warm-start, parallel Newton should match the sequential trajectory within a small absolute error.
"""
)

code(
    """
# Shorter T on CPU to keep Run All snappy.
seq_lengths = [128, 512, 1024, 2048] if device.type == "cuda" else [64, 128, 256]
dim = 128 if device.type == "cuda" else 64
trust_cell = ParaSLSTM(dim, dim, mix="diag", device=device, dtype=dtype).eval()

print(f"{'T':>6}  {'max|h_seq - h_newton|':>22}")
print("-" * 32)
for T in seq_lengths:
    xt = torch.randn(2, T, dim, device=device, dtype=dtype)
    with torch.no_grad():
        h_seq = sequential_apply(trust_cell, xt)
        h_par = newton_apply(trust_cell, xt, config=cfg)
        err = float((h_seq - h_par).abs().amax())
    print(f"{T:6d}  {err:22.4e}")
"""
)

md(
    """
## 3. Latency vs sequence length

Same Diag-sLSTM cell: sequential `step` unroll vs Newton (`scan_backend="auto"`). Medians over a few CUDA (or CPU) runs after warmup.
"""
)

code(
    """
def _sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median_ms(fn, *, warmup=5, runs=15) -> float:
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


T_list = [64, 256, 1024, 2048] if device.type == "cuda" else [64, 128, 256]
bench_dim, bench_B = (128, 8) if device.type == "cuda" else (64, 2)
bench_cell = ParaSLSTM(bench_dim, bench_dim, mix="diag", device=device, dtype=dtype).eval()

times_seq, times_newton = [], []
with torch.no_grad():
    for T in T_list:
        xb = torch.randn(bench_B, T, bench_dim, device=device, dtype=dtype)
        times_seq.append(median_ms(lambda xb=xb: sequential_apply(bench_cell, xb), warmup=3, runs=9))
        times_newton.append(
            median_ms(lambda xb=xb: newton_apply(bench_cell, xb, config=cfg), warmup=3, runs=9)
        )
        print(
            f"T={T:4d}  sequential={times_seq[-1]:7.2f} ms  "
            f"newton={times_newton[-1]:7.2f} ms"
        )

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
ax.plot(T_list, times_seq, "o--", label="sequential step unroll", color="#c0392b")
ax.plot(T_list, times_newton, "s-", label="Newton + scan (auto)", color="#27ae60")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("sequence length T")
ax.set_ylabel("forward median latency (ms)")
ax.set_title(f"Diag-sLSTM forward  B={bench_B} d_h={bench_dim}  ({device})")
ax.legend()
ax.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout()
plt.show()
plt.close(fig)
"""
)

md(
    r"""
## 4. Expressivity: \(\mathbb{Z}_2\) prefix tagging

Running XOR (Merrill et al.): train \(T{=}16\), eval also at \(T{=}32\). **ParaSLSTM** (K=3, P=3) vs **S4D-Real** SSM.
"""
)

code(
    """
# Models + data helpers (run once)
VOCAB, SEQ_LEN, EVAL_T, BATCH, D_H = 2, 16, 32, 16, 32
STEPS, WARMUP, LR, SEED = 2000, 200, 1e-3, 0
if device.type != "cuda":
    STEPS = 400


def sample_parity(n, t, *, g=None):
    bits = torch.randint(0, 2, (n, t), generator=g)
    return bits, bits.cumsum(1) % 2


class Residual(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        return x + self.inner(x)


class S6Block(nn.Module):
    \"\"\"Pre-norm residual S4D-Real (diagonal selective SSM).\"\"\"

    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm, self.dt_proj = nn.LayerNorm(d), nn.Linear(d, d)
        self.B_proj, self.C_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.log_A = nn.Parameter(torch.log(torch.arange(1, d + 1, dtype=torch.float32)))
        self.dt_bias = nn.Parameter(torch.linspace(math.log(1e-3), math.log(1e-1), d))

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        dt = F.softplus(self.dt_proj(z) + self.dt_bias)
        h = scan_diag(torch.exp(-dt * torch.exp(self.log_A)), self.B_proj(z) * z, backend="eager")
        return x + self.C_proj(h)


def parity_model(arm: str) -> nn.Module:
    cfg = NewtonConfig(max_iters=3, picard_iters=3, scan_backend="auto")
    if arm == "ssm":
        body: list[nn.Module] = [S6Block(D_H), S6Block(D_H)]
    else:
        def block() -> nn.Module:
            return Residual(
                nn.Sequential(
                    nn.LayerNorm(D_H),
                    ParaRNN(
                        ParaSLSTM(D_H, D_H, mix="diag", max_recurrent_norm=None),
                        config=cfg,
                        output_hidden=True,
                        solver="auto",
                    ),
                )
            )

        body = [block(), block()]
    return nn.Sequential(nn.Embedding(VOCAB, D_H), *body, nn.Linear(D_H, VOCAB))


print(f"parity helpers ready (STEPS={STEPS})")
"""
)

code(
    """
# Train both arms (~1–2 min on GPU)
@torch.no_grad()
def last_acc(model, bits, labels):
    was = model.training
    model.eval()
    acc = float((model(bits).argmax(-1) == labels)[:, -1].float().mean())
    model.train(was)
    return acc


def train_arm(arm: str) -> dict:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = parity_model(arm).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(STEPS - WARMUP, 1))
    g, eg = torch.Generator().manual_seed(SEED), torch.Generator().manual_seed(SEED + 1)
    ev16 = tuple(t.to(device) for t in sample_parity(BATCH * 4, SEQ_LEN, g=eg))
    ev32 = tuple(t.to(device) for t in sample_parity(BATCH * 4, EVAL_T, g=eg))
    steps, h16, h32 = [], [], []
    for step in range(STEPS):
        if step < WARMUP:
            for pg in opt.param_groups:
                pg["lr"] = LR * (step + 1) / WARMUP
        bits, lab = (t.to(device) for t in sample_parity(BATCH, SEQ_LEN, g=g))
        loss = F.cross_entropy(model(bits).reshape(-1, VOCAB), lab.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step >= WARMUP:
            sched.step()
        if step == 0 or (step + 1) % 200 == 0 or step + 1 == STEPS:
            a16, a32 = last_acc(model, *ev16), last_acc(model, *ev32)
            steps.append(step)
            h16.append(a16)
            h32.append(a32)
            print(f"{arm:6s} {step:04d}  ce={loss.item():.4f}  @16={a16:.3f}  @32={a32:.3f}")
    return {"steps": steps, "t16": h16, "t32": h32}


curves = {arm: train_arm(arm) for arm in ("newton", "ssm")}
"""
)

code(
    """
# Plot last-token accuracy
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=120)
for arm, style in (("newton", "-"), ("ssm", "--")):
    c = curves[arm]
    ax.plot(c["steps"], c["t16"], style, label=f"{arm} T=16")
    ax.plot(c["steps"], c["t32"], style, alpha=0.7, label=f"{arm} T=32")
ax.set(xlabel="step", ylabel="last-token accuracy", ylim=(-0.05, 1.05))
ax.set_title(r"$\\mathbb{Z}_2$ prefix tagging")
ax.legend(fontsize=8)
ax.grid(True, ls=":", alpha=0.5)
plt.tight_layout()
plt.show()
plt.close(fig)
"""
)

md(
    """
## 5. Decode step (T=1)

On CUDA, `decode_step` runs the recurrent step in one Triton kernel when the cell supports it. Agreement check vs eager `cell.step`.
"""
)

code(
    """
dec_cell = ParaSLSTM(64, 64, mix="diag", device=device, dtype=dtype).eval()
# Prefill a carry with a short sequential pass.
with torch.no_grad():
    carry = sequential_apply(
        dec_cell, torch.randn(4, 8, 64, device=device, dtype=dtype)
    )[:, -1]
    x1 = torch.randn(4, 64, device=device, dtype=dtype)
    wx = dec_cell.W_x(x1)

# can_decode_step(cell, ref) uses ``ref``'s device/dtype (CUDA fp32/fp16; bf16 needs SM≥8).
if can_decode_step(dec_cell, carry):
    with torch.no_grad():
        got = decode_step(dec_cell, carry, wx=wx)
        ref = dec_cell.step(carry, x1, wx=wx)
        err = float((got - ref).abs().amax())
    print(f"decode_step OK  max|got-ref|={err:.4e}  shape={tuple(got.shape)}")
else:
    with torch.no_grad():
        ref = dec_cell.step(carry, x1, wx=wx)
    print(
        f"decode_step unavailable on this device/dtype; "
        f"eager step shape={tuple(ref.shape)}"
    )
"""
)

md(
    r"""
## 6. Citation

```bibtex
@misc{sereda2026paraslstm,
  author       = {Sereda, Daniil},
  title        = {{ParaSLSTM}: Work-Efficient Parallel Training of Nonlinear {sLSTM} via Tropical Warm-Starts},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22302587},
  url          = {https://doi.org/10.5281/zenodo.22302587}
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

More benches and BabyLM training: [`scripts/README.md`](https://github.com/bugkira/pararnn-torch/blob/main/scripts/README.md) in the repo.
"""
)

nb.cells = cells
out = Path(__file__).resolve().parent / "paraslstm_demo.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
