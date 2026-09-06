#!/usr/bin/env python3
"""Regenerate README proof figures from documented lab bench numbers.

Source of truth for the numbers: README Benchmarks section (measured on lab
GPUs). Re-run after updating those tables::

    uv run python scripts/plot_readme_assets.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"

mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_slstm_speedup() -> Path:
    # B=8, d_h=256, float32, RTX 2080 Ti — scripts/slstm_vs_flashrnn.py
    t_vals = [256, 1024, 2048, 4096]
    fused = [6.4, 15.3, 29.1, 69.1]
    seq = [112, 429, 840, 1735]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.plot(
        t_vals,
        fused,
        "o-",
        color="#1f4e79",
        linewidth=2,
        markersize=7,
        label="fused Newton (Alg. 1)",
    )
    ax.plot(
        t_vals,
        seq,
        "s--",
        color="#8b4513",
        linewidth=2,
        markersize=7,
        label="sequential compiled (same cell)",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(t_vals)
    ax.set_xticklabels([str(t) for t in t_vals])
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("Forward median (ms)")
    ax.set_title("ParaSLSTM — parallel Newton train vs sequential time loop")
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.02,
        0.02,
        "B=8, d_h=256, float32, RTX 2080 Ti · scripts/slstm_vs_flashrnn.py",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = OUT / "slstm_fused_vs_sequential.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_k_star_wall() -> Path:
    # RTX 3060, τ=1e-4, B=1 — scripts/bench_k_star.py
    cells = ["ParaCfC", "ParaTitans", "ParaHopfield", "ParaRWKV7"]
    fused_ms = [11, 13, 250, 237]
    seq_ms = [52_000, 62_000, 38_000, 81_000]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    xs = range(len(cells))
    w = 0.36
    bars = ax.bar(
        [i - w / 2 for i in xs],
        fused_ms,
        width=w,
        color="#1f4e79",
        label="fused / scan",
    )
    ax.bar(
        [i + w / 2 for i in xs],
        seq_ms,
        width=w,
        color="#8b4513",
        label="sequential",
    )
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [f"{c}\n{k} Newton steps" for c, k in zip(cells, ["2", "2", "2", "0"], strict=True)]
    )
    ax.set_ylabel("Wall time at T=131072 (ms, log scale)")
    ax.set_title("Parallel train vs sequential loop at sequence length 131k")
    ax.legend(frameon=False)
    for bar, val in zip(bars, fused_ms, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.15,
            f"{val} ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.text(
        0.02,
        0.02,
        "RTX 3060 · agreement tol 1e-4 · B=1 · scripts/bench_k_star.py · lab GPU",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = OUT / "k_star_wall_131k.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for path in (plot_slstm_speedup(), plot_k_star_wall()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
