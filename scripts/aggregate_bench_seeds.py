"""Aggregate per-seed bench CSVs into mean±std of per-seed median_ms.

Reads ``outputs/multiseed/{picard,flashrnn,fused}_seed*.csv`` and writes
``outputs/multiseed/*_agg.csv`` plus a short markdown summary.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MULTI = ROOT / "outputs" / "multiseed"


def _load(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(MULTI.glob(pattern)):
        with path.open() as f:
            for row in csv.DictReader(f):
                row["_file"] = path.name
                rows.append(row)
    return rows


def _aggregate(rows: list[dict], out_name: str) -> None:
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["cell"], row["mode"], str(int(float(row["T"]))))
        buckets[key].append(float(row["median_ms"]))
    out_rows = []

    def _sort_key(item: tuple) -> tuple:
        (cell, mode, T), _vals = item
        return (cell, int(T), mode)

    for (cell, mode, T), vals in sorted(buckets.items(), key=_sort_key):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out_rows.append(
            {
                "cell": cell,
                "mode": mode,
                "T": T,
                "n_seeds": len(vals),
                "median_ms_mean": f"{mean:.6f}",
                "median_ms_std": f"{std:.6f}",
                "median_ms_min": f"{min(vals):.6f}",
                "median_ms_max": f"{max(vals):.6f}",
            }
        )
    out = MULTI / out_name
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            w.writeheader()
            w.writerows(out_rows)
    print(f"wrote {out} ({len(out_rows)} rows from {len(rows)} seed-rows)")


def main() -> None:
    MULTI.mkdir(parents=True, exist_ok=True)
    _aggregate(_load("picard_seed*.csv"), "picard_agg.csv")
    _aggregate(_load("flashrnn_seed*.csv"), "flashrnn_agg.csv")
    _aggregate(_load("fused_seed*.csv"), "fused_agg.csv")

    # Headline ratios at T=2048 for DRAFT
    picard_path = MULTI / "picard_agg.csv"
    flash_path = MULTI / "flashrnn_agg.csv"
    picard = (
        {(r["mode"], r["T"]): r for r in csv.DictReader(picard_path.open())}
        if picard_path.exists()
        else {}
    )
    flash = (
        {(r["mode"], r["T"]): r for r in csv.DictReader(flash_path.open())}
        if flash_path.exists()
        else {}
    )
    lines = ["# Multiseed aggregate (median_ms → mean±std)\n"]
    if ("newton_fused", "2048") in picard and ("sequential_eager", "2048") in picard:
        f = float(picard[("newton_fused", "2048")]["median_ms_mean"])
        e = float(picard[("sequential_eager", "2048")]["median_ms_mean"])
        c = float(picard[("sequential_compiled", "2048")]["median_ms_mean"])
        f_std = float(picard[("newton_fused", "2048")]["median_ms_std"])
        lines.append(
            f"ParaSLSTM T=2048: fused={f:.2f}±{f_std:.2f} ms; "
            f"eager={e:.1f}; compiled={c:.1f}; "
            f"fused/eager={e / f:.1f}×; fused/compiled={c / f:.1f}×\n"
        )
    fr_modes = [k for k in flash if k[1] == "2048" and k[0].startswith("flashrnn")]
    if fr_modes and ("newton_fused", "2048") in flash:
        fr = float(flash[fr_modes[0]]["median_ms_mean"])
        f = float(flash[("newton_fused", "2048")]["median_ms_mean"])
        lines.append(f"FlashRNN T=2048: flashrnn={fr:.2f} ms; fused={f:.2f} ms\n")
    (MULTI / "SUMMARY.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
