"""Why eager segmented scan grows faster than Blelloch with T.

Hillis–Steele does ~T log T composes and clones (B, T, *) each doubling.
Blelloch does ~2T composes on a power-of-two tree (strided gather).

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/diag_seg_scan.py
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import torch
from torch.profiler import ProfilerActivity, profile

from pararnn.solvers.scan import (
    _blelloch_inclusive,
    _compose_diag,
    _compose_seg,
    _fill_ident_diag,
    _hillis_steele_seg_inclusive,
    scan_diag,
)
from utils.cuda_timing import cuda_minmax

from gpu import select_device, setup_logging

log = logging.getLogger("diag_seg_scan")

BATCH = 8
D_H = 256
WARMUP = 10
RUNS = 30


def _hs_work(time: int) -> int:
    """Elements composed (right slice length) over Hillis–Steele rounds."""
    total = 0
    step = 1
    while step < time:
        total += time - step
        step *= 2
    return total


def _blelloch_work(time: int) -> int:
    """Compose sites in upsweep + downsweep on the padded length."""
    n = 1 << (time - 1).bit_length()
    up = 0
    step = 2
    while step <= n:
        up += n // step
        step *= 2
    return 2 * up


def _pingpong_hs(jac: torch.Tensor, residual: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """Same O(T log T) compose, double-buffer instead of clone-the-whole-tensor."""
    time = residual.shape[1]
    if time <= 1:
        return residual.clone()
    j_a, r_a, f_a = jac.clone(), residual.clone(), flags.clone()
    j_b, r_b, f_b = jac.clone(), residual.clone(), flags.clone()
    src_j, src_r, src_f = j_a, r_a, f_a
    dst_j, dst_r, dst_f = j_b, r_b, f_b
    step = 1
    while step < time:
        j_c, r_c, f_c = _compose_seg(
            _compose_diag,
            src_j[:, step:],
            src_r[:, step:],
            src_f[:, step:],
            src_j[:, :-step],
            src_r[:, :-step],
            src_f[:, :-step],
        )
        dst_j[:, :step] = src_j[:, :step]
        dst_r[:, :step] = src_r[:, :step]
        dst_f[:, :step] = src_f[:, :step]
        dst_j[:, step:] = j_c
        dst_r[:, step:] = r_c
        dst_f[:, step:] = f_c
        src_j, dst_j = dst_j, src_j
        src_r, dst_r = dst_r, src_r
        src_f, dst_f = dst_f, src_f
        step *= 2
    return src_r


def _seg_blelloch(jac: torch.Tensor, residual: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """Blelloch tree with the same head-flag drop as Hillis–Steele.

    Identity pad (T, n) is a fresh span so packed length stays one segment
    unless ``flags`` already has heads inside ``[:time]``.
    """
    time = residual.shape[1]
    if time <= 1:
        return residual.clone()
    n = 1 << (time - 1).bit_length()
    j = jac.new_empty(jac.shape[0], n, *jac.shape[2:])
    r = residual.new_empty(residual.shape[0], n, *residual.shape[2:])
    f = flags.new_zeros(flags.shape[0], n)
    j[:, :time] = jac
    r[:, :time] = residual
    f[:, :time] = flags
    if n > time:
        j[:, time:] = 1
        r[:, time:] = 0
        f[:, time] = True
    from pararnn.solvers.scan import _compose_seg as compose_seg

    step = 2
    while step <= n:
        right = torch.arange(step - 1, n, step, device=j.device)
        left = right - (step // 2)
        j[:, right], r[:, right], f[:, right] = compose_seg(
            _compose_diag,
            j[:, right],
            r[:, right],
            f[:, right],
            j[:, left],
            r[:, left],
            f[:, left],
        )
        step *= 2
    j[:, n - 1] = 1
    r[:, n - 1] = 0
    f[:, n - 1] = True
    step = n
    while step >= 2:
        right = torch.arange(step - 1, n, step, device=j.device)
        left = right - (step // 2)
        j_left, r_left, f_left = j[:, left].clone(), r[:, left].clone(), f[:, left].clone()
        j[:, left], r[:, left], f[:, left] = j[:, right], r[:, right], f[:, right]
        j[:, right], r[:, right], f[:, right] = compose_seg(
            _compose_diag,
            j_left,
            r_left,
            f_left,
            j[:, right],
            r[:, right],
            f[:, right],
        )
        step //= 2
    prefix = r[:, :time]
    return jac * prefix + residual


def _bytes_jr(time: int) -> int:
    return BATCH * time * D_H * 4 * 2


def main() -> None:
    setup_logging()
    device = select_device("3060")
    torch.cuda.set_device(device)
    log.info(
        "diag_seg_scan gpu=%s B=%d d_h=%d (cell_forward.yaml)",
        torch.cuda.get_device_name(device),
        BATCH,
        D_H,
    )
    for time in (512, 2048, 8192):
        hs_w = _hs_work(time)
        bl_w = _blelloch_work(time)
        rounds = math.ceil(math.log2(time))
        clone_mib = rounds * _bytes_jr(time) / (1024**2)
        log.info(
            "work T=%d hs_el=%d blelloch_el=%d ratio=%.2f hs_rounds=%d "
            "clone_jr=%.1f MiB (j+r × rounds)",
            time,
            hs_w,
            bl_w,
            hs_w / max(bl_w, 1),
            rounds,
            clone_mib,
        )

    for time in (512, 2048, 8192):
        g = torch.Generator(device=device)
        g.manual_seed(0)
        jac = torch.rand(BATCH, time, D_H, device=device, generator=g) * 0.3
        residual = torch.randn(BATCH, time, D_H, device=device, generator=g)
        flags = torch.zeros(BATCH, time, dtype=torch.bool, device=device)
        flags[:, 0] = True
        cs = torch.tensor([0, time], device=device, dtype=torch.long)
        cs2 = torch.tensor([0, time // 2, time], device=device, dtype=torch.long)

        # Default-arg binds close over this iteration's tensors (B023).
        arms = {
            "blelloch": lambda j=jac, r=residual: _blelloch_inclusive(
                j, r, _compose_diag, _fill_ident_diag
            ),
            "hs_clone": lambda j=jac, r=residual, f=flags: _hillis_steele_seg_inclusive(
                j, r, _compose_diag, f
            ),
            "hs_pingpong": lambda j=jac, r=residual, f=flags: _pingpong_hs(j, r, f),
            "scan_cu1": lambda j=jac, r=residual, c=cs: scan_diag(j, r, cu_seqlens=c),
            "scan_cu2": lambda j=jac, r=residual, c=cs2: scan_diag(j, r, cu_seqlens=c),
        }
        # segmented Blelloch is a prototype; skip if it disagrees
        ref = _blelloch_inclusive(jac, residual, _compose_diag, _fill_ident_diag)
        hs = _hillis_steele_seg_inclusive(jac, residual, _compose_diag, flags)
        torch.testing.assert_close(hs, ref, atol=1e-4, rtol=1e-4)
        pp = _pingpong_hs(jac, residual, flags)
        torch.testing.assert_close(pp, ref, atol=1e-4, rtol=1e-4)
        try:
            sb = _seg_blelloch(jac, residual, flags)
            torch.testing.assert_close(sb, ref, atol=1e-3, rtol=1e-3)
            arms["seg_blelloch"] = lambda j=jac, r=residual, f=flags: _seg_blelloch(j, r, f)
        except AssertionError as exc:
            log.warning("seg_blelloch_skip T=%d err=%s", time, exc)

        for name, fn in arms.items():
            tmin, tmed, tmean = cuda_minmax(fn, warmup=WARMUP, n_runs=RUNS)
            log.info(
                "time T=%d arm=%-12s min=%.3f ms median=%.3f mean=%.3f",
                time,
                name,
                tmin,
                tmed,
                tmean,
            )

    # Kernel split at T=8192: clone vs where vs mul
    time = 8192
    g = torch.Generator(device=device)
    g.manual_seed(1)
    jac = torch.rand(BATCH, time, D_H, device=device, generator=g) * 0.3
    residual = torch.randn(BATCH, time, D_H, device=device, generator=g)
    flags = torch.zeros(BATCH, time, dtype=torch.bool, device=device)
    flags[:, 0] = True
    for label, fn in (
        ("hs_clone", lambda: _hillis_steele_seg_inclusive(jac, residual, _compose_diag, flags)),
        ("blelloch", lambda: _blelloch_inclusive(jac, residual, _compose_diag, _fill_ident_diag)),
    ):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=acts, record_shapes=False) as prof:
            fn()
            torch.cuda.synchronize()
        rows = []
        for ev in prof.key_averages():
            cuda_us = ev.self_device_time_total
            if cuda_us <= 0:
                continue
            rows.append((cuda_us / 1000.0, ev.key))
        rows.sort(reverse=True)
        log.info("profiler T=%d arm=%s top CUDA kernels (ms):", time, label)
        for ms, key in rows[:12]:
            log.info("  %7.2f  %s", ms, key)


if __name__ == "__main__":
    main()
