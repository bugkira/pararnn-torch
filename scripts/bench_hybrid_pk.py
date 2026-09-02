"""Picard P × Newton K hybrid ablation, then optional S4D-Real SSM guess.

  uv run python scripts/bench_hybrid_pk.py

Does **not** add Mamba-2 weights to the cell. Picard is frozen-gate scans of
the same sLSTM (para-slstm.md). K=1 is the DEER hook: one Newton if the guess
is in the quadratic basin (Lim et al. 2024). Library contract stays P from T
and K=3; this script only measures.

Phase 1: P ∈ {0,1,3,5} × K ∈ {0,1,3}. P rungs: slstm_auto_picard / next.md.
K=0 is the guess vs sequential (predictor quality). K=3 is App. A / library.
Shapes: Dyck smoke (B=32 T=64 d_h=32) and fused table (B=8 d_h=256,
x_scale=1, T∈{64,256,1024,2048}). Protocol 10/50 smoke, not App. B.
GPU: 2080 Ti by name. Explicit P, picard_adapt off (bench path).

Win for K=1: max|par−seq| below agree_tol (1e-3 at d_h=256, 1e-4 at Dyck)
AND fwd min_ms < 0.85× library (P=1 K=3 at T≤64, P=3 K=3 otherwise).

Phase 2 (only if no K=1 win): S4D-Real diagonal SSM on the candidate chunk
(Gu & Dao 2023 Δ∈[10^{-3},10^{-1}]; A_n=n+1), one frozen-gate scan for
(c,n,m), then K=1. Untrained extra maps — solver guess, not a new LM.
Not the mamba-ssm package (Turing / no extra CUDA). Ours, not Apple.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import torch
from torch import Tensor
from torch.nn import functional as F

from examples.dyck_language import VOCAB, sample_dyck1
from examples.slstm_vs_flashrnn import _NewtonDyckLM
from pararnn import NewtonConfig, ParaSLSTM
from pararnn.kernels.newton_slstm import newton_slstm_fused
from pararnn.layout import SLSTM_HIDDEN, prepend_state
from pararnn.solvers import NewtonStats, newton_apply, sequential_apply
from pararnn.solvers.scan import scan_diag
from pararnn.solvers.slstm_picard import slstm_frozen_gate_scan
from utils.cuda_timing import cuda_minmax
from utils.mlflow_helper import git_commit, lock_hash, setup_logging, uv_export_hash

from gpu import (
    DEFAULT_EXPERIMENT_GPU_NAME,
    select_device,
    wait_until_free,
)

log = logging.getLogger("hybrid_pk")

WARMUP = 10
N_RUNS = 50
# Library Picard ladder (slstm_auto_picard). P=0 is zero-hidden (diverged
# long-T table in newton_slstm.yaml).
PICARD = (0, 1, 3, 5)
# K=0 = guess only; K=1 = hybrid claim; K=3 = App. A / library.
NEWTON_K = (0, 1, 3)
# Dyck yaml. 1e-4 is the numerics-test band at this width.
DYCK_BATCH, DYCK_T, DYCK_DH, DYCK_STEPS, DYCK_LR = 32, 64, 32, 50, 3e-3
DYCK_AGREE = 1e-4
# fused sLSTM smoke (newton_slstm_picard.yaml). agree_tol 1e-3: LM-length
# residual at d_h=256, not GRU 1e-4.
BENCH_BATCH, BENCH_DH, X_SCALE = 8, 256, 1.0
BENCH_TS = (64, 256, 1024, 2048)
BENCH_AGREE = 1e-3
# Win: at least 15% faster fwd than library K=3. Not a 2% clock wiggle.
SPEED_WIN = 0.85
# Mamba Δ range: Gu & Dao 2023 §3.2 / literature.md. Deterministic linspace,
# not a random draw per run.
MAMBA_DT_MIN, MAMBA_DT_MAX = 1e-3, 1e-1


def _cfg(picard: int, max_iters: int) -> NewtonConfig:
    return NewtonConfig(
        max_iters=max_iters,
        scan_backend="fused",
        picard_iters=picard,
        picard_adapt=False,
        residual_atol=None,
        residual_fail=None,
    )


def _library_pk(seq_len: int) -> tuple[int, int]:
    """Auto P from T, K=3. slstm_auto_picard / App. A."""
    from pararnn.solvers.newton import slstm_auto_picard

    return slstm_auto_picard(seq_len), 3


def _agree_tol(d_h: int) -> float:
    return DYCK_AGREE if d_h <= 32 else BENCH_AGREE


def _quality(cell: ParaSLSTM, x: Tensor, seq: Tensor, picard: int, max_iters: int) -> dict:
    st = NewtonStats()
    par = newton_apply(cell, x, _cfg(picard, max_iters), stats=st)
    err = float((par.detach() - seq.detach()).abs().amax())
    return {
        "err": err,
        "residual": st.max_residual,
        "iters": st.iters,
        "snaps": err < _agree_tol(cell.d_h) and math.isfinite(err),
    }


def _fwd_time(cell: ParaSLSTM, x: Tensor, picard: int, max_iters: int) -> dict:
    cfg = _cfg(picard, max_iters)

    def fn() -> None:
        with torch.no_grad():
            newton_apply(cell, x, cfg)

    tmin, tmed, tmean = cuda_minmax(fn, warmup=WARMUP, n_runs=N_RUNS)
    return {"min_ms": tmin, "median_ms": tmed, "mean_ms": tmean}


def _s4d_hidden(wx: Tensor) -> Tensor:
    """Untrained S4D-Real leaky integrator on the candidate chunk of ``wx``.

    A_n = n+1 (Gu et al. S4D-Real). Δ linspace in Mamba's [1e-3, 1e-1].
    h_t = a ⊙ h_{t-1} + (1-a) ⊙ z_t with a = exp(-Δ A). scan_diag, not a
    time loop. Does not train extra weights.
    """
    _, _, four_d = wx.shape
    d_h = four_d // 4
    _i, _f, z, _o = wx.chunk(4, dim=-1)
    idx = torch.arange(d_h, device=wx.device, dtype=torch.float32)
    a_hi = -(idx + 1.0)
    dt = torch.linspace(MAMBA_DT_MIN, MAMBA_DT_MAX, d_h, device=wx.device)
    decay = torch.exp(a_hi * dt).to(dtype=wx.dtype)
    a = decay.view(1, 1, d_h).expand_as(z).contiguous()
    drive = (1.0 - decay.to(dtype=wx.dtype)).view(1, 1, d_h) * z
    backend = "triton" if z.is_cuda else "eager"
    return scan_diag(a, drive.contiguous(), backend=backend)


def _ssm_guess(cell: ParaSLSTM, x: Tensor) -> Tensor:
    """SSM h, then one frozen-gate scan (Picard-style (c,n,m) given that h)."""
    wx = cell.W_x(x)
    h = _s4d_hidden(wx)
    fake = wx.new_zeros(wx.shape[0], wx.shape[1], 4, cell.d_h)
    fake[..., SLSTM_HIDDEN, :] = h
    h_prev = prepend_state(fake, None)[..., SLSTM_HIDDEN, :]
    pre = wx + cell._recurrent(h_prev)
    return slstm_frozen_gate_scan(pre, eps=cell.eps)


def _ssm_newton(cell: ParaSLSTM, x: Tensor, max_iters: int) -> Tensor:
    wx = cell.W_x(x)
    states = _ssm_guess(cell, x)
    return newton_slstm_fused(
        wx,
        cell.clipped_r(),
        max_iters=max_iters,
        omega=1.0,
        eps=cell.eps,
        states=states,
    )


def _ssm_quality(cell: ParaSLSTM, x: Tensor, seq: Tensor, max_iters: int) -> dict:
    par = _ssm_newton(cell, x, max_iters)
    with torch.no_grad():
        h_prev = prepend_state(par, None)
        pred = cell.step(h_prev, x, wx=cell.W_x(x))
        res = float((pred.detach() - par.detach()).abs().amax())
    err = float((par.detach() - seq.detach()).abs().amax())
    return {
        "err": err,
        "residual": res,
        "iters": max_iters,
        "snaps": err < _agree_tol(cell.d_h) and math.isfinite(err),
    }


def _ssm_fwd_time(cell: ParaSLSTM, x: Tensor, max_iters: int) -> dict:
    def fn() -> None:
        with torch.no_grad():
            _ssm_newton(cell, x, max_iters)

    tmin, tmed, tmean = cuda_minmax(fn, warmup=WARMUP, n_runs=N_RUNS)
    return {"min_ms": tmin, "median_ms": tmed, "mean_ms": tmean}


def _dyck_train(device: torch.device, picard: int | None, max_iters: int) -> dict:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cfg = NewtonConfig(
        max_iters=max_iters,
        scan_backend="fused",
        picard_iters=picard,
        picard_adapt=picard is None,
    )
    model = _NewtonDyckLM(DYCK_DH, cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=DYCK_LR, weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(0)
    losses: list[float] = []
    step_ms: list[float] = []
    for step in range(DYCK_STEPS):
        tokens = sample_dyck1(DYCK_BATCH, DYCK_T, generator=gen).to(device)

        def _step(tok: Tensor = tokens) -> Tensor:
            logits = model(tok[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tok[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return loss

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        loss = _step()
        end.record()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        step_ms.append(start.elapsed_time(end))
        log.info(
            "dyck train P=%s K=%d step=%02d ce=%.4f tot=%.2f ms",
            picard if picard is not None else "auto",
            max_iters,
            step,
            losses[-1],
            step_ms[-1],
        )
    warm = 5
    steady = step_ms[warm:]
    return {
        "ce0": losses[0],
        "ce_final": losses[-1],
        "min_ms": min(steady),
        "mean_ms": sum(steady) / len(steady),
        "losses": losses,
    }


def _shape_x(device: torch.device, batch: int, seq_len: int, d_h: int, scale: float) -> Tensor:
    return scale * torch.randn(batch, seq_len, d_h, device=device)


def _is_win(row: dict, baseline_ms: float) -> bool:
    if row["k"] != 1 or not row["snaps"]:
        return False
    return row["min_ms"] < SPEED_WIN * baseline_ms


def main() -> None:
    setup_logging()
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    torch.cuda.set_device(device)
    wait_until_free(device)
    log.info(
        "hybrid_pk gpu=%s warmup=%d n_runs=%d",
        torch.cuda.get_device_name(device),
        WARMUP,
        N_RUNS,
    )

    import mlflow

    shapes = [
        ("dyck", DYCK_BATCH, DYCK_T, DYCK_DH, 1.0),
        *[("bench", BENCH_BATCH, t, BENCH_DH, X_SCALE) for t in BENCH_TS],
    ]

    mlflow.set_experiment("newton-slstm-bench")
    with mlflow.start_run(run_name="hybrid-picard-k-then-ssm"):
        mlflow.set_tags(
            {
                "gpu": torch.cuda.get_device_name(device),
                "cell": "para_slstm",
                "mix": "diag",
                "mode": "hybrid_pk",
                "dtype": "fp32",
            }
        )
        mlflow.log_params(
            {
                "warmup": WARMUP,
                "n_runs": N_RUNS,
                "picard": ",".join(map(str, PICARD)),
                "newton_k": ",".join(map(str, NEWTON_K)),
                "speed_win": SPEED_WIN,
                "git_commit": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
            }
        )
        mlflow.log_text(
            "Picard P × Newton K, then S4D-Real guess only if K=1 is not a "
            "quality+speed win vs library P(T)+K=3. Guess is not a new cell.\n",
            "why.txt",
        )

        rows: list[dict] = []
        baselines: dict[str, float] = {}
        for name, batch, seq_len, d_h, scale in shapes:
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            cell = ParaSLSTM(d_in=d_h, d_h=d_h, mix="diag").to(device)
            x = _shape_x(device, batch, seq_len, d_h, scale)
            with torch.no_grad():
                seq = sequential_apply(cell, x)
            lib_p, lib_k = _library_pk(seq_len)
            key = f"{name}_T{seq_len}"
            log.info(
                "shape %s B=%d T=%d d_h=%d library P=%d K=%d agree_tol=%g",
                name,
                batch,
                seq_len,
                d_h,
                lib_p,
                lib_k,
                _agree_tol(d_h),
            )
            for p in PICARD:
                for k in NEWTON_K:
                    q = _quality(cell, x, seq, p, k)
                    t = _fwd_time(cell, x, p, k)
                    row = {
                        "phase": "picard",
                        "shape": name,
                        "T": seq_len,
                        "d_h": d_h,
                        "p": p,
                        "k": k,
                        **q,
                        **t,
                    }
                    rows.append(row)
                    log.info(
                        "picard P=%d K=%d T=%d  err=%.3e res=%.3e  min=%.3f med=%.3f snaps=%s",
                        p,
                        k,
                        seq_len,
                        q["err"],
                        q["residual"],
                        t["min_ms"],
                        t["median_ms"],
                        q["snaps"],
                    )
                    step = seq_len
                    tag = f"{name}/P{p}K{k}"
                    mlflow.log_metric(f"{tag}/err", q["err"], step=step)
                    mlflow.log_metric(f"{tag}/residual", q["residual"], step=step)
                    mlflow.log_metric(f"{tag}/min_ms", t["min_ms"], step=step)
                    if p == lib_p and k == lib_k:
                        baselines[key] = t["min_ms"]
                        mlflow.log_metric(f"{name}/library_min_ms", t["min_ms"], step=step)

        wins = []
        for row in rows:
            key = f"{row['shape']}_T{row['T']}"
            base = baselines.get(key)
            if base is None:
                continue
            row["vs_library"] = row["min_ms"] / base
            if _is_win(row, base):
                wins.append(row)
                log.info(
                    "WIN P=%d K=1 T=%d  %.3f ms vs library %.3f (%.2f×)",
                    row["p"],
                    row["T"],
                    row["min_ms"],
                    base,
                    base / row["min_ms"],
                )
        mlflow.log_param("k1_wins", len(wins))
        hard = [r for r in rows if r["k"] == 1 and r["T"] >= 1024]
        hard_snaps = [r for r in hard if r["snaps"]]
        long_wins = [w for w in wins if w["T"] >= 1024]
        need_ssm = not long_wins
        log.info(
            "phase1 K=1 snaps at T>=1024: %d / %d; wins vs library: %d (long T: %d); ssm=%s",
            len(hard_snaps),
            len(hard),
            len(wins),
            len(long_wins),
            need_ssm,
        )
        mlflow.log_param("need_ssm", int(need_ssm))

        if need_ssm:
            log.info("no K=1 speed+quality win; S4D-Real SSM guess + K in {0,1,3}")
            for name, batch, seq_len, d_h, scale in shapes:
                if name == "dyck":
                    continue
                if seq_len not in (1024, 2048) and seq_len != 64:
                    continue
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                cell = ParaSLSTM(d_in=d_h, d_h=d_h, mix="diag").to(device)
                x = _shape_x(device, batch, seq_len, d_h, scale)
                with torch.no_grad():
                    seq = sequential_apply(cell, x)
                for k in NEWTON_K:
                    q = _ssm_quality(cell, x, seq, k)
                    t = _ssm_fwd_time(cell, x, k)
                    log.info(
                        "ssm K=%d T=%d  err=%.3e res=%.3e  min=%.3f med=%.3f snaps=%s",
                        k,
                        seq_len,
                        q["err"],
                        q["residual"],
                        t["min_ms"],
                        t["median_ms"],
                        q["snaps"],
                    )
                    tag = f"ssm/T{seq_len}K{k}"
                    mlflow.log_metric(f"{tag}/err", q["err"], step=seq_len)
                    mlflow.log_metric(f"{tag}/residual", q["residual"], step=seq_len)
                    mlflow.log_metric(f"{tag}/min_ms", t["min_ms"], step=seq_len)
                    rows.append(
                        {
                            "phase": "ssm",
                            "shape": name,
                            "T": seq_len,
                            "d_h": d_h,
                            "p": -1,
                            "k": k,
                            **q,
                            **t,
                        }
                    )
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            cell = ParaSLSTM(d_in=DYCK_DH, d_h=DYCK_DH, mix="diag").to(device)
            x = _shape_x(device, DYCK_BATCH, DYCK_T, DYCK_DH, 1.0)
            with torch.no_grad():
                seq = sequential_apply(cell, x)
            q = _ssm_quality(cell, x, seq, 1)
            t = _ssm_fwd_time(cell, x, 1)
            log.info(
                "ssm K=1 dyck  err=%.3e res=%.3e  min=%.3f snaps=%s",
                q["err"],
                q["residual"],
                t["min_ms"],
                q["snaps"],
            )
            mlflow.log_metric("ssm/dyck/err", q["err"])
            mlflow.log_metric("ssm/dyck/min_ms", t["min_ms"])

        log.info("dyck 50-step train: library auto P K=3 vs explicit P=3 K=1")
        lib_train = _dyck_train(device, None, 3)
        k1_train = _dyck_train(device, 3, 1)
        log.info(
            "dyck library CE %.4f → %.4f mean_step=%.2f  "
            "P=3 K=1 CE %.4f → %.4f mean_step=%.2f  max|ΔCE|=%.3e",
            lib_train["ce0"],
            lib_train["ce_final"],
            lib_train["mean_ms"],
            k1_train["ce0"],
            k1_train["ce_final"],
            k1_train["mean_ms"],
            max(abs(a - b) for a, b in zip(lib_train["losses"], k1_train["losses"], strict=True)),
        )
        mlflow.log_metric("dyck/lib_ce0", lib_train["ce0"])
        mlflow.log_metric("dyck/lib_ce_final", lib_train["ce_final"])
        mlflow.log_metric("dyck/lib_mean_ms", lib_train["mean_ms"])
        mlflow.log_metric("dyck/p3k1_ce0", k1_train["ce0"])
        mlflow.log_metric("dyck/p3k1_ce_final", k1_train["ce_final"])
        mlflow.log_metric("dyck/p3k1_mean_ms", k1_train["mean_ms"])


if __name__ == "__main__":
    main()
