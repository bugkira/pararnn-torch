"""Replay Dyck Newton train: residual spike vs dt spikes (not a library API)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from examples.dyck_language import VOCAB, sample_dyck1
from examples.slstm_vs_flashrnn import _NewtonDyckLM
from pararnn import NewtonConfig
from pararnn.layout import SLSTM_HIDDEN
from pararnn.solvers import NewtonStats, sequential_apply
from pararnn.solvers.newton import newton_apply

from gpu import DEFAULT_EXPERIMENT_GPU_NAME, select_device

log = logging.getLogger("diag")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _cuda_ms(fn) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main() -> None:
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    torch.cuda.set_device(device)
    seed, batch, seq_len, d_h, steps, lr = 0, 32, 64, 32, 50, 3e-3
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = NewtonConfig(max_iters=3, scan_backend="fused")
    model = _NewtonDyckLM(d_h, cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cell = model.rnn.layers[0]
    slow: list[int] = []
    bad: list[int] = []

    for step in range(steps):
        tokens = sample_dyck1(batch, seq_len, generator=gen).to(device)
        loss_box: dict[str, torch.Tensor] = {}

        def _fwd(tok: Tensor = tokens, box: dict = loss_box) -> None:
            logits = model(tok[:, :-1])
            box["loss"] = F.cross_entropy(
                logits.reshape(-1, VOCAB), tok[:, 1:].reshape(-1)
            )

        ms_fwd = _cuda_ms(_fwd)
        st = model.rnn.last_stats[0]
        res = st.max_residual
        loss = loss_box["loss"]
        opt.zero_grad(set_to_none=True)

        def _bwd(loss_t: Tensor = loss) -> None:
            loss_t.backward()
            opt.step()

        if res > 1e-3:
            bad.append(step)
            _probe(cell, model.embed(tokens[:, :-1]).detach(), res)

        ms_bwd = _cuda_ms(_bwd)
        tot = ms_fwd + ms_bwd
        with torch.no_grad():
            r_abs = float(cell.R.detach().abs().amax())
            depth = (tokens == 0).int() - (tokens == 1).int()
            max_depth = int(depth.cumsum(1).amax())
        flag = ""
        if res > 1e-3:
            flag += " RESIDUAL"
        if tot > 18:
            flag += " SLOW"
            slow.append(step)
        log.info(
            "step=%02d res=%.3e |R|=%.3f depth=%d fwd=%.2f bwd=%.2f tot=%.2f P=%d%s",
            step,
            res,
            r_abs,
            max_depth,
            ms_fwd,
            ms_bwd,
            tot,
            st.picard_iters,
            flag,
        )

    log.info("residual_spikes=%s slow_steps=%s", bad, slow)


def _probe(cell, x: torch.Tensor, res: float) -> None:
    seq = sequential_apply(cell, x)
    cfg_f = NewtonConfig(max_iters=3, scan_backend="fused")
    cfg3 = NewtonConfig(max_iters=3, scan_backend="fused", picard_iters=3)
    cfg_e = NewtonConfig(max_iters=3, scan_backend="eager")
    st_f, st3, ste = NewtonStats(), NewtonStats(), NewtonStats()
    par_f = newton_apply(cell, x, cfg_f, stats=st_f)
    par3 = newton_apply(cell, x, cfg3, stats=st3)
    pare = newton_apply(cell, x, cfg_e, stats=ste)
    log.info(
        "  probe train-res=%.3e fused=%.3e vs_seq=%.3e | "
        "P=3 fused=%.3e vs_seq=%.3e | eager=%.3e vs_seq=%.3e |T|=%d |h|=%.3f",
        res,
        st_f.max_residual,
        float((par_f - seq).abs().amax()),
        st3.max_residual,
        float((par3 - seq).abs().amax()),
        ste.max_residual,
        float((pare - seq).abs().amax()),
        x.shape[1],
        float(seq[:, :, SLSTM_HIDDEN].abs().amax()),
    )


if __name__ == "__main__":
    main()
