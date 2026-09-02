"""Naive sequential unroll vs Newton+scan (Danieli et al. 2025).

Independent of ``sequential_apply``: the oracle is a Python loop over ``cell.step``.
Run: ``uv run python scripts/compare_naive.py``
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from pararnn.cells import ParaGRU, ParaLSTM
from pararnn.layout import prepend_zero_state
from pararnn.solvers import NewtonConfig, newton_apply
from utils.mlflow_helper import setup_logging

from gpu import DEFAULT_EXPERIMENT_GPU_NAME, select_device

log = logging.getLogger("compare")


def naive_unroll(cell: nn.Module, x: Tensor) -> Tensor:
    """Ground truth: h_0=0, then h_t = f(h_{t-1}, x_t) in a Python loop."""
    batch, time, _ = x.shape
    if isinstance(cell, ParaLSTM):
        h = x.new_zeros(batch, 2, cell.d_h)
    else:
        h = x.new_zeros(batch, cell.d_h)
    outs = []
    for t in range(time):
        h = cell.step(h, x[:, t])
        outs.append(h)
    return torch.stack(outs, dim=1)


def max_residual(cell: nn.Module, x: Tensor, states: Tensor) -> float:
    pred = cell.step(prepend_zero_state(states), x)
    return float((pred - states).abs().amax())


@torch.no_grad()
def compare_one(
    name: str,
    cell: nn.Module,
    x: Tensor,
    ks: tuple[int, ...] = (0, 1, 2, 3, 4, 6),
) -> None:
    naive = naive_unroll(cell, x)
    naive_res = max_residual(cell, x, naive)
    log.info(
        "%s naive: shape=%s residual=%.3e",
        name,
        tuple(naive.shape),
        naive_res,
    )
    for k in ks:
        if k == 0:
            h_prev0 = x.new_zeros(x.shape[0], x.shape[1], *naive.shape[2:])
            par, _ = cell.step_with_jacobian(h_prev0, x)
        else:
            par = newton_apply(
                cell,
                x,
                NewtonConfig(
                    max_iters=k,
                    scan_backend="eager",
                    residual_atol=None,
                    residual_fail=None,
                ),
            )
        err = (par - naive).abs()
        res = max_residual(cell, x, par)
        log.info(
            "%s K=%d  max|Δ|=%.3e  mean|Δ|=%.3e  residual=%.3e",
            name,
            k,
            float(err.amax()),
            float(err.mean()),
            res,
        )


def main() -> None:
    setup_logging()
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    if device.type != "cuda":
        raise RuntimeError("compare_naive needs the 2080 Ti (CUDA_VISIBLE_DEVICES to restrict)")
    log.info("device=%s (%s)", device, torch.cuda.get_device_name(device))
    torch.manual_seed(0)

    lengths = (1, 2, 7, 32, 64, 128, 256)
    batch, d_in, d_h = 4, 16, 32

    for T in lengths:
        x = torch.randn(batch, T, d_in, device=device)
        gru = ParaGRU(d_in, d_h).to(device)
        lstm = ParaLSTM(d_in, d_h).to(device)
        log.info("--- T=%d ---", T)
        compare_one(f"ParaGRU T={T}", gru, x)
        compare_one(f"ParaLSTM T={T}", lstm, x)


if __name__ == "__main__":
    main()
