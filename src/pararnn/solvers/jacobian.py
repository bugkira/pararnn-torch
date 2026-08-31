"""Jacobian of ``f(h_{t-1}, x_t)`` from Autograd (DEER / Lim et al.).

Any cell with ``step(h, x)`` is enough. Analytic ``step_with_jacobian`` is the
fast path for ParaGRU/ParaLSTM (paper §3), not a requirement.

``diag``: one JVP with a ones tangent. Exact Newton iff ``f`` is channelwise
in ``h``; otherwise this is the diagonal quasi-Newton (Gonzalez et al. 2024).
``block2``: two JVPs for a 2-slot channelwise state (CIFG-like).
``dense``: ``jacrev`` per ``(batch, time)`` — exact for any ``f``, ``O(d_h^3)``
scan. Not a paper hyperparameter; use it when the cell mixes channels.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.func import jacrev, jvp, vmap


def infer_jac_structure(state: Tensor) -> str:
    if state.dim() == 3:
        return "diag"
    if state.dim() == 4 and state.shape[-2] == 2:
        return "block2"
    raise TypeError(
        f"cannot infer Jacobian structure from state {tuple(state.shape)}; "
        "set NewtonConfig.jac_structure to 'diag', 'block2', or 'dense'"
    )


def step_and_jacobian(
    cell: nn.Module,
    h_prev: Tensor,
    x: Tensor,
    *,
    wx: Tensor | None,
    jacobian: str,
    jac_structure: str | None,
) -> tuple[Tensor, Tensor]:
    """``(f(h_prev, x), J)`` with ``J = ∂f/∂h_prev`` at this guess."""
    mode = jacobian
    if mode == "auto":
        mode = (
            "analytic" if hasattr(cell, "step_with_jacobian") else "autograd"
        )
    if mode == "analytic":
        if not hasattr(cell, "step_with_jacobian"):
            raise TypeError(
                f"{type(cell).__name__} has no step_with_jacobian; "
                "use NewtonConfig(jacobian='autograd')"
            )
        if wx is None:
            return cell.step_with_jacobian(h_prev, x)
        return cell.step_with_jacobian(h_prev, x, wx=wx)
    if mode != "autograd":
        raise ValueError(f"unknown jacobian {mode!r}")
    structure = jac_structure or infer_jac_structure(h_prev)
    return jacobian_autograd(cell, h_prev, x, structure=structure)


def jacobian_autograd(
    cell: nn.Module,
    h_prev: Tensor,
    x: Tensor,
    *,
    structure: str,
) -> tuple[Tensor, Tensor]:
    """``J`` from ``torch.func``; ``h_prev`` / ``x`` are treated as the linearization point."""
    h0 = h_prev.detach()
    x0 = x.detach()

    def f(h: Tensor) -> Tensor:
        return cell.step(h, x0)

    if structure == "diag":
        if h0.dim() != 3:
            raise ValueError("jac_structure='diag' needs state (batch, time, d_h)")
        pred, tan = jvp(f, (h0,), (torch.ones_like(h0),))
        return pred, tan
    if structure == "block2":
        return _jac_block2(f, h0)
    if structure == "dense":
        return _jac_dense(cell, h0, x0)
    raise ValueError(f"unknown jac_structure {structure!r}")


def _jac_block2(f, h0: Tensor) -> tuple[Tensor, Tensor]:
    if h0.dim() != 4 or h0.shape[-2] != 2:
        raise ValueError("jac_structure='block2' needs state (batch, time, 2, d_h)")
    v_c = torch.zeros_like(h0)
    v_h = torch.zeros_like(h0)
    v_c[..., 0, :] = 1
    v_h[..., 1, :] = 1
    pred, col_c = jvp(f, (h0,), (v_c,))
    _, col_h = jvp(f, (h0,), (v_h,))
    # col_*[..., out, d] = J_{out, in} for in in {c, h}. Layout (..., out, in, d).
    row_c = torch.stack((col_c[..., 0, :], col_h[..., 0, :]), dim=-2)
    row_h = torch.stack((col_c[..., 1, :], col_h[..., 1, :]), dim=-2)
    jac = torch.stack((row_c, row_h), dim=-3)
    return pred, jac


def _jac_dense(cell: nn.Module, h0: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
    if h0.dim() != 3:
        raise ValueError(
            "jac_structure='dense' needs state (batch, time, d_h); "
            "flatten a multi-slot state yourself or use block2"
        )

    def f_one(h: Tensor, xt: Tensor) -> Tensor:
        return cell.step(h, xt)

    jac = vmap(vmap(jacrev(f_one)))(h0, x0)
    pred = cell.step(h0, x0)
    return pred, jac
