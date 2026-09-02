"""Jacobian of ``f(h_{t-1}, x_t)`` from Autograd (DEER / Lim et al.).

Any cell with ``step(h, x)`` is enough. Analytic ``step_with_jacobian`` is the
fast path for ParaGRU/ParaLSTM (paper §3).

``diag``: one JVP with a ones tangent. Exact Newton iff ``f`` is channelwise
in ``h``; otherwise this is the diagonal quasi-Newton (Gonzalez et al. 2024).
``block2``: two JVPs for a 2-slot channelwise state (CIFG-like).
``block4``: four JVPs for a 4-slot channelwise state (sLSTM, diag mix).
``head``: ``jacrev`` per sLSTM head (``4 d_head × 4 d_head``). Exact for
xLSTM-style block-diagonal mixing.
``dense``: ``jacrev`` per ``(batch, time)`` — exact for any ``f``, ``O(d_h^3)``
scan. Use it when the cell mixes all channels.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.func import jacrev, jvp, vmap

from pararnn.layout import slstm_pack_heads


def infer_jac_structure(state: Tensor) -> str:
    if state.dim() == 3:
        return "diag"
    if state.dim() == 4 and state.shape[-2] == 2:
        return "block2"
    if state.dim() == 4 and state.shape[-2] == 4:
        return "block4"
    raise TypeError(
        f"cannot infer Jacobian structure from state {tuple(state.shape)}; "
        "set NewtonConfig.jac_structure to 'diag', 'block2', 'block4', 'head', or 'dense'"
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
        mode = "analytic" if hasattr(cell, "step_with_jacobian") else "autograd"
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
    structure = jac_structure or getattr(cell, "jac_structure", None) or infer_jac_structure(h_prev)
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
        return _jac_blockn(f, h0, slots=2)
    if structure == "block4":
        return _jac_blockn(f, h0, slots=4)
    if structure == "head":
        return _jac_head(cell, h0, x0)
    if structure == "dense":
        return _jac_dense(cell, h0, x0)
    raise ValueError(f"unknown jac_structure {structure!r}")


def _jac_blockn(f, h0: Tensor, *, slots: int) -> tuple[Tensor, Tensor]:
    if h0.dim() != 4 or h0.shape[-2] != slots:
        raise ValueError(f"jac_structure='block{slots}' needs state (batch, time, {slots}, d_h)")
    cols = []
    pred = None
    for s in range(slots):
        v = torch.zeros_like(h0)
        v[..., s, :] = 1
        pred, col = jvp(f, (h0,), (v,))
        cols.append(col)
    # (..., out, in, d)
    jac = torch.stack(cols, dim=-2)
    return pred, jac


def _jac_head(cell: nn.Module, h0: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
    n_heads = getattr(cell, "n_heads", None)
    d_head = getattr(cell, "d_head", None)
    if n_heads is None or d_head is None:
        raise ValueError("jac_structure='head' needs cell.n_heads and cell.d_head")
    packed = slstm_pack_heads(h0, n_heads, d_head)
    wx_slots = cell.W_x(x0).reshape(*x0.shape[:2], 4, n_heads * d_head)
    wx_p = slstm_pack_heads(wx_slots, n_heads, d_head)
    r_h = cell.clipped_r_head().permute(1, 0, 2, 3)

    def f_one(h: Tensor, wxh: Tensor, rh: Tensor) -> Tensor:
        s = h.reshape(4, d_head)
        return cell.step_head(s, wxh.reshape(4, d_head), rh).reshape(4 * d_head)

    inner = jacrev(f_one, argnums=0)
    per_head = vmap(inner, in_dims=(0, 0, 0))
    jac = vmap(vmap(per_head, in_dims=(0, 0, None)), in_dims=(0, 0, None))(packed, wx_p, r_h)
    pred = cell.step(h0, x0)
    return pred, jac


def _jac_dense(cell: nn.Module, h0: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
    if h0.dim() == 4:
        slots, d_h = h0.shape[-2], h0.shape[-1]
        h_flat = h0.reshape(*h0.shape[:2], slots * d_h)

        def f_one(h: Tensor, xt: Tensor) -> Tensor:
            y = cell.step(h.reshape(slots, d_h), xt)
            return y.reshape(slots * d_h)

        jac = vmap(vmap(jacrev(f_one)))(h_flat, x0)
        pred = cell.step(h0, x0)
        return pred, jac
    if h0.dim() != 3:
        raise ValueError(
            "jac_structure='dense' needs state (batch, time, d_h) or "
            "(batch, time, slots, d_h); flatten anything else yourself"
        )

    def f_one(h: Tensor, xt: Tensor) -> Tensor:
        return cell.step(h, xt)

    jac = vmap(vmap(jacrev(f_one)))(h0, x0)
    pred = cell.step(h0, x0)
    return pred, jac
