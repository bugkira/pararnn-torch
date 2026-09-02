"""Fused / Triton / eager backend pick and Jacobian-structure scan dispatch."""

from __future__ import annotations

import logging
from dataclasses import replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.kernels.precision import is_fused_dtype_supported
from pararnn.layout import slstm_pack_heads, slstm_unpack_heads
from pararnn.solvers.newton.config import NewtonConfig
from pararnn.solvers.newton.picard import _resolve_picard
from pararnn.solvers.scan import (
    reverse_scan_block2,
    reverse_scan_block4,
    reverse_scan_dense,
    reverse_scan_diag,
    scan_block2,
    scan_block4,
    scan_dense,
    scan_diag,
)

log = logging.getLogger("pararnn.solvers.newton")


def _can_triton_scan(x: Tensor) -> bool:
    return is_fused_dtype_supported(x.dtype, x.device)


def _can_fuse(cell: nn.Module, x: Tensor) -> bool:
    if not _can_triton_scan(x):
        return False
    if isinstance(cell, ParaSLSTM):
        return cell.mix == "diag" and getattr(cell, "W_x", None) is not None
    if not isinstance(cell, (ParaGRU, ParaLSTM)):
        return False
    return getattr(cell, "W_x", None) is not None


def _pick_auto(cell: nn.Module, x: Tensor) -> str:
    if _can_fuse(cell, x):
        return "fused"
    if _can_triton_scan(x):
        return "triton"
    return "eager"


def _resolve_backend(cell: nn.Module, x: Tensor, config: NewtonConfig) -> NewtonConfig:
    auto_p = config.picard_iters is None
    config = _resolve_picard(cell, x, config)
    if config.picard_adapt is None:
        config = replace(
            config,
            picard_adapt=auto_p and isinstance(cell, ParaSLSTM),
        )
    requested = config.scan_backend
    if config.coords == "log":
        if not isinstance(cell, ParaSLSTM):
            raise TypeError(
                f"NewtonConfig(coords='log') is ParaSLSTM only (got {type(cell).__name__})"
            )
        if requested == "fused" and not _can_fuse(cell, x):
            raise TypeError(_fused_error(cell, x))
        if requested == "auto":
            chosen = _pick_auto(cell, x)
            return replace(config, scan_backend=chosen)
        return config
    if requested == "auto":
        chosen = _pick_auto(cell, x)
        if not torch.compiler.is_compiling():
            log.debug(
                "scan_backend_auto",
                extra={
                    "chosen": chosen,
                    "cell": type(cell).__name__,
                    "device": str(x.device),
                    "dtype": str(x.dtype),
                },
            )
        return replace(config, scan_backend=chosen)
    if requested == "fused" and not _can_fuse(cell, x):
        raise TypeError(_fused_error(cell, x))
    return config


def _fused_error(cell: nn.Module, x: Tensor) -> str:
    if x.dtype == torch.bfloat16 and not is_fused_dtype_supported(x.dtype, x.device):
        major, minor = torch.cuda.get_device_capability(x.device)
        return (
            "fused Newton: bfloat16 needs CUDA compute capability >= 8.0 "
            f"(Ampere+ tensor cores); got sm_{major}{minor} on {x.device}. "
            "Use float16; cell+scan algebra stays fp32."
        )
    if isinstance(cell, ParaSLSTM) and cell.mix != "diag":
        return f"scan_backend='fused' is mix='diag' only (4x4 SRAM); got mix={cell.mix!r}"
    return (
        "scan_backend='fused' needs CUDA ParaGRU/ParaLSTM/ParaSLSTM(mix='diag') "
        f"in float16/float32/bfloat16 (got {type(cell).__name__} {x.dtype} {x.device})"
    )


def _scan(
    jac: Tensor,
    residual: Tensor,
    *,
    backend: str = "eager",
    structure: str | None = None,
) -> Tensor:
    if structure is not None:
        return _scan_named(jac, residual, backend=backend, structure=structure)
    return _scan_infer(jac, residual, backend=backend)


def _reverse_scan(
    jac: Tensor,
    partial: Tensor,
    *,
    backend: str = "eager",
    structure: str | None = None,
) -> Tensor:
    if structure is not None:
        return _reverse_scan_named(jac, partial, backend=backend, structure=structure)
    return _reverse_scan_infer(jac, partial, backend=backend)


def _scan_named(jac: Tensor, residual: Tensor, *, backend: str, structure: str) -> Tensor:
    if structure == "diag":
        return scan_diag(jac, residual, backend=backend)
    if structure == "block2":
        return scan_block2(jac, residual, backend=backend)
    if structure == "block4":
        return scan_block4(jac, residual, backend=backend)
    if structure == "head":
        packed_h = _head_slot_pack(jac, residual)
        if packed_h is None:
            raise ValueError(
                f"jac_structure='head' but jac {tuple(jac.shape)} residual {tuple(residual.shape)}"
            )
        jac_f, res_f, shape, n_heads, d_head = packed_h
        delta = scan_dense(jac_f, res_f, backend=backend)
        return _head_slot_unpack(delta, shape, n_heads, d_head)
    if structure == "dense":
        packed = _dense_slot_pack(jac, residual)
        if packed is not None:
            jac_f, res_f, shape = packed
            return scan_dense(jac_f, res_f, backend=backend).reshape(shape)
        return scan_dense(jac, residual, backend=backend)
    raise ValueError(f"unknown jac_structure {structure!r}")


def _reverse_scan_named(jac: Tensor, partial: Tensor, *, backend: str, structure: str) -> Tensor:
    if structure == "diag":
        return reverse_scan_diag(jac, partial, backend=backend)
    if structure == "block2":
        return reverse_scan_block2(jac, partial, backend=backend)
    if structure == "block4":
        return reverse_scan_block4(jac, partial, backend=backend)
    if structure == "head":
        packed_h = _head_slot_pack(jac, partial)
        if packed_h is None:
            raise ValueError(
                f"jac_structure='head' but jac {tuple(jac.shape)} partial {tuple(partial.shape)}"
            )
        jac_f, part_f, shape, n_heads, d_head = packed_h
        mu = reverse_scan_dense(jac_f, part_f, backend=backend)
        return _head_slot_unpack(mu, shape, n_heads, d_head)
    if structure == "dense":
        packed = _dense_slot_pack(jac, partial)
        if packed is not None:
            jac_f, part_f, shape = packed
            return reverse_scan_dense(jac_f, part_f, backend=backend).reshape(shape)
        return reverse_scan_dense(jac, partial, backend=backend)
    raise ValueError(f"unknown jac_structure {structure!r}")


def _scan_infer(jac: Tensor, residual: Tensor, *, backend: str) -> Tensor:
    packed = _dense_slot_pack(jac, residual)
    if packed is not None:
        jac_f, res_f, shape = packed
        delta = scan_dense(jac_f, res_f, backend=backend)
        return delta.reshape(shape)
    packed_h = _head_slot_pack(jac, residual)
    if packed_h is not None:
        jac_f, res_f, shape, n_heads, d_head = packed_h
        delta = scan_dense(jac_f, res_f, backend=backend)
        return _head_slot_unpack(delta, shape, n_heads, d_head)
    if jac.dim() == residual.dim():
        return scan_diag(jac, residual, backend=backend)
    if jac.dim() == 4:
        return scan_dense(jac, residual, backend=backend)
    if jac.dim() == 5 and jac.shape[-3] == 4:
        return scan_block4(jac, residual, backend=backend)
    if jac.dim() == 5 and jac.shape[-3] == 2:
        return scan_block2(jac, residual, backend=backend)
    raise ValueError(
        f"cannot dispatch scan for jac {tuple(jac.shape)} residual "
        f"{tuple(residual.shape)}; set NewtonConfig.jac_structure"
    )


def _reverse_scan_infer(jac: Tensor, partial: Tensor, *, backend: str) -> Tensor:
    packed = _dense_slot_pack(jac, partial)
    if packed is not None:
        jac_f, part_f, shape = packed
        mu = reverse_scan_dense(jac_f, part_f, backend=backend)
        return mu.reshape(shape)
    packed_h = _head_slot_pack(jac, partial)
    if packed_h is not None:
        jac_f, part_f, shape, n_heads, d_head = packed_h
        mu = reverse_scan_dense(jac_f, part_f, backend=backend)
        return _head_slot_unpack(mu, shape, n_heads, d_head)
    if jac.dim() == partial.dim():
        return reverse_scan_diag(jac, partial, backend=backend)
    if jac.dim() == 4:
        return reverse_scan_dense(jac, partial, backend=backend)
    if jac.dim() == 5 and jac.shape[-3] == 4:
        return reverse_scan_block4(jac, partial, backend=backend)
    if jac.dim() == 5 and jac.shape[-3] == 2:
        return reverse_scan_block2(jac, partial, backend=backend)
    raise ValueError(
        f"cannot dispatch reverse scan for jac {tuple(jac.shape)} partial "
        f"{tuple(partial.shape)}; set NewtonConfig.jac_structure"
    )


def _t0_state_vjp(jac: Tensor, mu: Tensor) -> Tensor:
    """``J_0^T μ_0`` — adjoint of paper ``h_0``. Layout matches ``_scan``."""
    packed = _dense_slot_pack(jac, mu)
    if packed is not None:
        jac_f, mu_f, shape = packed
        g = torch.matmul(jac_f[:, 0].transpose(-1, -2), mu_f[:, 0].unsqueeze(-1)).squeeze(-1)
        return g.reshape(shape[0], *shape[2:])
    packed_h = _head_slot_pack(jac, mu)
    if packed_h is not None:
        jac_f, mu_f, shape, n_heads, d_head = packed_h
        g = torch.matmul(jac_f[:, 0].transpose(-1, -2), mu_f[:, 0].unsqueeze(-1)).squeeze(-1)
        packed_h0 = g.reshape(shape[0], n_heads, 4 * d_head)
        return slstm_unpack_heads(packed_h0, n_heads, d_head)
    if jac.dim() == mu.dim():
        return jac[:, 0] * mu[:, 0]
    if jac.dim() == 4:
        return torch.matmul(jac[:, 0].transpose(-1, -2), mu[:, 0].unsqueeze(-1)).squeeze(-1)
    return torch.einsum("boid,bod->bid", jac[:, 0], mu[:, 0])


def _dense_slot_pack(jac: Tensor, vec: Tensor) -> tuple[Tensor, Tensor, tuple[int, ...]] | None:
    """4-slot state with a flattened dense J: ``jac`` is (B, T, S d, S d)."""
    if jac.dim() != 4 or vec.dim() != 4:
        return None
    slots, d_h = vec.shape[-2], vec.shape[-1]
    sd = slots * d_h
    if jac.shape[-1] != sd or jac.shape[-2] != sd:
        return None
    return jac, vec.reshape(*vec.shape[:2], sd), vec.shape


def _head_slot_pack(
    jac: Tensor, vec: Tensor
) -> tuple[Tensor, Tensor, tuple[int, ...], int, int] | None:
    """Per-head dense J: ``jac`` is (B, T, H, 4 d_head, 4 d_head), ``vec`` is (B, T, 4, d_h)."""
    if jac.dim() != 5 or vec.dim() != 4:
        return None
    if jac.shape[-1] != jac.shape[-2]:
        return None
    slots, d_h = vec.shape[-2], vec.shape[-1]
    n_heads = jac.shape[2]
    sd = jac.shape[-1]
    if slots != 4 or n_heads < 1 or d_h % n_heads != 0:
        return None
    d_head = d_h // n_heads
    if sd != 4 * d_head:
        return None
    # block4 is (B, T, 4, 4, d_h); last dim is the channel.
    if jac.shape[2] == 4 and jac.shape[3] == 4 and jac.shape[-1] == d_h:
        return None
    packed = slstm_pack_heads(vec, n_heads, d_head)
    b, t = vec.shape[:2]
    jac_f = jac.permute(0, 2, 1, 3, 4).reshape(b * n_heads, t, sd, sd).contiguous()
    vec_f = packed.permute(0, 2, 1, 3).reshape(b * n_heads, t, sd).contiguous()
    return jac_f, vec_f, vec.shape, n_heads, d_head


def _head_slot_unpack(folded: Tensor, shape: tuple[int, ...], n_heads: int, d_head: int) -> Tensor:
    b, t = shape[:2]
    sd = 4 * d_head
    packed = folded.reshape(b, n_heads, t, sd).permute(0, 2, 1, 3)
    return slstm_unpack_heads(packed, n_heads, d_head)
