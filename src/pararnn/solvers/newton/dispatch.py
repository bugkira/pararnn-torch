"""Fused / Triton / eager backend pick and Jacobian-structure scan dispatch."""

from __future__ import annotations

import logging
import warnings
from dataclasses import replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_nlru import ParaNLRU
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
        if getattr(cell, "W_x", None) is None:
            return False
        return cell.mix in ("diag", "head")
    if isinstance(cell, ParaGRU):
        if getattr(cell, "W_x", None) is None:
            return False
        return cell.mix in ("diag", "head")
    if isinstance(cell, ParaNLRU):
        return getattr(cell, "W_x", None) is not None
    if not isinstance(cell, ParaLSTM):
        return False
    return getattr(cell, "W_x", None) is not None


def _pick_auto(cell: nn.Module, x: Tensor) -> str:
    if _can_fuse(cell, x):
        return "fused"
    if _can_triton_scan(x):
        return "triton"
    return "eager"


def _resolve_backend(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    cu_seqlens: Tensor | None = None,
) -> NewtonConfig:
    auto_p = config.picard_iters is None
    config = _resolve_picard(cell, x, config)
    if config.picard_adapt is None:
        config = replace(
            config,
            picard_adapt=auto_p and isinstance(cell, ParaSLSTM),
        )
    requested = config.scan_backend
    # ParaM2RNN: factorized Kronecker; CUDA auto/fused → Triton SRAM (K,V≤64).
    if getattr(cell, "jac_structure", None) == "m2rnn":
        from pararnn.kernels.newton_m2rnn import can_fuse_m2rnn

        k_dim = int(getattr(cell, "k_dim", 0))
        v_dim = int(getattr(cell, "v_dim", 0))
        if requested == "fused" and not can_fuse_m2rnn(k_dim, v_dim, x):
            raise TypeError(
                "ParaM2RNN fused Newton needs CUDA float16/float32/bfloat16 "
                f"(got {k_dim}×{v_dim} {x.dtype} {x.device})"
            )
        if requested in ("auto", "fused") and can_fuse_m2rnn(k_dim, v_dim, x):
            if not torch.compiler.is_compiling() and requested == "auto":
                log.debug(
                    "scan_backend_auto",
                    extra={
                        "chosen": "fused",
                        "cell": "ParaM2RNN",
                        "device": str(x.device),
                        "dtype": str(x.dtype),
                        "k_dim": k_dim,
                        "v_dim": v_dim,
                    },
                )
            return replace(config, scan_backend="fused")
        if requested == "auto":
            return replace(config, scan_backend="eager")
        return config
    # ParaSLSTM head: auto/triton/fused → factorized fused path on CUDA;
    # eager keeps the dense-J oracle. Packed cu_seqlens: no silent fused→eager.
    if (
        isinstance(cell, ParaSLSTM)
        and cell.mix == "head"
        and requested in ("auto", "triton", "fused")
        and config.coords != "log"
    ):
        if cu_seqlens is not None:
            if requested == "fused":
                raise TypeError(
                    "ParaSLSTM(mix='head') fused Newton does not support cu_seqlens yet; "
                    "use scan_backend='eager' for ragged packs, or pad to a rectangular batch"
                )
            if requested == "auto":
                if not torch.compiler.is_compiling():
                    warnings.warn(
                        "ParaSLSTM(mix='head') + cu_seqlens: scan_backend auto → eager "
                        "(factorized fused kernels are rectangular-batch only)",
                        UserWarning,
                        stacklevel=2,
                    )
                return replace(config, scan_backend="eager")
            return config
        if _can_fuse(cell, x):
            if not torch.compiler.is_compiling() and requested == "auto":
                log.debug(
                    "scan_backend_auto",
                    extra={
                        "chosen": "fused",
                        "cell": "ParaSLSTM",
                        "mix": "head",
                        "device": str(x.device),
                        "dtype": str(x.dtype),
                    },
                )
            return replace(config, scan_backend="fused")
        if requested == "fused":
            raise TypeError(_fused_error(cell, x))
        if requested == "auto":
            return replace(config, scan_backend="eager")
        return config
    # ParaGRU head: auto/triton/fused → factorized fused path on CUDA;
    # eager keeps the dense-J oracle. Packed cu_seqlens: no silent fused→eager.
    if (
        isinstance(cell, ParaGRU)
        and cell.mix == "head"
        and requested in ("auto", "triton", "fused")
        and config.coords != "log"
    ):
        if cu_seqlens is not None:
            if requested == "fused":
                raise TypeError(
                    "ParaGRU(mix='head') fused Newton does not support cu_seqlens yet; "
                    "use scan_backend='eager' for ragged packs, or pad to a rectangular batch"
                )
            if requested == "auto":
                if not torch.compiler.is_compiling():
                    warnings.warn(
                        "ParaGRU(mix='head') + cu_seqlens: scan_backend auto → eager "
                        "(factorized fused kernels are rectangular-batch only)",
                        UserWarning,
                        stacklevel=2,
                    )
                return replace(config, scan_backend="eager")
            # Explicit triton: dense scan_dense path (supports ragged J=0 at heads).
            return config
        if _can_fuse(cell, x):
            if not torch.compiler.is_compiling() and requested == "auto":
                log.debug(
                    "scan_backend_auto",
                    extra={
                        "chosen": "fused",
                        "cell": "ParaGRU",
                        "mix": "head",
                        "device": str(x.device),
                        "dtype": str(x.dtype),
                    },
                )
            return replace(config, scan_backend="fused")
        if requested == "fused":
            raise TypeError(_fused_error(cell, x))
        if requested == "auto":
            # bf16 on pre-Ampere: dense eager (same class as other cells).
            return replace(config, scan_backend="eager")
        # Explicit triton: dense scan_dense path (factorized needs fused dtypes).
        return config
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
    if isinstance(cell, ParaSLSTM) and cell.mix not in ("diag", "head"):
        return f"scan_backend='fused' is mix='diag'|'head' for ParaSLSTM; got mix={cell.mix!r}"
    if isinstance(cell, ParaGRU) and cell.mix not in ("diag", "head"):
        return f"scan_backend='fused' is mix='diag'|'head' for ParaGRU; got mix={cell.mix!r}"
    return (
        "scan_backend='fused' needs CUDA ParaGRU/ParaLSTM/ParaSLSTM/ParaNLRU "
        f"in float16/float32/bfloat16 (got {type(cell).__name__} {x.dtype} {x.device})"
    )


def _scan(
    jac: Tensor,
    residual: Tensor,
    *,
    backend: str = "eager",
    structure: str | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    if structure is not None:
        return _scan_named(
            jac, residual, backend=backend, structure=structure, cu_seqlens=cu_seqlens
        )
    return _scan_infer(jac, residual, backend=backend, cu_seqlens=cu_seqlens)


def _reverse_scan(
    jac: Tensor,
    partial: Tensor,
    *,
    backend: str = "eager",
    structure: str | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    if structure is not None:
        return _reverse_scan_named(
            jac, partial, backend=backend, structure=structure, cu_seqlens=cu_seqlens
        )
    return _reverse_scan_infer(jac, partial, backend=backend, cu_seqlens=cu_seqlens)


def _scan_named(
    jac: Tensor,
    residual: Tensor,
    *,
    backend: str,
    structure: str,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    if structure == "diag":
        return scan_diag(jac, residual, backend=backend, cu_seqlens=cu_seqlens)
    if structure == "block2":
        return scan_block2(jac, residual, backend=backend, cu_seqlens=cu_seqlens)
    if structure == "block4":
        return scan_block4(jac, residual, backend=backend, cu_seqlens=cu_seqlens)
    if structure == "head":
        packed_h = _head_slot_pack(jac, residual)
        if packed_h is None:
            raise ValueError(
                f"jac_structure='head' but jac {tuple(jac.shape)} residual {tuple(residual.shape)}"
            )
        jac_f, res_f, shape, n_heads, d_head = packed_h
        delta = scan_dense(jac_f, res_f, backend=backend, cu_seqlens=cu_seqlens)
        return _head_slot_unpack(delta, shape, n_heads, d_head)
    if structure == "dense":
        packed = _dense_slot_pack(jac, residual)
        if packed is not None:
            jac_f, res_f, shape = packed
            return scan_dense(jac_f, res_f, backend=backend, cu_seqlens=cu_seqlens).reshape(shape)
        return scan_dense(jac, residual, backend=backend, cu_seqlens=cu_seqlens)
    raise ValueError(f"unknown jac_structure {structure!r}")


def _reverse_scan_named(
    jac: Tensor,
    partial: Tensor,
    *,
    backend: str,
    structure: str,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    if structure == "diag":
        return reverse_scan_diag(jac, partial, backend=backend, cu_seqlens=cu_seqlens)
    if structure == "block2":
        return reverse_scan_block2(jac, partial, backend=backend, cu_seqlens=cu_seqlens)
    if structure == "block4":
        return reverse_scan_block4(jac, partial, backend=backend, cu_seqlens=cu_seqlens)
    if structure == "head":
        packed_h = _head_slot_pack(jac, partial)
        if packed_h is None:
            raise ValueError(
                f"jac_structure='head' but jac {tuple(jac.shape)} partial {tuple(partial.shape)}"
            )
        jac_f, part_f, shape, n_heads, d_head = packed_h
        mu = reverse_scan_dense(jac_f, part_f, backend=backend, cu_seqlens=cu_seqlens)
        return _head_slot_unpack(mu, shape, n_heads, d_head)
    if structure == "dense":
        packed = _dense_slot_pack(jac, partial)
        if packed is not None:
            jac_f, part_f, shape = packed
            return reverse_scan_dense(
                jac_f, part_f, backend=backend, cu_seqlens=cu_seqlens
            ).reshape(shape)
        return reverse_scan_dense(jac, partial, backend=backend, cu_seqlens=cu_seqlens)
    raise ValueError(f"unknown jac_structure {structure!r}")


def _scan_infer(
    jac: Tensor,
    residual: Tensor,
    *,
    backend: str,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    packed = _dense_slot_pack(jac, residual)
    if packed is not None:
        jac_f, res_f, shape = packed
        delta = scan_dense(jac_f, res_f, backend=backend, cu_seqlens=cu_seqlens)
        return delta.reshape(shape)
    packed_h = _head_slot_pack(jac, residual)
    if packed_h is not None:
        jac_f, res_f, shape, n_heads, d_head = packed_h
        delta = scan_dense(jac_f, res_f, backend=backend, cu_seqlens=cu_seqlens)
        return _head_slot_unpack(delta, shape, n_heads, d_head)
    if jac.dim() == residual.dim():
        return scan_diag(jac, residual, backend=backend, cu_seqlens=cu_seqlens)
    if jac.dim() == 4:
        return scan_dense(jac, residual, backend=backend, cu_seqlens=cu_seqlens)
    if jac.dim() == 5 and jac.shape[-3] == 4:
        return scan_block4(jac, residual, backend=backend, cu_seqlens=cu_seqlens)
    if jac.dim() == 5 and jac.shape[-3] == 2:
        return scan_block2(jac, residual, backend=backend, cu_seqlens=cu_seqlens)
    raise ValueError(
        f"cannot dispatch scan for jac {tuple(jac.shape)} residual "
        f"{tuple(residual.shape)}; set NewtonConfig.jac_structure"
    )


def _reverse_scan_infer(
    jac: Tensor,
    partial: Tensor,
    *,
    backend: str,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    packed = _dense_slot_pack(jac, partial)
    if packed is not None:
        jac_f, part_f, shape = packed
        mu = reverse_scan_dense(jac_f, part_f, backend=backend, cu_seqlens=cu_seqlens)
        return mu.reshape(shape)
    packed_h = _head_slot_pack(jac, partial)
    if packed_h is not None:
        jac_f, part_f, shape, n_heads, d_head = packed_h
        mu = reverse_scan_dense(jac_f, part_f, backend=backend, cu_seqlens=cu_seqlens)
        return _head_slot_unpack(mu, shape, n_heads, d_head)
    if jac.dim() == partial.dim():
        return reverse_scan_diag(jac, partial, backend=backend, cu_seqlens=cu_seqlens)
    if jac.dim() == 4:
        return reverse_scan_dense(jac, partial, backend=backend, cu_seqlens=cu_seqlens)
    if jac.dim() == 5 and jac.shape[-3] == 4:
        return reverse_scan_block4(jac, partial, backend=backend, cu_seqlens=cu_seqlens)
    if jac.dim() == 5 and jac.shape[-3] == 2:
        return reverse_scan_block2(jac, partial, backend=backend, cu_seqlens=cu_seqlens)
    raise ValueError(
        f"cannot dispatch reverse scan for jac {tuple(jac.shape)} partial "
        f"{tuple(partial.shape)}; set NewtonConfig.jac_structure"
    )


def _t0_state_vjp(jac: Tensor, mu: Tensor) -> Tensor:
    """``J_0^T μ_0`` — adjoint of paper ``h_0``. Layout matches ``_scan``."""
    t0 = torch.zeros(1, dtype=torch.long, device=jac.device)
    return _state_vjp_at_times(jac, mu, t0)[:, 0]


def _state_vjp_at_times(jac: Tensor, mu: Tensor, times: Tensor) -> Tensor:
    """``J_t^T μ_t`` at each index in ``times``. Result time-axis is ``len(times)``."""
    packed = _dense_slot_pack(jac, mu)
    if packed is not None:
        jac_f, mu_f, shape = packed
        g = torch.matmul(
            jac_f[:, times].transpose(-1, -2),
            mu_f[:, times].unsqueeze(-1),
        ).squeeze(-1)
        return g.reshape(shape[0], times.numel(), *shape[2:])
    packed_h = _head_slot_pack(jac, mu)
    if packed_h is not None:
        jac_f, mu_f, shape, n_heads, d_head = packed_h
        g = torch.matmul(
            jac_f[:, times].transpose(-1, -2),
            mu_f[:, times].unsqueeze(-1),
        ).squeeze(-1)
        if len(shape) == 3:
            packed_hs = g.reshape(shape[0], n_heads, times.numel(), d_head).permute(0, 2, 1, 3)
            return packed_hs.reshape(shape[0], times.numel(), n_heads * d_head)
        packed_hs = g.reshape(shape[0], n_heads, times.numel(), 4 * d_head).permute(0, 2, 1, 3)
        return slstm_unpack_heads(packed_hs, n_heads, d_head)
    if jac.dim() == mu.dim():
        return jac[:, times] * mu[:, times]
    if jac.dim() == 4:
        return torch.matmul(
            jac[:, times].transpose(-1, -2),
            mu[:, times].unsqueeze(-1),
        ).squeeze(-1)
    return torch.einsum("btoid,btod->btid", jac[:, times], mu[:, times])


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
    """Per-head dense J for sLSTM (4-slot) or GRU (1-slot).

    sLSTM: ``jac`` ``(B, T, H, 4 d_head, 4 d_head)``, ``vec`` ``(B, T, 4, d_h)``.
    GRU: ``jac`` ``(B, T, H, d_head, d_head)``, ``vec`` ``(B, T, d_h)``.
    """
    if jac.dim() != 5 or jac.shape[-1] != jac.shape[-2]:
        return None
    n_heads = jac.shape[2]
    sd = jac.shape[-1]
    if n_heads < 1:
        return None
    b, t = vec.shape[:2]

    if vec.dim() == 3:
        d_h = vec.shape[-1]
        if d_h % n_heads != 0:
            return None
        d_head = d_h // n_heads
        if sd != d_head:
            return None
        heads = vec.reshape(b, t, n_heads, d_head)
        jac_f = jac.permute(0, 2, 1, 3, 4).reshape(b * n_heads, t, sd, sd).contiguous()
        vec_f = heads.permute(0, 2, 1, 3).reshape(b * n_heads, t, sd).contiguous()
        return jac_f, vec_f, vec.shape, n_heads, d_head

    if vec.dim() != 4:
        return None
    slots, d_h = vec.shape[-2], vec.shape[-1]
    if slots != 4 or d_h % n_heads != 0:
        return None
    d_head = d_h // n_heads
    if sd != 4 * d_head:
        return None
    # block4 is (B, T, 4, 4, d_h); last dim is the channel.
    if jac.shape[2] == 4 and jac.shape[3] == 4 and jac.shape[-1] == d_h:
        return None
    packed = slstm_pack_heads(vec, n_heads, d_head)
    jac_f = jac.permute(0, 2, 1, 3, 4).reshape(b * n_heads, t, sd, sd).contiguous()
    vec_f = packed.permute(0, 2, 1, 3).reshape(b * n_heads, t, sd).contiguous()
    return jac_f, vec_f, vec.shape, n_heads, d_head


def _head_slot_unpack(folded: Tensor, shape: tuple[int, ...], n_heads: int, d_head: int) -> Tensor:
    b, t = shape[:2]
    if len(shape) == 3:
        packed = folded.reshape(b, n_heads, t, d_head).permute(0, 2, 1, 3)
        return packed.reshape(b, t, n_heads * d_head)
    sd = 4 * d_head
    packed = folded.reshape(b, n_heads, t, sd).permute(0, 2, 1, 3)
    return slstm_unpack_heads(packed, n_heads, d_head)
