"""T=1 decode: one Triton launch for the recurrent step (gates + mix).

``W_x(x)`` stays a cuBLAS GEMM (eq. 3.1). This kernel is the rest of
``cell.step``: load ``h`` / slots, ``a_*`` / ``R``, and ``wx`` into SRAM,
write the next state. Serving path is ``.eval()`` + ``T=1``; training
gradients use eager ``cell.step``.

``out=`` reuses a buffer (CUDA graphs, decode loops). ``block_table`` is
``(B,)`` int32 slot ids into a pool-shaped ``state`` / ``out``
``(C, …)``; the cell algebra is unchanged. App. C.1 clip of ``a_*`` / ``R``
runs in-kernel so a captured graph does not allocate.

``BLOCK_D=128``: fused Newton uses 32 because a tile also holds ``(J, r)``
over ``BLOCK_T=128`` (32 KiB). Decode has no time tile; 128 channels is
four warps and ~4–6 KiB of fp32 vectors. CUDA float16/float32, and bf16 on
compute capability ≥ 8.0; algebra in fp32 (``load_acc`` / ``store_acc``).
"""

from __future__ import annotations

import inspect
import logging

import torch
import triton
import triton.language as tl
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.kernels._fused_common import _tanh
from pararnn.kernels.precision import (
    is_fused_dtype_supported,
    load_acc,
    store_acc,
    validate_cuda_tensors,
)
from pararnn.layout import (
    LSTM_CELL,
    LSTM_HIDDEN,
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_STABILIZER,
)

log = logging.getLogger(__name__)

# T=1: SRAM is a handful of ``(BLOCK_D,)`` vectors. fused Newton BLOCK_D=32
# is sized for ``BLOCK_T × BLOCK_D × (J, r)``.
_BLOCK_D = 128


def can_decode_step(cell: nn.Module, ref: Tensor) -> bool:
    """Whether ``decode_step`` can run the T=1 Triton kernel on ``ref``.

    Supports ParaGRU / ParaLSTM and ParaSLSTM with ``mix='diag'`` on CUDA
    with fp32/fp16 (bf16 needs SM ≥ 8.0). Head mix and CPU keep the eager
    ``cell.step`` path inside ``decode_step``.

    Parameters
    ----------
    cell : nn.Module
        Recurrent cell (``ParaGRU``, ``ParaLSTM``, or ``ParaSLSTM``).
    ref : Tensor
        Device / dtype reference (typically ``x`` or ``wx``).

    Returns
    -------
    bool
        ``True`` when the fused decode kernel is available.
    """
    if not ref.is_cuda:
        return False
    if not is_fused_dtype_supported(ref.dtype, ref.device):
        return False
    if isinstance(cell, ParaGRU):
        return cell.mix == "diag"
    if isinstance(cell, ParaLSTM):
        return True
    return isinstance(cell, ParaSLSTM) and cell.mix == "diag"


def decode_wx(cell: nn.Module, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Apply ``W_x(x)`` (eq. 3.1), optionally into a preallocated buffer.

    Parameters
    ----------
    cell : nn.Module
        Cell exposing ``W_x`` (``nn.Linear``).
    x : Tensor of shape (batch, d_in) or (batch, 1, d_in)
        Token / residual input for one step.
    out : Tensor of shape (batch, d_wx), optional
        GEMM destination for CUDA-graph capture; written in place.

    Returns
    -------
    Tensor of shape (batch, d_wx)
        Affine projection ``W_x(x)`` (+ bias when present).

    Raises
    ------
    TypeError
        If ``cell`` has no ``W_x``.
    """
    lin = getattr(cell, "W_x", None)
    if lin is None:
        raise TypeError(f"decode_wx needs cell.W_x; got {type(cell).__name__}")
    x2 = _squeeze_time(x) if x.dim() == 3 else x
    if out is None:
        return lin(x2)
    torch.mm(x2, lin.weight.t(), out=out)
    if lin.bias is not None:
        out.add_(lin.bias)
    return out


def decode_step(
    cell: nn.Module,
    state: Tensor,
    x: Tensor | None = None,
    *,
    wx: Tensor | None = None,
    out: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """One sequential recurrent step (Triton on CUDA when eligible).

    Else eager ``cell.step``. ``wx`` is optional precomputed ``W_x(x)``.
    ``block_table`` indexes pool-shaped ``state`` / ``out``.

    Parameters
    ----------
    cell : nn.Module
        ``ParaGRU``, ``ParaLSTM``, or ``ParaSLSTM``.
    state : Tensor
        Carry ``(batch, …)`` or pool ``(capacity, …)``.
    x : Tensor of shape (batch, d_in) or (batch, 1, d_in), optional
        Required when ``wx`` is omitted.
    wx : Tensor of shape (batch, d_wx), optional
        Precomputed ``W_x(x)``.
    out : Tensor, optional
        Next-state destination (CUDA graphs / decode loops).
    block_table : Tensor of shape (batch,), optional
        Slot ids into pool-shaped buffers.

    Returns
    -------
    Tensor
        Next state (``out`` when provided).

    Raises
    ------
    ValueError
        Both ``x`` and ``wx`` omitted.
    RuntimeError
        ``block_table`` outside CUDA Triton decode, or CUDA-graph
        contiguity / ``out`` requirements fail.
    TypeError
        No Triton kernel for ``cell``.

    See Also
    --------
    can_decode_step : Eligibility for the Triton path.
    decode_wx : Input projection GEMM.
    """
    ref = wx if wx is not None else x
    if ref is None:
        raise ValueError("decode_step needs x or wx (precomputed W_x(x))")
    use_triton = can_decode_step(cell, ref) and not torch.is_grad_enabled()
    if not use_triton:
        if block_table is not None:
            raise RuntimeError("block_table requires CUDA decode_step (Triton)")
        y = cell.step(state, x, wx=wx) if _accepts_wx(cell) else cell.step(state, x)
        if out is None:
            return y
        out.copy_(y)
        return out
    if wx is None:
        wx = decode_wx(cell, x)
    wx = _squeeze_time(wx)
    if _is_capturing():
        if not state.is_contiguous() or not wx.is_contiguous():
            raise RuntimeError("decode_step CUDA graph needs contiguous state and wx")
        if out is None and block_table is None:
            raise RuntimeError("decode_step CUDA graph needs out= (or block_table in-place)")
    elif not state.is_contiguous():
        state = state.contiguous()
    if not wx.is_contiguous():
        wx = wx.contiguous()
    out, bt, batch, has_bt = _resolve_out(state, out, block_table)
    if isinstance(cell, ParaGRU):
        _decode_gru(cell, state, wx, out, bt, batch, has_bt)
    elif isinstance(cell, ParaLSTM):
        _decode_lstm(cell, state, wx, out, bt, batch, has_bt)
    elif isinstance(cell, ParaSLSTM):
        _decode_slstm(cell, state, wx, out, bt, batch, has_bt)
    else:
        raise TypeError(f"decode_step Triton has no kernel for {type(cell).__name__}")
    compiling = torch.compiler.is_compiling()
    if log.isEnabledFor(logging.DEBUG) and not _is_capturing() and not compiling:
        log.debug(
            "decode_step",
            extra={
                "cell": type(cell).__name__,
                "batch": batch,
                "d_h": int(cell.d_h),
                "dtype": str(state.dtype),
                "device": str(state.device),
                "block_d": _BLOCK_D,
                "block_table": has_bt,
            },
        )
    return out


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _resolve_out(
    state: Tensor,
    out: Tensor | None,
    block_table: Tensor | None,
) -> tuple[Tensor, Tensor, int, bool]:
    if block_table is None:
        if out is None:
            out = torch.empty_like(state)
        elif out.shape != state.shape:
            raise ValueError(
                f"decode_step out shape {tuple(out.shape)} != state {tuple(state.shape)}"
            )
        elif out.dtype != state.dtype or out.device != state.device:
            raise ValueError("decode_step out must match state dtype and device")
        return out, state, int(state.shape[0]), False
    bt = block_table.to(device=state.device, dtype=torch.int32).contiguous()
    if bt.dim() != 1:
        raise ValueError(f"block_table must be 1-D (B,), got {tuple(bt.shape)}")
    batch = int(bt.numel())
    if out is None:
        out = state
    elif out.shape != state.shape:
        raise ValueError(
            "block_table decode writes pool-shaped out "
            f"(same as state {tuple(state.shape)}), got {tuple(out.shape)}"
        )
    elif out.dtype != state.dtype or out.device != state.device:
        raise ValueError("decode_step out must match state dtype and device")
    return out, bt, batch, True


def _accepts_wx(cell: nn.Module) -> bool:
    step = getattr(cell, "step", None)
    if step is None:
        return False
    try:
        return "wx" in inspect.signature(step).parameters
    except (TypeError, ValueError):
        return False


def _squeeze_time(wx: Tensor) -> Tensor:
    if wx.dim() == 3:
        if wx.shape[1] != 1:
            raise ValueError(f"decode_step wx time dim must be 1, got {tuple(wx.shape)}")
        return wx[:, 0]
    if wx.dim() != 2:
        raise ValueError(f"decode_step wx needs (B, G*d_h), got {tuple(wx.shape)}")
    return wx


def _cap_args(cell: nn.Module) -> tuple[bool, float]:
    cap = getattr(cell, "max_recurrent_norm", None)
    if cap is None:
        return False, 0.0
    return True, float(cap)


@triton.jit
def _slot_index(pid_b, bt_ptr, HAS_BT: tl.constexpr):
    if HAS_BT:
        return tl.load(bt_ptr + pid_b)
    return pid_b


@triton.jit
def _cap_vec(x, cap, HAS_CAP: tl.constexpr):
    if HAS_CAP:
        return tl.minimum(tl.maximum(x, -cap), cap)
    return x


@triton.jit
def _gru_decode_kernel(
    h_ptr,
    wx_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    out_ptr,
    bt_ptr,
    d_h,
    cap,
    stride_hb,
    stride_hd,
    stride_wb,
    stride_wd,
    stride_ob,
    stride_od,
    BLOCK_D: tl.constexpr,
    HAS_BT: tl.constexpr,
    HAS_CAP: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < d_h
    row = _slot_index(pid_b, bt_ptr, HAS_BT)
    h = load_acc(h_ptr + row * stride_hb + offs_d * stride_hd, mask, 0.0)
    az = _cap_vec(load_acc(az_ptr + offs_d, mask, 0.0), cap, HAS_CAP)
    ar = _cap_vec(load_acc(ar_ptr + offs_d, mask, 0.0), cap, HAS_CAP)
    an = _cap_vec(load_acc(an_ptr + offs_d, mask, 0.0), cap, HAS_CAP)
    base = wx_ptr + pid_b * stride_wb
    zx = load_acc(base + offs_d * stride_wd, mask, 0.0)
    rx = load_acc(base + (offs_d + d_h) * stride_wd, mask, 0.0)
    nx = load_acc(base + (offs_d + 2 * d_h) * stride_wd, mask, 0.0)
    z = tl.sigmoid(az * h + zx)
    r = tl.sigmoid(ar * h + rx)
    n = _tanh(an * (h * r) + nx)
    h_new = (1.0 - z) * h + z * n
    store_acc(out_ptr + row * stride_ob + offs_d * stride_od, h_new, mask)


@triton.jit
def _lstm_decode_kernel(
    s_ptr,
    wx_ptr,
    af_ptr,
    az_ptr,
    ao_ptr,
    cf_ptr,
    co_ptr,
    out_ptr,
    bt_ptr,
    d_h,
    cap,
    stride_sb,
    stride_ss,
    stride_sd,
    stride_wb,
    stride_wd,
    stride_ob,
    stride_os,
    stride_od,
    SLOT_C: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BT: tl.constexpr,
    HAS_CAP: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < d_h
    row = _slot_index(pid_b, bt_ptr, HAS_BT)
    c = load_acc(s_ptr + row * stride_sb + SLOT_C * stride_ss + offs_d * stride_sd, mask, 0.0)
    h = load_acc(s_ptr + row * stride_sb + SLOT_H * stride_ss + offs_d * stride_sd, mask, 0.0)
    a_f = _cap_vec(load_acc(af_ptr + offs_d, mask, 0.0), cap, HAS_CAP)
    a_z = _cap_vec(load_acc(az_ptr + offs_d, mask, 0.0), cap, HAS_CAP)
    a_o = _cap_vec(load_acc(ao_ptr + offs_d, mask, 0.0), cap, HAS_CAP)
    peephole_f = _cap_vec(load_acc(cf_ptr + offs_d, mask, 0.0), cap, HAS_CAP)
    peephole_o = _cap_vec(load_acc(co_ptr + offs_d, mask, 0.0), cap, HAS_CAP)
    base = wx_ptr + pid_b * stride_wb
    fx = load_acc(base + offs_d * stride_wd, mask, 0.0)
    zx = load_acc(base + (offs_d + d_h) * stride_wd, mask, 0.0)
    ox = load_acc(base + (offs_d + 2 * d_h) * stride_wd, mask, 0.0)
    f = tl.sigmoid(a_f * h + peephole_f * c + fx)
    z = _tanh(a_z * h + zx)
    c_new = f * c + (1.0 - f) * z
    o = tl.sigmoid(a_o * h + peephole_o * c_new + ox)
    h_act = _tanh(c_new)
    h_new = o * h_act
    store_acc(
        out_ptr + row * stride_ob + SLOT_C * stride_os + offs_d * stride_od,
        c_new,
        mask,
    )
    store_acc(
        out_ptr + row * stride_ob + SLOT_H * stride_os + offs_d * stride_od,
        h_new,
        mask,
    )


@triton.jit
def _slstm_decode_kernel(
    s_ptr,
    wx_ptr,
    r_ptr,
    out_ptr,
    bt_ptr,
    d_h,
    eps,
    cap,
    stride_sb,
    stride_ss,
    stride_sd,
    stride_wb,
    stride_wd,
    stride_rg,
    stride_rd,
    stride_ob,
    stride_os,
    stride_od,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BT: tl.constexpr,
    HAS_CAP: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < d_h
    row = _slot_index(pid_b, bt_ptr, HAS_BT)
    c = load_acc(s_ptr + row * stride_sb + SLOT_C * stride_ss + offs_d * stride_sd, mask, 0.0)
    n_prev = load_acc(s_ptr + row * stride_sb + SLOT_N * stride_ss + offs_d * stride_sd, mask, 0.0)
    m_prev = load_acc(s_ptr + row * stride_sb + SLOT_M * stride_ss + offs_d * stride_sd, mask, 0.0)
    h = load_acc(s_ptr + row * stride_sb + SLOT_H * stride_ss + offs_d * stride_sd, mask, 0.0)
    r_i = _cap_vec(load_acc(r_ptr + 0 * stride_rg + offs_d * stride_rd, mask, 0.0), cap, HAS_CAP)
    r_f = _cap_vec(load_acc(r_ptr + 1 * stride_rg + offs_d * stride_rd, mask, 0.0), cap, HAS_CAP)
    r_z = _cap_vec(load_acc(r_ptr + 2 * stride_rg + offs_d * stride_rd, mask, 0.0), cap, HAS_CAP)
    r_o = _cap_vec(load_acc(r_ptr + 3 * stride_rg + offs_d * stride_rd, mask, 0.0), cap, HAS_CAP)
    base = wx_ptr + pid_b * stride_wb
    zi_x = load_acc(base + offs_d * stride_wd, mask, 0.0)
    zf_x = load_acc(base + (offs_d + d_h) * stride_wd, mask, 0.0)
    zz_x = load_acc(base + (offs_d + 2 * d_h) * stride_wd, mask, 0.0)
    zo_x = load_acc(base + (offs_d + 3 * d_h) * stride_wd, mask, 0.0)
    z_i = r_i * h + zi_x
    z_f = r_f * h + zf_x
    z_z = r_z * h + zz_x
    z_o = r_o * h + zo_x
    left = z_f + m_prev
    m_new = tl.where(left > z_i, left, z_i)
    i_t = tl.exp(z_i - m_new)
    f_t = tl.exp(z_f + m_prev - m_new)
    z = _tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c + i_t * z
    o = tl.sigmoid(z_o)
    h_new = o * (c_new / (n_new + eps))
    store_acc(
        out_ptr + row * stride_ob + SLOT_C * stride_os + offs_d * stride_od,
        c_new,
        mask,
    )
    store_acc(
        out_ptr + row * stride_ob + SLOT_N * stride_os + offs_d * stride_od,
        n_new,
        mask,
    )
    store_acc(
        out_ptr + row * stride_ob + SLOT_M * stride_os + offs_d * stride_od,
        m_new,
        mask,
    )
    store_acc(
        out_ptr + row * stride_ob + SLOT_H * stride_os + offs_d * stride_od,
        h_new,
        mask,
    )


def _grid(batch: int, d_h: int) -> tuple[int, int]:
    return batch, (d_h + _BLOCK_D - 1) // _BLOCK_D


def _decode_gru(
    cell: ParaGRU,
    state: Tensor,
    wx: Tensor,
    out: Tensor,
    bt: Tensor,
    batch: int,
    has_bt: bool,
) -> None:
    a_z, a_r, a_n = cell.a_z, cell.a_r, cell.a_n
    d_h = int(cell.d_h)
    if wx.shape[-1] != 3 * d_h:
        raise ValueError(f"ParaGRU wx last dim {wx.shape[-1]} != 3 * d_h={3 * d_h}")
    if wx.shape[0] != batch:
        raise ValueError(f"ParaGRU wx batch {wx.shape[0]} != {batch}")
    validate_cuda_tensors(state, wx, a_z, a_r, a_n, out, name="decode_gru")
    has_cap, cap = _cap_args(cell)
    _gru_decode_kernel[_grid(batch, d_h)](
        state,
        wx,
        a_z,
        a_r,
        a_n,
        out,
        bt,
        d_h,
        cap,
        *state.stride(),
        *wx.stride(),
        *out.stride(),
        BLOCK_D=_BLOCK_D,
        HAS_BT=has_bt,
        HAS_CAP=has_cap,
    )


def _decode_lstm(
    cell: ParaLSTM,
    state: Tensor,
    wx: Tensor,
    out: Tensor,
    bt: Tensor,
    batch: int,
    has_bt: bool,
) -> None:
    d_h = int(cell.d_h)
    if wx.shape[-1] != 3 * d_h:
        raise ValueError(f"ParaLSTM wx last dim {wx.shape[-1]} != 3 * d_h={3 * d_h}")
    if wx.shape[0] != batch:
        raise ValueError(f"ParaLSTM wx batch {wx.shape[0]} != {batch}")
    validate_cuda_tensors(
        state, wx, cell.a_f, cell.a_z, cell.a_o, cell.c_f, cell.c_o, out, name="decode_lstm"
    )
    has_cap, cap = _cap_args(cell)
    _lstm_decode_kernel[_grid(batch, d_h)](
        state,
        wx,
        cell.a_f,
        cell.a_z,
        cell.a_o,
        cell.c_f,
        cell.c_o,
        out,
        bt,
        d_h,
        cap,
        *state.stride(),
        *wx.stride(),
        *out.stride(),
        SLOT_C=LSTM_CELL,
        SLOT_H=LSTM_HIDDEN,
        BLOCK_D=_BLOCK_D,
        HAS_BT=has_bt,
        HAS_CAP=has_cap,
    )


def _decode_slstm(
    cell: ParaSLSTM,
    state: Tensor,
    wx: Tensor,
    out: Tensor,
    bt: Tensor,
    batch: int,
    has_bt: bool,
) -> None:
    r = cell.R
    if r is None:
        raise TypeError("decode_step sLSTM is mix='diag' only")
    d_h = int(cell.d_h)
    if wx.shape[-1] != 4 * d_h:
        raise ValueError(f"ParaSLSTM wx last dim {wx.shape[-1]} != 4 * d_h={4 * d_h}")
    if wx.shape[0] != batch:
        raise ValueError(f"ParaSLSTM wx batch {wx.shape[0]} != {batch}")
    validate_cuda_tensors(state, wx, r, out, name="decode_slstm")
    has_cap, cap = _cap_args(cell)
    _slstm_decode_kernel[_grid(batch, d_h)](
        state,
        wx,
        r,
        out,
        bt,
        d_h,
        float(cell.eps),
        cap,
        *state.stride(),
        *wx.stride(),
        *r.stride(),
        *out.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_D=_BLOCK_D,
        HAS_BT=has_bt,
        HAS_CAP=has_cap,
    )
