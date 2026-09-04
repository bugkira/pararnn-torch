"""Megatron-style tensor parallel along channelwise ``d_h``.

``mix='diag'`` (and ParaGRU / ParaLSTM) keeps channels independent through
the Newton scan. Each rank owns ``d_h / N`` features: the cell's ``W_x`` is
already a column-parallel map (replicated ``x``, sharded output channels),
the fused Triton kernel runs locally, and ``RowParallelLinear`` maps
``d_h/N → d_out`` with one AllReduce. The split is ParaGRU, ParaLSTM, and
``ParaSLSTM(mix='diag')``.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.layers.para_rnn import ParaRNN
from pararnn.solvers.newton.config import NewtonConfig
from pararnn.weight_init import kaiming_uniform_linear_

log = logging.getLogger(__name__)

_CELL_GATES: dict[type, int] = {ParaGRU: 3, ParaLSTM: 3, ParaSLSTM: 4}


def tp_world_size(group: dist.ProcessGroup | None = None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size(group=group))


def tp_rank(group: dist.ProcessGroup | None = None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return int(dist.get_rank(group=group))


def local_hidden_size(d_h: int, world_size: int) -> int:
    """Per-rank channel count. ``d_h`` must divide evenly by ``world_size``."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if d_h % world_size != 0:
        raise ValueError(f"d_h={d_h} must be divisible by tensor-parallel size {world_size}")
    return d_h // world_size


def hidden_shard_slice(d_h: int, rank: int, world_size: int) -> slice:
    loc = local_hidden_size(d_h, world_size)
    start = rank * loc
    return slice(start, start + loc)


class _SumAllReduce(torch.autograd.Function):
    """Forward allreduce-sum; backward is identity.

    Megatron row-parallel: every rank holds the same ``y`` and the same
    replicated loss, so ``∂L/∂x_local = ∂L/∂y``.
    """

    @staticmethod
    def forward(ctx: object, tensor: Tensor, group: dist.ProcessGroup | None) -> Tensor:
        del ctx
        out = tensor.clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx: object, grad: Tensor) -> tuple[Tensor, None]:
        del ctx
        return grad, None


def _all_reduce(tensor: Tensor, group: dist.ProcessGroup | None) -> Tensor:
    if tp_world_size(group) == 1:
        return tensor
    return _SumAllReduce.apply(tensor, group)


class ColumnParallelLinear(nn.Module):
    """Split the output features of ``nn.Linear`` (Megatron column).

    ``out_features`` is the full width. This rank holds ``out_features / N``
    rows. Input ``x`` is replicated. No collective in forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group
        world = tp_world_size(process_group)
        self.out_features_local = local_hidden_size(out_features, world)
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.out_features_local, in_features, **factory))
        self.bias = nn.Parameter(torch.empty(self.out_features_local, **factory)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_linear_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features} (local_out={self.out_features_local})"

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """``y = allreduce(x @ Wᵀ) + b``. ``in_features`` is the local shard ``d_h/N``.

    Bias is replicated and added after the AllReduce so it is counted once.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        self.bias = nn.Parameter(torch.empty(out_features, **factory)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_linear_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}"

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, None)
        out = _all_reduce(out, self.process_group)
        if self.bias is not None:
            out = out + self.bias
        return out


def _require_diag_cell(cell: nn.Module) -> None:
    mix = getattr(cell, "mix", "diag")
    if mix != "diag":
        raise TypeError(
            "tensor parallel along d_h needs channelwise-diagonal mix "
            f"(got mix={mix!r} on {type(cell).__name__})"
        )


def shard_gated_linear_(dst: nn.Linear, src: nn.Linear, n_gates: int, sl: slice) -> None:
    """Copy gate-concat rows ``[z|r|…]`` so local layout stays ``n_gates * d_local``."""
    if src.out_features % n_gates != 0:
        raise ValueError(
            f"src.out_features={src.out_features} is not divisible by n_gates={n_gates}"
        )
    dst.weight.data.copy_(_shard_gate_rows(src.weight.detach(), n_gates, sl))
    if src.bias is None or dst.bias is None:
        return
    dst.bias.data.copy_(_shard_gate_rows(src.bias.detach(), n_gates, sl))


def _shard_gate_rows(t: Tensor, n_gates: int, sl: slice) -> Tensor:
    return torch.cat([chunk[sl] for chunk in t.chunk(n_gates, dim=0)], dim=0)


def copy_diag_cell_shard(src: nn.Module, dst: nn.Module, rank: int, world_size: int) -> None:
    """Copy the channel slice ``rank`` owns from a full-width diag cell into ``dst``."""
    if type(src) is not type(dst):
        raise TypeError(f"cell types differ: {type(src).__name__} vs {type(dst).__name__}")
    _require_diag_cell(src)
    _require_diag_cell(dst)
    d_h = int(src.d_h)
    sl = hidden_shard_slice(d_h, rank, world_size)
    if int(dst.d_h) != sl.stop - sl.start:
        raise ValueError(f"dst.d_h={dst.d_h} does not match shard {sl} of src.d_h={d_h}")
    n_gates = _CELL_GATES[type(src)]
    shard_gated_linear_(dst.W_x, src.W_x, n_gates, sl)
    if isinstance(src, ParaGRU):
        dst.a_z.data.copy_(src.a_z.detach()[sl])
        dst.a_r.data.copy_(src.a_r.detach()[sl])
        dst.a_n.data.copy_(src.a_n.detach()[sl])
        return
    if isinstance(src, ParaLSTM):
        for name in ("a_f", "a_z", "a_o", "c_f", "c_o"):
            getattr(dst, name).data.copy_(getattr(src, name).detach()[sl])
        return
    if isinstance(src, ParaSLSTM):
        dst.R.data.copy_(src.R.detach()[..., sl])
        return
    raise TypeError(f"copy_diag_cell_shard has no rule for {type(src).__name__}")


def copy_row_linear_shard(dst: RowParallelLinear, src: nn.Linear, sl: slice) -> None:
    """``src`` is ``Linear(d_h, d_out)``; ``dst`` holds columns ``sl``."""
    dst.weight.data.copy_(src.weight.detach()[:, sl])
    if src.bias is not None and dst.bias is not None:
        dst.bias.data.copy_(src.bias.detach())


class TensorParallelDiagBlock(nn.Module):
    """Column ``W_x`` (inside the cell) → local Newton scan → row ``W_out`` + AllReduce.

    Build the cell at ``hidden_size = local_hidden_size(d_h, N)``. Input ``x``
    is replicated ``(B, T, d_in)``. Scan communication is zero. The AllReduce
    sits on the output projection.
    """

    def __init__(
        self,
        cell: nn.Module,
        d_out: int,
        *,
        process_group: dist.ProcessGroup | None = None,
        config: NewtonConfig | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _require_diag_cell(cell)
        self.process_group = process_group
        self.d_in = int(cell.d_in)
        self.d_h_local = int(cell.d_h)
        self.d_out = d_out
        self.rnn = ParaRNN(cell, config=config, device=device, dtype=dtype)
        self.out = RowParallelLinear(
            self.d_h_local,
            d_out,
            process_group=process_group,
            device=device,
            dtype=dtype,
        )
        if not torch.compiler.is_compiling():
            log.debug(
                "tp_diag_block",
                extra={
                    "cell": type(cell).__name__,
                    "d_in": self.d_in,
                    "d_h_local": self.d_h_local,
                    "d_out": d_out,
                    "tp": tp_world_size(process_group),
                    "rank": tp_rank(process_group),
                },
            )

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_h_local={self.d_h_local}, d_out={self.d_out}, "
            f"tp={tp_world_size(self.process_group)}"
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.rnn(x))


def tensor_parallel_diag_block(
    kind: str,
    d_in: int,
    d_h: int,
    d_out: int | None = None,
    *,
    process_group: dist.ProcessGroup | None = None,
    config: NewtonConfig | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    **cell_kwargs: object,
) -> TensorParallelDiagBlock:
    """Factory: sharded diag cell + row-parallel output.

    ``d_h`` is the *full* hidden width. ``d_out`` defaults to ``d_in``
    (residual-shaped block).
    """
    world = tp_world_size(process_group)
    d_local = local_hidden_size(d_h, world)
    d_out = d_in if d_out is None else d_out
    cell_kwargs = dict(cell_kwargs)
    if kind == "gru":
        cell: nn.Module = ParaGRU(d_in, d_local, device=device, dtype=dtype, **cell_kwargs)
    elif kind == "lstm":
        cell = ParaLSTM(d_in, d_local, device=device, dtype=dtype, **cell_kwargs)
    elif kind == "slstm":
        cell_kwargs.setdefault("mix", "diag")
        cell = ParaSLSTM(d_in, d_local, device=device, dtype=dtype, **cell_kwargs)
        _require_diag_cell(cell)
    else:
        raise ValueError(f"kind must be gru, lstm, or slstm, got {kind!r}")
    return TensorParallelDiagBlock(
        cell,
        d_out,
        process_group=process_group,
        config=config,
        device=device,
        dtype=dtype,
    )
