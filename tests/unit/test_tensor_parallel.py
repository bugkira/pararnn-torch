"""Channel shard of a diag cell matches the full cell on those features."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pararnn import NewtonConfig, ParaGRU, ParaLSTM, ParaSLSTM, newton_apply
from pararnn.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelDiagBlock,
    copy_diag_cell_shard,
    hidden_shard_slice,
    local_hidden_size,
    tensor_parallel_diag_block,
)

_CFG = NewtonConfig(
    max_iters=3,
    scan_backend="eager",
    residual_atol=None,
    residual_fail=None,
    picard_adapt=False,
)


def test_local_hidden_size_divides() -> None:
    assert local_hidden_size(32, 2) == 16
    with pytest.raises(ValueError, match="divisible"):
        local_hidden_size(31, 2)


@pytest.mark.parametrize(
    ("make_src", "make_dst"),
    [
        (lambda: ParaGRU(6, 8), lambda: ParaGRU(6, 4)),
        (lambda: ParaLSTM(6, 8), lambda: ParaLSTM(6, 4)),
        (lambda: ParaSLSTM(6, 8, mix="diag"), lambda: ParaSLSTM(6, 4, mix="diag")),
    ],
)
def test_sharded_newton_matches_channel_slice(make_src, make_dst) -> None:
    torch.manual_seed(0)
    src = make_src()
    dst = make_dst()
    copy_diag_cell_shard(src, dst, rank=1, world_size=2)
    sl = hidden_shard_slice(src.d_h, rank=1, world_size=2)
    x = torch.randn(2, 12, src.d_in)
    y_full = newton_apply(src, x, _CFG)
    y_loc = newton_apply(dst, x, _CFG)
    torch.testing.assert_close(y_loc, y_full[..., sl], atol=1e-5, rtol=1e-5)


def test_copy_rejects_head_mix() -> None:
    src = ParaSLSTM(8, 8, mix="head", n_heads=2)
    dst = ParaSLSTM(8, 4, mix="diag")
    with pytest.raises(TypeError, match="channelwise-diagonal"):
        copy_diag_cell_shard(src, dst, rank=0, world_size=2)


def test_row_parallel_world1_matches_linear() -> None:
    torch.manual_seed(1)
    d_in, d_out = 7, 5
    ref = nn.Linear(d_in, d_out)
    row = RowParallelLinear(d_in, d_out)
    row.weight.data.copy_(ref.weight.detach())
    row.bias.data.copy_(ref.bias.detach())
    x = torch.randn(3, 9, d_in)
    torch.testing.assert_close(row(x), ref(x), atol=0, rtol=0)


def test_column_parallel_world1_matches_linear() -> None:
    torch.manual_seed(2)
    d_in, d_out = 6, 8
    ref = nn.Linear(d_in, d_out)
    col = ColumnParallelLinear(d_in, d_out)
    col.weight.data.copy_(ref.weight.detach())
    col.bias.data.copy_(ref.bias.detach())
    x = torch.randn(2, 5, d_in)
    torch.testing.assert_close(col(x), ref(x), atol=0, rtol=0)


def test_tp_block_world1_forward_shape() -> None:
    torch.manual_seed(3)
    block = tensor_parallel_diag_block("gru", d_in=8, d_h=8, config=_CFG)
    assert isinstance(block, TensorParallelDiagBlock)
    y = block(torch.randn(2, 10, 8))
    assert y.shape == (2, 10, 8)
