"""Public numerics-contract helper: parallel vs sequential agreement."""

from __future__ import annotations

import pytest
import torch

from pararnn import (
    AgreementError,
    AgreementReport,
    NewtonConfig,
    ParaGRU,
    ParaRNN,
    ParaSLSTM,
    ParaSLSTMBlock,
    default_agreement_atol,
    verify_agreement,
)


@torch.no_grad()
def test_verify_agreement_cell_ok():
    torch.manual_seed(0)
    cell = ParaGRU(8, 16)
    x = torch.randn(2, 32, 8)
    report = verify_agreement(cell, x, config=NewtonConfig(max_iters=3))
    assert isinstance(report, AgreementReport)
    assert report.ok
    assert report.path == "cell"
    assert report.max_abs < 1e-4
    assert report.newton_iters is not None
    assert report.to_dict()["ok"] is True


@torch.no_grad()
def test_verify_agreement_para_rnn_ok():
    torch.manual_seed(1)
    model = ParaRNN(ParaSLSTM(16, 16, mix="diag"), config=NewtonConfig(max_iters=3))
    x = torch.randn(2, 24, 16)
    report = verify_agreement(model, x)
    assert report.ok
    assert report.path == "para_rnn"
    assert model.solver == "auto"  # restored


@torch.no_grad()
def test_verify_agreement_block_ok():
    torch.manual_seed(2)
    block = ParaSLSTMBlock(32, mlp_ratio=2.0, config=NewtonConfig(max_iters=3))
    x = torch.randn(2, 16, 32)
    report = verify_agreement(block, x)
    assert report.ok
    assert report.path == "para_rnn"


@torch.no_grad()
def test_verify_agreement_raise_on_fail():
    torch.manual_seed(3)
    cell = ParaGRU(4, 8)
    x = torch.randn(2, 16, 4)
    with pytest.raises(AgreementError) as ei:
        verify_agreement(
            cell,
            x,
            config=NewtonConfig(max_iters=3),
            atol=0.0,
            rtol=0.0,
            raise_on_fail=True,
        )
    assert ei.value.report.max_abs > 0.0
    assert ei.value.report.ok is False


def test_default_agreement_atol():
    assert default_agreement_atol(torch.float32) == 1e-4
    assert default_agreement_atol(torch.bfloat16) == 1e-2


def test_verify_agreement_rejects_unknown_module():
    with pytest.raises(TypeError, match="expects an RNN cell"):
        verify_agreement(torch.nn.Linear(4, 4), torch.randn(2, 8, 4))


@torch.no_grad()
def test_verify_first_step_runs_once():
    torch.manual_seed(4)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager", verify_first_step=True)
    model = ParaRNN(ParaGRU(8, 8), config=cfg)
    x = torch.randn(2, 16, 8)
    model.train()
    _ = model(x)
    assert model._agreement_verified
    # Second forward must not re-run the sequential oracle path.
    model.train()
    y = model(x)
    assert y.shape == (2, 16, 8)
    model.reset_agreement_check()
    assert not model._agreement_verified


@torch.no_grad()
def test_verify_first_step_off_by_default():
    model = ParaRNN(ParaGRU(4, 4), config=NewtonConfig(max_iters=2, scan_backend="eager"))
    assert model.config.verify_first_step is False
    model.train()
    _ = model(torch.randn(1, 8, 4))
    assert not model._agreement_verified
