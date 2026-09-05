"""``torch.amp.autocast`` smoke around ``ParaRNN``.

Newton disables outer autocast and runs in the module/input dtype (see
``_newton_precision_region``). Half-precision training uses an explicit
``.to(dtype)`` on the module and ``x``. This file checks that wrapping a
float32 ``ParaRNN`` in autocast still completes forward+backward and matches
the float32 reference.
"""

from __future__ import annotations

import pytest
import torch

from pararnn import NewtonConfig, ParaGRU, ParaRNN

_ATOL = 1e-5
_RTOL = 1e-5


def _eager_cfg() -> NewtonConfig:
    return NewtonConfig(max_iters=3, scan_backend="eager")


@pytest.mark.cuda
@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_autocast_pararnn_matches_fp32_reference(
    cuda_device: torch.device,
    amp_dtype: torch.dtype,
) -> None:
    if amp_dtype is torch.bfloat16:
        major, _ = torch.cuda.get_device_capability(cuda_device)
        if major < 8:
            pytest.skip("bf16 autocast needs CC >= 8.0")

    torch.manual_seed(0)
    model_ref = ParaRNN(ParaGRU(8, 16, device=cuda_device), config=_eager_cfg())
    model_ref.train()
    model_amp = ParaRNN(ParaGRU(8, 16, device=cuda_device), config=_eager_cfg())
    model_amp.load_state_dict(model_ref.state_dict())
    model_amp.train()

    x_ref = torch.randn(2, 16, 8, device=cuda_device, requires_grad=True)
    x_amp = x_ref.detach().clone().requires_grad_(True)

    y_ref = model_ref(x_ref)
    (y_ref.sum()).backward()

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        y_amp = model_amp(x_amp)
        (y_amp.float().sum()).backward()

    # Newton opts out of autocast: activations stay float32.
    assert y_amp.dtype == torch.float32
    torch.testing.assert_close(y_amp, y_ref, atol=_ATOL, rtol=_RTOL)
    assert x_ref.grad is not None
    assert x_amp.grad is not None
    torch.testing.assert_close(x_amp.grad, x_ref.grad, atol=_ATOL, rtol=_RTOL)
    for p_a, p_r in zip(model_amp.parameters(), model_ref.parameters(), strict=True):
        assert p_a.grad is not None
        assert p_r.grad is not None
        torch.testing.assert_close(p_a.grad, p_r.grad, atol=_ATOL, rtol=_RTOL)
