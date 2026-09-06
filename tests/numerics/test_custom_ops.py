"""``pararnn::scan_*`` custom ops: fake meta + compile smoke."""

from __future__ import annotations

import pytest
import torch

from pararnn.kernels import scan_block2_triton, scan_block4_triton, scan_diag_triton
from pararnn.solvers.scan import scan_block2, scan_block4, scan_diag


@pytest.mark.cuda
@torch.no_grad()
def test_scan_custom_ops_match_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(0)
    b, t, d = 2, 64, 8
    jac = torch.randn(b, t, d, device=cuda_device) * 0.3
    residual = torch.randn(b, t, d, device=cuda_device)
    torch.testing.assert_close(
        scan_diag_triton(jac, residual),
        scan_diag(jac, residual, backend="eager"),
        atol=1e-5,
        rtol=1e-5,
    )

    jac2 = torch.randn(b, t, 2, 2, d, device=cuda_device) * 0.2
    res2 = torch.randn(b, t, 2, d, device=cuda_device)
    torch.testing.assert_close(
        scan_block2_triton(jac2, res2),
        scan_block2(jac2, res2, backend="eager"),
        atol=1e-5,
        rtol=1e-5,
    )

    jac4 = torch.randn(b, t, 4, 4, d, device=cuda_device) * 0.12
    res4 = torch.randn(b, t, 4, d, device=cuda_device)
    torch.testing.assert_close(
        scan_block4_triton(jac4, res4),
        scan_block4(jac4, res4, backend="eager"),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.cuda
@torch.no_grad()
def test_scan_diag_custom_op_compiles(cuda_device: torch.device) -> None:
    """``register_fake`` lets Dynamo treat the scan as an opaque op."""
    torch.compiler.reset()
    torch.manual_seed(1)
    jac = torch.randn(2, 32, 8, device=cuda_device) * 0.3
    residual = torch.randn(2, 32, 8, device=cuda_device)
    ref = scan_diag_triton(jac, residual)

    def fn(j: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return scan_diag_triton(j, r)

    compiled = torch.compile(fn, fullgraph=True)
    torch.testing.assert_close(compiled(jac, residual), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
@torch.no_grad()
def test_scan_diag_opcheck(cuda_device: torch.device) -> None:
    torch.manual_seed(2)
    jac = torch.randn(2, 16, 4, device=cuda_device) * 0.3
    residual = torch.randn(2, 16, 4, device=cuda_device)
    torch.library.opcheck(scan_diag_triton, (jac, residual), test_utils="test_schema")


@pytest.mark.cuda
@torch.no_grad()
def test_newton_gru_head_fused_custom_op(cuda_device: torch.device) -> None:
    """``register_fake`` + opcheck for head fused Newton."""
    from pararnn.kernels.custom_ops import newton_gru_head_fused
    from pararnn.kernels.newton_gru_head import _newton_gru_head_fused_impl

    torch.manual_seed(3)
    torch.compiler.reset()
    d_h, n_heads, d_head = 32, 4, 8
    wx = torch.randn(2, 16, 3 * d_h, device=cuda_device)
    a = torch.randn(n_heads, d_head, d_head, device=cuda_device) * 0.1
    ref = _newton_gru_head_fused_impl(wx, a, a, a, max_iters=3, omega=1.0)
    got = newton_gru_head_fused(wx, a, a, a, None, None, max_iters=3, omega=1.0)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    torch.library.opcheck(
        newton_gru_head_fused,
        (wx, a, a, a, None, None),
        kwargs={"max_iters": 2, "omega": 1.0},
        test_utils="test_schema",
    )

    def fn(w: torch.Tensor) -> torch.Tensor:
        return newton_gru_head_fused(w, a, a, a, None, None, max_iters=2, omega=1.0)

    compiled = torch.compile(fn, fullgraph=True)
    torch.testing.assert_close(compiled(wx), fn(wx), atol=1e-5, rtol=1e-5)
