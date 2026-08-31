import pytest
import torch


def test_package_device_is_torch_device():
    try:
        from pararnn import device
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert isinstance(device, torch.device)


def test_device_is_2080_ti_when_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        from pararnn import device
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if device.type != "cuda":
        pytest.skip("no matching GPU")
    name = torch.cuda.get_device_name(device)
    assert "2080 Ti" in name, f"expected 2080 Ti, got cuda:{device.index} ({name})"
