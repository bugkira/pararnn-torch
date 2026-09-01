import torch

from pararnn.hw import select_device


def test_package_device_is_torch_device():
    from pararnn import device

    assert isinstance(device, torch.device)


def test_select_device_cpu_when_allowed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = select_device(allow_cpu=True)
    assert device.type == "cpu"


def test_select_device_env_override(monkeypatch):
    monkeypatch.setenv("PARARNN_DEVICE", "cpu")
    device = select_device("2080 Ti", allow_cpu=True)
    assert device.type == "cpu"
