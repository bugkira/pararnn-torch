import torch

from pararnn.device import experiment_device


def test_experiment_device_is_2080_ti():
    device = experiment_device()
    name = torch.cuda.get_device_name(device)
    assert "2080 Ti" in name, f"expected 2080 Ti, got cuda:{device.index} ({name})"
