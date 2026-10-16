import importlib
import sys
from unittest.mock import patch

import torch


def _reload_torch_setup():
    sys.modules.pop("rct.torch_setup", None)
    return importlib.import_module("rct.torch_setup")


def test_device_override_prefers_cpu(monkeypatch):
    monkeypatch.setenv("RCT_DEVICE", "cpu")

    with patch("torch.cuda.is_available", return_value=True):
        module = _reload_torch_setup()

    assert module.DEVICE == torch.device("cpu")
