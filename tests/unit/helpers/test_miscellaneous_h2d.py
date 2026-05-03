"""PERF-004: rgb_hwc_uint8_numpy_to_torch_chw pinned / async H2D helper."""

import numpy as np
import pytest
import torch

from app.helpers.miscellaneous import rgb_hwc_uint8_numpy_to_torch_chw


def test_rgb_hwc_chw_shape_cpu():
    x = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    t = rgb_hwc_uint8_numpy_to_torch_chw(x, "cpu")
    assert t.shape == (3, 4, 5)
    assert t.dtype == torch.uint8
    assert t.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rgb_hwc_cuda_contiguous():
    x = np.random.default_rng(0).integers(0, 255, size=(8, 12, 3), dtype=np.uint8)
    t = rgb_hwc_uint8_numpy_to_torch_chw(x, "cuda:0")
    assert t.shape == (3, 8, 12)
    assert t.is_contiguous()
    assert t.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rgb_hwc_respects_disable_env(monkeypatch):
    monkeypatch.setenv("VISIOMASTER_DISABLE_PINNED_H2D", "1")
    x = np.zeros((2, 3, 3), dtype=np.uint8)
    t = rgb_hwc_uint8_numpy_to_torch_chw(x, "cuda:0")
    assert t.shape == (3, 2, 3)
    monkeypatch.delenv("VISIOMASTER_DISABLE_PINNED_H2D", raising=False)
