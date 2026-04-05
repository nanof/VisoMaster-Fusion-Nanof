"""Smoke tests for swap_light_touch_chw_uint8."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from app.processors.utils import faceutil


def test_swap_light_touch_usm_shape_and_dtype():
    x = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    p = {
        "SwapLightTouchUSMAmountDecimalSlider": "0.5",
        "SwapLightTouchClaheEnableToggle": False,
    }
    y = faceutil.swap_light_touch_chw_uint8(x, p)
    assert y.shape == x.shape
    assert y.dtype == torch.uint8


def test_swap_light_touch_noop_when_amounts_zero():
    x = torch.ones(3, 32, 32, dtype=torch.uint8) * 128
    p = {
        "SwapLightTouchUSMAmountDecimalSlider": "0",
        "SwapLightTouchClaheEnableToggle": False,
    }
    y = faceutil.swap_light_touch_chw_uint8(x, p)
    assert torch.equal(y, x)


def test_swap_light_touch_clahe_runs_when_enabled():
    x = torch.randint(40, 200, (3, 64, 64), dtype=torch.uint8)
    p = {
        "SwapLightTouchUSMAmountDecimalSlider": "0",
        "SwapLightTouchClaheEnableToggle": True,
        "SwapLightTouchClaheClipDecimalSlider": "1.0",
        "SwapLightTouchClaheBlendDecimalSlider": "0.5",
    }
    y = faceutil.swap_light_touch_chw_uint8(x, p)
    assert y.shape == x.shape
    assert y.dtype == torch.uint8
