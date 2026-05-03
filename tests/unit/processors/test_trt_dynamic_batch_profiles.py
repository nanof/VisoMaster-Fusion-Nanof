"""PERF-005: TensorRT dynamic batch profiles (env vs UI control)."""

import pytest

from app.processors.trt_dynamic_batch_profiles import tensorrt_dynamic_shape_profile_opts


def test_trt_swap_profiles_use_env_when_ui_toggle_off(monkeypatch):
    monkeypatch.setenv("VISIOMASTER_TRT_MAX_BATCH_SWAP", "5")
    monkeypatch.setenv("VISIOMASTER_TRT_OPT_BATCH_SWAP", "3")
    p = tensorrt_dynamic_shape_profile_opts(
        "Inswapper128", control={"TrtDynamicBatchTuningToggle": False}
    )
    assert p is not None
    assert "target:5x3x128x128" in p["trt_profile_max_shapes"]
    assert "target:3x3x128x128" in p["trt_profile_opt_shapes"]


def test_trt_swap_profiles_ui_overrides_env_when_toggle_on(monkeypatch):
    monkeypatch.setenv("VISIOMASTER_TRT_MAX_BATCH_SWAP", "16")
    monkeypatch.setenv("VISIOMASTER_TRT_OPT_BATCH_SWAP", "4")
    ctrl = {
        "TrtDynamicBatchTuningToggle": True,
        "TrtMaxBatchSwapSlider": 3,
        "TrtOptBatchSwapSlider": 2,
        "TrtMaxBatchArcfaceSlider": 16,
        "TrtOptBatchArcfaceSlider": 8,
        "TrtMaxBatchLpMotionSlider": 8,
        "TrtOptBatchLpMotionSlider": 2,
        "TrtMaxBatchLpStitchSlider": 12,
        "TrtOptBatchLpStitchSlider": 4,
    }
    p = tensorrt_dynamic_shape_profile_opts("Inswapper128", control=ctrl)
    assert p is not None
    assert "target:3x3x128x128" in p["trt_profile_max_shapes"]
    assert "target:2x3x128x128" in p["trt_profile_opt_shapes"]
    assert "source:3x512" in p["trt_profile_max_shapes"]


def test_trt_lp_motion_respects_ui(monkeypatch):
    monkeypatch.delenv("VISIOMASTER_TRT_NO_DYNAMIC_PROFILES", raising=False)
    monkeypatch.delenv("VISIOMASTER_LP_MOTION_TRT_STATIC_BATCH", raising=False)
    ctrl = {
        "TrtDynamicBatchTuningToggle": True,
        "TrtMaxBatchSwapSlider": 16,
        "TrtOptBatchSwapSlider": 4,
        "TrtMaxBatchArcfaceSlider": 16,
        "TrtOptBatchArcfaceSlider": 8,
        "TrtMaxBatchLpMotionSlider": 4,
        "TrtOptBatchLpMotionSlider": 2,
        "TrtMaxBatchLpStitchSlider": 12,
        "TrtOptBatchLpStitchSlider": 4,
    }
    p = tensorrt_dynamic_shape_profile_opts(
        "LivePortraitMotionExtractor", control=ctrl
    )
    assert p is not None
    assert "img:4x3x256x256" in p["trt_profile_max_shapes"]
    assert "img:2x3x256x256" in p["trt_profile_opt_shapes"]


def test_trt_profiles_none_when_disabled(monkeypatch):
    monkeypatch.setenv("VISIOMASTER_TRT_NO_DYNAMIC_PROFILES", "1")
    p = tensorrt_dynamic_shape_profile_opts("Inswapper128", control=None)
    assert p is None
