"""Last-stage pipeline upscaler scale helper (no GPU)."""

from __future__ import annotations

from app.processors.pytorch_extras.pipeline_upscaler.scale import (
    PIPELINE_UPSCALER_TYPE_FLASHVSR_TINY,
    PIPELINE_UPSCALER_TYPE_HYPIR,
    PIPELINE_UPSCALER_TYPE_PMRF,
    pipeline_upscaler_enabled,
    pipeline_upscaler_output_dims,
    pipeline_upscaler_scale_factor,
    pipeline_upscaler_type,
)


def test_disabled_scale_is_one():
    assert pipeline_upscaler_scale_factor({}) == 1
    assert pipeline_upscaler_scale_factor({"PipelineUpscalerEnableToggle": False}) == 1


def test_enabled_hypir_default_is_1x():
    ctrl = {"PipelineUpscalerEnableToggle": True}
    assert pipeline_upscaler_type(ctrl) == PIPELINE_UPSCALER_TYPE_HYPIR
    assert pipeline_upscaler_scale_factor(ctrl) == 1
    h, w = pipeline_upscaler_output_dims(ctrl, frame_height=270, frame_width=480)
    assert (h, w) == (270, 480)


def test_pmrf_default_is_2x():
    ctrl = {
        "PipelineUpscalerEnableToggle": True,
        "PipelineUpscalerTypeSelection": PIPELINE_UPSCALER_TYPE_PMRF,
    }
    assert pipeline_upscaler_scale_factor(ctrl) == 2
    h, w = pipeline_upscaler_output_dims(ctrl, frame_height=270, frame_width=480)
    assert (h, w) == (540, 960)


def test_flashvsr_without_scale_is_4x():
    ctrl = {
        "PipelineUpscalerEnableToggle": True,
        "PipelineUpscalerTypeSelection": PIPELINE_UPSCALER_TYPE_FLASHVSR_TINY,
    }
    assert pipeline_upscaler_scale_factor(ctrl) == 4
    h, w = pipeline_upscaler_output_dims(ctrl, frame_height=270, frame_width=480)
    assert (h, w) == (1080, 1920)


def test_2x_scale_and_even_dims():
    ctrl = {
        "PipelineUpscalerEnableToggle": True,
        "PipelineUpscalerScaleSelection": "2x",
    }
    assert pipeline_upscaler_scale_factor(ctrl) == 2
    h, w = pipeline_upscaler_output_dims(ctrl, frame_height=271, frame_width=481)
    assert h % 2 == 0 and w % 2 == 0
    assert h == 542 and w == 962


def test_explicit_1x():
    ctrl = {
        "PipelineUpscalerEnableToggle": True,
        "PipelineUpscalerTypeSelection": PIPELINE_UPSCALER_TYPE_FLASHVSR_TINY,
        "PipelineUpscalerScaleSelection": "1x",
    }
    assert pipeline_upscaler_scale_factor(ctrl) == 1


def test_type_fallback():
    assert pipeline_upscaler_type(None) == PIPELINE_UPSCALER_TYPE_HYPIR
    assert pipeline_upscaler_type({"PipelineUpscalerTypeSelection": "nope"}) == (
        PIPELINE_UPSCALER_TYPE_HYPIR
    )
    assert pipeline_upscaler_enabled({"PipelineUpscalerEnableToggle": True}) is True
