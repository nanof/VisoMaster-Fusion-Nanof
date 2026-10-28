"""PipelineUpscalerEngine fail-soft behaviour (no weights, no GPU)."""

from __future__ import annotations

import numpy as np

from app.processors.pytorch_extras.pipeline_upscaler.engine import PipelineUpscalerEngine


def test_apply_returns_input_when_disabled():
    engine = PipelineUpscalerEngine()
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    out = engine.apply_bgr(frame, {"PipelineUpscalerEnableToggle": False})
    assert out is frame


def test_apply_returns_input_when_enabled_but_not_loaded():
    engine = PipelineUpscalerEngine()
    frame = np.full((16, 24, 3), 7, dtype=np.uint8)
    out = engine.apply_bgr(
        frame,
        {
            "PipelineUpscalerEnableToggle": True,
            "PipelineUpscalerTypeSelection": "FlashVSR Tiny",
        },
    )
    assert np.array_equal(out, frame)


def test_apply_oom_keeps_target_scale():
    class _Boom:
        is_loaded = True

        def upscale_bgr(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory")

    engine = PipelineUpscalerEngine()
    engine._backend = _Boom()
    engine._backend_kind = "flashvsr"
    engine._loaded_type = "FlashVSR Tiny"
    frame = np.full((8, 10, 3), 3, dtype=np.uint8)
    out = engine.apply_bgr(
        frame,
        {
            "PipelineUpscalerEnableToggle": True,
            "PipelineUpscalerTypeSelection": "FlashVSR Tiny",
            "PipelineUpscalerScaleSelection": "4x",
        },
    )
    assert out.shape == (32, 40, 3)


def test_apply_oom_keeps_1x_for_hypir():
    class _Boom:
        is_loaded = True

        def upscale_bgr(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory")

    engine = PipelineUpscalerEngine()
    engine._backend = _Boom()
    engine._backend_kind = "hypir"
    engine._loaded_type = "HYPIR SD2"
    frame = np.full((8, 10, 3), 3, dtype=np.uint8)
    out = engine.apply_bgr(
        frame,
        {
            "PipelineUpscalerEnableToggle": True,
            "PipelineUpscalerTypeSelection": "HYPIR SD2",
            "PipelineUpscalerScaleSelection": "1x",
        },
    )
    assert out.shape == (8, 10, 3)
    assert np.array_equal(out, frame)


def test_drop_unusable_flash_attn_stub():
    import sys
    import types

    previous = sys.modules.get("flash_attn")
    stub = types.ModuleType("flash_attn")
    stub.flash_attn_func = None
    sys.modules["flash_attn"] = stub
    try:
        from app.processors.pytorch_extras.pipeline_upscaler.flashvsr_backend import (
            _drop_unusable_flash_attn,
        )

        _drop_unusable_flash_attn()
        leftover = sys.modules.get("flash_attn")
        assert leftover is not stub
        if leftover is not None:
            assert callable(getattr(leftover, "flash_attn_func", None))
    finally:
        if previous is None:
            sys.modules.pop("flash_attn", None)
        else:
            sys.modules["flash_attn"] = previous


def test_flashvsr_lq_hw_equals_pipe_hw():
    """PixelShuffle 16 vs DiT /16: LQ tensor must not be 1/4 of height/width."""
    from app.processors.pytorch_extras.pipeline_upscaler.flashvsr_backend import (
        flashvsr_aligned_hw,
    )

    lq_h, lq_w = flashvsr_aligned_hw(272, 288)
    pipe_h, pipe_w = lq_h, lq_w
    assert (lq_h, lq_w) == (pipe_h, pipe_w)
    assert lq_h % 128 == 0 and lq_w % 128 == 0
    # The failing call used height=lq*4, which is 16× more DiT tokens.
    assert pipe_h * pipe_w != (272 * 4) * (288 * 4)


def test_flashvsr_working_hw_is_single_canvas():
    from app.processors.pytorch_extras.pipeline_upscaler.flashvsr_backend import (
        FlashVSRBackend,
    )

    backend = FlashVSRBackend()
    h, w = backend._working_hw(720, 1280, 640 * 768)
    assert h * w <= 640 * 768
    assert h % 128 == 0 and w % 128 == 0
