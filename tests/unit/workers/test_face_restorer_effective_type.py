"""Tests for FrameWorker._face_restorer_effective_type (ultra-light GPEN path)."""

from app.processors.workers.frame_worker import FrameWorker


def test_effective_type_passthrough_when_ultra_light_off():
    p = {"FaceRestorerUltraLightOnnxToggle": False}
    assert (
        FrameWorker._face_restorer_effective_type(
            p, "CodeFormer", 2.5, is_live_stream=True
        )
        == "CodeFormer"
    )


def test_effective_type_live_triggers_fast_fp16():
    p = {
        "FaceRestorerUltraLightOnnxToggle": True,
        "FaceRestorerUltraLightOnLiveToggle": True,
        "FaceRestorerUltraLightOnSmallFaceToggle": False,
        "FaceRestorerUltraLightPreferFp16Toggle": True,
    }
    assert (
        FrameWorker._face_restorer_effective_type(
            p, "GFPGAN-v1.4", 1.0, is_live_stream=True
        )
        == "GPEN-256 Fast FP16 (128→256)"
    )


def test_effective_type_small_face_triggers_fast_fp32():
    p = {
        "FaceRestorerUltraLightOnnxToggle": True,
        "FaceRestorerUltraLightOnLiveToggle": False,
        "FaceRestorerUltraLightOnSmallFaceToggle": True,
        "FaceRestorerUltraLightScaleGeDecimalSlider": "2.0",
        "FaceRestorerUltraLightPreferFp16Toggle": False,
    }
    assert (
        FrameWorker._face_restorer_effective_type(
            p, "VQFR-v2", 2.1, is_live_stream=False
        )
        == "GPEN-256 Fast (128→256)"
    )


def test_effective_type_manual_fast_unchanged():
    p = {
        "FaceRestorerUltraLightOnnxToggle": True,
        "FaceRestorerUltraLightOnLiveToggle": True,
        "FaceRestorerUltraLightPreferFp16Toggle": True,
    }
    assert (
        FrameWorker._face_restorer_effective_type(
            p, "GPEN-256 Fast (128→256)", 1.0, is_live_stream=True
        )
        == "GPEN-256 Fast (128→256)"
    )
