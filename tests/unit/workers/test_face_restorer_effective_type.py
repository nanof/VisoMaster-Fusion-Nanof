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


def test_effective_type_osdface_not_replaced_on_live_or_small_face():
    p = {
        "FaceRestorerUltraLightOnnxToggle": True,
        "FaceRestorerUltraLightOnLiveToggle": True,
        "FaceRestorerUltraLightOnSmallFaceToggle": True,
        "FaceRestorerUltraLightScaleGeDecimalSlider": "2.0",
        "FaceRestorerUltraLightPreferFp16Toggle": True,
    }
    assert (
        FrameWorker._face_restorer_effective_type(
            p, "OSDFace", 3.0, is_live_stream=True
        )
        == "OSDFace"
    )


def test_restorer_infer_cache_key_changes_with_osdface_sliders():
    base_params = {
        "FaceRestorerDetTypeSelection": "Original",
        "FaceFidelityWeightDecimalSlider": 0.9,
        "FaceRestorerUltraLightOnnxToggle": False,
        "FaceRestorerUltraLightOnLiveToggle": True,
        "FaceRestorerUltraLightOnSmallFaceToggle": False,
        "FaceRestorerUltraLightScaleGeDecimalSlider": 2.0,
        "FaceRestorerUltraLightPreferFp16Toggle": True,
        "SwapModelSelection": "Inswapper128",
        "SwapperResSelection": "128",
        "OSDFaceTimestepSlider": 399,
        "OSDFaceLatentStrengthDecimalSlider": 1.0,
        "OSDFaceTimestep2Slider": 399,
        "OSDFaceLatentStrength2DecimalSlider": 1.0,
    }
    control = {"DetectorScoreSlider": 0.5}
    key = FrameWorker._restorer_infer_cache_key(base_params, control, "OSDFace")
    key_ts1 = FrameWorker._restorer_infer_cache_key(
        {**base_params, "OSDFaceTimestepSlider": 450}, control, "OSDFace"
    )
    key_ls1 = FrameWorker._restorer_infer_cache_key(
        {**base_params, "OSDFaceLatentStrengthDecimalSlider": 0.8}, control, "OSDFace"
    )
    key_ts2 = FrameWorker._restorer_infer_cache_key(
        {**base_params, "OSDFaceTimestep2Slider": 350}, control, "OSDFace"
    )
    key_ls2 = FrameWorker._restorer_infer_cache_key(
        {**base_params, "OSDFaceLatentStrength2DecimalSlider": 0.75},
        control,
        "OSDFace",
    )
    assert key != key_ts1
    assert key != key_ls1
    assert key != key_ts2
    assert key != key_ls2


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


def test_restorer_infer_cache_key_ignores_blend_slider():
    base_params = {
        "FaceRestorerDetTypeSelection": "Original",
        "FaceFidelityWeightDecimalSlider": 0.9,
        "FaceRestorerUltraLightOnnxToggle": False,
        "FaceRestorerUltraLightOnLiveToggle": True,
        "FaceRestorerUltraLightOnSmallFaceToggle": False,
        "FaceRestorerUltraLightScaleGeDecimalSlider": 2.0,
        "FaceRestorerUltraLightPreferFp16Toggle": True,
        "SwapModelSelection": "Inswapper128",
        "SwapperResSelection": "128",
    }
    control = {"DetectorScoreSlider": 0.5}
    key_a = FrameWorker._restorer_infer_cache_key(
        {**base_params, "FaceRestorerBlendSlider": 40}, control, "GPEN-512"
    )
    key_b = FrameWorker._restorer_infer_cache_key(
        {**base_params, "FaceRestorerBlendSlider": 90}, control, "GPEN-512"
    )
    assert key_a == key_b
    assert "GPEN-512" in key_a


def test_restorer_infer_cache_key_changes_with_swapper():
    base_params = {
        "FaceRestorerDetTypeSelection": "Original",
        "FaceFidelityWeightDecimalSlider": 0.9,
        "FaceRestorerUltraLightOnnxToggle": False,
        "FaceRestorerUltraLightOnLiveToggle": True,
        "FaceRestorerUltraLightOnSmallFaceToggle": False,
        "FaceRestorerUltraLightScaleGeDecimalSlider": 2.0,
        "FaceRestorerUltraLightPreferFp16Toggle": True,
        "SwapperResSelection": "128",
    }
    control = {"DetectorScoreSlider": 0.5}
    key_a = FrameWorker._restorer_infer_cache_key(
        {**base_params, "SwapModelSelection": "Inswapper128"}, control, "GPEN-512"
    )
    key_b = FrameWorker._restorer_infer_cache_key(
        {**base_params, "SwapModelSelection": "GhostFace-v2"}, control, "GPEN-512"
    )
    assert key_a != key_b


def test_restorer_infer_cache_key_changes_with_secondary_swapper():
    base_params = {
        "FaceRestorerDetTypeSelection": "Original",
        "FaceFidelityWeightDecimalSlider": 0.9,
        "FaceRestorerUltraLightOnnxToggle": False,
        "FaceRestorerUltraLightOnLiveToggle": True,
        "FaceRestorerUltraLightOnSmallFaceToggle": False,
        "FaceRestorerUltraLightScaleGeDecimalSlider": 2.0,
        "FaceRestorerUltraLightPreferFp16Toggle": True,
        "SwapModelSelection": "Inswapper128",
        "SwapperResSelection": "128",
    }
    control = {"DetectorScoreSlider": 0.5}
    key_off = FrameWorker._restorer_infer_cache_key(base_params, control, "GPEN-512")
    key_on = FrameWorker._restorer_infer_cache_key(
        {**base_params, "SecondarySwapperEnableToggle": True},
        control,
        "GPEN-512",
    )
    assert key_off != key_on


def test_restorer_infer_swap_fingerprint_changes_with_content():
    import torch

    a = torch.zeros(3, 8, 8)
    b = a.clone()
    b[0, 4, 4] = 255
    assert FrameWorker._restorer_infer_swap_fingerprint(
        a
    ) != FrameWorker._restorer_infer_swap_fingerprint(b)
