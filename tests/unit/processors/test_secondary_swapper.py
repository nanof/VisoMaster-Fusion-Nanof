"""Secondary Swapper helpers and blend math (upstream #337, adapted)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.processors.workers.frame_worker import FrameWorker


ARCFACE = FrameWorker._SECONDARY_SWAPPER_ARCFACE_MODELS


def test_arcface_family_includes_fork_models_but_not_ghost_dfm():
    for name in (
        "Inswapper128",
        "AlphaFace",
        "SimSwap512-CrossFace",
        "ReHiFace-S",
        "BlendSwap-256",
        "UniFace-256",
    ):
        assert name in ARCFACE
    assert "HyperSwap-v1" not in ARCFACE
    assert "GhostFace-v1" not in ARCFACE
    assert "DeepFaceLive (DFM)" not in ARCFACE
    assert "SimSwap512" not in ARCFACE


def test_pair_allowed_arcface_without_hyperswap_toggle():
    params = {}
    assert (
        FrameWorker._secondary_pair_allowed("Inswapper128", "AlphaFace", params) is True
    )
    assert (
        FrameWorker._secondary_pair_allowed("GhostFace-v1", "AlphaFace", params)
        is False
    )


def test_pair_allowed_hyperswap_requires_mix_toggle():
    off = {"SecondarySwapperHyperSwapMixEnableToggle": False}
    on = {"SecondarySwapperHyperSwapMixEnableToggle": True}
    assert (
        FrameWorker._secondary_pair_allowed("HyperSwap-v1", "Inswapper128", off)
        is False
    )
    assert (
        FrameWorker._secondary_pair_allowed("Inswapper128", "HyperSwap-v2", off)
        is False
    )
    assert (
        FrameWorker._secondary_pair_allowed("HyperSwap-v1", "Inswapper128", on) is True
    )


def test_secondary_blend_alpha_clamps_and_skips_zero():
    assert FrameWorker._secondary_blend_alpha({}) == pytest.approx(0.5)
    assert FrameWorker._secondary_blend_alpha(
        {"SecondarySwapperBlendAmountSlider": 0}
    ) == pytest.approx(0.0)
    assert FrameWorker._secondary_blend_alpha(
        {"SecondarySwapperBlendAmountSlider": 150}
    ) == pytest.approx(1.0)


def test_secondary_requested_defaults_off():
    assert FrameWorker._secondary_swapper_requested({}) is False
    assert (
        FrameWorker._secondary_swapper_requested(
            {"SecondarySwapperEnableToggle": False}
        )
        is False
    )
    assert (
        FrameWorker._secondary_swapper_requested({"SecondarySwapperEnableToggle": True})
        is True
    )


def test_alignment_only_256_stays_true_when_secondary_off():
    params = {"SecondarySwapperEnableToggle": False}
    assert FrameWorker._alignment_only_256("AlphaFace", params) is True
    assert FrameWorker._alignment_only_256("Inswapper128", params) is False


def test_alignment_only_256_false_when_secondary_needs_other_crops():
    params = {
        "SecondarySwapperEnableToggle": True,
        "SecondarySwapModelSelection": "Inswapper128",
    }
    assert FrameWorker._alignment_only_256("AlphaFace", params) is False
    params["SecondarySwapModelSelection"] = "AlphaFace"
    assert FrameWorker._alignment_only_256("AlphaFace", params) is True


def test_plane_batch_disabled_when_secondary_on():
    worker = SimpleNamespace(
        models_processor=SimpleNamespace(provider_name="Custom"),
        _secondary_swapper_requested=FrameWorker._secondary_swapper_requested,
    )
    s_e = np.zeros(512, dtype=np.float32)
    off = {
        "SwapModelSelection": "Inswapper128",
        "SecondarySwapperEnableToggle": False,
    }
    on = {**off, "SecondarySwapperEnableToggle": True}
    assert (
        FrameWorker._plane_multi_face_batch_key(worker, off, {}, True, s_e)
        == "Inswapper128"
    )
    assert FrameWorker._plane_multi_face_batch_key(worker, on, {}, True, s_e) is None


def test_secondary_runtime_params_map_instyle_resolution():
    worker = SimpleNamespace()
    tform = SimpleNamespace(scale=1.0)
    params = {
        "SwapModelSelection": "Inswapper128",
        "SecondarySwapperResSelection": "512",
        "StrengthEnableToggle": True,
        "SecondaryStrengthAmountSlider": 200,
        "InStyleResAEnableToggle": False,
    }
    mapped, strength_on, amount = FrameWorker._secondary_swapper_runtime_parameters(
        worker, params, "InStyleSwapper256 Version A", tform
    )
    assert mapped["SwapModelSelection"] == "InStyleSwapper256 Version A"
    assert mapped["InStyleResAEnableToggle"] is True
    assert mapped["AlphaFaceResSelection"] == "512"
    assert strength_on is True
    assert amount == 200.0


def test_preview_cache_key_ignores_blend_amount_and_mode():
    worker = SimpleNamespace()
    base = {
        "SwapModelSelection": "Inswapper128",
        "SecondarySwapModelSelection": "AlphaFace",
    }
    k0 = FrameWorker._secondary_preview_cache_key(worker, base, "Inswapper128")
    k1 = FrameWorker._secondary_preview_cache_key(
        worker,
        {**base, "SecondarySwapperBlendAmountSlider": 10},
        "Inswapper128",
    )
    k2 = FrameWorker._secondary_preview_cache_key(
        worker,
        {
            **base,
            "SecondarySwapperBlendAmountSlider": 90,
            "SecondarySwapperBlendModeSelection": "Center weighted",
        },
        "Inswapper128",
    )
    assert k0 == k1 == k2
    k3 = FrameWorker._secondary_preview_cache_key(
        worker,
        {**base, "SecondarySwapModelSelection": "Inswapper128"},
        "Inswapper128",
    )
    assert k3 != k0


def test_center_weighted_mix_prefers_secondary_in_the_interior():
    primary = torch.zeros(3, 16, 16)
    secondary = torch.ones(3, 16, 16)
    out = FrameWorker._mix_center_weighted(primary, secondary, 1.0)
    assert out[:, 8, 8].mean().item() > out[:, 0, 0].mean().item()


def test_linear_mix_at_zero_keeps_primary():
    class _W:
        _secondary_blend_alpha = staticmethod(FrameWorker._secondary_blend_alpha)
        _ensure_swap_crop_chw = staticmethod(FrameWorker._ensure_swap_crop_chw)
        _swap_crop_to_unit01 = staticmethod(FrameWorker._swap_crop_to_unit01)
        _swap_crop_from_unit01 = staticmethod(FrameWorker._swap_crop_from_unit01)

    primary = torch.zeros(3, 8, 8, dtype=torch.uint8)
    secondary = torch.full((3, 8, 8), 200, dtype=torch.uint8)
    out = FrameWorker._mix_secondary_swap_tensors(
        _W(),
        primary,
        secondary,
        {
            "SecondarySwapperBlendAmountSlider": 0,
            "SecondarySwapperBlendModeSelection": "Linear",
        },
    )
    assert torch.equal(out, primary)


def test_linear_mix_preserves_float_0_255_range():
    """get_swapped_and_prev_face returns float CHW in 0-255, not uint8.

    Clamping the mix to 0-1 left a solid black face crop (the user-visible square).
    """

    class _W:
        _secondary_blend_alpha = staticmethod(FrameWorker._secondary_blend_alpha)
        _ensure_swap_crop_chw = staticmethod(FrameWorker._ensure_swap_crop_chw)
        _swap_crop_to_unit01 = staticmethod(FrameWorker._swap_crop_to_unit01)
        _swap_crop_from_unit01 = staticmethod(FrameWorker._swap_crop_from_unit01)

    primary = torch.full((3, 8, 8), 200.0)
    secondary = torch.full((3, 8, 8), 100.0)
    out = FrameWorker._mix_secondary_swap_tensors(
        _W(),
        primary,
        secondary,
        {
            "SecondarySwapperBlendAmountSlider": 50,
            "SecondarySwapperBlendModeSelection": "Linear",
        },
    )
    assert out.dtype == torch.float32
    assert out.mean().item() == pytest.approx(150.0, abs=0.5)
