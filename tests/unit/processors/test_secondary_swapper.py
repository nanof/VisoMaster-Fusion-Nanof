"""Secondary Swapper helpers and blend math (upstream #337, adapted)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.processors.workers.frame_worker import FrameWorker


COMPAT = FrameWorker._SECONDARY_SWAPPER_COMPATIBLE_MODELS


def test_compatible_models_are_inswapper_arcface_family():
    assert COMPAT == frozenset(
        {
            "Inswapper128",
            "AlphaFace",
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
        }
    )
    assert "HyperSwap-v1" not in COMPAT
    assert "GhostFace-v1" not in COMPAT
    assert "DeepFaceLive (DFM)" not in COMPAT


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


def test_secondary_blend_lerp_weights_primary_and_secondary():
    primary = torch.zeros(3, 8, 8)
    secondary = torch.full((3, 8, 8), 100.0)
    blended = torch.lerp(primary.float(), secondary.float(), 0.5)
    assert blended.mean().item() == pytest.approx(50.0)
    full_sec = torch.lerp(primary.float(), secondary.float(), 1.0)
    assert full_sec.mean().item() == pytest.approx(100.0)
    full_pri = torch.lerp(primary.float(), secondary.float(), 0.0)
    assert full_pri.mean().item() == pytest.approx(0.0)
