"""Lip-sync exposes the identity/motion trade-off as a control.

MuseTalk derives lip shape from audio, so it normalises away much of a face's own
lip character: five very different mouths fed through it came back near-identical.
That trade-off has no universally right setting, so strength and mask geometry are
user-facing, and the plumbing from slider to mask is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.processors.pytorch_extras.musetalk.blending import soft_lower_face_mask
from app.processors.pytorch_extras.musetalk.engine import MuseTalkEngine
from app.processors.workers.frame_worker import FrameWorker

H = W = 200


def test_full_strength_repaints_the_mouth_completely() -> None:
    assert soft_lower_face_mask(H, W).max() == pytest.approx(1.0, abs=0.02)


def test_lower_strength_mixes_the_original_mouth_back() -> None:
    assert soft_lower_face_mask(H, W, strength=0.5).max() == pytest.approx(
        0.5, abs=0.02
    )


def test_zero_strength_leaves_the_frame_untouched() -> None:
    assert soft_lower_face_mask(H, W, strength=0.0).max() == pytest.approx(
        0.0, abs=1e-6
    )


def test_a_narrower_width_repaints_less_of_the_face() -> None:
    wide = soft_lower_face_mask(H, W, radius_x=0.50).sum()
    narrow = soft_lower_face_mask(H, W, radius_x=0.25).sum()
    assert narrow < wide


def test_a_shorter_height_spares_the_chin() -> None:
    tall = soft_lower_face_mask(H, W, radius_y=0.40)
    short = soft_lower_face_mask(H, W, radius_y=0.18)
    assert short[-30:].sum() < tall[-30:].sum()


def test_the_centre_control_moves_the_repainted_band_vertically() -> None:
    def centroid(mask: np.ndarray) -> float:
        ys = np.arange(mask.shape[0], dtype=np.float32)
        return float((mask.sum(axis=1) * ys).sum() / max(mask.sum(), 1e-6))

    assert centroid(soft_lower_face_mask(H, W, centre_y=0.56)) < centroid(
        soft_lower_face_mask(H, W, centre_y=0.74)
    )


def test_sliders_are_read_as_percentages() -> None:
    opts = FrameWorker._musetalk_mask_options(
        {
            "MuseTalkBlendStrengthSlider": 60,
            "MuseTalkMouthWidthSlider": 35,
            "MuseTalkMouthHeightSlider": 25,
            "MuseTalkMouthCentreSlider": 70,
        }
    )
    assert MuseTalkEngine._ellipse_options(opts) == pytest.approx(
        {"strength": 0.60, "radius_x": 0.35, "radius_y": 0.25, "centre_y": 0.70}
    )


def test_missing_sliders_fall_back_to_the_tuned_defaults() -> None:
    """An old saved workspace has none of these keys and must still behave."""
    opts = FrameWorker._musetalk_mask_options({})
    assert MuseTalkEngine._ellipse_options(opts) == pytest.approx(
        {"strength": 1.0, "radius_x": 0.42, "radius_y": 0.30, "centre_y": 0.64}
    )


def test_parsing_sliders_are_read_in_their_own_units() -> None:
    """Cheek width is pixels on the 512 parse; the top limit is a fraction."""
    opts = FrameWorker._musetalk_mask_options(
        {"MuseTalkRepaintTopSlider": 55, "MuseTalkCheekWidthSlider": 70}
    )
    assert opts["upper_boundary_ratio"] == pytest.approx(0.55)
    assert opts["left_cheek_width"] == 70
    assert opts["right_cheek_width"] == 70


def test_parsing_defaults_are_upstreams() -> None:
    opts = FrameWorker._musetalk_mask_options({})
    assert opts["upper_boundary_ratio"] == pytest.approx(0.5)
    assert opts["left_cheek_width"] == 90
    assert opts["right_cheek_width"] == 90


def test_bbox_shift_is_read_as_signed_rows() -> None:
    """Upstream's bbox_shift is a signed pixel offset, not a percentage."""
    assert FrameWorker._musetalk_mask_options({"MuseTalkBboxShiftSlider": -8})[
        "bbox_shift"
    ] == -8
    assert FrameWorker._musetalk_mask_options({"MuseTalkBboxShiftSlider": 12})[
        "bbox_shift"
    ] == 12


def test_bbox_shift_defaults_to_zero() -> None:
    """Zero is MuseTalk's own default: the mask splits the crop in half."""
    assert FrameWorker._musetalk_mask_options({})["bbox_shift"] == 0


def test_lip_colour_strength_reads_as_a_fraction() -> None:
    opts = FrameWorker._musetalk_mask_options({"MuseTalkLipColorStrengthSlider": 40})
    assert opts["lip_color_strength"] == pytest.approx(0.40)


def test_lip_colour_defaults_on_at_seventy_percent() -> None:
    assert FrameWorker._musetalk_mask_options({})["lip_color_strength"] == pytest.approx(
        0.70
    )


def test_the_toggle_off_zeroes_the_lip_colour_strength() -> None:
    """Zero strength is the engine's off signal, so the toggle must produce it."""
    opts = FrameWorker._musetalk_mask_options(
        {"MuseTalkLipColorToggle": False, "MuseTalkLipColorStrengthSlider": 80}
    )
    assert opts["lip_color_strength"] == 0.0


def test_local_mouth_repaint_is_on_by_default() -> None:
    """The default has to be the path without the doubled mouth and chin."""
    opts = FrameWorker._musetalk_mask_options({})
    assert opts["mouth_only"] is True
    assert opts["mouth_padding"] == 6


def test_local_mouth_repaint_needs_face_parsing() -> None:
    """Both halves of the union come from the parser; without it there is no union."""
    opts = FrameWorker._musetalk_mask_options({"MuseTalkFaceParsingToggle": False})
    assert opts["mouth_only"] is False


def test_local_mouth_repaint_can_be_turned_off_to_compare() -> None:
    opts = FrameWorker._musetalk_mask_options({"MuseTalkMouthOnlyToggle": False})
    assert opts["mouth_only"] is False


def test_the_padding_slider_is_read_as_pixels() -> None:
    opts = FrameWorker._musetalk_mask_options({"MuseTalkMouthPaddingSlider": 14})
    assert opts["mouth_padding"] == 14


@pytest.mark.parametrize("junk", ["", None, "abc"])
def test_unusable_slider_values_do_not_break_the_frame(junk) -> None:
    opts = FrameWorker._musetalk_mask_options({"MuseTalkBlendStrengthSlider": junk})
    assert opts["strength"] == pytest.approx(1.0)


def test_mask_options_accept_what_the_mask_produces() -> None:
    """Guards the slider-to-mask contract against a rename on either side.

    The options dict feeds two different masks, so the split has to keep the
    geometric fallback from being handed keys it cannot take.
    """
    opts = FrameWorker._musetalk_mask_options({})
    mask = soft_lower_face_mask(H, W, **MuseTalkEngine._ellipse_options(opts))
    assert mask.shape == (H, W)
    assert 0.0 <= mask.min() and mask.max() <= 1.0


def test_bypass_skips_only_the_frame_hook() -> None:
    assert FrameWorker._musetalk_should_apply(
        {"MuseTalkEnableToggle": True, "MuseTalkBypassToggle": False}
    )
    assert not FrameWorker._musetalk_should_apply(
        {"MuseTalkEnableToggle": True, "MuseTalkBypassToggle": True}
    )


def test_bypass_defaults_off_for_old_workspaces() -> None:
    """Saved workspaces predating the comparison toggle keep lip-sync active."""
    assert FrameWorker._musetalk_should_apply({"MuseTalkEnableToggle": True})


def test_disabled_musetalk_stays_disabled_even_if_bypass_is_off() -> None:
    assert not FrameWorker._musetalk_should_apply(
        {"MuseTalkEnableToggle": False, "MuseTalkBypassToggle": False}
    )
