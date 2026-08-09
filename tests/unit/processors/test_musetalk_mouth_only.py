"""Repainting only the mouth is what removes the doubled mouth and chin.

Two failures produce that artifact and both are geometric, not perceptual: an
alpha below 1.0 leaves the original mouth visible *through* the generated one, and
a mask that does not cover both mouth poses leaves the original mouth visible
*beside* it. The union built here is the smallest region that avoids both, which
is also the region that touches the least identity.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.processors.pytorch_extras.musetalk.engine import MuseTalkEngine
from app.processors.pytorch_extras.musetalk.parsing import (
    crop_mouth_mask,
    mouth_only_blend_mask,
    mouth_region_mask,
    parsed_masks,
)

FRAME = np.full((720, 1280, 3), 120, dtype=np.uint8)
BBOX = (500, 200, 700, 460)
REGION = (BBOX[3] - BBOX[1], BBOX[2] - BBOX[0])


def _labels() -> np.ndarray:
    labels = np.full((512, 512), 17, dtype=np.uint8)
    cv2.ellipse(labels, (256, 270), (150, 200), 0, 0, 360, 1, -1)  # skin
    cv2.ellipse(labels, (256, 280), (28, 55), 0, 0, 360, 10, -1)  # nose
    cv2.ellipse(labels, (256, 380), (70, 16), 0, 0, 360, 11, -1)  # cavity
    cv2.ellipse(labels, (256, 364), (72, 10), 0, 0, 360, 12, -1)  # upper lip
    cv2.ellipse(labels, (256, 396), (72, 12), 0, 0, 360, 13, -1)  # lower lip
    return labels.astype(np.int64)


def _mask(shape, box) -> np.ndarray:
    """A rectangular footprint standing in for one mouth pose."""
    out = np.zeros(shape, dtype=np.float32)
    y1, y2, x1, x2 = box
    out[y1:y2, x1:x2] = 1.0
    return out


# --- the label reduction -----------------------------------------------------


def test_the_mouth_region_is_the_cavity_and_both_lips():
    labels = _labels()
    mask = mouth_region_mask(labels)
    for cls in (11, 12, 13):
        assert mask[labels == cls].min() == 255, f"class {cls} must be repainted"
    # Skin and nose stay with the swapped face: that is the identity being kept.
    assert mask[labels == 1].max() == 0
    assert mask[labels == 10].max() == 0


def test_all_three_masks_come_from_a_single_parse():
    calls = {"n": 0}

    def parser(_rgb):
        calls["n"] += 1
        return _labels()

    masks = parsed_masks(FRAME, BBOX, parser)
    assert calls["n"] == 1, "one BiSeNet call per frame, not one per mask"
    assert masks.jaw is not None and masks.lip is not None
    assert masks.mouth is not None
    assert masks.mouth.shape == masks.jaw.shape == REGION
    # The mouth region is a small part of what the jaw mask would repaint.
    assert float(masks.mouth.sum()) < float(masks.jaw.sum()) / 2


def test_a_failed_parse_yields_no_mouth_mask():
    assert parsed_masks(FRAME, BBOX, lambda _rgb: None).mouth is None


# --- the generated half of the union ----------------------------------------


def test_the_generated_crop_is_parsed_as_rgb_and_lands_in_region_space():
    seen = {}

    def parser(rgb):
        seen["shape"] = rgb.shape
        seen["first_px"] = tuple(int(v) for v in rgb[0, 0])
        return _labels()

    crop = np.zeros((256, 256, 3), dtype=np.uint8)
    crop[:, :] = (10, 20, 30)  # BGR
    mask = crop_mouth_mask(crop, parser, REGION)
    assert seen["shape"] == (256, 256, 3)
    assert seen["first_px"] == (30, 20, 10), "the parser expects RGB"
    assert mask is not None and mask.shape == REGION
    assert float(mask.max()) == 1.0


def test_an_unusable_crop_or_parse_is_refused_quietly():
    assert (
        crop_mouth_mask(np.zeros((0, 0, 3), np.uint8), lambda r: _labels(), REGION)
        is None
    )
    assert (
        crop_mouth_mask(np.zeros((256, 256, 3), np.uint8), lambda r: None, REGION)
        is None
    )


# --- the union itself, which is where the artifact is fixed ------------------


def test_the_union_covers_both_mouth_poses():
    """The regression: a closed original and a wide-open generated mouth.

    Covering only one of them is what left a second mouth beside the new one.
    """
    shape = (200, 200)
    closed = _mask(shape, (100, 110, 70, 130))
    open_wide = _mask(shape, (90, 140, 60, 140))
    mask = mouth_only_blend_mask(closed, open_wide, padding_px=0, feather_px=0)
    assert mask is not None
    assert float(mask[closed > 0.5].min()) == 1.0
    assert float(mask[open_wide > 0.5].min()) == 1.0


def test_the_repaint_is_fully_opaque_inside_the_region():
    """No alpha below 1.0 anywhere the mouth actually is, at any padding.

    Partial opacity is what made the original mouth show through, and feathering
    must only soften the border, never the middle.
    """
    shape = (200, 200)
    mouth = _mask(shape, (95, 125, 65, 135))
    mask = mouth_only_blend_mask(mouth, None, padding_px=8, feather_px=11)
    assert mask is not None
    assert float(mask[mouth > 0.5].min()) == 1.0


def test_the_border_is_feathered_rather_than_cut():
    shape = (200, 200)
    mouth = _mask(shape, (95, 125, 65, 135))
    hard = mouth_only_blend_mask(mouth, None, padding_px=4, feather_px=0)
    soft = mouth_only_blend_mask(mouth, None, padding_px=4, feather_px=9)
    partial = np.logical_and(soft > 0.01, soft < 0.99)
    assert partial.sum() > 0, "a hard edge in skin reads as a scar"
    assert np.logical_and(hard > 0.01, hard < 0.99).sum() < partial.sum()


def test_padding_grows_the_region_without_moving_the_mouth():
    shape = (200, 200)
    mouth = _mask(shape, (95, 125, 65, 135))
    tight = mouth_only_blend_mask(mouth, None, padding_px=0, feather_px=0)
    wide = mouth_only_blend_mask(mouth, None, padding_px=10, feather_px=0)
    assert float(wide.sum()) > float(tight.sum())
    assert float(wide[mouth > 0.5].min()) == 1.0


def test_nothing_to_repaint_is_reported_as_no_mask():
    shape = (64, 64)
    assert mouth_only_blend_mask(None, None) is None
    assert mouth_only_blend_mask(np.zeros(shape, np.float32), None) is None
    assert (
        mouth_only_blend_mask(np.ones(shape, np.float32), np.ones((32, 32), np.float32))
        is None
    )


# --- the engine's choice between the two masks ------------------------------


def _engine() -> MuseTalkEngine:
    engine = MuseTalkEngine.__new__(MuseTalkEngine)
    engine._warn_once = set()
    return engine


def _recon() -> np.ndarray:
    return np.full((256, 256, 3), 130, dtype=np.uint8)


def _original_mouth() -> np.ndarray:
    return _mask(REGION, (150, 180, 80, 130))


def test_the_engine_builds_the_union_when_the_option_is_on():
    mask = _engine()._mouth_only_mask(
        _original_mouth(),
        _recon(),
        lambda _rgb: _labels(),
        REGION,
        {"mouth_only": True, "mouth_padding": 6},
    )
    assert mask is not None and mask.shape == REGION
    # Both halves of the union survive: the original footprint and the parsed one.
    assert float(mask[_original_mouth() > 0.5].min()) == 1.0
    assert float(mask.max()) == 1.0


def test_the_jaw_mask_is_kept_when_the_option_is_off_or_unusable():
    engine = _engine()
    opts = {"mouth_only": True, "mouth_padding": 6}
    parser = lambda _rgb: _labels()  # noqa: E731 - one-liner stub, read inline
    assert engine._mouth_only_mask(None, _recon(), parser, REGION, {}) is None
    assert engine._mouth_only_mask(None, _recon(), None, REGION, opts) is None
    # Zero strength must still mean untouched, not "opaque local repaint".
    assert (
        engine._mouth_only_mask(
            _original_mouth(), _recon(), parser, REGION, {**opts, "strength": 0.0}
        )
        is None
    )


def test_the_hybrid_alpha_scales_the_mouth_opacity():
    """The hybrid after-pass re-sharpens at partial opacity; the pose is aligned.

    Full opacity would overwrite the swap's identity-correct mouth, so the alpha
    must actually reach the mask rather than being ignored the way it is for the
    full-opacity passes.
    """
    engine = _engine()
    opts = {"mouth_only": True, "mouth_padding": 6}
    full = engine._mouth_only_mask(
        _original_mouth(), _recon(), lambda _rgb: _labels(), REGION, opts
    )
    half = engine._mouth_only_mask(
        _original_mouth(),
        _recon(),
        lambda _rgb: _labels(),
        REGION,
        {**opts, "mouth_alpha": 0.5},
    )
    assert full is not None and half is not None
    assert float(half.max()) == pytest.approx(float(full.max()) * 0.5, abs=1e-4)


def test_a_failing_parse_of_the_generated_mouth_warns_once_and_degrades():
    def boom(_rgb):
        raise RuntimeError("no session")

    engine = _engine()
    opts = {"mouth_only": True}
    assert engine._mouth_only_mask(None, _recon(), boom, REGION, opts) is None
    assert engine._mouth_only_mask(None, _recon(), boom, REGION, opts) is None
    assert engine._warn_once == {"mouth_only"}
