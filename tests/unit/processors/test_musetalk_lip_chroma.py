"""Lip colour recovery must fix chroma without disturbing the lip-sync.

MuseTalk normalises the lips toward a magenta prior whatever the face beneath.
The correction pulls only the chrominance of the parsed lips back toward the
original, so the checks here are about what it must *not* touch — luminance and
teeth — as much as what it does.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.processors.pytorch_extras.musetalk.blending import transfer_lip_chroma
from app.processors.pytorch_extras.musetalk.parsing import (
    lip_region_mask,
    parsed_face_masks,
)


def _bgr(b: int, g: int, r: int, shape=(64, 64)) -> np.ndarray:
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:] = (b, g, r)
    return img


def _luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.int16)


def _cr(img: np.ndarray) -> float:
    """Mean redness (Cr), the channel MuseTalk pushes toward magenta."""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 1].mean())


def test_the_generated_lips_move_toward_the_original_colour():
    generated = _bgr(150, 120, 210)  # magenta-ish, high red
    original = _bgr(150, 130, 170)  # the swapped face's own, less red
    mask = np.ones((64, 64), dtype=np.float32)
    # A partial correction so the direction, not just the endpoint, is checked.
    out = transfer_lip_chroma(generated, original, mask, strength=0.6)
    assert _cr(original) < _cr(out) < _cr(generated)
    # At full strength the lips reach the original colour exactly.
    full = transfer_lip_chroma(generated, original, mask, strength=1.0)
    assert _cr(full) == pytest.approx(_cr(original), abs=1.5)


def test_luminance_is_left_untouched():
    """The lip-sync lives in luma; the correction must not move it at all."""
    generated = _bgr(150, 120, 210)
    original = _bgr(90, 80, 110)  # much darker, to tempt a luma change
    mask = np.ones((64, 64), dtype=np.float32)
    out = transfer_lip_chroma(generated, original, mask, strength=1.0)
    assert np.array_equal(_luma(out), _luma(generated))


def test_teeth_inside_the_lip_footprint_are_not_tinted():
    """An opened mouth reveals near-neutral teeth where the original had lip.

    The chroma gate must keep them from being dragged toward lip colour.
    """
    generated = _bgr(150, 120, 210)
    generated[24:40, 24:40] = (200, 200, 202)  # bright, near-neutral teeth
    original = _bgr(150, 130, 170)
    mask = np.ones((64, 64), dtype=np.float32)
    out = transfer_lip_chroma(generated, original, mask, strength=1.0)
    teeth_before = generated[24:40, 24:40].astype(np.int16)
    teeth_after = out[24:40, 24:40].astype(np.int16)
    assert np.abs(teeth_after - teeth_before).mean() < 3.0


def test_strength_zero_is_a_no_op():
    generated = _bgr(150, 120, 210)
    original = _bgr(150, 130, 170)
    mask = np.ones((64, 64), dtype=np.float32)
    assert np.array_equal(
        transfer_lip_chroma(generated, original, mask, strength=0.0), generated
    )


def test_an_empty_mask_is_a_no_op():
    generated = _bgr(150, 120, 210)
    original = _bgr(150, 130, 170)
    mask = np.zeros((64, 64), dtype=np.float32)
    assert np.array_equal(
        transfer_lip_chroma(generated, original, mask, strength=1.0), generated
    )


def test_a_mismatched_mask_is_ignored():
    generated = _bgr(150, 120, 210)
    original = _bgr(150, 130, 170)
    mask = np.ones((32, 32), dtype=np.float32)
    assert np.array_equal(
        transfer_lip_chroma(generated, original, mask, strength=1.0), generated
    )


def test_strength_scales_the_amount_of_correction():
    generated = _bgr(150, 120, 210)
    original = _bgr(150, 130, 170)
    mask = np.ones((64, 64), dtype=np.float32)
    gentle = _cr(transfer_lip_chroma(generated, original, mask, strength=0.3))
    strong = _cr(transfer_lip_chroma(generated, original, mask, strength=1.0))
    # More strength pulls the redness further down toward the original.
    assert strong < gentle < _cr(generated)


# --- lip mask sourced from the same parse as the repaint mask ----------------


def _labels() -> np.ndarray:
    labels = np.full((512, 512), 17, dtype=np.uint8)
    cv2.ellipse(labels, (256, 270), (150, 200), 0, 0, 360, 1, -1)  # skin
    cv2.ellipse(labels, (256, 280), (28, 55), 0, 0, 360, 10, -1)  # nose
    cv2.ellipse(labels, (256, 380), (70, 16), 0, 0, 360, 11, -1)  # mouth cavity
    cv2.ellipse(labels, (256, 364), (72, 10), 0, 0, 360, 12, -1)  # upper lip
    cv2.ellipse(labels, (256, 396), (72, 12), 0, 0, 360, 13, -1)  # lower lip
    return labels.astype(np.int64)


def test_lip_mask_is_the_lips_and_not_the_cavity():
    labels = _labels()
    mask = lip_region_mask(labels)
    assert mask[labels == 12].min() == 255
    assert mask[labels == 13].min() == 255
    # The oral cavity is excluded so teeth cannot be tinted.
    assert mask[labels == 11].max() == 0
    assert mask[labels == 1].max() == 0


def test_both_masks_come_from_one_parse():
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    labels = _labels()
    calls = {"n": 0}

    def parser(rgb):
        calls["n"] += 1
        return labels

    bbox = (500, 200, 700, 460)
    jaw, lip = parsed_face_masks(frame, bbox, parser)
    assert calls["n"] == 1, "the head must be segmented once, not per mask"
    assert jaw is not None and lip is not None
    assert jaw.shape == lip.shape == (260, 200)
    # The lip mask reaches full strength somewhere and covers less than the jaw.
    assert float(lip.max()) > 0.5
    assert float(lip.sum()) < float(jaw.sum())


def test_a_failed_parse_yields_no_masks():
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    jaw, lip = parsed_face_masks(frame, (500, 200, 700, 460), lambda rgb: None)
    assert jaw is None and lip is None
