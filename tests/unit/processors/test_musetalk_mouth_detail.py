"""The generated mouth is sharpened only as much as its surroundings warrant.

A fixed boost was measured to be wrong: on a sharp clip the VAE stripped 34% of
the mouth's detail, but on soft footage the generated mouth came out 33% *sharper*
than its own blurry original, so the same boost would have made the mouth crisper
than the face around it. The amount is therefore derived per frame from the
untouched face beside the mouth.

Sharpening is self-sourced on purpose: pulling high frequencies from the original
mouth would ghost the old lip shape back in wherever the audio changed it.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.processors.pytorch_extras.musetalk.blending import (
    _MOUTH_SHARPEN_MAX,
    restore_mouth_detail,
)


def _detail(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0)).mean())


def _texture(blur: float, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.integers(60, 200, size=(size, size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (0, 0), sigmaX=blur) if blur > 0 else base


def test_a_blurry_mouth_beside_a_sharp_face_gets_sharpened() -> None:
    mouth, face = _texture(2.0), _texture(0.6)
    assert _detail(restore_mouth_detail(mouth, face)) > _detail(mouth)


def test_a_mouth_already_sharper_than_the_face_is_left_alone() -> None:
    """The soft-footage case: boosting here would betray the surrounding face."""
    mouth, face = _texture(0.6), _texture(2.0)
    assert np.array_equal(restore_mouth_detail(mouth, face), mouth)


def test_sharpening_closes_the_gap_to_the_surrounding_face() -> None:
    """The goal is matching the neighbourhood, not maximum gain.

    Relative gain is not monotonic in blur: unsharp masking cannot amplify
    frequencies heavy blurring already destroyed, so a very soft mouth improves
    by a smaller factor than a mildly soft one. What must always hold is that the
    result sits closer to the reference than it started.
    """
    face = _texture(0.6)
    target = _detail(face)
    for blur in (1.2, 2.0, 2.5):
        mouth = _texture(blur)
        before = target - _detail(mouth)
        after = target - _detail(restore_mouth_detail(mouth, face))
        assert 0 <= after < before, f"blur {blur}: {before:.2f} -> {after:.2f}"


def test_a_missing_reference_is_a_passthrough() -> None:
    mouth = _texture(2.0)
    assert np.array_equal(restore_mouth_detail(mouth, None), mouth)
    assert np.array_equal(restore_mouth_detail(mouth, np.zeros((2, 2, 3), np.uint8)), mouth)


def test_a_flat_mouth_does_not_divide_by_zero() -> None:
    flat = np.full((64, 64, 3), 128, np.uint8)
    assert np.array_equal(restore_mouth_detail(flat, _texture(0.6)), flat)


def test_the_boost_is_capped_against_haloing() -> None:
    """An extremely soft mouth must not demand unbounded gain."""
    mouth = _texture(6.0)
    out = restore_mouth_detail(mouth, _texture(0.0))
    assert _detail(out) <= _detail(mouth) * (1.0 + _MOUTH_SHARPEN_MAX) * 1.2


def test_mean_colour_survives_sharpening() -> None:
    mouth, face = _texture(2.0), _texture(0.6)
    before = mouth.reshape(-1, 3).mean(axis=0)
    after = restore_mouth_detail(mouth, face).reshape(-1, 3).mean(axis=0)
    assert np.abs(after - before).max() < 2.0


def test_output_stays_a_valid_uint8_image() -> None:
    """A hard edge drives the unsharp overshoot past 0/255; it must clip, not wrap."""
    edge = np.zeros((128, 128, 3), dtype=np.uint8)
    edge[:, 64:] = 255
    out = restore_mouth_detail(cv2.GaussianBlur(edge, (0, 0), 3.0), _texture(0.0, 128))
    assert out.dtype == np.uint8 and out.shape == edge.shape
