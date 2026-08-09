"""The parsing mask must match MuseTalk's reference implementation.

The reference is transcribed inline from ``musetalk/utils/face_parsing/__init__.py``
and ``musetalk/utils/blending.py`` so a divergence shows up as a test failure
rather than as an artefact someone has to notice on screen. Our earlier ellipse
put a visible edge through the mouth, which is what these constants avoid.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.processors.pytorch_extras.musetalk.parsing import (
    _JAW_KERNEL,
    crop_box_for,
    jaw_region_mask,
    parsed_lower_face_mask,
)


def _upstream_kernel() -> np.ndarray:
    cone_height = 21
    tail_height = 12
    total_size = cone_height + tail_height
    kernel = np.zeros((total_size, total_size), dtype=np.uint8)
    center_x = total_size // 2
    for row in range(cone_height):
        if row < cone_height // 2:
            continue
        width = int(2 * (row - cone_height // 2) + 1)
        start = int(center_x - (width // 2))
        end = int(center_x + (width // 2) + 1)
        kernel[row, start:end] = 1
    base_width = int(kernel[cone_height - 1].sum()) if cone_height > 0 else 1
    for row in range(cone_height, total_size):
        start = max(0, int(center_x - (base_width // 2)))
        end = min(total_size, int(center_x + (base_width // 2) + 1))
        kernel[row, start:end] = 1
    return kernel


def _upstream_jaw(parsing: np.ndarray, cheek: int = 90) -> np.ndarray:
    parsing = parsing.copy()
    cheek_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 3))
    cheek_mask = np.zeros((512, 512), dtype=np.uint8)
    center = 512 // 2
    cv2.rectangle(cheek_mask, (0, 0), (center - cheek, 512), 255, -1)
    cv2.rectangle(cheek_mask, (center + cheek, 0), (512, 512), 255, -1)

    face_region = np.isin(parsing, [1]) * 255
    face_region = face_region.astype(np.uint8)
    original_dilated = cv2.dilate(face_region, _upstream_kernel(), iterations=1)
    eroded = cv2.erode(original_dilated, cheek_kernel, iterations=2)
    face_region = cv2.bitwise_and(eroded, cheek_mask)
    face_region = cv2.bitwise_or(
        face_region, cv2.bitwise_and(original_dilated, ~cheek_mask)
    )
    parsing[(face_region == 255) & (~np.isin(parsing, [10]))] = 255
    parsing[np.isin(parsing, [11, 12, 13])] = 255
    parsing[np.where(parsing != 255)] = 0
    return parsing.astype(np.uint8)


def _synthetic_labels() -> np.ndarray:
    """A crude face: skin oval, nose, mouth and lips, hair around it."""
    labels = np.full((512, 512), 17, dtype=np.uint8)  # hair
    cv2.ellipse(labels, (256, 270), (150, 200), 0, 0, 360, 1, -1)  # skin
    cv2.ellipse(labels, (256, 280), (28, 55), 0, 0, 360, 10, -1)  # nose
    cv2.ellipse(labels, (256, 380), (70, 16), 0, 0, 360, 11, -1)  # mouth
    cv2.ellipse(labels, (256, 364), (72, 10), 0, 0, 360, 12, -1)  # upper lip
    cv2.ellipse(labels, (256, 396), (72, 12), 0, 0, 360, 13, -1)  # lower lip
    # argmax hands back int64 in the real pipeline.
    return labels.astype(np.int64)


def test_the_dilation_kernel_is_upstreams():
    assert np.array_equal(_JAW_KERNEL, _upstream_kernel())


def test_the_kernel_reaches_downwards_only():
    """Mass below the anchor is what grows the mask past the jaw to the chin."""
    centre = _JAW_KERNEL.shape[0] // 2
    assert _JAW_KERNEL[:centre].sum() < _JAW_KERNEL[centre:].sum()


def test_jaw_region_matches_upstream_exactly():
    labels = _synthetic_labels()
    assert np.array_equal(jaw_region_mask(labels), _upstream_jaw(labels))


@pytest.mark.parametrize("cheek", [60, 90, 120])
def test_jaw_region_matches_upstream_for_other_cheek_widths(cheek):
    labels = _synthetic_labels()
    ours = jaw_region_mask(labels, left_cheek_width=cheek, right_cheek_width=cheek)
    assert np.array_equal(ours, _upstream_jaw(labels, cheek))


def test_the_nose_is_excluded_and_the_mouth_is_kept():
    labels = _synthetic_labels()
    mask = jaw_region_mask(labels)
    assert mask[labels == 11].min() == 255
    assert mask[labels == 12].min() == 255
    assert mask[labels == 13].min() == 255
    # The nose interior is dropped even though it sits inside the dilated skin.
    assert mask[labels == 10].mean() < 64


def test_crop_box_matches_upstream_get_crop_box():
    for box in [(10, 20, 110, 160), (0, 0, 51, 40), (300, 200, 460, 500)]:
        x, y, x1, y1 = box
        x_c, y_c = (x + x1) // 2, (y + y1) // 2
        s = int(max(x1 - x, y1 - y) // 2 * 1.5)
        assert crop_box_for(box) == (x_c - s, y_c - s, x_c + s, y_c + s)


def _frame_and_parser():
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    labels = _synthetic_labels()
    return frame, lambda rgb: labels


def test_mask_covers_the_bbox_and_stays_in_range():
    frame, parser = _frame_and_parser()
    bbox = (500, 200, 700, 460)
    mask = parsed_lower_face_mask(frame, bbox, parser)
    assert mask is not None
    assert mask.shape == (260, 200)
    assert mask.dtype == np.float32
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
    assert float(mask.max()) > 0.5, "nothing would be repainted"


def test_the_top_of_the_crop_is_never_repainted():
    """Eyes and forehead must survive: they come from the swap, not the model."""
    frame, parser = _frame_and_parser()
    bbox = (500, 200, 700, 460)
    mask = parsed_lower_face_mask(frame, bbox, parser)
    assert float(mask[0].max()) == 0.0


def test_the_mask_has_no_hard_edge():
    """The ellipse's clipped plateau is what cut through the mouth on screen."""
    frame, parser = _frame_and_parser()
    mask = parsed_lower_face_mask(frame, (500, 200, 700, 460), parser)
    gy, gx = np.gradient(mask.astype(np.float64))
    steepest = float(np.hypot(gy, gx).max())
    assert steepest < 0.2, f"mask jumps by {steepest:.2f} per pixel"


def test_strength_scales_the_whole_mask():
    frame, parser = _frame_and_parser()
    bbox = (500, 200, 700, 460)
    full = parsed_lower_face_mask(frame, bbox, parser)
    half = parsed_lower_face_mask(frame, bbox, parser, strength=0.5)
    assert np.allclose(half, full * 0.5, atol=1e-6)


def test_a_parser_that_fails_is_reported_rather_than_guessed():
    frame, _ = _frame_and_parser()
    assert parsed_lower_face_mask(frame, (500, 200, 700, 460), lambda rgb: None) is None


def test_a_face_at_the_frame_edge_still_produces_a_mask():
    """The expanded square runs off-frame; upstream zero-fills, so must we."""
    frame, parser = _frame_and_parser()
    mask = parsed_lower_face_mask(frame, (0, 0, 180, 240), parser)
    assert mask is not None
    assert mask.shape == (240, 180)
