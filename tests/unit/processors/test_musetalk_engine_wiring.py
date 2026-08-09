"""The engine must choose the parsed mask when it can and degrade quietly.

These are the seams between the reference-faithful pieces and the pipeline: which
mask wins, which options reach which mask, and which face is picked. A break here
silently reverts lip-sync to the worse geometric path.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.processors.pytorch_extras.musetalk.engine import MuseTalkEngine

FRAME = np.full((720, 1280, 3), 120, dtype=np.uint8)
BBOX = (500, 200, 700, 460)


def _labels(_rgb):
    labels = np.full((512, 512), 17, dtype=np.int64)
    labels[120:430, 140:380] = 1  # skin
    labels[300:340, 220:300] = 11  # mouth
    return labels


def _engine() -> MuseTalkEngine:
    engine = MuseTalkEngine.__new__(MuseTalkEngine)
    engine._warn_once = set()
    return engine


def test_the_parsed_mask_is_used_when_a_parser_is_available():
    mask = _engine()._blend_mask(FRAME, BBOX, _labels, {"strength": 1.0})
    assert mask is not None
    assert mask.shape == (260, 200)
    assert float(mask.max()) > 0.5


def test_without_a_parser_the_caller_is_told_to_use_the_fallback():
    assert _engine()._blend_mask(FRAME, BBOX, None, {}) is None


def test_a_failing_parser_degrades_instead_of_dropping_lip_sync():
    def boom(_rgb):
        raise RuntimeError("no session")

    engine = _engine()
    assert engine._blend_mask(FRAME, BBOX, boom, {}) is None
    assert "parse" in engine._warn_once


def test_the_failure_is_only_reported_once():
    def boom(_rgb):
        raise RuntimeError("no session")

    engine = _engine()
    engine._blend_mask(FRAME, BBOX, boom, {})
    engine._blend_mask(FRAME, BBOX, boom, {})
    assert engine._warn_once == {"parse"}


def test_strength_reaches_the_parsed_mask():
    engine = _engine()
    full = engine._blend_mask(FRAME, BBOX, _labels, {"strength": 1.0})
    half = engine._blend_mask(FRAME, BBOX, _labels, {"strength": 0.5})
    assert np.allclose(half, full * 0.5, atol=1e-6)


def test_the_fallback_never_receives_the_parsing_keys():
    """soft_lower_face_mask would raise TypeError on them."""
    opts = {
        "strength": 0.8,
        "upper_boundary_ratio": 0.5,
        "left_cheek_width": 90,
        "radius_x": 0.4,
    }
    assert MuseTalkEngine._ellipse_options(opts) == {"strength": 0.8, "radius_x": 0.4}


@pytest.mark.parametrize(
    "items, index, expected",
    [
        ([10, 20, 30], 1, 20),
        ([10, 20, 30], 9, 30),
        ([10, 20, 30], -4, 10),
        (None, 0, None),
        ([], 0, None),
    ],
)
def test_face_selection_stays_in_range(items, index, expected):
    assert MuseTalkEngine._pick(items, index) == expected


def test_face_selection_accepts_numpy_landmark_stacks():
    """Detectors hand back arrays, where truthiness raises."""
    stack = np.zeros((2, 68, 2), dtype=np.float32)
    stack[1] = 5.0
    assert float(MuseTalkEngine._pick(stack, 1).mean()) == 5.0
    assert MuseTalkEngine._pick(np.zeros((0, 68, 2), dtype=np.float32), 0) is None
