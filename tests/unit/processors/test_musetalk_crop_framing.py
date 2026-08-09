"""The lip-sync crop framing is calibrated, not arbitrary.

``expand_bbox``'s defaults were measured against ground truth: driving a clip
with its own audio means the correct mouth is the one already on screen, so the
framing minimising the error against it matches MuseTalk's training convention.
Padding the detector box scored 18.9/255 at 0.05 and 25.1/255 at 0.12 versus
15.8/255 unpadded, and lifting the window by 6% of the box height dropped it to
11.0/255. Both defaults look like harmless cosmetics, so they are pinned here:
restoring the padding or dropping the shift roughly doubles the mouth error.
"""

from __future__ import annotations

import inspect

from app.processors.pytorch_extras.musetalk.blending import expand_bbox

FRAME_SHAPE = (1080, 1920, 3)


def _defaults() -> dict:
    return {
        name: p.default
        for name, p in inspect.signature(expand_bbox).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_detector_box_is_not_padded_sideways() -> None:
    assert _defaults()["pad_ratio"] == 0.0


def test_crop_window_is_lifted_so_the_mouth_sits_lower_in_it() -> None:
    assert _defaults()["vertical_shift"] < 0.0


def test_no_padding_keeps_the_horizontal_span_of_the_detector_box() -> None:
    x1, _, x2, _ = expand_bbox([800.0, 300.0, 950.0, 520.0], FRAME_SHAPE)
    assert (x1, x2) == (800, 950)


def test_shift_moves_the_window_up_by_a_fraction_of_the_box_height() -> None:
    box = [800.0, 300.0, 950.0, 500.0]
    _, y1, _, _ = expand_bbox(box, FRAME_SHAPE, extra_margin=0)
    # 200 px tall box, -6% shift => 12 px higher than the detector's own top.
    assert y1 == 288


def test_extra_margin_still_extends_the_bottom_edge() -> None:
    box = [800.0, 300.0, 950.0, 500.0]
    _, _, _, plain = expand_bbox(box, FRAME_SHAPE, extra_margin=0)
    _, _, _, padded = expand_bbox(box, FRAME_SHAPE, extra_margin=30)
    assert padded - plain == 30


def test_window_stays_inside_the_frame_for_a_face_at_the_top_edge() -> None:
    x1, y1, x2, y2 = expand_bbox([10.0, 4.0, 160.0, 220.0], FRAME_SHAPE)
    assert 0 <= x1 < x2 <= FRAME_SHAPE[1]
    assert 0 <= y1 < y2 <= FRAME_SHAPE[0]
