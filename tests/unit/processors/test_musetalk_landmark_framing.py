"""The crop must be framed the way MuseTalk frames it.

Upstream builds the window from the 68 face landmarks, not from the detector
box, so the reference construction is transcribed here and compared against
ours. Framing the model differently from its training is what made the generated
mouth come back small and generic whatever the detector box was nudged to.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.processors.dmdnet_landmarks import landmarks106_to_68_xy
from app.processors.pytorch_extras.musetalk.framing import (
    IBUG68_NOSE_BRIDGE,
    as_ibug68,
    landmark_crop_bbox,
)

FRAME = (720, 1280, 3)


def _upstream(land: np.ndarray, extra_margin: int = 10, frame_h: int = 720):
    """Transcribed from ``get_landmark_and_bbox`` plus inference.py's v15 margin."""
    land = land.astype(np.int32)
    half_face_coord = land[29]
    half_face_dist = np.max(land[:, 1]) - half_face_coord[1]
    upper_bond = max(0, half_face_coord[1] - half_face_dist)
    f_landmark = (
        np.min(land[:, 0]),
        int(upper_bond),
        np.max(land[:, 0]),
        np.max(land[:, 1]),
    )
    x1, y1, x2, y2 = f_landmark
    if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
        return None
    y2 = min(y2 + extra_margin, frame_h)
    return int(x1), int(y1), int(x2), int(y2)


def _face_68(cx: float = 640, cy: float = 360, scale: float = 1.0) -> np.ndarray:
    """A plausible 68-point face: jaw arc, bridge running down, mouth, eyes."""
    pts = np.zeros((68, 2), dtype=np.float32)
    # Jaw contour 0..16, widest at the sides, lowest in the middle (the chin).
    for i in range(17):
        t = (i - 8) / 8.0
        pts[i] = (cx + t * 110 * scale, cy + (60 - 34 * t * t) * scale)
    # Brows and eyes sit above the bridge.
    for i in range(17, 27):
        pts[i] = (cx - 60 + (i - 17) * 13, cy - 70 * scale)
    # Nose bridge 27..30 running down to the tip.
    for i in range(27, 31):
        pts[i] = (cx, cy + (-60 + (i - 27) * 22) * scale)
    for i in range(31, 36):
        pts[i] = (cx - 20 + (i - 31) * 10, cy + 12 * scale)
    for i in range(36, 48):
        pts[i] = (cx - 55 + (i - 36) * 9, cy - 55 * scale)
    for i in range(48, 68):
        pts[i] = (cx - 40 + (i - 48) * 4, cy + 30 * scale)
    return pts


def test_matches_upstream_for_a_68_point_face():
    land = _face_68()
    assert landmark_crop_bbox(land, None, FRAME) == _upstream(land)


@pytest.mark.parametrize("scale", [0.6, 1.0, 1.7])
@pytest.mark.parametrize("cy", [220, 360, 500])
def test_matches_upstream_across_sizes_and_positions(scale, cy):
    land = _face_68(cy=cy, scale=scale)
    assert landmark_crop_bbox(land, None, FRAME) == _upstream(land)


def test_the_window_is_centred_on_the_nose_bridge():
    """Upstream's whole point: the bridge lands at the vertical middle."""
    land = _face_68()
    x1, y1, x2, y2 = landmark_crop_bbox(land, None, FRAME, extra_margin=0)
    centre = (y1 + y2) / 2.0
    assert centre == pytest.approx(float(land[IBUG68_NOSE_BRIDGE, 1]), abs=1.0)


def test_the_width_is_the_landmark_extent_not_a_padded_box():
    land = _face_68()
    x1, _, x2, _ = landmark_crop_bbox(land, None, FRAME)
    assert x1 == pytest.approx(float(land[:, 0].min()), abs=1.0)
    assert x2 == pytest.approx(float(land[:, 0].max()), abs=1.0)


def test_the_bottom_is_the_chin_plus_the_margin():
    land = _face_68()
    _, _, _, y2 = landmark_crop_bbox(land, None, FRAME, extra_margin=10)
    assert y2 == pytest.approx(float(land[:, 1].max()) + 10, abs=1.0)


def _face_106(cx: float = 640, cy: float = 360) -> np.ndarray:
    """106 distinguishable points: jaw arc 0..32, the rest spread over the face.

    The values only have to be unique and plausibly placed; what is under test is
    that the framing re-indexes them the same way the shared map does.
    """
    pts = np.zeros((106, 2), dtype=np.float32)
    for i in range(33):
        t = (i - 16) / 16.0
        pts[i] = (cx + t * 110, cy + 60 - 34 * t * t)
    for i in range(33, 106):
        pts[i] = (cx - 60 + (i % 13) * 10, cy - 70 + ((i - 33) % 11) * 9)
    return pts


def test_106_landmarks_are_reindexed_to_the_68_scheme():
    pts = _face_106()
    mapped = as_ibug68(pts)
    assert mapped.shape == (68, 2)
    np.testing.assert_allclose(mapped, landmarks106_to_68_xy(pts))


def test_106_landmarks_frame_the_same_window_as_their_68_mapping():
    """The cheap detector must land on upstream's window, not an approximation."""
    pts = _face_106()
    assert landmark_crop_bbox(pts, None, FRAME) == _upstream(landmarks106_to_68_xy(pts))


def test_106_landmarks_use_the_exact_bridge_not_the_five_point_guess():
    """A 5-point set pointing elsewhere must not move the window."""
    pts = _face_106()
    misleading_kps_5 = np.array(
        [
            [500.0, 100.0],
            [560.0, 100.0],
            [530.0, 300.0],
            [510.0, 340.0],
            [550.0, 340.0],
        ],
        dtype=np.float32,
    )
    assert landmark_crop_bbox(pts, misleading_kps_5, FRAME) == landmark_crop_bbox(
        pts, None, FRAME
    )


def test_other_schemes_are_left_untouched_by_the_reindexing():
    land = _face_68()
    assert as_ibug68(land) is land
    dense = np.repeat(land, 3, axis=0)
    assert as_ibug68(dense) is dense


def test_a_dense_scheme_without_a_bridge_index_uses_the_five_point_set():
    """203-point landmarks are the project default and have no documented bridge."""
    land = _face_68()
    dense = np.repeat(land, 3, axis=0)  # 204 points, no usable index 29
    kps_5 = np.array(
        [
            [600.0, 305.0],
            [680.0, 305.0],
            [640.0, 372.0],
            [615.0, 390.0],
            [665.0, 390.0],
        ],
        dtype=np.float32,
    )
    box = landmark_crop_bbox(dense, kps_5, FRAME)
    assert box is not None
    x1, y1, x2, y2 = box
    # Still framed on the landmark extent and the chin.
    assert x1 == pytest.approx(float(dense[:, 0].min()), abs=1.0)
    assert y2 == pytest.approx(float(dense[:, 1].max()) + 10, abs=1.0)
    # Centre falls between the eye line and the nose tip, as documented.
    assert 305.0 < (y1 + y2) / 2.0 < 372.0


def test_five_points_alone_are_refused():
    """Their extent stops at the mouth corners, nowhere near jaw or chin."""
    kps_5 = np.array(
        [
            [600.0, 305.0],
            [680.0, 305.0],
            [640.0, 372.0],
            [615.0, 390.0],
            [665.0, 390.0],
        ],
        dtype=np.float32,
    )
    assert landmark_crop_bbox(kps_5, kps_5, FRAME) is None


def test_missing_landmarks_fall_back_to_the_detector_box():
    assert landmark_crop_bbox(None, None, FRAME) is None


def test_garbage_landmarks_are_refused_rather_than_framed():
    land = _face_68()
    land[5] = (np.nan, np.nan)
    assert landmark_crop_bbox(land, None, FRAME) is None


def test_a_face_above_the_top_edge_is_clamped_like_upstream():
    """Upstream clamps upper_bond at 0; the window must stay inside the frame."""
    land = _face_68(cy=90)
    box = landmark_crop_bbox(land, None, FRAME)
    assert box == _upstream(land)
    if box is not None:
        assert box[1] >= 0


def test_landmarks_from_another_coordinate_space_are_refused():
    """Small frames get upsampled with the cached boxes rescaled but not the
    landmarks, and a crop framed from the stale ones would land off the face."""
    land = _face_68(cx=640, cy=360)
    stale = land * 0.4  # what the landmarks look like before the upsample
    assert (
        landmark_crop_bbox(stale, None, FRAME, reference_bbox=(530, 290, 750, 430))
        is None
    )
    assert (
        landmark_crop_bbox(land, None, FRAME, reference_bbox=(530, 290, 750, 430))
        is not None
    )


def test_a_missing_or_degenerate_reference_box_does_not_block_framing():
    land = _face_68()
    assert landmark_crop_bbox(land, None, FRAME, reference_bbox=None) is not None
    assert (
        landmark_crop_bbox(land, None, FRAME, reference_bbox=(0, 0, 0, 0)) is not None
    )


def test_the_bottom_never_leaves_the_frame():
    land = _face_68(cy=690)
    box = landmark_crop_bbox(land, None, FRAME, extra_margin=40)
    assert box is not None
    assert box[3] <= FRAME[0]
