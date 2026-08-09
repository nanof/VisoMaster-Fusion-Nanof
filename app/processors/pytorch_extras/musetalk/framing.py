"""MuseTalk's crop convention, ported from ``musetalk/utils/preprocessing.py``.

The reference implementation never crops with the detector box. It builds the
window from the 68 face landmarks:

    half_face_coord = face_land_mark[29]                       # mid-lower nose bridge
    half_face_dist  = max(face_land_mark[:,1]) - half_face_coord[1]
    upper_bond      = max(0, half_face_coord[1] - half_face_dist)
    f_landmark      = (min_x, upper_bond, max_x, max_y)

So the window spans jaw to jaw horizontally, ends at the chin, and is centred
vertically on the nose bridge. Feeding the model a detector box instead puts the
face at a different scale and offset than it saw in training, which is why the
generated mouth came back normalised and too small however the box was nudged.

``bbox_shift`` is upstream's one tuning knob and it acts *here*, on the bridge
point, not on the mask the model repaints::

    half_face_coord[1] = half_face_coord[1] + bbox_shift

Because the mask is always the exact lower half of the crop, moving the bridge
moves the crop's top edge and so changes where that half line lands on the face.
Shifting the mask inside a fixed crop is *not* equivalent: it detaches the hard
edge of the model's input mask from the blend's upper boundary and leaves a
visible horizontal seam across the nose.
"""

from __future__ import annotations

import numpy as np

# Index of the mid-lower nose bridge point in the iBUG 68-point scheme.
IBUG68_NOSE_BRIDGE = 29

# Where the bridge point sits between the eye line and the nose tip, used only
# when the landmarks are not the 68-point scheme. In iBUG-68 the bridge runs
# 27 (between the brows, level with the eyes) down to 30 (the tip), and 29 is the
# point before the tip, so it lands roughly two thirds of the way down.
BRIDGE_FRACTION = 0.66


def _nose_bridge_y(
    landmarks: np.ndarray | None,
    kps_5: np.ndarray | None,
    bridge_fraction: float = BRIDGE_FRACTION,
) -> float | None:
    """Height of the crop's vertical centre.

    Exact for 68-point landmarks, which is the scheme upstream uses. Other
    schemes have no documented bridge index in this project, so the point is
    interpolated between the eye line and the nose tip of the 5-point set, which
    every scheme can produce.
    """
    if landmarks is not None:
        pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] == 68:
            return float(pts[IBUG68_NOSE_BRIDGE, 1])
    if kps_5 is None:
        return None
    five = np.asarray(kps_5, dtype=np.float32).reshape(-1, 2)
    if five.shape[0] < 3:
        return None
    eye_y = float((five[0, 1] + five[1, 1]) * 0.5)
    tip_y = float(five[2, 1])
    return eye_y + (tip_y - eye_y) * float(bridge_fraction)


def _agrees_with(pts: np.ndarray, reference_bbox) -> bool:
    """Whether the landmarks describe the same face the detector box does.

    Landmarks and boxes do not always arrive in the same coordinate space: the
    pipeline rescales cached boxes when it upsamples a small frame but leaves the
    landmarks alone, and a rotated pass can disagree too. Framing the crop from
    landmarks belonging to another space would paint a mouth somewhere else
    entirely, so a mismatch falls back instead.
    """
    if reference_bbox is None:
        return True
    try:
        bx1, by1, bx2, by2 = (float(v) for v in list(reference_bbox)[:4])
    except (TypeError, ValueError):
        return True
    if bx2 <= bx1 or by2 <= by1:
        return True
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    # Generous: landmark sets cover different extents, and the mean of a dense
    # set sits well inside the box. This only has to catch gross disagreement.
    mx, my = (bx2 - bx1), (by2 - by1)
    return (bx1 - mx) <= cx <= (bx2 + mx) and (by1 - my) <= cy <= (by2 + my)


def landmark_crop_bbox(
    landmarks: np.ndarray | None,
    kps_5: np.ndarray | None,
    frame_shape: tuple[int, ...],
    *,
    extra_margin: int = 10,
    bridge_fraction: float = BRIDGE_FRACTION,
    reference_bbox=None,
    bbox_shift: int = 0,
) -> tuple[int, int, int, int] | None:
    """Upstream's landmark window, or None when the landmarks cannot support it.

    Returning None rather than a guess matters: the caller then keeps the
    detector-box approximation, which is worse but predictable.
    """
    if landmarks is None:
        return None
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    # 5 points cover eyes, nose and mouth corners, so their extent is nowhere
    # near the jaw or the chin and would frame the model far too tightly.
    if pts.shape[0] < 20 or not np.isfinite(pts).all():
        return None
    if not _agrees_with(pts, reference_bbox):
        return None

    bridge_y = _nose_bridge_y(pts, kps_5, bridge_fraction)
    if bridge_y is None:
        return None

    # Upstream truncates the landmarks to int32 before any of this arithmetic, so
    # matching it means truncating here too rather than rounding at the end.
    ipts = pts.astype(np.int32)
    # Upstream shifts the bridge point itself, which drags the crop's top edge
    # with it: positive moves the half line toward the mouth (more openness),
    # negative away from it, leaving more of the real face standing.
    bridge = int(bridge_y) + int(bbox_shift)
    h, w = int(frame_shape[0]), int(frame_shape[1])
    chin_y = int(ipts[:, 1].max())
    half = chin_y - bridge
    if half <= 0:
        return None

    x1 = int(ipts[:, 0].min())
    x2 = int(ipts[:, 0].max())
    y1 = max(0, bridge - half)
    y2 = chin_y + int(extra_margin)

    # Upstream discards the landmark window on these and reuses the detector box.
    if x2 - x1 <= 0 or y2 - y1 <= 0 or x1 < 0:
        return None

    x2 = min(x2, w)
    y2 = min(y2, h)
    if x2 - x1 <= 2 or y2 - y1 <= 2:
        return None
    return x1, y1, x2, y2
