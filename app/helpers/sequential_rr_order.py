"""Spatial ordering for Swap all by index (round-robin) temporal alignment."""

from __future__ import annotations

import numpy as np


def rr_spatial_order_key(
    bbox: np.ndarray,
    list_index: int,
    track_id: int,
) -> tuple[float, float, int]:
    """Sort key: left→right, top→bottom, then stable tie-break.

    Without a tie-break, two faces with nearly equal center-x can alternate order
    frame-to-frame when the detector jitters, which reshuffles input assignments.
    ByteTrack id breaks ties when valid; otherwise the stable face list index does.
    """
    bb = np.asarray(bbox, dtype=np.float64)
    cx = float((bb[0] + bb[2]) * 0.5)
    cy = float((bb[1] + bb[3]) * 0.5)
    tid = int(track_id)
    tb = tid if tid >= 0 else int(list_index)
    return (cx, cy, tb)
