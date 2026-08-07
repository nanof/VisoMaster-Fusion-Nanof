"""GenderAge preprocessing must not scale with frame resolution.

The first implementation warped the full-resolution frame per face, so a 4K frame
cost ~9 ms per face just to produce a 96x96 crop. The sub-window warp keeps the
geometry identical while touching only pixels near the face.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from app.processors.face_attributes import (
    GENDERAGE_INPUT_SIZE,
    FaceAttributes,
)
from app.processors.utils import faceutil


def _attrs() -> FaceAttributes:
    return FaceAttributes(SimpleNamespace())


def _full_frame_reference(img: torch.Tensor, bbox) -> torch.Tensor:
    """Original full-frame warp, kept here as the geometric ground truth."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    scale = float(GENDERAGE_INPUT_SIZE) / (max(w, h) * 1.5)
    crop, _ = faceutil.transform(img, center, GENDERAGE_INPUT_SIZE, scale, 0.0)
    return crop.to(dtype=torch.float32)


def test_row_shape_is_model_input():
    attrs = _attrs()
    img = torch.randint(0, 255, (3, 720, 1280), dtype=torch.uint8)
    row = attrs._genderage_input_row(img, np.array([400, 300, 520, 460], np.float32))
    assert row is not None
    assert tuple(row.shape) == (3, GENDERAGE_INPUT_SIZE, GENDERAGE_INPUT_SIZE)
    assert row.dtype == torch.float32
    # mean=0 / std=1 preprocessing keeps the 0-255 pixel scale.
    assert float(row.max()) > 1.5


def test_subwindow_crop_matches_full_frame_warp_geometry():
    attrs = _attrs()
    torch.manual_seed(0)
    img = torch.randint(0, 255, (3, 1080, 1920), dtype=torch.uint8)
    for bbox in (
        (700.0, 400.0, 880.0, 620.0),  # interior
        (-30.0, -20.0, 120.0, 160.0),  # clipped top-left
        (1860.0, 1010.0, 1990.0, 1130.0),  # clipped bottom-right
    ):
        row = attrs._genderage_input_row(img, np.array(bbox, np.float32))
        reference = _full_frame_reference(img, bbox)
        assert row is not None
        # Both paths round through uint8, so a single quantisation level may differ.
        assert float((row - reference).abs().max()) <= 1.0


def test_crop_cost_is_independent_of_frame_resolution():
    import time

    attrs = _attrs()
    bbox = np.array([500, 400, 680, 620], np.float32)

    def measure(h, w):
        img = torch.randint(0, 255, (3, h, w), dtype=torch.uint8)
        attrs._genderage_input_row(img, bbox)
        start = time.perf_counter()
        for _ in range(10):
            attrs._genderage_input_row(img, bbox)
        return (time.perf_counter() - start) / 10

    small = measure(720, 1280)
    large = measure(2160, 3840)
    # 4K has 9x the pixels; cost must stay in the same ballpark as 720p.
    assert large < small * 4.0, f"720p={small * 1000:.2f}ms 4K={large * 1000:.2f}ms"


def test_out_of_frame_bbox_returns_none():
    attrs = _attrs()
    img = torch.randint(0, 255, (3, 240, 320), dtype=torch.uint8)
    assert (
        attrs._genderage_input_row(img, np.array([-900, -900, -800, -800], np.float32))
        is None
    )
