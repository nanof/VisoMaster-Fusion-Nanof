"""Spatial tile grid for FlashVSR (no GPU)."""

from __future__ import annotations

import numpy as np

from app.processors.pytorch_extras.pipeline_upscaler.tiles import (
    axis_starts,
    choose_tile_hw,
    overlap_weights,
    tile_boxes,
)


def test_single_tile_when_frame_fits():
    assert choose_tile_hw(480, 640, max_pixels=768 * 1280) == (480, 640)


def test_large_4k_uses_sub_tiles():
    h, w = 2880, 5120
    th, tw = choose_tile_hw(h, w, max_pixels=768 * 1280)
    assert th * tw <= 768 * 1280
    boxes = tile_boxes(h, w, th, tw, overlap=128)
    covered = np.zeros((h, w), dtype=np.uint8)
    for y0, y1, x0, x1 in boxes:
        covered[y0:y1, x0:x1] = 1
    assert int(covered.min()) == 1
    assert len(boxes) > 1


def test_720p_4x_hr_tiles_stay_under_budget():
    """Bicubic-to-HR then same-size FlashVSR tiles (LQ HxW == pipe H/W)."""
    h, w = 2880, 5120
    th, tw = choose_tile_hw(h, w, max_pixels=640 * 768, multiple=128)
    assert th * tw <= 640 * 768
    boxes = tile_boxes(h, w, th, tw, overlap=64)
    assert len(boxes) > 1


def test_axis_starts_covers_end():
    starts = axis_starts(2880, 768, 128)
    assert starts[0] == 0
    assert starts[-1] + 768 == 2880


def test_overlap_weights_full_on_image_edges():
    w = overlap_weights(768, 1280, y0=0, x0=0, full_h=768, full_w=1280, overlap=128)
    assert w.shape == (768, 1280)
    assert float(w.min()) > 0.0
    assert abs(float(w[0, 0]) - 1.0) < 1e-5
