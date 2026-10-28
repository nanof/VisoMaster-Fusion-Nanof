"""Spatial tile grid + overlap weights for last-stage upscalers (no ML imports)."""

from __future__ import annotations

import numpy as np


def axis_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start indices so tiles of ``tile`` cover ``[0, length)`` with ``overlap``."""
    length = max(int(length), 1)
    tile = max(min(int(tile), length), 1)
    if length <= tile:
        return [0]
    overlap = max(0, min(int(overlap), tile - 1))
    step = max(tile - overlap, 1)
    starts: list[int] = []
    start = 0
    while start + tile < length:
        starts.append(start)
        start += step
    last = length - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def tile_boxes(
    height: int, width: int, tile_h: int, tile_w: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """Return ``(y0, y1, x0, x1)`` boxes covering the image."""
    boxes: list[tuple[int, int, int, int]] = []
    th = max(min(int(tile_h), int(height)), 1)
    tw = max(min(int(tile_w), int(width)), 1)
    for y0 in axis_starts(height, th, overlap):
        for x0 in axis_starts(width, tw, overlap):
            boxes.append((y0, y0 + th, x0, x0 + tw))
    return boxes


def choose_tile_hw(
    height: int, width: int, max_pixels: int, *, multiple: int = 128
) -> tuple[int, int]:
    """Pick a tile that stays under ``max_pixels`` (full frame if it already fits)."""
    h = max(int(height), 1)
    w = max(int(width), 1)
    multiple = max(int(multiple), 1)
    max_pixels = max(int(max_pixels), multiple * multiple)
    if h * w <= max_pixels:
        return h, w
    if w >= h:
        tile_h, tile_w = 768, 1280
    else:
        tile_h, tile_w = 1280, 768
    tile_h = min(tile_h, h)
    tile_w = min(tile_w, w)
    while tile_h * tile_w > max_pixels and (tile_h > multiple or tile_w > multiple):
        if tile_w >= tile_h and tile_w > multiple:
            tile_w = max(multiple, tile_w - multiple)
        elif tile_h > multiple:
            tile_h = max(multiple, tile_h - multiple)
        else:
            break
        tile_h = min(tile_h, h)
        tile_w = min(tile_w, w)
    return max(tile_h, 1), max(tile_w, 1)


def overlap_weights(
    tile_h: int,
    tile_w: int,
    y0: int,
    x0: int,
    full_h: int,
    full_w: int,
    overlap: int,
) -> np.ndarray:
    """Linear fade on interior overlaps so neighboring tiles blend."""
    weight = np.ones((int(tile_h), int(tile_w)), dtype=np.float32)
    ov = max(int(overlap), 0)
    if ov <= 0:
        return weight
    th, tw = weight.shape
    if y0 > 0:
        n = min(ov, th)
        ramp = (np.arange(n, dtype=np.float32) + 1.0) / float(n + 1)
        weight[:n, :] *= ramp[:, None]
    if y0 + th < full_h:
        n = min(ov, th)
        ramp = (np.arange(n, dtype=np.float32) + 1.0) / float(n + 1)
        weight[-n:, :] *= ramp[::-1, None]
    if x0 > 0:
        n = min(ov, tw)
        ramp = (np.arange(n, dtype=np.float32) + 1.0) / float(n + 1)
        weight[:, :n] *= ramp[None, :]
    if x0 + tw < full_w:
        n = min(ov, tw)
        ramp = (np.arange(n, dtype=np.float32) + 1.0) / float(n + 1)
        weight[:, -n:] *= ramp[None, ::-1]
    return np.maximum(weight, 1e-6)
