"""Tests for Poisson / seamless edge blend helper (requires torch via faceutil)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from app.processors.utils.faceutil import poisson_ring_edge_blend_numpy


def test_poisson_ring_amount_zero_returns_standard():
    h, w = 32, 32
    orig = np.zeros((h, w, 3), dtype=np.uint8)
    standard = np.full((h, w, 3), 128, dtype=np.uint8)
    opaque = standard.copy()
    mask = np.ones((h, w), dtype=np.float32)
    out = poisson_ring_edge_blend_numpy(orig, standard, opaque, mask, 0.0)
    assert out is standard


def test_poisson_ring_runs_small_roi():
    h, w = 64, 64
    orig = np.zeros((h, w, 3), dtype=np.uint8)
    orig[:] = (40, 50, 60)
    standard = orig.copy()
    standard[16:48, 16:48] = (200, 180, 160)
    opaque = standard.copy()
    mask = np.zeros((h, w), dtype=np.float32)
    mask[16:48, 16:48] = 1.0
    import cv2

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0)
    out = poisson_ring_edge_blend_numpy(
        orig, standard, opaque, mask, 0.5, mode=cv2.MIXED_CLONE
    )
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
