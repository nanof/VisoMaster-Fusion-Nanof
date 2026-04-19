"""Unit tests for Swap all by index (round-robin) spatial ordering helpers."""

import numpy as np

from app.helpers.sequential_rr_order import rr_spatial_order_key


def test_rr_spatial_order_key_tie_break_track_id():
    bb = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    k0 = rr_spatial_order_key(bb, 0, 7)
    k1 = rr_spatial_order_key(bb, 1, 3)
    # Same bbox: primary keys equal; track id breaks tie (deterministic order).
    assert k0[:2] == k1[:2]
    assert k0[2] == 7 and k1[2] == 3


def test_rr_spatial_order_key_tie_break_list_index_without_track():
    bb = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    k0 = rr_spatial_order_key(bb, 0, -1)
    k1 = rr_spatial_order_key(bb, 1, -1)
    assert k0[:2] == k1[:2]
    assert k0[2] == 0 and k1[2] == 1


def test_rr_spatial_order_key_left_to_right():
    left = np.array([0.0, 0.0, 50.0, 100.0])
    right = np.array([200.0, 0.0, 250.0, 100.0])
    assert rr_spatial_order_key(left, 0, -1)[0] < rr_spatial_order_key(right, 1, -1)[0]
