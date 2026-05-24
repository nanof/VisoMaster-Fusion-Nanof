"""Unit tests for Swap all by index (round-robin) spatial ordering helpers."""

import numpy as np

from app.helpers.sequential_rotate_stabilizer import SequentialRotateStabilizer
from app.helpers.sequential_rr_order import (
    rr_greedy_assign_from_memory,
    rr_spatial_order_key,
    rr_spatial_sort_indices,
)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


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


def test_rr_spatial_sort_indices_ignores_detector_list_order():
    left = np.array([0.0, 0.0, 50.0, 100.0])
    right = np.array([200.0, 0.0, 250.0, 100.0])
    boxes = [right, left]
    assert rr_spatial_sort_indices(boxes) == [1, 0]


def test_rr_greedy_assign_keeps_input_when_bbox_jitters():
    """Prev-frame slot + small jitter must keep the same input index."""
    stable = np.array([100.0, 80.0, 200.0, 220.0], dtype=np.float64)
    jitter = np.array([102.0, 82.0, 198.0, 218.0], dtype=np.float64)
    mem = [(stable, 2, 10_000_000_000)]
    assign, matched = rr_greedy_assign_from_memory(
        [jitter], mem, n_in=3, centroid_max=80.0, iou_fn=_iou_xyxy
    )
    assert assign == [2]
    assert matched == [True]


def test_rr_greedy_assign_stable_under_reordered_detections():
    """Two faces: detector order swaps but geometry is unchanged → same inputs."""
    face_a = np.array([50.0, 50.0, 150.0, 150.0], dtype=np.float64)
    face_b = np.array([300.0, 50.0, 400.0, 150.0], dtype=np.float64)
    mem = [(face_a, 0, 10_000_000_000), (face_b, 1, 10_000_000_000)]
    order1 = rr_greedy_assign_from_memory(
        [face_a, face_b], mem, n_in=2, centroid_max=120.0, iou_fn=_iou_xyxy
    )[0]
    order2 = rr_greedy_assign_from_memory(
        [face_b, face_a], mem, n_in=2, centroid_max=120.0, iou_fn=_iou_xyxy
    )[0]
    assert order1 == [0, 1]
    assert order2 == [1, 0]


def test_stabilizer_no_track_keeps_input_across_jitter():
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    stabilizer = SequentialRotateStabilizer()
    checked = [object(), object(), object()]
    f0 = [{"bbox": np.array([80.0, 60.0, 160.0, 140.0]), "track_id": -1}]
    stabilizer.apply(f0, checked, 0, (640, 480), iou, memory_without_tracking=True)
    inp0 = int(f0[0]["_rr_input_idx"])

    f1 = [{"bbox": np.array([82.0, 62.0, 158.0, 138.0]), "track_id": -1}]
    stabilizer.apply(f1, checked, 1, (640, 480), iou, memory_without_tracking=True)
    assert int(f1[0]["_rr_input_idx"]) == inp0
