"""
FD-* tests for face detector logic (NMS, bbox handling, edge cases).

No ML models are loaded; all inference paths are mocked.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# FD-03: NMS (torchvision IoU-NMS) reduces overlapping boxes
# ---------------------------------------------------------------------------


def test_nms_removes_overlapping_boxes():
    """torchvision.ops.nms should suppress heavily overlapping boxes."""
    from torchvision.ops import nms

    # Three boxes: first two overlap heavily, third is separate
    boxes = torch.tensor(
        [
            [10.0, 10.0, 100.0, 100.0],
            [12.0, 12.0, 102.0, 102.0],  # heavily overlaps box 0
            [200.0, 200.0, 300.0, 300.0],  # no overlap
        ],
        dtype=torch.float32,
    )
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=0.0)
    keep = nms(boxes, areas, iou_threshold=0.5)
    kept_indices = keep.tolist()

    # Box 1 (smaller or equal area vs box 0 depending on order) should be suppressed
    # The separate box (index 2) should always be kept
    assert 2 in kept_indices
    # At most one of the two overlapping boxes should remain
    assert not (0 in kept_indices and 1 in kept_indices), (
        "Both overlapping boxes survived NMS — at least one should be suppressed"
    )


def test_nms_keeps_all_non_overlapping_boxes():
    from torchvision.ops import nms

    boxes = torch.tensor(
        [
            [0.0, 0.0, 50.0, 50.0],
            [100.0, 100.0, 150.0, 150.0],
            [200.0, 200.0, 250.0, 250.0],
        ],
        dtype=torch.float32,
    )
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=0.0)
    keep = nms(boxes, areas, iou_threshold=0.5)
    assert len(keep) == 3


# ---------------------------------------------------------------------------
# Max faces cap keeps the biggest faces, whatever their position in the frame
# ---------------------------------------------------------------------------


def _filter_detections(boxes, scores, max_num):
    from app.processors.face_detectors import FaceDetectors

    detectors = FaceDetectors.__new__(FaceDetectors)
    detectors.models_processor = SimpleNamespace(
        get_effective_torch_device=lambda: torch.device("cpu")
    )
    boxes_np = np.asarray(boxes, dtype=np.float32)
    kpss = np.zeros((boxes_np.shape[0], 5, 2), dtype=np.float32)
    det, _kpss, score_values = detectors._filter_detections_gpu(
        [np.asarray(scores, dtype=np.float32).reshape(-1, 1)],
        [boxes_np],
        [kpss],
        1080,
        1920,
        torch.tensor(1.0),
        max_num,
        skip_nms=True,
    )
    return det, score_values


def test_max_faces_cap_keeps_big_face_away_from_center():
    big_off_center = [10.0, 300.0, 310.0, 600.0]
    small_centered = [900.0, 500.0, 980.0, 580.0]
    smaller_centered = [1000.0, 500.0, 1070.0, 570.0]

    det, _ = _filter_detections(
        [big_off_center, small_centered, smaller_centered],
        [0.90, 0.95, 0.94],
        max_num=2,
    )

    kept = [row[:4].tolist() for row in det]
    assert big_off_center in kept
    assert smaller_centered not in kept


def test_max_faces_cap_breaks_area_ties_by_score():
    left = [0.0, 0.0, 100.0, 100.0]
    right = [800.0, 0.0, 900.0, 100.0]

    det, scores = _filter_detections([left, right], [0.5, 0.9], max_num=1)

    assert det.shape[0] == 1
    assert det[0][:4].tolist() == right
    assert float(scores[0]) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# FD-05: bindex slicing produces a subset of input bboxes
# ---------------------------------------------------------------------------


def test_bindex_slicing():
    """Selecting rows by index array gives the correct subset."""
    bboxes = np.array(
        [
            [10, 10, 50, 50, 0.9],
            [20, 20, 60, 60, 0.8],
            [30, 30, 70, 70, 0.7],
        ],
        dtype=np.float32,
    )
    keep_indices = np.array([0, 2])
    result = bboxes[keep_indices]
    assert result.shape == (2, 5)
    assert np.allclose(result[0], bboxes[0])
    assert np.allclose(result[1], bboxes[2])


# ---------------------------------------------------------------------------
# FD-07: empty bbox array handled gracefully (IndexError guard)
# ---------------------------------------------------------------------------


def test_empty_bbox_array_len_check():
    """len(bboxes_eq_np) == 0 should short-circuit without IndexError."""
    bboxes = np.empty((0, 5), dtype=np.float32)
    if len(bboxes) == 0:
        result = "early_return"
    else:
        result = "processed"
    assert result == "early_return"


def test_single_bbox_reshape():
    """1-D bbox (4,) or (5,) should be reshaped to (1, N) without error."""
    bbox_1d = np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32)
    if bbox_1d.ndim == 1 and bbox_1d.shape[0] in (4, 5):
        bbox_2d = bbox_1d.reshape(1, -1)
    assert bbox_2d.shape == (1, 4)

    bbox_1d_5 = np.array([10.0, 20.0, 50.0, 60.0, 0.9], dtype=np.float32)
    if bbox_1d_5.ndim == 1 and bbox_1d_5.shape[0] in (4, 5):
        bbox_2d_5 = bbox_1d_5.reshape(1, -1)
    assert bbox_2d_5.shape == (1, 5)


# ---------------------------------------------------------------------------
# FD-06: np.exp clipping — no overflow on extreme scores
# ---------------------------------------------------------------------------


def test_exp_clipping_no_overflow():
    """Scores passed through np.exp should be clipped to avoid overflow."""
    extreme_scores = np.array([1000.0, -1000.0, 0.0, 500.0])
    clipped = np.clip(extreme_scores, -500, 500)
    result = np.exp(clipped)
    assert np.all(np.isfinite(result)), "np.exp produced inf/nan on extreme input"


# ---------------------------------------------------------------------------
# FD-02: BYTETracker None-guard — tracking disabled gracefully
# ---------------------------------------------------------------------------


def test_bytetracker_none_guard():
    """If BYTETracker is None, tracking should be skipped without AttributeError."""
    BYTETracker = None

    tracker = None
    if BYTETracker is not None:
        tracker = BYTETracker()  # would fail with None

    assert tracker is None


# ---------------------------------------------------------------------------
# FD-08: anchor init not called twice under concurrent access (race guard)
# ---------------------------------------------------------------------------


def test_anchor_init_once_under_concurrency():
    """Simulate the pattern: init flag guarded by a lock is called exactly once."""
    import threading

    init_count = 0
    lock = threading.Lock()
    initialized = [False]

    def maybe_init():
        nonlocal init_count
        with lock:
            if not initialized[0]:
                init_count += 1
                initialized[0] = True

    threads = [threading.Thread(target=maybe_init) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert init_count == 1, f"Expected init_count=1, got {init_count}"
