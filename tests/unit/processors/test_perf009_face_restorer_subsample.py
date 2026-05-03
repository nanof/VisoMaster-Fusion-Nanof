"""PERF-009: Face restorer subsample interval + motion gate helpers."""

import torch

from app.processors.workers.frame_worker import FrameWorker


def test_perf009_interval_first_frame_and_period() -> None:
    assert FrameWorker._perf009_interval_run_neural(1, 3) is True
    assert FrameWorker._perf009_interval_run_neural(2, 3) is False
    assert FrameWorker._perf009_interval_run_neural(3, 3) is False
    assert FrameWorker._perf009_interval_run_neural(4, 3) is True
    assert FrameWorker._perf009_interval_run_neural(5, 3) is False


def test_perf009_interval_clamps() -> None:
    # interval 1 clamps to iv=2 (same as N=2 cadence)
    assert FrameWorker._perf009_interval_run_neural(1, 1) is True
    assert FrameWorker._perf009_interval_run_neural(2, 1) is False
    assert FrameWorker._perf009_interval_run_neural(3, 1) is True


def test_perf009_motion_zero_threshold_never_triggers() -> None:
    a = torch.zeros(3, 8, 8, dtype=torch.uint8)
    b = torch.ones(3, 8, 8, dtype=torch.uint8) * 255
    assert FrameWorker._perf009_motion_exceeds(b, a, 0.0) is False


def test_perf009_motion_detects_change() -> None:
    a = torch.zeros(3, 8, 8, dtype=torch.uint8)
    b = torch.ones(3, 8, 8, dtype=torch.uint8) * 255
    assert FrameWorker._perf009_motion_exceeds(b, a, 0.01) is True


def test_perf009_motion_shape_mismatch_safe() -> None:
    a = torch.zeros(3, 8, 8, dtype=torch.uint8)
    b = torch.zeros(3, 4, 4, dtype=torch.uint8)
    assert FrameWorker._perf009_motion_exceeds(b, a, 0.01) is False
