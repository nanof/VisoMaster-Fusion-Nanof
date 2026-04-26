"""Smoke tests for the proportional multi-GPU dispatch path.

These exercise the ``VideoProcessor`` queue + scheduler plumbing without
spinning up real GPUs: we drive ``_resolve_assigned_gpu_index`` with an
emulated ``ModelsProcessor`` and check:

* **Round-Robin** preserves 50/50 distribution and loses zero frames.
* **Weighted Manual (3:1)** skews the dispatch toward the fast GPU while
  still delivering every frame (no drops, no duplicates).

The goal is to simulate the "2 GPUs (or emulated)" smoke run requested in
the plan without requiring an actual CUDA environment in CI.
"""
from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from app.processors.gpu_scheduler import WeightedScheduler
from app.processors.video_processor import VideoProcessor


def _make_dummy_vp(mode: str, weights: dict[int, int]):
    targets = sorted(weights.keys())
    mp = SimpleNamespace(
        gpu_index=0,
        device="cuda",
        emulate_multi_gpu=False,
        ui_multi_gpu_routing_enabled=True,
        _physical_cuda_device_count=lambda: len(targets),
        get_ui_routing_targets_sorted=lambda: list(targets),
        get_configured_gpu_count=lambda: len(targets),
        clamp_gpu_index=lambda idx: idx,
        load_balancing_mode=mode,
        gpu_weights=dict(weights),
        resolve_effective_weights=lambda: dict(weights),
    )
    subqueues: dict[int, queue.Queue] = {g: queue.Queue() for g in targets}
    dummy = SimpleNamespace(
        main_window=SimpleNamespace(models_processor=mp),
        scheduler=WeightedScheduler(targets=targets, weights=weights),
        frame_queues_by_gpu=subqueues,
        frame_queue=subqueues[targets[0]],
    )
    return dummy, subqueues


def _dispatch_frames(dummy, subqueues, n: int) -> list[int]:
    assignments = []
    for i in range(n):
        g = VideoProcessor._resolve_assigned_gpu_index(dummy, frame_number=i)
        assignments.append(int(g))
        subqueues[int(g)].put(f"frame-{i}")
    return assignments


def test_round_robin_smoke_zero_loss_and_balanced():
    dummy, subqueues = _make_dummy_vp(mode="round_robin", weights={0: 1, 1: 1})
    N = 200
    assignments = _dispatch_frames(dummy, subqueues, N)

    assert len(assignments) == N
    # Zero-loss: every dispatched frame is still enqueued somewhere.
    enqueued = sum(q.qsize() for q in subqueues.values())
    assert enqueued == N

    # Round-robin: exactly balanced when N is even.
    assert assignments.count(0) == N // 2
    assert assignments.count(1) == N // 2


def test_weighted_3_1_smoke_skews_to_fast_gpu_without_loss():
    dummy, subqueues = _make_dummy_vp(mode="weighted_manual", weights={0: 3, 1: 1})
    N = 400
    assignments = _dispatch_frames(dummy, subqueues, N)

    enqueued = sum(q.qsize() for q in subqueues.values())
    assert enqueued == N

    # 3:1 DRR -> exactly 3/4 go to gpu0.
    assert assignments.count(0) == N * 3 // 4
    assert assignments.count(1) == N * 1 // 4


@pytest.mark.parametrize("mode", ["hybrid", "weighted_auto"])
def test_proportional_modes_route_by_weights(mode: str):
    dummy, subqueues = _make_dummy_vp(mode=mode, weights={0: 2, 1: 1})
    N = 90
    assignments = _dispatch_frames(dummy, subqueues, N)

    enqueued = sum(q.qsize() for q in subqueues.values())
    assert enqueued == N
    assert assignments.count(0) == 60  # 2/3
    assert assignments.count(1) == 30  # 1/3
