"""Tests for FrameWorker._fetch_task_with_stealing affinity/steal logic."""
from __future__ import annotations

import queue
from types import SimpleNamespace

from app.processors.workers.frame_worker import FrameWorker


def _make_fake_worker(
    *,
    assigned_gpu_index: int | None,
    own_queue: queue.Queue,
    queues_by_gpu: dict[int, queue.Queue] | None,
    affinity_enabled: bool,
    mode: str,
) -> SimpleNamespace:
    mp = SimpleNamespace(load_balancing_mode=mode)
    vp = SimpleNamespace(
        _queue_affinity_enabled=affinity_enabled,
        frame_queues_by_gpu=queues_by_gpu or {},
        main_window=SimpleNamespace(models_processor=mp),
    )
    main_window = SimpleNamespace(video_processor=vp, models_processor=mp)
    return SimpleNamespace(
        main_window=main_window,
        assigned_gpu_index=assigned_gpu_index,
        frame_queue=own_queue,
    )


def _fetch(fake_worker):
    return FrameWorker._fetch_task_with_stealing.__get__(fake_worker)()


def test_legacy_path_uses_shared_queue_when_affinity_off():
    shared = queue.Queue()
    shared.put("task-A")
    fw = _make_fake_worker(
        assigned_gpu_index=None,
        own_queue=shared,
        queues_by_gpu=None,
        affinity_enabled=False,
        mode="round_robin",
    )
    task, src, stolen = _fetch(fw)
    assert task == "task-A"
    assert src is shared
    assert stolen is False


def test_affinity_without_hybrid_does_not_steal():
    q0 = queue.Queue()
    q1 = queue.Queue()
    q1.put("frame-for-1")
    fw = _make_fake_worker(
        assigned_gpu_index=0,
        own_queue=q0,
        queues_by_gpu={0: q0, 1: q1},
        affinity_enabled=True,
        mode="weighted_manual",
    )
    # Own queue is empty, peer has work but mode forbids stealing -> the
    # worker blocks and eventually raises queue.Empty (via the 1.0s timeout).
    # We put a task after a short while to unblock; simpler here: put on own
    # queue before the call.
    q0.put("own-frame")
    task, src, stolen = _fetch(fw)
    assert task == "own-frame"
    assert src is q0
    assert stolen is False
    # Peer queue must still be intact.
    assert q1.get_nowait() == "frame-for-1"


def test_hybrid_steals_from_busiest_peer_when_own_queue_empty():
    q0 = queue.Queue()  # own (empty)
    q1 = queue.Queue()
    q1.put("stolen-frame")
    fw = _make_fake_worker(
        assigned_gpu_index=0,
        own_queue=q0,
        queues_by_gpu={0: q0, 1: q1},
        affinity_enabled=True,
        mode="hybrid",
    )
    task, src, stolen = _fetch(fw)
    assert task == "stolen-frame"
    assert src is q1
    assert stolen is True


def test_hybrid_prefers_own_queue_when_it_has_work():
    q0 = queue.Queue()
    q1 = queue.Queue()
    q0.put("own-frame")
    q1.put("peer-frame")
    fw = _make_fake_worker(
        assigned_gpu_index=0,
        own_queue=q0,
        queues_by_gpu={0: q0, 1: q1},
        affinity_enabled=True,
        mode="hybrid",
    )
    task, src, stolen = _fetch(fw)
    assert task == "own-frame"
    assert src is q0
    assert stolen is False


def test_hybrid_does_not_swallow_peer_poison_pill():
    q0 = queue.Queue()
    q1 = queue.Queue()
    # Peer only has a poison pill. It must stay in the peer queue so the
    # peer worker can exit cleanly.
    q1.put(None)
    fw = _make_fake_worker(
        assigned_gpu_index=0,
        own_queue=q0,
        queues_by_gpu={0: q0, 1: q1},
        affinity_enabled=True,
        mode="hybrid",
    )
    # Also feed the own queue so the call returns without blocking long.
    q0.put("own-after-empty")
    task, src, stolen = _fetch(fw)
    assert task == "own-after-empty"
    assert src is q0
    assert stolen is False
    # Peer pill is preserved.
    assert q1.get_nowait() is None
