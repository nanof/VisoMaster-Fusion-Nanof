"""Seeking must select absolute audio chunks and cancel stale batch requests."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.processors.pytorch_extras.musetalk.engine import (  # noqa: E402
    MuseTalkEngine,
    _CropRequest,
)


def _bare_engine() -> MuseTalkEngine:
    engine = MuseTalkEngine.__new__(MuseTalkEngine)
    engine._probe = None
    engine._whisper_chunks = torch.zeros((4, 50, 384))
    engine._batch_queue = queue.Queue()
    engine._bbox_shift = 0
    return engine


@pytest.mark.parametrize("frame_index", [-1, 4, 9])
def test_out_of_range_seek_never_wraps_audio_to_frame_zero(frame_index: int) -> None:
    engine = _bare_engine()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    result = engine._apply_frame_bgr_unlocked(
        frame,
        frame_index,
        [np.array([0, 0, 32, 32])],
        extra_margin=0,
        face_index=0,
    )

    assert result is frame
    assert engine._batch_queue.empty(), "an unrelated wrapped chunk was enqueued"


def test_seek_cancellation_releases_waiter_without_batch_timeout() -> None:
    engine = _bare_engine()
    request = _CropRequest(np.zeros((256, 256, 3), dtype=np.uint8), 2)
    cancel = threading.Event()
    result: list[str] = []

    waiter = threading.Thread(
        target=lambda: result.append(
            engine._wait_for_request(request, cancel, timeout_s=5.0)
        )
    )
    started = time.perf_counter()
    waiter.start()
    cancel.set()
    waiter.join(timeout=0.5)

    assert not waiter.is_alive()
    assert result == ["cancelled"]
    assert request.cancelled.is_set()
    assert time.perf_counter() - started < 0.5


def test_recording_wait_ignores_soft_timeout_until_batch_finishes() -> None:
    """Export must not punch lip-sync holes when the batcher is slow."""
    engine = _bare_engine()
    request = _CropRequest(np.zeros((256, 256, 3), dtype=np.uint8), 1)
    result: list[str] = []

    def _finish_later() -> None:
        time.sleep(0.08)
        request.recon = np.zeros((256, 256, 3), dtype=np.uint8)
        request.done.set()

    threading.Thread(target=_finish_later, daemon=True).start()
    started = time.perf_counter()
    # A 20 ms soft timeout would have fired; recording passes timeout_s=None.
    result.append(engine._wait_for_request(request, None, timeout_s=None))
    assert result == ["done"]
    assert time.perf_counter() - started >= 0.05
    assert not request.cancelled.is_set()


def test_preview_soft_timeout_still_abandons_a_stuck_batch() -> None:
    engine = _bare_engine()
    request = _CropRequest(np.zeros((256, 256, 3), dtype=np.uint8), 0)
    started = time.perf_counter()
    outcome = engine._wait_for_request(request, None, timeout_s=0.05)
    assert outcome == "timeout"
    assert request.cancelled.is_set()
    assert time.perf_counter() - started < 0.5


def test_batcher_discards_a_request_cancelled_by_scrub() -> None:
    engine = _bare_engine()
    engine._batch_stop = threading.Event()
    engine._max_batch = 2
    engine._warn_once = set()
    inferred: list[list[_CropRequest]] = []
    engine._infer_batch = lambda batch: inferred.append(batch)

    stale = _CropRequest(np.zeros((1, 1, 3), dtype=np.uint8), 0)
    stale.cancelled.set()
    current = _CropRequest(np.zeros((1, 1, 3), dtype=np.uint8), 3)
    engine._batch_queue.put(stale)
    engine._batch_queue.put(current)

    batcher = threading.Thread(target=engine._batch_loop)
    batcher.start()
    assert current.done.wait(timeout=1.0)
    engine._batch_stop.set()
    engine._batch_queue.put(None)
    batcher.join(timeout=1.0)

    assert inferred == [[current]]
    assert stale.done.is_set()
