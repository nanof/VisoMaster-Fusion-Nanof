"""Lip-sync must accept the bbox container the frame worker actually passes.

Regression: ``FrameWorker`` hands over ``precomputed_bboxes`` as a numpy array, and
the engine gated on ``if bboxes:``. NumPy raises "truth value of an array with more
than one element is ambiguous" for that, so every frame hit the broad except in
``apply_frame_bgr`` and was returned untouched — lip-sync silently did nothing in
the app while passing tests that handed it a plain Python list.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.processors.pytorch_extras.musetalk.engine import MuseTalkEngine

RECON_VALUE = 255


class _ImmediateQueue:
    """Stands in for the batcher: answers each request in place."""

    def __init__(self) -> None:
        self.seen = 0

    def put(self, request) -> None:
        self.seen += 1
        request.recon = np.full((256, 256, 3), RECON_VALUE, dtype=np.uint8)
        request.done.set()


class _Chunks:
    shape = (40,)

    def __len__(self) -> int:
        return self.shape[0]


@pytest.fixture
def engine():
    eng = MuseTalkEngine()
    eng._loaded = True
    eng._whisper_chunks = _Chunks()
    eng._batch_queue = _ImmediateQueue()
    return eng


@pytest.fixture
def frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


BOX = [638.0, 127.0, 776.0, 335.0]


def _changed(before: np.ndarray, after: np.ndarray) -> int:
    return int(np.abs(after.astype(np.int16) - before.astype(np.int16)).sum())


@pytest.mark.parametrize(
    "bboxes",
    [
        pytest.param(np.array([BOX], dtype=np.float32), id="numpy_2d_array"),
        pytest.param([np.array(BOX, dtype=np.float32)], id="list_of_arrays"),
        pytest.param([BOX], id="list_of_lists"),
        pytest.param(np.array([BOX, BOX], dtype=np.float32), id="numpy_two_faces"),
    ],
)
def test_every_bbox_container_reaches_the_blend(engine, frame, bboxes):
    out = engine.apply_frame_bgr(frame, 3, bboxes)
    assert engine._batch_queue.seen == 1, "engine bailed out before the forward pass"
    assert _changed(frame, out) > 0, "frame came back untouched"


def test_no_faces_leaves_the_frame_alone(engine, frame):
    """Without a detection there is nothing to lip-sync.

    A centred fallback box used to smear a generated mouth over ~700x640 px of
    the frame whenever detection came up empty.
    """
    for empty in (None, [], np.empty((0, 4), dtype=np.float32)):
        engine._batch_queue = _ImmediateQueue()
        out = engine.apply_frame_bgr(frame, 0, empty)
        assert engine._batch_queue.seen == 0, "ran inference without a face"
        assert _changed(frame, out) == 0


def test_face_index_past_the_end_clamps(engine, frame):
    out = engine.apply_frame_bgr(
        frame, 0, np.array([BOX], dtype=np.float32), face_index=7
    )
    assert engine._batch_queue.seen == 1
    assert _changed(frame, out) > 0
