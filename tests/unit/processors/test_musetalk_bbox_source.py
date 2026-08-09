"""Lip-sync must find the face whichever stage detected it.

The feeder fills ``precomputed_bboxes`` during playback, but a paused preview
detects inside the worker and leaves it empty. Reading only the feeder's copy
made lip-sync skip every paused frame while reporting nothing, which is the one
situation where a user is actually studying the mouth.
"""

from __future__ import annotations

import numpy as np

from app.processors.workers.frame_worker import FrameWorker


def _worker(feeder, local):
    w = FrameWorker.__new__(FrameWorker)
    w.precomputed_bboxes = feeder
    w._detected_bboxes = local
    return w


def test_playback_uses_the_feeder_boxes():
    feeder = [np.array([10.0, 20.0, 110.0, 140.0])]
    local = [np.array([0.0, 0.0, 5.0, 5.0])]
    assert _worker(feeder, local)._musetalk_bboxes() is feeder


def test_paused_preview_falls_back_to_locally_detected_boxes():
    local = [np.array([10.0, 20.0, 110.0, 140.0])]
    assert _worker(None, local)._musetalk_bboxes() is local


def test_an_empty_feeder_list_is_not_mistaken_for_a_detection():
    """A frame the feeder scanned without finding faces must not mask a local hit."""
    local = [np.array([10.0, 20.0, 110.0, 140.0])]
    assert _worker([], local)._musetalk_bboxes() is local


def test_no_faces_anywhere_stays_empty():
    assert _worker(None, None)._musetalk_bboxes() is None


def test_numpy_arrays_are_accepted_as_the_feeder_payload():
    """Detectors hand back arrays; truthiness on those raises, so length is used."""
    feeder = np.array([[10.0, 20.0, 110.0, 140.0]])
    got = _worker(feeder, None)._musetalk_bboxes()
    assert got is feeder

    empty = np.zeros((0, 4), dtype=np.float32)
    local = [np.array([1.0, 2.0, 3.0, 4.0])]
    assert _worker(empty, local)._musetalk_bboxes() is local
