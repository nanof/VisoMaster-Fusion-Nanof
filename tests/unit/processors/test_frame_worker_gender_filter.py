"""FrameWorker gender-appearance filter: annotation + skip decision."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.processors.workers.frame_worker import FrameWorker

FEMALE_ONLY = {
    "GenderAppearanceFilterSelection": "Female appearance only",
    "GenderAppearanceMinConfidenceSlider": 60,
}
ALL_MODE = {"GenderAppearanceFilterSelection": "All"}


def _worker(classify_result, calls=None):
    def classify_batch(img, faces):
        if calls is not None:
            calls.append([dict(f) for f in faces])
        return [classify_result] * len(faces)

    return SimpleNamespace(
        models_processor=SimpleNamespace(classify_faces_gender=classify_batch),
        _gender_filter_last_report=None,
    )


def _annotate(worker, faces, control):
    FrameWorker._annotate_detected_gender_appearance(worker, None, faces, control)


def _face(bbox=(10.0, 10.0, 50.0, 60.0), track_id=-1):
    return {
        "bbox": np.array(bbox, dtype=np.float32),
        "kps_5": np.zeros((5, 2), dtype=np.float32),
        "track_id": track_id,
    }


def test_annotation_skipped_entirely_when_filter_is_all():
    calls: list = []
    worker = _worker(("male", 0.99), calls)
    faces = [_face()]
    _annotate(worker, faces, ALL_MODE)
    assert calls == []
    assert "detected_gender" not in faces[0]


def test_annotation_runs_and_stores_gender_when_filter_active():
    calls: list = []
    worker = _worker(("male", 0.93), calls)
    faces = [_face(track_id=7)]
    _annotate(worker, faces, FEMALE_ONLY)
    assert len(calls) == 1
    assert calls[0][0]["track_id"] == 7
    assert faces[0]["detected_gender"] == "male"
    assert faces[0]["detected_gender_confidence"] == pytest.approx(0.93)


def test_all_faces_are_classified_in_a_single_batched_call():
    calls: list = []
    worker = _worker(("female", 0.9), calls)
    faces = [_face(track_id=1), _face(track_id=2), _face(track_id=3)]
    _annotate(worker, faces, FEMALE_ONLY)
    assert len(calls) == 1, "expected one batched inference per frame"
    assert len(calls[0]) == 3
    assert all(f["detected_gender"] == "female" for f in faces)


def test_male_face_is_skipped_under_female_only():
    worker = _worker(("male", 0.93))
    faces = [_face()]
    _annotate(worker, faces, FEMALE_ONLY)
    assert FrameWorker._skip_face_for_gender_filter(faces[0], FEMALE_ONLY) is True


def test_female_face_is_kept_under_female_only():
    worker = _worker(("female", 0.93))
    faces = [_face()]
    _annotate(worker, faces, FEMALE_ONLY)
    assert FrameWorker._skip_face_for_gender_filter(faces[0], FEMALE_ONLY) is False


def test_low_confidence_face_is_kept_fail_open():
    worker = _worker(("male", 0.51))
    faces = [_face()]
    _annotate(worker, faces, FEMALE_ONLY)
    assert FrameWorker._skip_face_for_gender_filter(faces[0], FEMALE_ONLY) is False


def test_unannotated_face_is_never_skipped():
    # Face never passed through annotation (e.g. filter off) must not be dropped.
    assert FrameWorker._skip_face_for_gender_filter(_face(), FEMALE_ONLY) is False


def test_annotation_is_not_repeated_for_same_face():
    calls: list = []
    worker = _worker(("female", 0.90), calls)
    faces = [_face()]
    _annotate(worker, faces, FEMALE_ONLY)
    _annotate(worker, faces, FEMALE_ONLY)
    assert len(calls) == 1
