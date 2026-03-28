from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.processors.video_processor import VideoProcessor


class _DummyCapture:
    def isOpened(self):
        return True

    def set(self, *_args, **_kwargs):
        return True


class _DummyTargetFace:
    def __init__(self, face_id, embeddings):
        self.face_id = face_id
        self._embeddings = embeddings

    def get_embedding(self, recognition_model):
        return self._embeddings.get(recognition_model)


def test_scan_issue_frames_restores_dense_smoothing_state():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = [{"id": 1}]
    processor._smoothed_kps = {1: np.array([[1.0, 2.0]], dtype=np.float32)}
    processor._smoothed_dense_kps = {1: np.array([[3.0, 4.0]], dtype=np.float32)}
    processor.last_detected_faces = [{"id": 1}]
    processor.main_window = SimpleNamespace(
        control={},
        parameters={},
        target_faces={},
        dropped_frames=set(),
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 7),
    )
    processor._get_target_input_height = lambda: 256

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
    ):
        result = processor.scan_issue_frames(
            scan_ranges=[(1, 0)],
            base_control={},
            base_params={},
            target_faces_snapshot={},
            reset_frame_number=3,
        )

    assert result == {
        "issue_frames_by_face": {},
        "frames_scanned": 0,
        "faces_with_issues": 0,
    }
    np.testing.assert_array_equal(
        processor._smoothed_dense_kps[1], np.array([[3.0, 4.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        processor._smoothed_kps[1], np.array([[1.0, 2.0]], dtype=np.float32)
    )
    assert processor.last_detected_faces == [{"id": 1}]
    assert processor.current_frame_number == 3


def test_describe_issue_scan_scope_uses_normalized_effective_ranges():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.max_frame_number = 100
    processor.main_window = SimpleNamespace(
        job_marker_pairs=[(20, 30), (10, 25), (40, None)]
    )

    scope_text = processor.describe_issue_scan_scope([(10, 30), (40, 100)])

    assert scope_text == "Scanning 1 marked range and record start frame 40 to end"


def test_describe_issue_scan_scope_uses_raw_open_start_when_ranges_merge():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.max_frame_number = 100
    processor.main_window = SimpleNamespace(job_marker_pairs=[(10, 30), (20, None)])

    scope_text = processor.describe_issue_scan_scope([(10, 100)])

    assert scope_text == "Scanning 1 marked range and record start frame 20 to end"


def test_build_issue_scan_state_segments_switches_only_at_marker_boundaries():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.main_window = SimpleNamespace(markers={5: {"id": 5}, 8: {"id": 8}})
    resolved_frames = []

    def fake_resolve(frame_number, *_args, **_kwargs):
        resolved_frames.append(frame_number)
        return {"frame": frame_number}, {"params": frame_number}

    processor._resolve_scan_state_for_frame = fake_resolve

    segments = processor._build_issue_scan_state_segments(
        [(3, 10)],
        {},
        {},
        {},
    )

    assert resolved_frames == [3, 5, 8]
    assert segments == [
        (3, 4, {"frame": 3}, {"params": 3}),
        (5, 7, {"frame": 5}, {"params": 5}),
        (8, 10, {"frame": 8}, {"params": 8}),
    ]


def test_scan_issue_frames_reports_progress_per_frame_and_skips_dropped_runs():
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.media_path = "dummy.mp4"
    processor.media_rotation = 0
    processor.fps = 30.0
    processor.current_frame_number = 0
    processor.last_detected_faces = []
    processor._smoothed_kps = {}
    processor._smoothed_dense_kps = {}
    processor.main_window = SimpleNamespace(
        control={},
        parameters={},
        target_faces={
            "1": _DummyTargetFace(
                "1", {"arcface_128": np.array([1.0], dtype=np.float32)}
            )
        },
        dropped_frames={2, 3, 4, 11},
        markers={},
        videoSeekSlider=SimpleNamespace(value=lambda: 0),
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 50}),
        models_processor=SimpleNamespace(device="cpu"),
    )
    processor._get_target_input_height = lambda: 256
    progress_updates = []
    seek_calls = []
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def fake_read_frame(*_args, **_kwargs):
        return True, frame.copy()

    with (
        patch(
            "app.processors.video_processor.cv2.VideoCapture",
            return_value=_DummyCapture(),
        ),
        patch(
            "app.processors.video_processor.misc_helpers.read_frame",
            side_effect=fake_read_frame,
        ),
        patch(
            "app.processors.video_processor.misc_helpers.seek_frame",
            side_effect=lambda _capture, frame_number: seek_calls.append(frame_number),
        ),
        patch("app.processors.video_processor.misc_helpers.release_capture"),
        patch.object(
            processor,
            "_build_issue_scan_state_segments",
            return_value=[(0, 24, {}, {})],
        ),
        patch.object(
            processor,
            "_prepare_issue_scan_match_context",
            return_value={
                "recognition_model": "arcface_128",
                "similarity_type": "Opal",
                "prepared_targets": [],
            },
        ),
        patch.object(
            processor,
            "_run_sequential_detection",
            return_value=(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
                np.empty((0, 68, 2), dtype=np.float32),
            ),
        ),
    ):
        result = processor.scan_issue_frames(
            scan_ranges=[(0, 24)],
            base_control={},
            base_params={},
            target_faces_snapshot=processor.main_window.target_faces,
            progress_callback=lambda processed, total, frame_number: (
                progress_updates.append((processed, total, frame_number))
            ),
        )

    assert result == {
        "issue_frames_by_face": {
            "1": list(range(0, 2)) + list(range(5, 11)) + list(range(12, 25))
        },
        "frames_scanned": 21,
        "faces_with_issues": 1,
    }
    assert progress_updates == [
        (1, 21, 0),
        (2, 21, 1),
        (3, 21, 5),
        (4, 21, 6),
        (5, 21, 7),
        (6, 21, 8),
        (7, 21, 9),
        (8, 21, 10),
        (9, 21, 12),
        (10, 21, 13),
        (11, 21, 14),
        (12, 21, 15),
        (13, 21, 16),
        (14, 21, 17),
        (15, 21, 18),
        (16, 21, 19),
        (17, 21, 20),
        (18, 21, 21),
        (19, 21, 22),
        (20, 21, 23),
        (21, 21, 24),
    ]
    assert seek_calls == [0, 5, 12]
