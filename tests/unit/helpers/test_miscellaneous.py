"""
MISC-* tests for pure utility functions in app.helpers.miscellaneous
"""

import numpy as np
import pytest
from app.helpers.miscellaneous import (
    coerce_similarity_threshold,
    count_issue_scan_frames,
    ParametersDict,
    detector_input_size_from_control,
    find_best_target_match,
    is_av1_fourcc_tag,
    is_image_file,
    is_detected_face_eligible_for_matching,
    is_video_file,
    get_file_type,
    get_scaling_transforms,
    MediaMetadata,
    format_target_media_tooltip,
    probe_media_metadata,
    target_media_path_sort_key,
    image_extensions,
    normalize_issue_scan_ranges,
    refresh_target_media_file_stats,
    video_extensions,
    _transform_cache,
)


# ---------------------------------------------------------------------------
# MISC-03/04 — file type detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "photo.png",
        "photo.jpg",
        "photo.jpeg",
        "photo.webp",
        "scan.tif",
        "scan.tiff",
        "image.jp2",
    ],
)
def test_is_image_file_true(name):
    assert is_image_file(name) is True


@pytest.mark.parametrize(
    "name",
    ["video.mp4", "clip.avi", "movie.mkv", "record.mov", "doc.txt", "archive.zip", ""],
)
def test_is_image_file_false(name):
    assert is_image_file(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "video.mp4",
        "clip.avi",
        "movie.mkv",
        "record.mov",
        "stream.webm",
        "anim.gif",
    ],
)
def test_is_video_file_true(name):
    assert is_video_file(name) is True


@pytest.mark.parametrize(
    "name", ["photo.png", "photo.jpg", "doc.txt", "archive.zip", ""]
)
def test_is_video_file_false(name):
    assert is_video_file(name) is False


def test_get_file_type_image():
    assert get_file_type("photo.png") == "image"


def test_get_file_type_video():
    assert get_file_type("clip.mp4") == "video"


def test_get_file_type_unknown():
    assert get_file_type("notes.txt") is None


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("av01", True),
        ("AV01", True),
        ("dav1", True),
        ("h264", False),
        ("avc1", False),
        ("", False),
    ],
)
def test_is_av1_fourcc_tag(tag, expected):
    assert is_av1_fourcc_tag(tag) is expected


# ---------------------------------------------------------------------------
# MISC-01/02 — get_scaling_transforms cache behaviour
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_transform_cache():
    """Ensure transform cache is empty before each cache test."""
    _transform_cache.clear()
    yield
    _transform_cache.clear()


def _make_control(**overrides):
    base = {
        "get_cropped_face_kpsTypeSelection": "BILINEAR",
        "original_face_128_384TypeSelection": "BILINEAR",
        "original_face_512TypeSelection": "BILINEAR",
        "UntransformTypeSelection": "BILINEAR",
        "ScalebackFrameTypeSelection": "BILINEAR",
        "expression_faceeditor_t256TypeSelection": "BILINEAR",
        "expression_faceeditor_backTypeSelection": "BILINEAR",
        "block_shiftTypeSelection": "NEAREST",
        "AntialiasTypeSelection": "False",
    }
    base.update(overrides)
    return base


def test_same_params_returns_cached_object():
    ctrl = _make_control()
    result1 = get_scaling_transforms(ctrl)
    result2 = get_scaling_transforms(ctrl)
    # Exact same tuple object — came from cache
    assert result1 is result2


def test_different_params_returns_different_object():
    ctrl_a = _make_control(original_face_512TypeSelection="BILINEAR")
    ctrl_b = _make_control(original_face_512TypeSelection="BICUBIC")
    result_a = get_scaling_transforms(ctrl_a)
    result_b = get_scaling_transforms(ctrl_b)
    assert result_a is not result_b


def test_cache_populated_after_first_call():
    ctrl = _make_control()
    assert len(_transform_cache) == 0
    get_scaling_transforms(ctrl)
    assert len(_transform_cache) == 1


def test_returns_tuple_of_expected_length():
    """get_scaling_transforms returns a 12-element tuple."""
    ctrl = _make_control()
    result = get_scaling_transforms(ctrl)
    assert isinstance(result, tuple)
    assert len(result) == 12


def test_resize_objects_have_correct_size():
    from torchvision.transforms import v2

    ctrl = _make_control()
    t512, t384, t256, t128 = get_scaling_transforms(ctrl)[:4]
    # Each Resize object should have the right target size
    assert isinstance(t512, v2.Resize)
    assert isinstance(t384, v2.Resize)
    assert isinstance(t256, v2.Resize)
    assert isinstance(t128, v2.Resize)


# ---------------------------------------------------------------------------
# Extension tuple completeness sanity checks
# ---------------------------------------------------------------------------


def test_image_extensions_are_lowercase_dotted():
    for ext in image_extensions:
        assert ext.startswith("."), f"{ext} should start with '.'"
        assert ext == ext.lower(), f"{ext} should be lowercase"


def test_video_extensions_are_lowercase_dotted():
    for ext in video_extensions:
        assert ext.startswith("."), f"{ext} should start with '.'"
        assert ext == ext.lower(), f"{ext} should be lowercase"


def test_no_overlap_between_image_and_video_extensions():
    assert set(image_extensions).isdisjoint(set(video_extensions))


# ---------------------------------------------------------------------------
# MISC-05/06 - shared scan/render matching helpers
# ---------------------------------------------------------------------------


def test_is_detected_face_eligible_for_matching_rejects_small_bbox():
    kps = np.array(
        [[10.0, 10.0], [20.0, 10.0], [15.0, 15.0], [11.0, 20.0], [19.0, 20.0]],
        dtype=np.float32,
    )
    tiny_bbox = np.array([0.0, 0.0, 18.0, 30.0], dtype=np.float32)
    assert is_detected_face_eligible_for_matching(kps, tiny_bbox, 20) is False


def test_is_detected_face_eligible_for_matching_accepts_valid_face():
    kps = np.array(
        [[10.0, 10.0], [20.0, 10.0], [15.0, 15.0], [11.0, 20.0], [19.0, 20.0]],
        dtype=np.float32,
    )
    bbox = np.array([0.0, 0.0, 30.0, 30.0], dtype=np.float32)
    assert is_detected_face_eligible_for_matching(kps, bbox, 20) is True


class _DummyTargetFace:
    def __init__(self, face_id: int, embedding_store: dict[str, np.ndarray]):
        self.face_id = face_id
        self._embedding_store = embedding_store

    def get_embedding(self, recognition_model: str) -> np.ndarray | None:
        return self._embedding_store.get(recognition_model)


class _DummyModelsProcessor:
    @staticmethod
    def findCosineDistance(
        detected_embedding: np.ndarray, target_embedding: np.ndarray
    ) -> float:
        return float(np.dot(detected_embedding, target_embedding) * 100.0)


def test_find_best_target_match_respects_parameters_dict_thresholds():
    defaults = {"SimilarityThresholdSlider": 60}
    face_params = {
        "1": ParametersDict({"SimilarityThresholdSlider": 95}, defaults),
        "2": ParametersDict({"SimilarityThresholdSlider": 70}, defaults),
    }
    targets = {
        1: _DummyTargetFace(1, {"arcface_128": np.array([0.90], dtype=np.float32)}),
        2: _DummyTargetFace(2, {"arcface_128": np.array([0.80], dtype=np.float32)}),
    }

    best_target, best_params, best_score = find_best_target_match(
        np.array([1.0], dtype=np.float32),
        _DummyModelsProcessor(),
        targets,
        face_params,
        defaults,
        "arcface_128",
    )

    assert best_target is not None
    assert best_target.face_id == 2
    assert best_params is not None
    assert best_params["SimilarityThresholdSlider"] == 70
    assert best_score == pytest.approx(80.0)


def test_find_best_target_match_returns_none_for_invalid_embedding():
    defaults = {"SimilarityThresholdSlider": 60}
    targets = {
        1: _DummyTargetFace(1, {"arcface_128": np.array([0.90], dtype=np.float32)}),
    }
    best_target, best_params, best_score = find_best_target_match(
        None,
        _DummyModelsProcessor(),
        targets,
        {},
        defaults,
        "arcface_128",
    )
    assert best_target is None
    assert best_params is None
    assert best_score == -1.0


def test_find_best_target_match_coerces_string_threshold():
    defaults = {"SimilarityThresholdSlider": 60}
    face_params = {"1": {"SimilarityThresholdSlider": "85"}}
    targets = {
        1: _DummyTargetFace(1, {"arcface_128": np.array([1.0], dtype=np.float32)}),
    }
    best_target, best_params, best_score = find_best_target_match(
        np.array([1.0], dtype=np.float32),
        _DummyModelsProcessor(),
        targets,
        face_params,
        defaults,
        "arcface_128",
    )
    assert best_target is not None
    assert best_params is not None
    assert best_score == pytest.approx(100.0)


def test_coerce_similarity_threshold_falls_back_to_defaults():
    assert coerce_similarity_threshold("72", {"SimilarityThresholdSlider": 50}) == 72.0
    assert coerce_similarity_threshold(None, {"SimilarityThresholdSlider": 55}) == 55.0


def test_count_issue_scan_frames_excludes_dropped_frames():
    scan_ranges = [(0, 9), (20, 24)]
    dropped_frames = {2, 3, 22}
    assert count_issue_scan_frames(scan_ranges, dropped_frames) == 12


def test_count_issue_scan_frames_returns_zero_when_all_frames_dropped():
    scan_ranges = [(10, 12)]
    dropped_frames = {10, 11, 12}
    assert count_issue_scan_frames(scan_ranges, dropped_frames) == 0


def test_normalize_issue_scan_ranges_sorts_open_ended_style_ranges():
    scan_ranges = [(20, 30), (5, 50)]
    assert normalize_issue_scan_ranges(scan_ranges) == [(5, 50)]


def test_normalize_issue_scan_ranges_merges_overlaps():
    scan_ranges = [(20, 25), (10, 15), (12, 18), (24, 30), (40, 45)]
    assert normalize_issue_scan_ranges(scan_ranges) == [(10, 18), (20, 30), (40, 45)]


def test_count_issue_scan_frames_does_not_double_count_overlaps():
    scan_ranges = [(10, 20), (15, 25)]
    dropped_frames = {12, 18, 22}
    assert count_issue_scan_frames(scan_ranges, dropped_frames) == 13


# ---------------------------------------------------------------------------
# UT-04: keypoints_adjustments — guard against fewer than 5 keypoints
# ---------------------------------------------------------------------------

from app.helpers.miscellaneous import keypoints_adjustments  # noqa: E402


class TestKeypointsAdjustmentsGuard:
    """UT-04: When LandmarksPositionAdjEnableToggle is True and fewer than 5
    keypoints are present, keypoints_adjustments must return the input unchanged
    rather than raising IndexError.
    """

    def _params_with_toggle(self) -> dict:
        return {
            "LandmarksPositionAdjEnableToggle": True,
            "EyeLeftXAmountSlider": 5.0,
            "EyeLeftYAmountSlider": 3.0,
            "EyeRightXAmountSlider": -2.0,
            "EyeRightYAmountSlider": 1.0,
            "NoseXAmountSlider": 0.0,
            "NoseYAmountSlider": 0.0,
            "MouthLeftXAmountSlider": 0.0,
            "MouthLeftYAmountSlider": 0.0,
            "MouthRightXAmountSlider": 0.0,
            "MouthRightYAmountSlider": 0.0,
        }

    def test_fewer_than_5_keypoints_returns_input_unchanged(self):
        kps = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)  # only 2 points
        params = self._params_with_toggle()
        result = keypoints_adjustments(kps, params)
        np.testing.assert_array_equal(result, kps)

    def test_zero_keypoints_returns_input_unchanged(self):
        kps = np.zeros((0, 2), dtype=np.float32)
        params = self._params_with_toggle()
        result = keypoints_adjustments(kps, params)
        assert result.shape == (0, 2)

    def test_exactly_4_keypoints_returns_input_unchanged(self):
        kps = np.ones((4, 2), dtype=np.float32)
        params = self._params_with_toggle()
        result = keypoints_adjustments(kps, params)
        np.testing.assert_array_equal(result, kps)

    def test_exactly_5_keypoints_applies_adjustments(self):
        kps = np.zeros((5, 2), dtype=np.float32)
        params = self._params_with_toggle()
        result = keypoints_adjustments(kps, params)
        # EyeLeft X should be shifted by EyeLeftXAmountSlider=5
        assert result[0][0] == pytest.approx(5.0)

    def test_toggle_off_with_fewer_kps_does_not_raise(self):
        """When toggle is off, the guard should not matter — confirm no crash."""
        kps = np.array([[1.0, 2.0]], dtype=np.float32)
        params = {"LandmarksPositionAdjEnableToggle": False}
        result = keypoints_adjustments(kps, params)
        np.testing.assert_array_equal(result, kps)


def test_detector_input_size_fast_cap_clamps():
    c = {
        "DetectorInternalSizeSelection": "512",
        "PerformanceFastDetectEnableToggle": True,
        "PerformanceFastDetectCapSideSelection": "320",
    }
    assert detector_input_size_from_control(c) == (320, 320)


def test_detector_input_size_fast_cap_min_of_user_and_cap():
    c = {
        "DetectorInternalSizeSelection": "256",
        "PerformanceFastDetectEnableToggle": True,
        "PerformanceFastDetectCapSideSelection": "384",
    }
    assert detector_input_size_from_control(c) == (256, 256)


def test_target_media_path_sort_key_orders_by_name_date_and_size(tmp_path):
    older = tmp_path / "alpha.png"
    newer = tmp_path / "beta.png"
    older.write_bytes(b"a")
    newer.write_bytes(b"abcd")

    name_keys = [
        target_media_path_sort_key(str(older), "name"),
        target_media_path_sort_key(str(newer), "name"),
    ]
    assert name_keys == sorted(name_keys)

    date_keys = [
        target_media_path_sort_key(str(older), "date"),
        target_media_path_sort_key(str(newer), "date"),
    ]
    assert date_keys == sorted(date_keys)

    size_keys = [
        target_media_path_sort_key(str(older), "size"),
        target_media_path_sort_key(str(newer), "size"),
    ]
    assert size_keys == sorted(size_keys)
    assert size_keys[0][1] < size_keys[1][1]


def test_target_media_path_sort_key_orders_by_metadata_modes():
    small = MediaMetadata(width=100, height=50, total_frames=10)
    large = MediaMetadata(width=200, height=100, total_frames=40)

    dim_keys = [
        target_media_path_sort_key("a.png", "dimensions", small),
        target_media_path_sort_key("b.png", "dimensions", large),
    ]
    assert dim_keys == sorted(dim_keys)

    pixel_keys = [
        target_media_path_sort_key("a.png", "pixels", small),
        target_media_path_sort_key("b.png", "pixels", large),
    ]
    assert pixel_keys[0][1] < pixel_keys[1][1]

    frame_keys = [
        target_media_path_sort_key("a.mp4", "frames", small),
        target_media_path_sort_key("b.mp4", "frames", large),
    ]
    assert frame_keys[0][1] < frame_keys[1][1]


def test_probe_media_metadata_image_header_only(tmp_path):
    from PIL import Image

    image_path = tmp_path / "shot.png"
    Image.new("RGB", (320, 180), color=(12, 34, 56)).save(image_path)

    metadata = probe_media_metadata(str(image_path), "image")
    assert metadata is not None
    assert metadata.width == 320
    assert metadata.height == 180
    assert metadata.total_frames == 1
    assert metadata.pixels == 320 * 180


def test_format_target_media_tooltip_includes_metadata():
    tip = format_target_media_tooltip(
        "C:/clips/demo.mp4",
        "video",
        MediaMetadata(
            width=1920,
            height=1080,
            total_frames=240,
            frame_rate=24.0,
            bitrate_kbits=5000.0,
        ),
        file_size=2048,
    )
    assert "demo.mp4" in tip
    assert "Type: video" in tip
    assert "1920×1080" in tip
    assert "Frames: 240" in tip


def test_refresh_target_media_file_stats_reads_file_metadata(tmp_path):
    from types import SimpleNamespace

    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"12345678")

    button = SimpleNamespace(
        media_path=str(media_file),
        is_webcam=False,
        is_screen_capture=False,
        file_type="video",
        _media_metadata=None,
        setToolTip=lambda *_args, **_kwargs: None,
    )
    refresh_target_media_file_stats(button)

    assert button._file_stats_loaded is True
    assert button._file_size == 8
    assert button._file_mtime > 0


def test_detector_input_size_fast_cap_off_uses_user_only():
    c = {
        "DetectorInternalSizeSelection": "416",
        "PerformanceFastDetectEnableToggle": False,
        "PerformanceFastDetectCapSideSelection": "256",
    }
    assert detector_input_size_from_control(c) == (416, 416)
