"""PERF-018: preview vs record output resolution (VideoProcessor helpers)."""

import numpy as np

from app.processors.video_processor import VideoProcessor


def test_apply_record_decouple_noop_when_disabled():
    h, w = VideoProcessor.apply_record_output_resize_decouple_to_dims(
        {"GlobalInputResizeToggle": True, "RecordOutputDecoupleResizeToggle": False},
        frame_height=540,
        frame_width=960,
    )
    assert (h, w) == (540, 960)


def test_apply_record_decouple_noop_when_global_resize_off():
    h, w = VideoProcessor.apply_record_output_resize_decouple_to_dims(
        {
            "GlobalInputResizeToggle": False,
            "RecordOutputDecoupleResizeToggle": True,
            "RecordOutputResizeSizeSelection": "1080p",
        },
        frame_height=540,
        frame_width=960,
    )
    assert (h, w) == (540, 960)


def test_apply_record_decouple_upscales_height_preserves_aspect():
    ctrl = {
        "GlobalInputResizeToggle": True,
        "RecordOutputDecoupleResizeToggle": True,
        "RecordOutputResizeSizeSelection": "1080p",
    }
    h, w = VideoProcessor.apply_record_output_resize_decouple_to_dims(
        ctrl, frame_height=540, frame_width=960
    )
    assert h == 1080
    assert w == 1920


def test_apply_record_decouple_even_dimensions_odd_rounding():
    ctrl = {
        "GlobalInputResizeToggle": True,
        "RecordOutputDecoupleResizeToggle": True,
        "RecordOutputResizeSizeSelection": "720p",
    }
    h, w = VideoProcessor.apply_record_output_resize_decouple_to_dims(
        ctrl, frame_height=360, frame_width=641
    )
    assert h % 2 == 0 and w % 2 == 0
    assert h == 720
    assert abs(w / h - 641 / 360) < 0.02


def test_resize_numpy_bgr_noop_when_dims_match():
    ctrl = {
        "GlobalInputResizeToggle": True,
        "RecordOutputDecoupleResizeToggle": True,
        "RecordOutputResizeSizeSelection": "540p",
    }
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    out = VideoProcessor.resize_numpy_bgr_for_recording_stdin(ctrl, frame)
    assert out is frame


def test_resize_numpy_bgr_upscales():
    ctrl = {
        "GlobalInputResizeToggle": True,
        "RecordOutputDecoupleResizeToggle": True,
        "RecordOutputResizeSizeSelection": "1080p",
    }
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[:, :, 1] = 200
    out = VideoProcessor.resize_numpy_bgr_for_recording_stdin(ctrl, frame)
    assert out.shape == (1080, 1920, 3)
    assert out.flags["C_CONTIGUOUS"]
