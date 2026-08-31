from types import SimpleNamespace
from unittest.mock import patch

import cv2

from app.processors.video_processor import VideoProcessor
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA


def _capped_processor() -> VideoProcessor:
    processor = VideoProcessor.__new__(VideoProcessor)
    processor._used_ffmpeg_cap = True
    processor._preview_fps_cap_active = True
    processor._ffmpeg_scale_input_active = False
    processor.recording_source_fps = 60.0
    processor.fps = 30.0
    processor.media_rotation = 0
    return processor


def test_preview_fps_cap_controls_default_to_thirty() -> None:
    settings = SETTINGS_LAYOUT_DATA["Video Playback Settings"]

    assert settings["PreviewFpsCapEnableToggle"]["default"] is False
    assert settings["PreviewMaxFpsSlider"]["default"] == "30"
    assert (
        settings["PreviewMaxFpsSlider"]["parentToggle"]
        == "PreviewFpsCapEnableToggle"
    )


def test_preview_fps_cap_maps_between_source_and_processing_frames() -> None:
    processor = _capped_processor()

    assert processor._timeline_to_processing_frame(120) == 60
    assert processor._processing_to_timeline_frame(60) == 120


def test_preview_fps_cap_applies_marker_skipped_by_sampling() -> None:
    processor = _capped_processor()
    marker = {"parameters": {}, "control": {}}
    processor.main_window = SimpleNamespace(markers={3: marker})

    marker_frame, marker_data = processor._marker_data_for_processing_frame(2)

    assert marker_frame == 3
    assert marker_data is marker


def test_audio_wall_clock_advances_in_capped_processing_space() -> None:
    processor = _capped_processor()
    processor._playback_use_wall_clock = True
    processor._playback_clock_t0 = 10.0
    processor._playback_clock_anchor_frame = 15
    processor._wall_clock_use_audio_file_rate = True
    processor._audio_sync_rate = 1.0
    processor._audio_sync_fps_file = 30.0
    processor.next_frame_to_display = 15
    processor.max_frame_number = 300

    with patch(
        "app.processors.video_processor.time.perf_counter", return_value=12.0
    ):
        assert processor._expected_frame_from_wall_clock() == 75


def test_preview_fps_cap_ffmpeg_stream_seeks_in_output_time() -> None:
    processor = _capped_processor()
    processor.media_path = "input.mp4"
    processor.media_capture = SimpleNamespace(
        get=lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
    )
    processor.ffmpeg_input_sp = None
    processor.ffmpeg_input_width = 0
    processor.ffmpeg_input_height = 0
    processor.ffmpeg_input_prefetched_frame = None
    fake_process = SimpleNamespace(stdout=SimpleNamespace())

    with patch(
        "app.processors.video_processor.subprocess.Popen", return_value=fake_process
    ) as popen:
        assert processor._start_recording_ffmpeg_input_stream(60, 30.0, None)

    args = popen.call_args.args[0]
    assert args[args.index("-ss") + 1] == "2.000000"
    assert args[args.index("-vf") + 1] == "fps=30.000000"


def test_preview_fps_cap_ffmpeg_stream_scales_with_lanczos() -> None:
    processor = _capped_processor()
    processor.media_path = "input.mp4"
    processor.media_capture = SimpleNamespace(
        get=lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
    )
    processor.ffmpeg_input_sp = None
    processor.ffmpeg_input_width = 0
    processor.ffmpeg_input_height = 0
    processor.ffmpeg_input_prefetched_frame = None
    fake_process = SimpleNamespace(stdout=SimpleNamespace())

    with patch(
        "app.processors.video_processor.subprocess.Popen", return_value=fake_process
    ) as popen:
        assert processor._start_recording_ffmpeg_input_stream(60, 30.0, 540)

    vf = popen.call_args.args[0][popen.call_args.args[0].index("-vf") + 1]
    assert vf.startswith("fps=30.000000,")
    assert "scale=960:540:flags=lanczos+accurate_rnd+full_chroma_int" in vf
    assert processor._used_ffmpeg_cap is True


def test_scaled_decode_ffmpeg_omits_fps_filter_and_does_not_remap_frames() -> None:
    processor = VideoProcessor.__new__(VideoProcessor)
    processor._used_ffmpeg_cap = False
    processor._preview_fps_cap_active = False
    processor._ffmpeg_scale_input_active = True
    processor.recording_source_fps = 30.0
    processor.fps = 30.0
    processor.media_rotation = 0
    processor.media_path = "input.mp4"
    processor.media_capture = SimpleNamespace(
        get=lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
    )
    processor.ffmpeg_input_sp = None
    processor.ffmpeg_input_width = 0
    processor.ffmpeg_input_height = 0
    processor.ffmpeg_input_prefetched_frame = None
    fake_process = SimpleNamespace(stdout=SimpleNamespace())

    with patch(
        "app.processors.video_processor.subprocess.Popen", return_value=fake_process
    ) as popen:
        assert processor._start_recording_ffmpeg_input_stream(
            90, 30.0, 540, resample_fps=False
        )

    args = popen.call_args.args[0]
    assert args[args.index("-ss") + 1] == "3.000000"
    assert "-noautorotate" in args
    vf = args[args.index("-vf") + 1]
    assert "fps=" not in vf
    assert vf == "scale=960:540:flags=fast_bilinear"
    assert processor._used_ffmpeg_cap is False
    assert processor.ffmpeg_input_width == 960
    assert processor.ffmpeg_input_height == 540
