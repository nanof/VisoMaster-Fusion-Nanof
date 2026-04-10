from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from app.processors.video_processor import VideoProcessor


class _RunResult:
    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def test_clear_single_frame_preview_caches_resets_all_preview_state():
    dummy = SimpleNamespace(
        _last_requested_frame_num=7,
        _cached_raw_frame_media_path="video_a.mp4",
        _cached_raw_frame_number=12,
        _cached_raw_frame_target_height=720,
        _cached_raw_frame_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        _cached_raw_image_path="image_a.png",
        _cached_raw_image_target_height=1080,
        _cached_raw_image_bgr=np.ones((2, 2, 3), dtype=np.uint8),
        _seek_cached_frame=(12, np.ones((2, 2, 3), dtype=np.uint8)),
    )

    VideoProcessor._clear_single_frame_preview_caches(dummy)

    assert dummy._last_requested_frame_num is None
    assert dummy._cached_raw_frame_media_path is None
    assert dummy._cached_raw_frame_number is None
    assert dummy._cached_raw_frame_target_height is None
    assert dummy._cached_raw_frame_bgr is None
    assert dummy._cached_raw_image_path is None
    assert dummy._cached_raw_image_target_height is None
    assert dummy._cached_raw_image_bgr is None
    assert dummy._seek_cached_frame is None


def test_process_current_frame_ignores_cached_video_frame_from_other_media(monkeypatch):
    read_calls = []
    displayed_frames = []
    started_workers = []
    fresh_frame_bgr = np.full((2, 3, 3), 77, dtype=np.uint8)

    dummy = SimpleNamespace(
        processing=False,
        is_processing_segments=False,
        main_window=SimpleNamespace(
            control={"DenoiserBaseSeedSlider": 220},
            videoSeekSlider=SimpleNamespace(value=lambda: 0),
            last_seek_read_failed=False,
        ),
        file_type="video",
        media_capture=object(),
        current_frame_number=0,
        next_frame_to_display=0,
        media_path="video_b.mp4",
        media_rotation=0,
        max_frame_number=20,
        _last_requested_frame_num=None,
        _cached_raw_frame_media_path="video_a.mp4",
        _cached_raw_frame_number=0,
        _cached_raw_frame_target_height=None,
        _cached_raw_frame_bgr=np.full((2, 3, 3), 11, dtype=np.uint8),
        _seek_cached_frame=None,
        _get_target_input_height=lambda: None,
        display_current_frame=lambda **kwargs: displayed_frames.append(kwargs),
        start_frame_worker=lambda frame_number, frame, is_single_frame, synchronous, fit_on_complete: (
            started_workers.append(
                (
                    frame_number,
                    frame.copy(),
                    is_single_frame,
                    synchronous,
                    fit_on_complete,
                )
            )
            or "worker"
        ),
    )

    monkeypatch.setattr(
        "app.processors.video_processor.misc_helpers.seek_frame",
        lambda *_args, **_kwargs: None,
    )

    def fake_read_frame(*_args, **_kwargs):
        read_calls.append(True)
        return True, fresh_frame_bgr.copy()

    monkeypatch.setattr(
        "app.processors.video_processor.misc_helpers.read_frame", fake_read_frame
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = VideoProcessor.process_current_frame(dummy, synchronous=True)

    assert result == "worker"
    assert read_calls == [True]
    assert dummy._cached_raw_frame_media_path == "video_b.mp4"
    assert len(displayed_frames) == 1
    assert np.array_equal(displayed_frames[0]["frame"], fresh_frame_bgr)
    assert len(started_workers) == 1
    assert np.array_equal(started_workers[0][1], fresh_frame_bgr[..., ::-1])


def test_mark_skipped_frame_tracks_reason_counts():
    dummy = SimpleNamespace(
        skipped_frames=set(),
        total_skipped_frames=0,
        manual_dropped_skip_count=0,
        read_error_skip_count=0,
    )

    VideoProcessor._mark_skipped_frame(dummy, 10, "manual_drop")
    VideoProcessor._mark_skipped_frame(dummy, 11, "read_error")

    assert dummy.skipped_frames == {10, 11}
    assert dummy.total_skipped_frames == 2
    assert dummy.manual_dropped_skip_count == 1
    assert dummy.read_error_skip_count == 1


def test_identify_frame_segments_without_skips_uses_processing_start_frame():
    dummy = SimpleNamespace(processing_start_frame=100, skipped_frames=set())

    segments = VideoProcessor._identify_frame_segments(dummy, 120)

    assert segments == [(100, 120)]


def test_identify_frame_segments_handles_boundary_and_pre_start_skips():
    dummy = SimpleNamespace(
        processing_start_frame=100,
        skipped_frames={95, 100, 101, 105, 112},
    )

    segments = VideoProcessor._identify_frame_segments(dummy, 112)

    assert segments == [(102, 104), (106, 111)]


def test_extract_audio_segments_always_normalizes_to_containerized_aac(
    tmp_path, monkeypatch
):
    calls: list[list[str]] = []
    validation_results = iter([False, True])

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _RunResult()

    dummy = SimpleNamespace(
        fps=30.0,
        media_path=str(tmp_path / "input.mkv"),
        _validate_audio_file=lambda _: next(validation_results),
    )

    monkeypatch.setattr("subprocess.run", fake_run)

    ok, audio_files = VideoProcessor._extract_audio_segments(
        dummy, [(0, 29)], str(tmp_path)
    )

    assert ok is True
    assert len(audio_files) == 1
    assert audio_files[0].endswith(".m4a")

    first_call = calls[0]
    retry_call = calls[1]

    assert first_call[-1].endswith(".m4a")
    assert first_call[first_call.index("-c:a") + 1] == "aac"
    assert first_call[first_call.index("-af") + 1] == "aresample=async=1:first_pts=0"
    assert first_call[first_call.index("-map") + 1] == "0:a:0?"

    assert retry_call[-1].endswith(".m4a")
    assert retry_call[retry_call.index("-c:a") + 1] == "aac"
    assert retry_call[retry_call.index("-af") + 1] == "aresample=async=1:first_pts=0"


def test_extract_audio_segments_returns_failure_after_double_validation_failure(
    tmp_path, monkeypatch
):
    calls: list[list[str]] = []
    removed_paths: list[str] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _RunResult()

    dummy = SimpleNamespace(
        fps=30.0,
        media_path=str(tmp_path / "input.mkv"),
        _validate_audio_file=lambda _: False,
    )

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("os.remove", lambda path: removed_paths.append(path))

    ok, audio_files = VideoProcessor._extract_audio_segments(
        dummy, [(0, 29)], str(tmp_path)
    )

    assert ok is False
    assert audio_files == []
    assert len(calls) == 2
    assert removed_paths == [str(tmp_path / "audio_segment_0000.m4a")]


def test_concatenate_audio_segments_reencodes_concat_output_to_m4a(
    tmp_path, monkeypatch
):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _RunResult()

    audio_files = []
    for name in ("seg_a.m4a", "seg_b.m4a"):
        path = tmp_path / name
        path.write_bytes(b"stub")
        audio_files.append(str(path))

    monkeypatch.setattr("subprocess.run", fake_run)

    output_path = VideoProcessor._concatenate_audio_segments(
        SimpleNamespace(), audio_files, str(tmp_path)
    )

    assert output_path == str(tmp_path / "audio_concatenated.m4a")
    assert len(calls) == 1

    concat_call = calls[0]
    manifest_path = Path(tmp_path / "concat_manifest.txt")
    manifest_text = manifest_path.read_text(encoding="utf-8")

    assert "file '" in manifest_text
    assert concat_call[concat_call.index("-c:a") + 1] == "aac"
    assert concat_call[concat_call.index("-af") + 1] == "aresample=async=1:first_pts=0"
    assert concat_call[-1].endswith(".m4a")


def test_finalize_default_style_recording_uses_rebuilt_audio_when_frames_skipped(
    tmp_path, monkeypatch
):
    temp_file = tmp_path / "temp_output.mp4"
    temp_file.write_bytes(b"temp-video")
    rebuilt_audio = tmp_path / "rebuilt_audio.m4a"
    rebuilt_audio.write_bytes(b"temp-audio")
    final_output = tmp_path / "final_output.mp4"

    identify_calls: list[int] = []
    extract_calls: list[tuple[list[tuple[int, int]], str]] = []
    concat_calls: list[tuple[list[str], str]] = []
    ffmpeg_calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        ffmpeg_calls.append(list(args))
        return _RunResult()

    q = queue.Queue()
    dummy = SimpleNamespace(
        feeder_thread=None,
        frames_to_display=[],
        frame_queue=q,
        join_and_clear_threads=lambda: None,
        gpu_memory_update_timer=SimpleNamespace(stop=lambda: None),
        preroll_timer=SimpleNamespace(stop=lambda: None),
        stop_live_sound=lambda: None,
        media_capture=None,
        recording_sp=None,
        recording=True,
        processing=True,
        is_processing_segments=False,
        next_frame_to_display=31,
        max_frame_number=100,
        frames_written=30,
        fps=30.0,
        play_start_time=0.0,
        play_end_time=0.0,
        total_skipped_frames=2,
        manual_dropped_skip_count=1,
        read_error_skip_count=1,
        temp_file=str(temp_file),
        media_path=str(tmp_path / "input.mkv"),
        stopped_by_error_limit=False,
        triggered_by_job_manager=False,
        processing_start_frame=10,
        last_displayed_frame=29,
        main_window=SimpleNamespace(
            control={
                "OutputMediaFolder": str(tmp_path),
                "AutoSaveWorkspaceToggle": False,
                "OpenOutputToggle": False,
            }
        ),
        _apply_job_timestamp_to_output_name=lambda *args: (None, None),
        _identify_frame_segments=lambda actual_end_frame: (
            identify_calls.append(actual_end_frame) or [(10, 14), (16, 29)]
        ),
        _extract_audio_segments=lambda segments, temp_audio_dir: (
            extract_calls.append((segments, temp_audio_dir))
            or (True, [str(tmp_path / "seg_0000.m4a"), str(tmp_path / "seg_0001.m4a")])
        ),
        _concatenate_audio_segments=lambda audio_files, temp_audio_dir: (
            concat_calls.append((audio_files, temp_audio_dir)) or str(rebuilt_audio)
        ),
        _write_video_only_output=lambda *args: True,
        _log_processing_summary=lambda *args: None,
        disable_virtualcam=lambda: None,
        processing_stopped_signal=SimpleNamespace(emit=lambda: None),
        file_type="image",
        start_time=0.0,
    )

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "app.processors.video_processor.misc_helpers.get_output_file_path",
        lambda *args, **kwargs: str(final_output),
    )
    monkeypatch.setattr(
        "app.processors.video_processor.layout_actions.enable_all_parameters_and_control_widget",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.processors.video_processor.video_control_actions.reset_media_buttons",
        lambda *args, **kwargs: None,
    )

    VideoProcessor._finalize_default_style_recording(dummy)

    assert identify_calls == [29]
    assert extract_calls and extract_calls[0][0] == [(10, 14), (16, 29)]
    assert concat_calls

    final_mux_call = ffmpeg_calls[-1]
    assert str(rebuilt_audio) in final_mux_call
    assert str(temp_file) in final_mux_call
    assert "-ss" not in final_mux_call
    assert dummy.temp_file == ""


def _make_finalize_default_style_recording_dummy(
    tmp_path,
    *,
    active_output_folder="",
    auto_save=False,
    total_skipped_frames=0,
):
    temp_file = tmp_path / "temp_output.mp4"
    temp_file.write_bytes(b"temp-video")
    return SimpleNamespace(
        feeder_thread=None,
        frames_to_display=[],
        frame_queue=queue.Queue(),
        join_and_clear_threads=lambda: None,
        gpu_memory_update_timer=SimpleNamespace(stop=lambda: None),
        preroll_timer=SimpleNamespace(stop=lambda: None),
        stop_live_sound=lambda: None,
        media_capture=None,
        recording_sp=None,
        recording=True,
        processing=True,
        is_processing_segments=False,
        next_frame_to_display=31,
        max_frame_number=100,
        frames_written=30,
        fps=30.0,
        play_start_time=0.0,
        play_end_time=0.0,
        total_skipped_frames=total_skipped_frames,
        manual_dropped_skip_count=0,
        read_error_skip_count=0,
        temp_file=str(temp_file),
        media_path=str(tmp_path / "input.mkv"),
        stopped_by_error_limit=False,
        triggered_by_job_manager=False,
        processing_start_frame=10,
        last_displayed_frame=29,
        active_output_folder=active_output_folder,
        main_window=SimpleNamespace(
            control={
                "OutputMediaFolder": str(tmp_path / "fallback-output"),
                "AutoSaveWorkspaceToggle": auto_save,
                "OpenOutputToggle": False,
            }
        ),
        _apply_job_timestamp_to_output_name=lambda *args: (None, None),
        _identify_frame_segments=lambda *_args, **_kwargs: [(10, 29)],
        _extract_audio_segments=lambda *_args, **_kwargs: (True, []),
        _concatenate_audio_segments=lambda *_args, **_kwargs: None,
        _write_video_only_output=lambda *args: True,
        _log_processing_summary=lambda *args: None,
        disable_virtualcam=lambda: None,
        processing_stopped_signal=SimpleNamespace(emit=lambda: None),
        file_type="image",
        start_time=0.0,
    )


def test_finalize_default_style_recording_uses_active_output_folder_for_output_and_autosave(
    tmp_path, monkeypatch
):
    resolved_output_folder = str(tmp_path / "clustered" / "Embedding A")
    dummy = _make_finalize_default_style_recording_dummy(
        tmp_path,
        active_output_folder=resolved_output_folder,
        auto_save=True,
    )
    ffmpeg_calls: list[list[str]] = []
    output_path_calls: list[dict] = []
    saved_workspaces: list[str] = []

    def fake_run(args, **kwargs):
        ffmpeg_calls.append(list(args))
        return _RunResult()

    def fake_get_output_file_path(media_path, output_folder, **kwargs):
        output_path_calls.append(
            {
                "media_path": media_path,
                "output_folder": output_folder,
                "kwargs": kwargs,
            }
        )
        return str(Path(output_folder) / "resolved_output.mp4")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "app.processors.video_processor.misc_helpers.get_output_file_path",
        fake_get_output_file_path,
    )
    monkeypatch.setattr(
        "app.processors.video_processor.save_load_actions.save_current_workspace",
        lambda _main_window, json_path: saved_workspaces.append(json_path),
    )
    monkeypatch.setattr(
        "app.processors.video_processor.layout_actions.enable_all_parameters_and_control_widget",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.processors.video_processor.video_control_actions.reset_media_buttons",
        lambda *args, **kwargs: None,
    )

    VideoProcessor._finalize_default_style_recording(dummy)

    assert [call["output_folder"] for call in output_path_calls] == [
        resolved_output_folder,
        resolved_output_folder,
    ]
    assert ffmpeg_calls[-1][-1] == str(
        Path(resolved_output_folder) / "resolved_output.mp4"
    )
    assert saved_workspaces == [
        str(Path(resolved_output_folder) / "resolved_output.mp4.json")
    ]


def test_finalize_default_style_recording_falls_back_to_output_media_folder_when_active_output_folder_empty(
    tmp_path, monkeypatch
):
    dummy = _make_finalize_default_style_recording_dummy(
        tmp_path,
        active_output_folder="",
        auto_save=False,
    )
    output_path_calls: list[str] = []

    def fake_run(args, **kwargs):
        return _RunResult()

    def fake_get_output_file_path(media_path, output_folder, **kwargs):
        output_path_calls.append(output_folder)
        return str(Path(output_folder) / "resolved_output.mp4")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "app.processors.video_processor.misc_helpers.get_output_file_path",
        fake_get_output_file_path,
    )
    monkeypatch.setattr(
        "app.processors.video_processor.layout_actions.enable_all_parameters_and_control_widget",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.processors.video_processor.video_control_actions.reset_media_buttons",
        lambda *args, **kwargs: None,
    )

    VideoProcessor._finalize_default_style_recording(dummy)

    assert output_path_calls == [str(tmp_path / "fallback-output")]
