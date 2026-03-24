from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.processors.video_processor import VideoProcessor


class _RunResult:
    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


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


def test_extract_audio_segments_reencodes_to_containerized_aac_when_validation_fails(
    tmp_path, monkeypatch
):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _RunResult()

    dummy = SimpleNamespace(
        fps=30.0,
        media_path=str(tmp_path / "input.mkv"),
        _validate_audio_file=lambda _: False,
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
    assert first_call[first_call.index("-c:a") + 1] == "copy"
    assert first_call[first_call.index("-map") + 1] == "0:a:0?"

    assert retry_call[-1].endswith(".m4a")
    assert retry_call[retry_call.index("-c:a") + 1] == "aac"
    assert retry_call[retry_call.index("-af") + 1] == "aresample=async=1:first_pts=0"


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
