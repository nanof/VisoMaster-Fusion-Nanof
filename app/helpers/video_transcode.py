"""
FFmpeg helpers for transcoding video to H.264 (NVENC or libx264) with safe in-place replace.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from typing import Callable, Iterable, List, Optional

from app.helpers import miscellaneous as misc_helpers

_ffprobe_timeout_sec = 60

_nvenc_h264_cache: Optional[bool] = None


def _subprocess_kwargs() -> dict:
    kw: dict = {}
    if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return kw


def ffmpeg_h264_nvenc_available() -> bool:
    global _nvenc_h264_cache
    if _nvenc_h264_cache is not None:
        return _nvenc_h264_cache
    if not misc_helpers.cmd_exist("ffmpeg"):
        _nvenc_h264_cache = False
        return False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            **_subprocess_kwargs(),
        )
        out = (r.stdout or "") + (r.stderr or "")
        _nvenc_h264_cache = "h264_nvenc" in out
        return bool(_nvenc_h264_cache)
    except (OSError, subprocess.TimeoutExpired):
        _nvenc_h264_cache = False
        return False


def ffprobe_video_codec_name(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path) or not misc_helpers.cmd_exist("ffprobe"):
        return None
    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        os.path.normpath(path),
    ]
    try:
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_ffprobe_timeout_sec,
            check=False,
            **_subprocess_kwargs(),
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        data = json.loads(r.stdout)
        streams = data.get("streams") or []
        if not streams:
            return None
        name = streams[0].get("codec_name")
        return str(name).lower() if name else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def iter_candidate_video_paths(folder: str, recursive: bool) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    paths = misc_helpers.get_video_files(folder, include_subfolders=recursive)
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


def filter_av1_paths(paths: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        c = ffprobe_video_codec_name(p)
        if c == "av1":
            out.append(p)
    return out


def ffprobe_video_duration_sec(path: str) -> Optional[float]:
    """Best-effort duration in seconds (format or first video stream)."""
    if not path or not os.path.isfile(path) or not misc_helpers.cmd_exist("ffprobe"):
        return None
    np = os.path.normpath(path)

    def _probe(args: List[str]) -> Optional[float]:
        try:
            r = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=_ffprobe_timeout_sec,
                check=False,
                **_subprocess_kwargs(),
            )
            if r.returncode != 0:
                return None
            s = (r.stdout or "").strip()
            if not s or s == "N/A":
                return None
            d = float(s)
            return d if d > 0.1 else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None

    d0 = _probe(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            np,
        ]
    )
    if d0 is not None:
        return d0
    return _probe(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            np,
        ]
    )


_TIME_HMS_RE = re.compile(
    r"time=(\d+):(\d+):(\d+(?:\.\d+)?)\b"
)
_TIME_SEC_RE = re.compile(r"time=(\d+(?:\.\d+)?)\s")


def _parse_ffmpeg_time_seconds(line: str) -> Optional[float]:
    m = _TIME_HMS_RE.search(line)
    if m:
        h, mi, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mi * 60 + sec
    m = _TIME_SEC_RE.search(line)
    if m:
        return float(m.group(1))
    return None


def _video_filter_for_scale(max_height: Optional[int]) -> str:
    # Even dimensions for H.264; optional cap on height (preserve aspect).
    even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if max_height is None or max_height <= 0:
        return even
    mh = max(64, min(int(max_height), 4320))
    return (
        f"scale=-2:min(ih\\,{mh}):force_original_aspect_ratio=decrease,"
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )


def build_ffmpeg_h264_command(
    input_path: str,
    output_path: str,
    *,
    max_height: Optional[int],
    prefer_nvenc: bool,
    use_aac_audio: bool,
) -> List[str]:
    vf = _video_filter_for_scale(max_height)
    use_nvenc = bool(prefer_nvenc and ffmpeg_h264_nvenc_available())

    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        # info: stderr gets periodic encoding lines with time=… for progress parsing
        "-loglevel",
        "info",
        "-stats_period",
        "0.5",
        "-nostdin",
        "-y",
        "-i",
        os.path.normpath(input_path),
        "-map",
        "0:v:0",
    ]
    # Optional audio (skip if missing)
    cmd.extend(["-map", "0:a?"])

    if use_nvenc:
        cmd.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p5",
                "-profile:v",
                "high",
                "-cq",
                "21",
                "-pix_fmt",
                "yuv420p",
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
            ]
        )

    cmd.extend(["-vf", vf])

    if use_aac_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.extend(["-c:a", "copy"])

    cmd.append(os.path.normpath(output_path))
    return cmd


def _unique_temp_path(original: str) -> str:
    d = os.path.dirname(os.path.abspath(original)) or "."
    base_name = os.path.basename(original)
    root, ext = os.path.splitext(base_name)
    ext = ext if ext else ".mp4"
    return os.path.join(d, f".{root}{ext}.vmf_tmp.{uuid.uuid4().hex}{ext}")


def _run_ffmpeg_with_stderr_progress(
    cmd: List[str],
    *,
    process_holder: Optional[List],
    cancel_event: Optional[threading.Event],
    duration_sec: Optional[float],
    progress_callback: Optional[Callable[[float], None]],
) -> tuple[int, str]:
    """
    Run FFmpeg, drain stderr on a side thread (avoids pipe deadlock), parse ``time=``
    for optional ``progress_callback`` in [0, 1].
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        **_subprocess_kwargs(),
    )
    if process_holder is not None:
        process_holder.clear()
        process_holder.append(proc)

    stderr_lines: List[str] = []
    dur = float(duration_sec) if duration_sec and duration_sec > 0.01 else 0.0

    def _reader() -> None:
        assert proc.stderr is not None
        for line in iter(proc.stderr.readline, ""):
            stderr_lines.append(line)
            if cancel_event and cancel_event.is_set():
                break
            if progress_callback and dur > 0:
                t = _parse_ffmpeg_time_seconds(line)
                if t is not None:
                    progress_callback(min(1.0, max(0.0, t / dur)))

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    try:
        rc = proc.wait()
    finally:
        reader.join(timeout=5.0)
        if process_holder is not None:
            process_holder.clear()

    return int(rc), "".join(stderr_lines)


def transcode_replace_in_place(
    input_path: str,
    *,
    max_height: Optional[int],
    prefer_nvenc: bool,
    process_holder: Optional[List] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> tuple[bool, str]:
    """
    Transcode to H.264 MP4 into a temp file, then replace ``input_path``.

    ``process_holder`` may be a single-element list; the running Popen is stored
    in ``process_holder[0]`` so the caller can kill it on cancel.

    Returns (ok, message).
    """
    if not misc_helpers.cmd_exist("ffmpeg"):
        return False, "FFmpeg was not found on PATH."
    path = os.path.normpath(os.path.abspath(input_path))
    if not os.path.isfile(path):
        return False, f"File does not exist: {path}"

    temp_out = _unique_temp_path(path)
    last_stderr = ""
    replaced_ok = False
    duration_sec = ffprobe_video_duration_sec(path)

    try:
        for use_aac in (False, True):
            if cancel_event and cancel_event.is_set():
                return False, "Canceled."
            if os.path.isfile(temp_out):
                try:
                    os.remove(temp_out)
                except OSError:
                    pass

            cmd = build_ffmpeg_h264_command(
                path,
                temp_out,
                max_height=max_height,
                prefer_nvenc=prefer_nvenc,
                use_aac_audio=use_aac,
            )
            rc, stderr_full = _run_ffmpeg_with_stderr_progress(
                cmd,
                process_holder=process_holder,
                cancel_event=cancel_event,
                duration_sec=duration_sec,
                progress_callback=progress_callback,
            )
            last_stderr = stderr_full.strip()

            if cancel_event and cancel_event.is_set():
                if os.path.isfile(temp_out):
                    try:
                        os.remove(temp_out)
                    except OSError:
                        pass
                return False, "Canceled."

            if rc == 0 and os.path.isfile(temp_out):
                if progress_callback:
                    progress_callback(1.0)
                break
            if use_aac:
                if os.path.isfile(temp_out):
                    try:
                        os.remove(temp_out)
                    except OSError:
                        pass
                err = last_stderr or f"exit code {rc}"
                return False, f"FFmpeg failed: {err}"
        else:
            return False, "FFmpeg did not produce valid output."

        try:
            os.replace(temp_out, path)
        except OSError as e:
            try:
                os.remove(temp_out)
            except OSError:
                pass
            return False, f"Could not replace file: {e}"
        replaced_ok = True
        return True, ""
    finally:
        if not replaced_ok and temp_out and os.path.isfile(temp_out):
            try:
                os.remove(temp_out)
            except OSError:
                pass
