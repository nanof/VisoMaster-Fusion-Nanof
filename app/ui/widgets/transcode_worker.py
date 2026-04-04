"""Background worker for H.264 transcode (in-place or sibling output)."""

from __future__ import annotations

import threading
from typing import List, Optional

from PySide6 import QtCore as qtc

from app.helpers import video_transcode as vt


class Av1ScanWorker(qtc.QThread):
    """Enumerate AV1 files in a folder without blocking the GUI thread."""

    found = qtc.Signal(list)
    failed = qtc.Signal(str)

    def __init__(self, folder: str, recursive: bool, parent=None):
        super().__init__(parent)
        self._folder = folder
        self._recursive = recursive

    def run(self) -> None:
        try:
            candidates = vt.iter_candidate_video_paths(self._folder, self._recursive)
            av1_list = vt.filter_av1_paths(candidates)
            self.found.emit(av1_list)
        except Exception as e:  # noqa: BLE001 — surface probe errors in UI
            self.failed.emit(str(e))


class H264TranscodeWorker(qtc.QThread):
    # file_index 0-based, total_files, current path, fraction within current file [0,1]
    progress = qtc.Signal(int, int, str, float)
    finished_ok = qtc.Signal(int, str)  # count, last path or ""
    failed = qtc.Signal(str)

    def __init__(
        self,
        paths: List[str],
        max_height: Optional[int],
        prefer_nvenc: bool,
        *,
        overwrite: bool = True,
        target_fps: Optional[float] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._paths = list(paths)
        self._max_height = max_height
        self._prefer_nvenc = prefer_nvenc
        self._overwrite = overwrite
        self._target_fps = target_fps
        self._cancel_event = threading.Event()
        self._proc_holder: list = []

    def cancel(self) -> None:
        self._cancel_event.set()
        self._kill_active()

    def _kill_active(self) -> None:
        if not self._proc_holder:
            return
        proc = self._proc_holder[0]
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
        except (OSError, AttributeError):
            pass

    def run(self) -> None:
        total = len(self._paths)
        done = 0
        last_output_path = ""
        for i, p in enumerate(self._paths):
            if self._cancel_event.is_set():
                self.failed.emit("Canceled.")
                return
            self.progress.emit(i, total, p, 0.0)

            def _file_progress(fr: float, idx: int = i) -> None:
                self.progress.emit(idx, total, p, fr)

            self._proc_holder.clear()
            ok, msg = vt.transcode_replace_in_place(
                p,
                max_height=self._max_height,
                target_fps=self._target_fps,
                prefer_nvenc=self._prefer_nvenc,
                overwrite=self._overwrite,
                process_holder=self._proc_holder,
                cancel_event=self._cancel_event,
                progress_callback=_file_progress,
            )
            self._proc_holder.clear()
            if self._cancel_event.is_set():
                self.failed.emit("Canceled.")
                return
            if not ok:
                self.failed.emit(f"{p}\n{msg}")
                return
            last_output_path = msg
            done += 1
            self.progress.emit(i, total, p, 1.0)
        self.finished_ok.emit(done, last_output_path)
