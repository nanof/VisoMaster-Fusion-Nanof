"""UI entry points for H.264 transcode (single file + batch AV1 folder)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from app.helpers import miscellaneous as misc_helpers
from app.helpers import video_transcode as vt
from app.ui.widgets.transcode_options_dialog import TranscodeOptionsDialog
from app.ui.widgets.transcode_worker import H264TranscodeWorker

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def _worker_attr(main_window: MainWindow) -> str:
    return "_h264_transcode_worker"


def _active_worker(main_window: MainWindow) -> H264TranscodeWorker | None:
    w = getattr(main_window, _worker_attr(main_window), None)
    if w is not None and w.isRunning():
        return w
    return None


def _run_transcode_worker(
    main_window: MainWindow,
    paths: List[str],
    *,
    batch_av1: bool,
    encode_options: Optional[
        Tuple[Optional[int], bool, bool, Optional[float]]
    ] = None,
) -> None:
    if _active_worker(main_window):
        QtWidgets.QMessageBox.information(
            main_window,
            "Transcode in progress",
            "An H.264 conversion is already running.",
        )
        return

    if encode_options is None:
        dlg = TranscodeOptionsDialog(main_window, batch_av1=batch_av1)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        max_height, prefer_nvenc, _recursive, overwrite, target_fps = dlg.options()
    else:
        max_height, prefer_nvenc, overwrite, target_fps = encode_options

    prog = QtWidgets.QProgressDialog(main_window)
    prog.setWindowTitle("Converting to H.264")
    prog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    prog.setMinimumDuration(0)
    prog.setRange(0, 1000)
    prog.setValue(0)
    prog.setAutoClose(True)
    prog.setAutoReset(True)
    prog.setCancelButtonText("Cancel")

    worker = H264TranscodeWorker(
        paths,
        max_height,
        prefer_nvenc,
        overwrite=overwrite,
        target_fps=target_fps,
        parent=main_window,
    )
    setattr(main_window, _worker_attr(main_window), worker)

    def on_progress(idx: int, total: int, path: str, frac: float) -> None:
        tp = max(1, int(total))
        overall = (float(idx) + max(0.0, min(1.0, frac))) / float(tp)
        prog.setValue(min(1000, int(overall * 1000)))
        prog.setLabelText(os.path.basename(path))

    def on_ok(count: int, last_output: str) -> None:
        prog.close()
        msg = f"Finished: {count} file(s)."
        if (
            count == 1
            and paths
            and last_output
            and os.path.normpath(last_output)
            != os.path.normpath(os.path.abspath(paths[0]))
        ):
            msg += f"\n\nSaved as:\n{last_output}"
        QtWidgets.QMessageBox.information(main_window, "Transcode", msg)

    def on_fail(msg: str) -> None:
        prog.close()
        low = msg.lower()
        if "cancel" in low:
            return
        QtWidgets.QMessageBox.critical(main_window, "Transcode error", msg)

    qc = QtCore.Qt.ConnectionType.QueuedConnection
    worker.progress.connect(on_progress, qc)
    worker.finished_ok.connect(on_ok, qc)
    worker.failed.connect(on_fail, qc)
    prog.canceled.connect(worker.cancel)
    worker.finished.connect(
        lambda: setattr(main_window, _worker_attr(main_window), None)
    )

    prog.setValue(0)
    if paths:
        prog.setLabelText(os.path.basename(paths[0]))
    prog.show()
    QtWidgets.QApplication.processEvents()
    worker.start()


def convert_target_video_to_h264(main_window: MainWindow, media_path: str) -> None:
    if not misc_helpers.is_ffmpeg_in_path():
        QtWidgets.QMessageBox.warning(
            main_window,
            "FFmpeg",
            "FFmpeg was not found on PATH.",
        )
        return
    path = os.path.normpath(media_path)
    if not os.path.isfile(path):
        return

    vp = main_window.video_processor
    if vp.media_path and os.path.normpath(str(vp.media_path)) == path:
        QtWidgets.QMessageBox.warning(
            main_window,
            "File in use",
            "This video is loaded in the player. To avoid file locks on Windows, "
            "stop playback or switch to another video before converting.",
        )

    _run_transcode_worker(main_window, [path], batch_av1=False)


def batch_convert_av1_in_folder(main_window: MainWindow) -> None:
    if not misc_helpers.is_ffmpeg_in_path():
        QtWidgets.QMessageBox.warning(
            main_window,
            "FFmpeg",
            "FFmpeg was not found on PATH.",
        )
        return
    if not misc_helpers.cmd_exist("ffprobe"):
        QtWidgets.QMessageBox.warning(
            main_window,
            "ffprobe",
            "ffprobe was not found on PATH (it is usually installed with FFmpeg).",
        )
        return

    folder = main_window.last_target_media_folder_path or ""
    if not folder or not os.path.isdir(folder):
        picked = QtWidgets.QFileDialog.getExistingDirectory(
            main_window,
            "Folder containing AV1 videos",
            folder or QtCore.QDir.homePath(),
        )
        if not picked:
            return
        folder = picked

    recursive_default = bool(
        main_window.control.get("TargetMediaFolderRecursiveToggle", False)
    )
    dlg = TranscodeOptionsDialog(
        main_window,
        batch_av1=True,
        recursive_default=recursive_default,
    )
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return

    max_height, prefer_nvenc, recursive, _overwrite, target_fps = dlg.options()
    candidates = vt.iter_candidate_video_paths(folder, recursive)
    av1_list = vt.filter_av1_paths(candidates)
    if not av1_list:
        QtWidgets.QMessageBox.information(
            main_window,
            "AV1 → H.264",
            "No AV1-encoded videos were found for the selected folder "
            + ("(including subfolders)." if recursive else "."),
        )
        return

    confirm = QtWidgets.QMessageBox.question(
        main_window,
        "Confirm replace",
        f"{len(av1_list)} AV1 file(s) will be converted to H.264 and originals will be "
        f"replaced after a successful transcode.\n\nContinue?",
        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.No,
    )
    if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
        return

    _run_transcode_worker(
        main_window,
        av1_list,
        batch_av1=True,
        encode_options=(max_height, prefer_nvenc, True, target_fps),
    )
