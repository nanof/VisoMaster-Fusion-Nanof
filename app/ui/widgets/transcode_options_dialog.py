"""Dialog for H.264 transcode options (resize, FPS, NVENC, overwrite, batch)."""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6 import QtWidgets


class TranscodeOptionsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        batch_av1: bool = False,
        recursive_default: bool = False,
    ):
        super().__init__(parent)
        self._batch_av1 = batch_av1
        self.setWindowTitle(
            "Batch AV1 folder → H.264" if batch_av1 else "Convert to H.264"
        )
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        if batch_av1:
            warn = QtWidgets.QLabel(
                "All AV1 videos matching your search will be transcoded and "
                "<b>original files will be replaced</b> (no backup). "
                "Close any of those files in other apps (and preferably unload them here) before continuing."
            )
            warn.setWordWrap(True)
            layout.addWidget(warn)

            self._recursive_check = QtWidgets.QCheckBox("Include subfolders")
            self._recursive_check.setChecked(recursive_default)
            layout.addWidget(self._recursive_check)
        else:
            self._recursive_check = None

        self._resize_check = QtWidgets.QCheckBox("Limit max height (keep aspect ratio)")
        self._resize_check.setChecked(False)
        layout.addWidget(self._resize_check)

        height_row = QtWidgets.QHBoxLayout()
        height_row.addWidget(QtWidgets.QLabel("Max height (px):"))
        self._height_spin = QtWidgets.QSpinBox()
        self._height_spin.setRange(256, 4320)
        self._height_spin.setSingleStep(16)
        self._height_spin.setValue(1080)
        self._height_spin.setEnabled(False)
        height_row.addWidget(self._height_spin)
        height_row.addStretch()
        layout.addLayout(height_row)
        self._resize_check.toggled.connect(self._height_spin.setEnabled)

        self._fps_check = QtWidgets.QCheckBox("Set output frame rate (constant FPS)")
        self._fps_check.setChecked(False)
        layout.addWidget(self._fps_check)

        fps_row = QtWidgets.QHBoxLayout()
        fps_row.addWidget(QtWidgets.QLabel("FPS:"))
        self._fps_spin = QtWidgets.QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 240.0)
        self._fps_spin.setDecimals(3)
        self._fps_spin.setSingleStep(1.0)
        self._fps_spin.setValue(30.0)
        self._fps_spin.setEnabled(False)
        fps_row.addWidget(self._fps_spin)
        fps_row.addStretch()
        layout.addLayout(fps_row)
        self._fps_check.toggled.connect(self._fps_spin.setEnabled)

        self._gpu_check = QtWidgets.QCheckBox("Use GPU (NVIDIA NVENC) when available")
        self._gpu_check.setChecked(True)
        layout.addWidget(self._gpu_check)

        if batch_av1:
            self._overwrite_check = None
        else:
            self._overwrite_check = QtWidgets.QCheckBox(
                "Overwrite original file (if unchecked, saves as name_h264.ext next to it)"
            )
            self._overwrite_check.setChecked(True)
            layout.addWidget(self._overwrite_check)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> Tuple[Optional[int], bool, bool, bool, Optional[float]]:
        """Returns (max_height, prefer_nvenc, recursive, overwrite, target_fps or None)."""
        mh: Optional[int] = None
        if self._resize_check.isChecked():
            mh = int(self._height_spin.value())
        rec = (
            bool(self._recursive_check.isChecked())
            if self._recursive_check is not None
            else False
        )
        overwrite = (
            bool(self._overwrite_check.isChecked())
            if self._overwrite_check is not None
            else True
        )
        target_fps: Optional[float] = None
        if self._fps_check.isChecked():
            target_fps = float(self._fps_spin.value())
        return mh, self._gpu_check.isChecked(), rec, overwrite, target_fps
