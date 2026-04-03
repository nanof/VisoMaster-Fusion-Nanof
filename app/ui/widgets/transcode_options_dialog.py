"""Dialog for H.264 transcode options (resize + NVENC + batch subfolders)."""

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

        self._gpu_check = QtWidgets.QCheckBox("Use GPU (NVIDIA NVENC) when available")
        self._gpu_check.setChecked(True)
        layout.addWidget(self._gpu_check)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> Tuple[Optional[int], bool, bool]:
        """Returns (max_height or None, prefer_nvenc, include_subfolders for batch)."""
        mh: Optional[int] = None
        if self._resize_check.isChecked():
            mh = int(self._height_spin.value())
        rec = (
            bool(self._recursive_check.isChecked())
            if self._recursive_check is not None
            else False
        )
        return mh, self._gpu_check.isChecked(), rec
