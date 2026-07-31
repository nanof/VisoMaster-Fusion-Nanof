from __future__ import annotations

from collections import deque
from functools import partial
from typing import TYPE_CHECKING, Dict, Type
from pathlib import Path
import sys
import os
import uuid
import subprocess
import time
import traceback
import faulthandler

import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore

from app.helpers.app_metadata import AppDisplayMetadata, get_app_display_metadata
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets import widget_components
import app.helpers.miscellaneous as misc_helpers
from app.helpers import input_face_favorites_storage
from app.ui.widgets import ui_workers
from app.helpers.screen_capture import SCREEN_CAPTURE_MEDIA_LABEL, mss_available

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_WORKER_STOP_TIMEOUT_MS = 1000
_TARGET_BUTTON_SIZE = (90, 90)
_SMALL_FACE_BUTTON_SIZE = (70, 70)
_LARGE_FACE_BUTTON_SIZE = (96, 96)
_FACE_BUTTON_SIZE = _SMALL_FACE_BUTTON_SIZE
_EMBED_BUTTON_SIZE = (120, 25)
_EMBED_LIST_HEIGHT = 140
_TARGET_MEDIA_BATCH_SIZE = 24
_TARGET_MEDIA_BATCH_INTERVAL_MS = 1
_THUMB_ZOOM_MIN = 0.5
_THUMB_ZOOM_MAX = 3.0
TARGET_MEDIA_SORT_NAME = "name"
TARGET_MEDIA_SORT_DATE = "date"
TARGET_MEDIA_SORT_SIZE = "size"
TARGET_MEDIA_SORT_DIMENSIONS = "dimensions"
TARGET_MEDIA_SORT_PIXELS = "pixels"
TARGET_MEDIA_SORT_FRAMES = "frames"
TARGET_MEDIA_SORT_DEFAULT = TARGET_MEDIA_SORT_NAME
TARGET_MEDIA_SORT_MODES = (
    TARGET_MEDIA_SORT_NAME,
    TARGET_MEDIA_SORT_DATE,
    TARGET_MEDIA_SORT_SIZE,
    TARGET_MEDIA_SORT_DIMENSIONS,
    TARGET_MEDIA_SORT_PIXELS,
    TARGET_MEDIA_SORT_FRAMES,
)
_TARGET_SORT_LOG_PROGRESS_EVERY = 25


def _target_sort_debug_enabled() -> bool:
    return os.environ.get("VISIOMASTER_TARGET_SORT_DEBUG", "").strip() in (
        "1",
        "true",
        "yes",
    )


def _target_sort_log(phase: str, **details) -> None:
    if not details and not _target_sort_debug_enabled():
        return
    parts = " ".join(f"{k}={v}" for k, v in details.items())
    message = f"[TARGET-SORT] {phase}"
    if parts:
        message = f"{message} {parts}"
    print(message, flush=True)


def get_target_media_sort_mode(main_window: "MainWindow") -> str:
    combo = getattr(main_window, "targetMediaSortComboBox", None)
    if combo is None:
        return TARGET_MEDIA_SORT_DEFAULT
    mode = combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
    if mode in TARGET_MEDIA_SORT_MODES:
        return str(mode)
    return TARGET_MEDIA_SORT_DEFAULT


def get_target_media_sort_descending(main_window: "MainWindow") -> bool:
    button = getattr(main_window, "targetMediaSortDirectionButton", None)
    if button is None:
        return False
    return bool(button.isChecked())


def target_media_path_sort_key(media_path: str, mode: str) -> tuple:
    return misc_helpers.target_media_path_sort_key(media_path, mode)


def _target_media_button_sort_key(button, mode: str) -> tuple:
    if getattr(button, "is_webcam", False) or getattr(button, "file_type", None) == "webcam":
        return (2, getattr(button, "webcam_index", 0), str(button.media_path).lower())
    if getattr(button, "is_screen_capture", False) or getattr(
        button, "file_type", None
    ) == "screen":
        return (2, 999, str(button.media_path).lower())

    path = str(button.media_path)
    metadata = getattr(button, "_media_metadata", None)
    if mode == TARGET_MEDIA_SORT_DATE:
        if not getattr(button, "_file_stats_loaded", False):
            misc_helpers.refresh_target_media_file_stats(button)
        return (0, float(getattr(button, "_file_mtime", 0.0) or 0.0), path.lower())
    if mode == TARGET_MEDIA_SORT_SIZE:
        if not getattr(button, "_file_stats_loaded", False):
            misc_helpers.refresh_target_media_file_stats(button)
        return (0, int(getattr(button, "_file_size", 0) or 0), path.lower())
    if mode in (
        TARGET_MEDIA_SORT_DIMENSIONS,
        TARGET_MEDIA_SORT_PIXELS,
        TARGET_MEDIA_SORT_FRAMES,
    ):
        if metadata is None and getattr(button, "file_type", None) in ("image", "video"):
            metadata = misc_helpers.probe_media_metadata(path, button.file_type)
            button._media_metadata = metadata
        return misc_helpers.target_media_path_sort_key(path, mode, metadata)
    return (0, os.path.basename(path).lower())


def sort_target_media_list(main_window: "MainWindow") -> None:
    if getattr(main_window, "_target_media_sort_in_progress", False):
        _target_sort_log("skip_reentrant")
        return

    list_widget = main_window.targetVideosList
    count = list_widget.count()
    if count <= 1:
        _target_sort_log("skip_small_list", count=count)
        return

    mode = get_target_media_sort_mode(main_window)
    descending = get_target_media_sort_descending(main_window)
    started = time.perf_counter()
    _target_sort_log("start", mode=mode, descending=descending, count=count)

    if _target_sort_debug_enabled():
        faulthandler.enable()

    entries: list[tuple[tuple, QtWidgets.QListWidgetItem, QtWidgets.QWidget]] = []
    missing_widgets = 0
    try:
        phase_started = time.perf_counter()
        if mode in (TARGET_MEDIA_SORT_DATE, TARGET_MEDIA_SORT_SIZE):
            for i in range(count):
                item = list_widget.item(i)
                if item is None:
                    continue
                button = list_widget.itemWidget(item)
                if button is not None:
                    misc_helpers.refresh_target_media_file_stats(button)

        for i in range(count):
            item = list_widget.item(i)
            if item is None:
                missing_widgets += 1
                _target_sort_log("missing_item", index=i)
                continue
            button = list_widget.itemWidget(item)
            if button is None:
                missing_widgets += 1
                if _target_sort_debug_enabled():
                    _target_sort_log("missing_widget", index=i)
                continue
            entries.append((_target_media_button_sort_key(button, mode), item, button))

        _target_sort_log(
            "keys_collected",
            entries=len(entries),
            missing_widgets=missing_widgets,
            elapsed_ms=int((time.perf_counter() - phase_started) * 1000),
        )
        if mode == TARGET_MEDIA_SORT_DATE and entries:
            unique_mtimes = len({entry[0][1] for entry in entries})
            _target_sort_log("date_stats", unique_mtimes=unique_mtimes)
        elif mode == TARGET_MEDIA_SORT_SIZE and entries:
            unique_sizes = len({entry[0][1] for entry in entries})
            _target_sort_log("size_stats", unique_sizes=unique_sizes)

        if len(entries) <= 1:
            _target_sort_log("skip_few_entries", entries=len(entries))
            return

        phase_started = time.perf_counter()
        entries.sort(key=lambda entry: entry[0])
        if descending:
            # Keep webcam/screen sinks (group 2) at the bottom.
            media_entries = [entry for entry in entries if entry[0][0] == 0]
            other_entries = [entry for entry in entries if entry[0][0] != 0]
            media_entries.reverse()
            entries = media_entries + other_entries
        _target_sort_log(
            "sorted",
            entries=len(entries),
            elapsed_ms=int((time.perf_counter() - phase_started) * 1000),
        )

        sorted_items = [item for _, item, _ in entries]

        main_window._target_media_sort_in_progress = True
        list_widget.setUpdatesEnabled(False)
        try:
            phase_started = time.perf_counter()
            moves = 0
            failed_moves = 0
            model = list_widget.model()
            root = QtCore.QModelIndex()
            # Reorder through the model. takeItem()/insertItem() removes the row,
            # which makes the view release (deleteLater) the item widget; re-setting
            # that widget afterwards leaves the view with a dangling pointer and the
            # process dies with an access violation on the next event loop pass.
            for target_index, item in enumerate(sorted_items):
                current_index = list_widget.row(item)
                if current_index == target_index:
                    continue
                if current_index < 0:
                    _target_sort_log(
                        "warn_item_not_in_list",
                        target_index=target_index,
                    )
                    continue

                if not model.moveRow(root, current_index, root, target_index):
                    failed_moves += 1
                    _target_sort_log(
                        "warn_move_failed",
                        target_index=target_index,
                        current_index=current_index,
                    )
                    continue
                moves += 1

                if (
                    _target_sort_debug_enabled()
                    and moves % _TARGET_SORT_LOG_PROGRESS_EVERY == 0
                ):
                    _target_sort_log(
                        "move_progress",
                        moves=moves,
                        target_index=target_index,
                    )

            _target_sort_log(
                "reordered",
                moves=moves,
                failed_moves=failed_moves,
                elapsed_ms=int((time.perf_counter() - phase_started) * 1000),
            )
        finally:
            list_widget.setUpdatesEnabled(True)
            list_widget.viewport().update()
            main_window._target_media_sort_in_progress = False

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _target_sort_log("done", mode=mode, count=count, elapsed_ms=elapsed_ms)
    except Exception:
        _target_sort_log(
            "error",
            mode=mode,
            count=count,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        traceback.print_exc()
        main_window._target_media_sort_in_progress = False
        raise


def on_target_media_sort_changed(main_window: "MainWindow", *args) -> None:
    combo_index = args[0] if args else "?"
    _target_sort_log(
        "combo_changed",
        index=combo_index,
        mode=get_target_media_sort_mode(main_window),
        descending=get_target_media_sort_descending(main_window),
    )
    if getattr(main_window, "_target_media_sort_in_progress", False):
        _target_sort_log("skip_combo_reentrant", index=combo_index)
        return
    try:
        sort_target_media_list(main_window)
    except Exception:
        _target_sort_log("combo_handler_failed", index=combo_index)
        raise


def initialize_target_media_sort_combo(main_window: "MainWindow") -> None:
    combo = getattr(main_window, "targetMediaSortComboBox", None)
    if combo is None:
        return
    # Re-populate when older workspaces only had name/date/size.
    expected_modes = set(TARGET_MEDIA_SORT_MODES)
    existing_modes = {
        combo.itemData(i, QtCore.Qt.ItemDataRole.UserRole)
        for i in range(combo.count())
    }
    if combo.count() > 0 and expected_modes.issubset(existing_modes):
        return
    current_mode = get_target_media_sort_mode(main_window)
    combo.blockSignals(True)
    try:
        combo.clear()
        for label, mode in (
            ("Name", TARGET_MEDIA_SORT_NAME),
            ("Date", TARGET_MEDIA_SORT_DATE),
            ("Size", TARGET_MEDIA_SORT_SIZE),
            ("Dimensions", TARGET_MEDIA_SORT_DIMENSIONS),
            ("Pixels", TARGET_MEDIA_SORT_PIXELS),
            ("Frames", TARGET_MEDIA_SORT_FRAMES),
        ):
            combo.addItem(label)
            combo.setItemData(
                combo.count() - 1,
                mode,
                QtCore.Qt.ItemDataRole.UserRole,
            )
        for index in range(combo.count()):
            if (
                combo.itemData(index, QtCore.Qt.ItemDataRole.UserRole)
                == current_mode
            ):
                combo.setCurrentIndex(index)
                break
        else:
            combo.setCurrentIndex(0)
    finally:
        combo.blockSignals(False)


def set_target_media_sort_mode(main_window: "MainWindow", mode: str) -> None:
    combo = getattr(main_window, "targetMediaSortComboBox", None)
    if combo is None:
        return
    initialize_target_media_sort_combo(main_window)
    if mode not in TARGET_MEDIA_SORT_MODES:
        mode = TARGET_MEDIA_SORT_DEFAULT
    for index in range(combo.count()):
        if (
            combo.itemData(index, QtCore.Qt.ItemDataRole.UserRole)
            == mode
        ):
            combo.blockSignals(True)
            try:
                combo.setCurrentIndex(index)
            finally:
                combo.blockSignals(False)
            return


def set_target_media_sort_descending(main_window: "MainWindow", descending: bool) -> None:
    button = getattr(main_window, "targetMediaSortDirectionButton", None)
    if button is None:
        return
    button.blockSignals(True)
    try:
        button.setChecked(bool(descending))
        button.setText("↓" if descending else "↑")
        button.setToolTip(
            "Sort descending" if descending else "Sort ascending"
        )
    finally:
        button.blockSignals(False)


def on_target_media_sort_direction_changed(main_window: "MainWindow", *_args) -> None:
    button = getattr(main_window, "targetMediaSortDirectionButton", None)
    if button is not None:
        set_target_media_sort_descending(main_window, button.isChecked())
    on_target_media_sort_changed(main_window)


def initialize_target_media_min_dimension_spinboxes(main_window: "MainWindow") -> None:
    for name in ("targetMediaMinWidthSpinBox", "targetMediaMinHeightSpinBox"):
        spin = getattr(main_window, name, None)
        if spin is None:
            continue
        spin.blockSignals(True)
        try:
            spin.setRange(0, 8192)
            spin.setSingleStep(16)
            if spin.value() < 0:
                spin.setValue(0)
            if name.endswith("WidthSpinBox"):
                spin.setToolTip("Minimum media width (0 = no filter)")
                spin.setPrefix("W ")
            else:
                spin.setToolTip("Minimum media height (0 = no filter)")
                spin.setPrefix("H ")
        finally:
            spin.blockSignals(False)


def set_target_media_min_dimensions(
    main_window: "MainWindow", min_width: int = 0, min_height: int = 0
) -> None:
    initialize_target_media_min_dimension_spinboxes(main_window)
    for name, value in (
        ("targetMediaMinWidthSpinBox", min_width),
        ("targetMediaMinHeightSpinBox", min_height),
    ):
        spin = getattr(main_window, name, None)
        if spin is None:
            continue
        spin.blockSignals(True)
        try:
            spin.setValue(max(0, int(value)))
        finally:
            spin.blockSignals(False)


def clear_target_videos_search(main_window: "MainWindow", *_args) -> None:
    search = getattr(main_window, "targetVideosSearchBox", None)
    if search is None:
        return
    search.clear()


def thumbnail_size_for_zoom(base_size: tuple[int, int], zoom: float) -> QtCore.QSize:
    bw, bh = base_size
    z = max(_THUMB_ZOOM_MIN, min(_THUMB_ZOOM_MAX, zoom))
    return QtCore.QSize(max(24, int(round(bw * z))), max(24, int(round(bh * z))))


def _update_list_grid_for_thumbnail_size(
    list_widget: QtWidgets.QListWidget, button_size: QtCore.QSize
) -> None:
    list_widget.setGridSize(button_size + QtCore.QSize(4, 4))


def _apply_scaled_list_thumbnail_icon(
    button: QtWidgets.QAbstractButton, icon_size: QtCore.QSize
) -> None:
    """Scale pixmap to icon_size so the image grows with zoom (QIcon+setIconSize alone can cap size)."""
    base = getattr(button, "_thumbnail_base_pixmap", None)
    if base is None or base.isNull():
        button.setIconSize(icon_size)
        return
    scaled = base.scaled(
        icon_size.width(),
        icon_size.height(),
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation,
    )
    button.setIcon(QtGui.QIcon(scaled))
    button.setIconSize(icon_size)


def refresh_thumbnail_sizes_for_list(
    main_window: "MainWindow", list_widget: QtWidgets.QListWidget
) -> None:
    if list_widget == main_window.targetVideosList:
        zoom = main_window.target_videos_thumbnail_zoom
        base = _TARGET_BUTTON_SIZE
        buttons = main_window.target_videos
    elif list_widget in (
        main_window.inputFacesList,
        main_window.inputFacesFavoritesList,
    ):
        zoom = main_window.input_faces_thumbnail_zoom
        base = _FACE_BUTTON_SIZE
        buttons = main_window.input_faces
    else:
        return
    button_size = thumbnail_size_for_zoom(base, zoom)
    icon_size = button_size - QtCore.QSize(8, 8)
    for btn in buttons.values():
        btn.setFixedSize(button_size)
        _apply_scaled_list_thumbnail_icon(btn, icon_size)
        li = getattr(btn, "list_item", None)
        if li is not None:
            li.setSizeHint(button_size)
    _update_list_grid_for_thumbnail_size(list_widget, button_size)


def apply_wheel_zoom_to_thumbnail_list(
    main_window: "MainWindow", list_widget: QtWidgets.QListWidget, delta_y: int
) -> bool:
    """Ctrl+wheel zoom for target videos or input faces lists. Returns True if handled."""
    if list_widget == main_window.targetVideosList:
        attr = "target_videos_thumbnail_zoom"
    elif list_widget in (
        main_window.inputFacesList,
        main_window.inputFacesFavoritesList,
    ):
        attr = "input_faces_thumbnail_zoom"
    else:
        return False
    if delta_y == 0:
        return True
    current = getattr(main_window, attr, 1.0)
    new_zoom = current * (1.1 if delta_y > 0 else 1.0 / 1.1)
    new_zoom = max(_THUMB_ZOOM_MIN, min(_THUMB_ZOOM_MAX, new_zoom))
    setattr(main_window, attr, new_zoom)
    refresh_thumbnail_sizes_for_list(main_window, list_widget)
    return True


def apply_face_thumbnail_size(
    main_window: "MainWindow", button_size_tuple: tuple[int, int]
) -> None:
    """Fixed presets (small/large) for face lists; keeps zoom state in sync for input faces."""
    main_window.face_thumbnail_button_size = button_size_tuple
    tw = button_size_tuple[0]
    bw = _FACE_BUTTON_SIZE[0]
    z = max(_THUMB_ZOOM_MIN, min(_THUMB_ZOOM_MAX, tw / float(bw)))
    main_window.input_faces_thumbnail_zoom = z
    refresh_thumbnail_sizes_for_list(main_window, main_window.inputFacesList)
    refresh_thumbnail_sizes_for_list(main_window, main_window.inputFacesFavoritesList)

    button_size = QtCore.QSize(*button_size_tuple)
    grid_size_with_padding = button_size + QtCore.QSize(4, 4)
    icon_size = button_size - QtCore.QSize(8, 8)
    for list_widget in (main_window.targetFacesList,):
        list_widget.setGridSize(grid_size_with_padding)
        for i in range(list_widget.count()):
            list_item = list_widget.item(i)
            button = list_widget.itemWidget(list_item)
            if button is None:
                continue
            button.setFixedSize(button_size)
            if getattr(button, "_thumbnail_base_pixmap", None) is not None:
                _apply_scaled_list_thumbnail_icon(button, icon_size)
            else:
                button.setIconSize(icon_size)
            list_item.setSizeHint(button_size)
        list_widget.doItemsLayout()
        list_widget.viewport().update()


def _get_target_media_batch_size(pending_count: int) -> int:
    if pending_count >= 1500:
        return 64
    if pending_count >= 900:
        return 48
    if pending_count >= 400:
        return 36
    return _TARGET_MEDIA_BATCH_SIZE


def _ensure_target_media_batch_timer(main_window: "MainWindow") -> QtCore.QTimer:
    timer = getattr(main_window, "_target_media_batch_timer", None)
    if timer is None:
        timer = QtCore.QTimer(main_window)
        timer.setSingleShot(True)
        timer.timeout.connect(partial(_flush_target_media_thumbnail_batch, main_window))
        main_window._target_media_batch_timer = timer
    return timer


def _add_target_media_progress_total(main_window: "MainWindow", n: int) -> None:
    """Grow the loading progress bar's total. GUI thread only."""
    bar = getattr(main_window, "targetVideosListProgressBar", None)
    if bar is None or n <= 0:
        return
    bar.setMaximum(bar.maximum() + n)
    bar.setVisible(True)


def _advance_target_media_progress(main_window: "MainWindow", n: int) -> None:
    """Advance the loading progress bar, hiding it once the queue has drained."""
    bar = getattr(main_window, "targetVideosListProgressBar", None)
    if bar is None:
        return
    bar.setValue(min(bar.value() + n, bar.maximum()))
    if bar.value() >= bar.maximum():
        _reset_target_media_progress(main_window)


def _reset_target_media_progress(main_window: "MainWindow") -> None:
    """Hide and zero the loading progress bar (also used when loading is cancelled)."""
    bar = getattr(main_window, "targetVideosListProgressBar", None)
    if bar is None:
        return
    bar.setVisible(False)
    bar.setMaximum(0)
    bar.reset()


def _flush_target_media_thumbnail_batch(main_window: "MainWindow") -> None:
    pending_items = getattr(main_window, "_pending_target_media_thumbnails", None)
    if not pending_items:
        return

    list_widget = main_window.targetVideosList
    pending_before = len(pending_items)
    batch_size = 0
    list_widget.setUpdatesEnabled(False)
    try:
        batch_size = min(_get_target_media_batch_size(pending_before), pending_before)
        for _ in range(batch_size):
            media_path, q_image, file_type, media_id, metadata = pending_items.popleft()
            add_media_thumbnail_button(
                main_window,
                widget_components.TargetMediaCardButton,
                list_widget,
                main_window.target_videos,
                q_image,
                media_path=media_path,
                file_type=file_type,
                media_id=media_id,
                media_metadata=metadata,
            )
    finally:
        list_widget.setUpdatesEnabled(True)
        list_widget.viewport().update()
        if batch_size:
            _advance_target_media_progress(main_window, batch_size)

    if pending_items:
        _ensure_target_media_batch_timer(main_window).start(
            _TARGET_MEDIA_BATCH_INTERVAL_MS
        )


def _queue_target_media_thumbnail(
    main_window: "MainWindow",
    media_path,
    q_image,
    file_type,
    media_id,
    media_metadata=None,
) -> None:
    pending_items = getattr(main_window, "_pending_target_media_thumbnails", None)
    if pending_items is None:
        pending_items = deque()
        main_window._pending_target_media_thumbnails = pending_items

    pending_items.append((media_path, q_image, file_type, media_id, media_metadata))
    _add_target_media_progress_total(main_window, 1)
    timer = _ensure_target_media_batch_timer(main_window)
    if not timer.isActive():
        timer.start(_TARGET_MEDIA_BATCH_INTERVAL_MS)


def _has_pending_target_media_thumbnail_work(main_window: "MainWindow") -> bool:
    timer = getattr(main_window, "_target_media_batch_timer", None)
    pending_items = getattr(main_window, "_pending_target_media_thumbnails", None)
    return bool(pending_items) or bool(timer and timer.isActive())


def _ensure_input_face_batch_timer(main_window: "MainWindow") -> QtCore.QTimer:
    timer = getattr(main_window, "_input_face_batch_timer", None)
    if timer is None:
        timer = QtCore.QTimer(main_window)
        timer.setSingleShot(True)
        timer.timeout.connect(partial(_flush_input_face_thumbnail_batch, main_window))
        main_window._input_face_batch_timer = timer
    return timer


def _flush_input_face_thumbnail_batch(main_window: "MainWindow") -> None:
    pending_items = getattr(main_window, "_pending_input_face_thumbnails", None)
    if not pending_items:
        return

    list_widget = main_window.inputFacesList
    pending_before = len(pending_items)
    list_widget.setUpdatesEnabled(False)
    try:
        batch_size = min(24, pending_before)
        for _ in range(batch_size):
            media_path, cropped_face, embedding_store, q_image, face_id = (
                pending_items.popleft()
            )
            add_media_thumbnail_button(
                main_window,
                widget_components.InputFaceCardButton,
                list_widget,
                main_window.input_faces,
                q_image,
                media_path=media_path,
                cropped_face=cropped_face,
                embedding_store=embedding_store,
                face_id=face_id,
            )
    finally:
        list_widget.setUpdatesEnabled(True)
        list_widget.viewport().update()

    if pending_items:
        _ensure_input_face_batch_timer(main_window).start(
            _TARGET_MEDIA_BATCH_INTERVAL_MS
        )


def _queue_input_face_thumbnail(
    main_window: "MainWindow",
    media_path,
    cropped_face,
    embedding_store,
    q_image,
    face_id,
) -> None:
    pending_items = getattr(main_window, "_pending_input_face_thumbnails", None)
    if pending_items is None:
        pending_items = deque()
        main_window._pending_input_face_thumbnails = pending_items

    pending_items.append((media_path, cropped_face, embedding_store, q_image, face_id))
    timer = _ensure_input_face_batch_timer(main_window)
    if not timer.isActive():
        timer.start(_TARGET_MEDIA_BATCH_INTERVAL_MS)


# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QImage, str, str, object)
def add_media_thumbnail_to_target_videos_list(
    main_window: "MainWindow",
    media_path,
    q_image,
    file_type,
    media_id,
    media_metadata=None,
):
    _queue_target_media_thumbnail(
        main_window, media_path, q_image, file_type, media_id, media_metadata
    )


# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QImage, str, str, int, int)
def add_webcam_thumbnail_to_target_videos_list(
    main_window: "MainWindow",
    media_path,
    q_image,
    file_type,
    media_id,
    webcam_index,
    webcam_backend,
):
    add_media_thumbnail_button(
        main_window,
        widget_components.TargetMediaCardButton,
        main_window.targetVideosList,
        main_window.target_videos,
        q_image,
        media_path=media_path,
        file_type=file_type,
        media_id=media_id,
        is_webcam=True,
        webcam_index=webcam_index,
        webcam_backend=webcam_backend,
    )


def scroll_target_videos_list_to_media_id(
    main_window: "MainWindow", media_id: str | bool
) -> None:
    """Scroll ``targetVideosList`` so the card for ``media_id`` is visible (e.g. after workspace restore)."""
    if not media_id or not isinstance(media_id, str):
        return
    button = main_window.target_videos.get(media_id)
    if button is None:
        return
    list_item = getattr(button, "list_item", None)
    if list_item is None:
        return
    list_w = main_window.targetVideosList
    list_w.doItemsLayout()
    list_w.scrollToItem(
        list_item,
        QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
    )


def add_screen_capture_thumbnail_to_target_videos_list(main_window: "MainWindow"):
    if not mss_available():
        print("[WARN] mss is not installed; screen capture is unavailable.")
        return
    q_image = common_widget_actions.extract_frame_as_image(
        main_window,
        SCREEN_CAPTURE_MEDIA_LABEL,
        "screen",
    )
    if not q_image:
        print("[WARN] Could not grab a screen preview for the target list.")
        return
    media_id = str(uuid.uuid1().int)
    add_media_thumbnail_button(
        main_window,
        widget_components.TargetMediaCardButton,
        main_window.targetVideosList,
        main_window.target_videos,
        q_image,
        media_path=SCREEN_CAPTURE_MEDIA_LABEL,
        file_type="screen",
        media_id=media_id,
        is_screen_capture=True,
    )


@QtCore.Slot()
def add_media_thumbnail_to_target_faces_list(
    main_window: "MainWindow", cropped_face, embedding_store, image_data, face_id
):
    add_media_thumbnail_button(
        main_window,
        widget_components.TargetFaceCardButton,
        main_window.targetFacesList,
        main_window.target_faces,
        image_data,
        cropped_face=cropped_face,
        embedding_store=embedding_store,
        face_id=face_id,
    )


@QtCore.Slot(str, object, object, QtGui.QImage, str)
def add_media_thumbnail_to_source_faces_list(
    main_window: "MainWindow",
    media_path,
    cropped_face,
    embedding_store,
    q_image,
    face_id,
):
    _queue_input_face_thumbnail(
        main_window,
        media_path,
        cropped_face,
        embedding_store,
        q_image,
        face_id,
    )


def _copy_payload_to_favorites_list(
    main_window: MainWindow,
    cropped_bgr: np.ndarray,
    embedding_store: dict,
    media_path: str,
) -> None:
    cropped_bgr = np.ascontiguousarray(cropped_bgr)
    face_id = str(uuid.uuid1().int)
    h, w = cropped_bgr.shape[:2]
    bytes_per_line = 3 * w
    q_image = QtGui.QImage(
        cropped_bgr.data,
        w,
        h,
        bytes_per_line,
        QtGui.QImage.Format.Format_BGR888,
    ).copy()

    add_media_thumbnail_button(
        main_window,
        widget_components.InputFaceCardButton,
        main_window.inputFacesFavoritesList,
        main_window.input_faces,
        q_image,
        media_path=media_path,
        cropped_face=cropped_bgr,
        embedding_store=embedding_store,
        face_id=face_id,
        is_favorite_clip=True,
    )
    input_face_favorites_storage.save_favorite(
        main_window,
        face_id,
        media_path,
        cropped_bgr,
        embedding_store,
    )
    main_window.placeholder_update_signal.emit(main_window.inputFacesFavoritesList, False)


def add_input_faces_selection_to_favorites(
    main_window: MainWindow,
    source_button: widget_components.InputFaceCardButton,
):
    """Add checked faces from the main Input Faces list to Favorites, or the clicked face if none checked."""
    main_list = main_window.inputFacesList
    candidates = [
        b
        for b in main_window.input_faces.values()
        if b.isChecked()
        and b.list_widget is main_list
        and not getattr(b, "is_favorite_clip", False)
    ]
    if not candidates:
        if (
            source_button.list_widget is main_list
            and not source_button.is_favorite_clip
        ):
            candidates = [source_button]

    if not candidates:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "No faces to add",
            "Select one or more faces in the Faces list, or right‑click a face card.",
            source_button,
        )
        return

    added = 0
    for btn in candidates:
        cropped = btn.cropped_face
        if cropped is None or getattr(cropped, "size", 0) == 0:
            continue
        embedding_store: dict = {}
        for key, val in btn.embedding_store.items():
            if isinstance(val, np.ndarray):
                embedding_store[key] = val.copy()
            else:
                embedding_store[key] = val
        mp = btn.media_path
        if not isinstance(mp, str):
            mp = str(mp)
        label = f"Favorite (Input Faces · {mp})"
        _copy_payload_to_favorites_list(main_window, cropped, embedding_store, label)
        added += 1

    if added == 0:
        common_widget_actions.create_and_show_messagebox(
            main_window,
            "Cannot add favorite",
            "The selected faces have no cropped image to save.",
            source_button,
        )


def add_media_thumbnail_button(
    main_window: "MainWindow",
    buttonClass: "Type[widget_components.CardButton]",
    listWidget: QtWidgets.QListWidget,
    buttons_list: Dict,
    image_data,  # Accepts QImage (from workers) or QPixmap (from main thread)
    **kwargs,
):
    if buttonClass == widget_components.TargetMediaCardButton:
        constructor_args = [
            kwargs.get("media_path"),
            kwargs.get("file_type"),
            kwargs.get("media_id"),
            kwargs.get("is_webcam", False),
            kwargs.get("webcam_index", -1),
            kwargs.get("webcam_backend", -1),
            kwargs.get("is_screen_capture", False),
        ]
    elif buttonClass in (
        widget_components.TargetFaceCardButton,
        widget_components.InputFaceCardButton,
    ):
        constructor_args = [
            kwargs.get("media_path", ""),
            kwargs.get("cropped_face"),
            kwargs.get("embedding_store"),
            kwargs.get("face_id"),
        ]

    if buttonClass == widget_components.TargetMediaCardButton:
        button_size = thumbnail_size_for_zoom(
            _TARGET_BUTTON_SIZE, main_window.target_videos_thumbnail_zoom
        )
    elif buttonClass == widget_components.InputFaceCardButton:
        button_size = thumbnail_size_for_zoom(
            _FACE_BUTTON_SIZE, main_window.input_faces_thumbnail_zoom
        )
    else:
        button_size = QtCore.QSize(*_FACE_BUTTON_SIZE)

    button_kw: dict = {"main_window": main_window}
    if buttonClass == widget_components.InputFaceCardButton and kwargs.get(
        "is_favorite_clip"
    ):
        button_kw["is_favorite_clip"] = True
    button: widget_components.CardButton = buttonClass(
        *constructor_args, **button_kw
    )

    # --- Main thread conversion ---
    if isinstance(image_data, QtGui.QImage):
        pixmap = QtGui.QPixmap.fromImage(image_data)
    else:
        pixmap = image_data

    icon_size = button_size - QtCore.QSize(8, 8)
    if buttonClass in (
        widget_components.TargetMediaCardButton,
        widget_components.InputFaceCardButton,
    ):
        button._thumbnail_base_pixmap = pixmap.copy()
        _apply_scaled_list_thumbnail_icon(button, icon_size)
    else:
        button.setIcon(QtGui.QIcon(pixmap))
        button.setIconSize(icon_size)
    button.setFixedSize(button_size)
    button.setCheckable(True)

    if buttonClass in [
        widget_components.TargetFaceCardButton,
        widget_components.InputFaceCardButton,
    ]:
        buttons_list[button.face_id] = button
    elif buttonClass == widget_components.TargetMediaCardButton:
        buttons_list[button.media_id] = button
        metadata = kwargs.get("media_metadata")
        button._media_metadata = metadata
        misc_helpers.refresh_target_media_file_stats(button)
    elif buttonClass == widget_components.EmbeddingCardButton:
        buttons_list[button.embedding_id] = button

    # Create a QListWidgetItem and set the button as its widget
    list_item = QtWidgets.QListWidgetItem(listWidget)
    list_item.setSizeHint(button_size)
    button.list_item = list_item
    button.list_widget = listWidget
    if buttonClass == widget_components.InputFaceCardButton:
        button.create_context_menu()
    # Align the item to center
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    listWidget.setItemWidget(list_item, button)


def initialize_media_list_widgets(main_window: "MainWindow"):
    """One-time configuration for target/input media and face list widgets."""
    if not hasattr(main_window, "face_thumbnail_button_size"):
        main_window.face_thumbnail_button_size = _FACE_BUTTON_SIZE
    for listWidget, button_size_tuple, zoom_attr in [
        (main_window.targetVideosList, _TARGET_BUTTON_SIZE, "target_videos_thumbnail_zoom"),
        (main_window.targetFacesList, _FACE_BUTTON_SIZE, None),
        (main_window.inputFacesList, _FACE_BUTTON_SIZE, "input_faces_thumbnail_zoom"),
        (
            main_window.inputFacesFavoritesList,
            _FACE_BUTTON_SIZE,
            "input_faces_thumbnail_zoom",
        ),
    ]:
        if zoom_attr is not None:
            button_size = thumbnail_size_for_zoom(
                button_size_tuple, getattr(main_window, zoom_attr, 1.0)
            )
        else:
            button_size = QtCore.QSize(*button_size_tuple)
        grid_size_with_padding = button_size + QtCore.QSize(4, 4)
        listWidget.setGridSize(grid_size_with_padding)
        listWidget.setWrapping(True)
        listWidget.setFlow(QtWidgets.QListView.LeftToRight)
        listWidget.setResizeMode(QtWidgets.QListView.Adjust)

    _set_up_panel_context_menu(
        main_window, main_window.targetVideosList, "target_media"
    )
    _set_up_panel_context_menu(main_window, main_window.inputFacesList, "input_faces")
    initialize_target_media_sort_combo(main_window)
    set_target_media_sort_descending(main_window, False)
    initialize_target_media_min_dimension_spinboxes(main_window)
    _reset_target_media_progress(main_window)


def initialize_embeddings_list_widget(main_window: "MainWindow"):
    """One-time configuration for the inputEmbeddingsList widget."""
    inputEmbeddingsList = main_window.inputEmbeddingsList
    button_size = QtCore.QSize(*_EMBED_BUTTON_SIZE)
    grid_size_with_padding = button_size + QtCore.QSize(4, 4)

    inputEmbeddingsList.setGridSize(grid_size_with_padding)
    inputEmbeddingsList.setWrapping(True)
    inputEmbeddingsList.setFlow(QtWidgets.QListView.TopToBottom)
    inputEmbeddingsList.setResizeMode(QtWidgets.QListView.Fixed)
    inputEmbeddingsList.setSpacing(2)
    inputEmbeddingsList.setUniformItemSizes(True)
    inputEmbeddingsList.setViewMode(QtWidgets.QListView.IconMode)
    inputEmbeddingsList.setMovement(QtWidgets.QListView.Static)

    inputEmbeddingsList.setFixedHeight(_EMBED_LIST_HEIGHT)

    col_width = grid_size_with_padding.width()
    min_width = (3 * col_width) + 16
    inputEmbeddingsList.setMinimumWidth(min_width)

    inputEmbeddingsList.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    inputEmbeddingsList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    inputEmbeddingsList.setVerticalScrollMode(
        QtWidgets.QAbstractItemView.ScrollPerPixel
    )
    inputEmbeddingsList.setHorizontalScrollMode(
        QtWidgets.QAbstractItemView.ScrollPerPixel
    )

    inputEmbeddingsList.setLayoutDirection(QtCore.Qt.LeftToRight)
    inputEmbeddingsList.setLayoutMode(QtWidgets.QListView.Batched)
    _set_up_panel_context_menu(main_window, inputEmbeddingsList, "embeddings")


def create_and_add_embed_button_to_list(
    main_window: "MainWindow", embedding_name, embedding_store, embedding_id
):
    inputEmbeddingsList = main_window.inputEmbeddingsList
    embed_button = widget_components.EmbeddingCardButton(
        main_window=main_window,
        embedding_name=embedding_name,
        embedding_store=embedding_store,
        embedding_id=embedding_id,
    )

    button_size = QtCore.QSize(*_EMBED_BUTTON_SIZE)
    embed_button.setFixedSize(button_size)

    list_item = QtWidgets.QListWidgetItem(inputEmbeddingsList)
    list_item.setSizeHint(button_size)
    embed_button.list_item = list_item
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    inputEmbeddingsList.setItemWidget(list_item, embed_button)

    main_window.merged_embeddings[embed_button.embedding_id] = embed_button


def clear_stop_loading_target_media(main_window: "MainWindow", clear_list: bool = True):
    batch_timer = getattr(main_window, "_target_media_batch_timer", None)
    if batch_timer is not None:
        batch_timer.stop()
    main_window._pending_target_media_thumbnails = deque()
    _reset_target_media_progress(main_window)

    if main_window.video_loader_worker is not None:
        worker = main_window.video_loader_worker
        worker.blockSignals(True)
        worker._running = False
        worker.quit()
        if not worker.wait(_WORKER_STOP_TIMEOUT_MS):
            worker.terminate()
            worker.wait()
        main_window.video_loader_worker = None
        if clear_list:
            main_window.targetVideosList.clear()


@QtCore.Slot()
def select_target_medias(
    main_window: "MainWindow", source_type="folder", folder_name=False, files_list=None
):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "change target media"
    ):
        return

    files_list = files_list or []
    if source_type == "folder":
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(
            dir=main_window.last_target_media_folder_path
        )
        if not folder_name:
            return
        main_window.labelTargetVideosPath.setText(
            misc_helpers.truncate_text(folder_name)
        )
        main_window.labelTargetVideosPath.setToolTip(folder_name)
        main_window.last_target_media_folder_path = folder_name

    elif source_type == "files":
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        # Get Folder name from the first file
        file_dir = misc_helpers.get_dir_of_file(files_list[0])
        main_window.labelTargetVideosPath.setText(
            file_dir
        )  # Just a temp text until i think of something better
        main_window.labelTargetVideosPath.setToolTip(file_dir)
        main_window.last_target_media_folder_path = file_dir

    clear_stop_loading_target_media(main_window)
    card_actions.clear_target_faces(main_window)

    main_window.selected_video_button = None
    apply_main_window_title_for_selected_media(main_window)
    main_window.target_videos = {}

    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(
        main_window=main_window,
        folder_name=folder_name,
        files_list=files_list,
        sort_mode=get_target_media_sort_mode(main_window),
    )
    main_window.video_loader_worker.thumbnail_ready.connect(
        partial(add_media_thumbnail_to_target_videos_list, main_window)
    )
    main_window.video_loader_worker.finished.connect(
        partial(filter_target_videos, main_window)
    )
    main_window.video_loader_worker.start()


@QtCore.Slot()
def filter_target_videos(main_window):
    from app.ui.widgets.actions import video_control_actions

    if _has_pending_target_media_thumbnail_work(main_window):
        QtCore.QTimer.singleShot(0, partial(filter_target_videos, main_window))
        return

    if video_control_actions.is_issue_scan_active(main_window):
        video_control_actions._mark_pending_target_media_refresh(main_window)
        return
    try:
        sort_target_media_list(main_window)
    except Exception:
        _target_sort_log("filter_wrapper_sort_failed")
        raise
    filter_actions.filter_target_videos(main_window)
    load_target_webcams(main_window)
    load_target_screen_capture(main_window)


def load_target_screen_capture(main_window: "MainWindow"):
    if main_window.filterScreenCaptureCheckBox.isChecked():
        has_screen = any(
            getattr(b, "is_screen_capture", False)
            for b in main_window.target_videos.values()
        )
        if not has_screen:
            add_screen_capture_thumbnail_to_target_videos_list(main_window)
            main_window.placeholder_update_signal.emit(main_window.targetVideosList, False)
    else:
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, True)
        for _, target_video in main_window.target_videos.copy().items():
            if target_video.file_type == "screen":
                target_video.remove_target_media_from_list()
                if target_video == main_window.selected_video_button:
                    main_window.selected_video_button = None
                    apply_main_window_title_for_selected_media(main_window)
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, False)


@QtCore.Slot()
def load_target_webcams(
    main_window: "MainWindow",
):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.is_issue_scan_active(main_window):
        video_control_actions._mark_pending_target_media_refresh(main_window)
        return
    if main_window.filterWebcamsCheckBox.isChecked():
        main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(
            main_window=main_window, webcam_mode=True
        )
        main_window.video_loader_worker.webcam_thumbnail_ready.connect(
            partial(add_webcam_thumbnail_to_target_videos_list, main_window)
        )
        main_window.video_loader_worker.start()
    else:
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, True)
        for (
            _,
            target_video,
        ) in main_window.target_videos.copy().items():  # Use a copy of the dict to prevent Dictionary changed during iteration exceptions
            if target_video.file_type == "webcam":
                target_video.remove_target_media_from_list()
                if target_video == main_window.selected_video_button:
                    main_window.selected_video_button = None
                    apply_main_window_title_for_selected_media(main_window)
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, False)


def clear_stop_loading_input_media(main_window: "MainWindow", clear_list: bool = True):
    batch_timer = getattr(main_window, "_input_face_batch_timer", None)
    if batch_timer is not None:
        batch_timer.stop()
    main_window._pending_input_face_thumbnails = deque()

    if main_window.input_faces_loader_worker is not None:
        worker = main_window.input_faces_loader_worker
        worker.blockSignals(True)
        worker._running = False
        worker.quit()
        if not worker.wait(_WORKER_STOP_TIMEOUT_MS):
            worker.terminate()
            worker.wait()
        main_window.input_faces_loader_worker = None
        if clear_list:
            main_window.inputFacesList.clear()
            main_window.inputFacesFavoritesList.clear()


def _set_folder_path_display(
    main_window: "MainWindow",
    *,
    line_edit_attr: str,
    label_attr: str,
    path: str,
) -> None:
    """Support dev QLineEdit path widgets and nanof QLabel path widgets."""
    line_edit = getattr(main_window, line_edit_attr, None)
    if line_edit is not None:
        line_edit.setText(path)
        line_edit.setToolTip(path)
        return
    label = getattr(main_window, label_attr, None)
    if label is not None:
        label.setText(misc_helpers.truncate_text(path) if path else "")
        label.setToolTip(path)


def set_target_media_path_display(main_window: "MainWindow", path: str) -> None:
    _set_folder_path_display(
        main_window,
        line_edit_attr="targetVideosPathLineEdit",
        label_attr="labelTargetVideosPath",
        path=path,
    )


def set_input_faces_path_display(main_window: "MainWindow", path: str) -> None:
    _set_folder_path_display(
        main_window,
        line_edit_attr="inputFacesPathLineEdit",
        label_attr="labelInputFacesPath",
        path=path,
    )


def _set_path_line_edit_value(line_edit: QtWidgets.QLineEdit, path: str) -> None:
    line_edit.setText(path)
    line_edit.setToolTip(path)


def _confirm_panel_clear(main_window: "MainWindow", title: str, message: str) -> bool:
    reply = QtWidgets.QMessageBox.question(
        main_window,
        title,
        message,
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No,
    )
    return reply == QtWidgets.QMessageBox.Yes


def clear_all_target_media(main_window: "MainWindow") -> bool:
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(main_window, "clear all media"):
        return False

    if not main_window.target_videos:
        return False

    confirmed = _confirm_panel_clear(
        main_window,
        "Clear All Media",
        "This will remove all target media, including webcams, and reset the "
        "Target Media panel.\n\nFiles on disk will not be deleted.",
    )
    if not confirmed:
        return False

    clear_stop_loading_target_media(main_window, clear_list=False)

    for target_media_button in list(main_window.target_videos.values()):
        target_media_button.remove_target_media_from_list()

    if main_window.target_faces:
        card_actions.clear_target_faces(main_window, refresh_frame=False)

    main_window.target_videos.clear()
    main_window.selected_video_button = None
    apply_main_window_title_for_selected_media(main_window)
    set_target_media_path_display(main_window, "")
    main_window.last_target_media_folder_path = ""
    main_window.placeholder_update_signal.emit(main_window.targetVideosList, False)
    return True


def clear_all_input_faces(main_window: "MainWindow") -> bool:
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(main_window, "clear all faces"):
        return False

    if not main_window.input_faces:
        return False

    confirmed = _confirm_panel_clear(
        main_window,
        "Clear All Faces",
        "This will remove all input faces and reset the Input Faces panel.\n\n"
        "Files on disk will not be deleted.",
    )
    if not confirmed:
        return False

    clear_stop_loading_input_media(main_window, clear_list=False)

    for input_face_button in list(main_window.input_faces.values()):
        input_face_button.remove_kv_data_file()
        input_face_button._remove_face_from_lists()
        input_face_button.deleteLater()

    common_widget_actions.refresh_frame(main_window)
    set_input_faces_path_display(main_window, "")
    main_window.last_input_media_folder_path = ""
    main_window.placeholder_update_signal.emit(main_window.inputFacesList, False)
    return True


def clear_all_embeddings(main_window: "MainWindow") -> bool:
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "clear all embeddings"
    ):
        return False

    if not main_window.merged_embeddings:
        return False

    confirmed = _confirm_panel_clear(
        main_window,
        "Clear All Embeddings",
        "This will remove all embeddings and reset the Embeddings panel.\n\n"
        "Files on disk will not be deleted.",
    )
    if not confirmed:
        return False

    card_actions.clear_merged_embeddings(main_window)
    return True


def _build_panel_context_menu(
    main_window: "MainWindow",
    list_widget: QtWidgets.QListWidget,
    panel_type: str,
) -> QtWidgets.QMenu:
    from app.ui.widgets.actions import video_control_actions

    scan_active = video_control_actions.is_issue_scan_active(main_window)
    menu = QtWidgets.QMenu(list_widget)

    if panel_type == "target_media":
        clear_action = QtGui.QAction("Clear All Media", menu)
        clear_action.setEnabled(bool(main_window.target_videos) and not scan_active)
        clear_action.triggered.connect(partial(clear_all_target_media, main_window))
    elif panel_type == "input_faces":
        clear_action = QtGui.QAction("Clear All Faces", menu)
        clear_action.setEnabled(bool(main_window.input_faces) and not scan_active)
        clear_action.triggered.connect(partial(clear_all_input_faces, main_window))
    else:
        clear_action = QtGui.QAction("Clear All Embeddings", menu)
        clear_action.setEnabled(bool(main_window.merged_embeddings) and not scan_active)
        clear_action.triggered.connect(partial(clear_all_embeddings, main_window))

    menu.addAction(clear_action)
    return menu


def _show_panel_context_menu(
    main_window: "MainWindow",
    list_widget: QtWidgets.QListWidget,
    panel_type: str,
    position: QtCore.QPoint,
) -> None:
    if list_widget.itemAt(position) is not None:
        return

    menu = _build_panel_context_menu(main_window, list_widget, panel_type)
    menu.exec(list_widget.viewport().mapToGlobal(position))


def _set_up_panel_context_menu(
    main_window: "MainWindow",
    list_widget: QtWidgets.QListWidget,
    panel_type: str,
) -> None:
    list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    list_widget.customContextMenuRequested.connect(
        partial(_show_panel_context_menu, main_window, list_widget, panel_type)
    )


@QtCore.Slot()
def select_input_face_images(
    main_window: "MainWindow", source_type="folder", folder_name=False, files_list=None
):
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "load input faces"
    ):
        return

    files_list = files_list or []
    if source_type == "folder":
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(
            dir=main_window.last_input_media_folder_path
        )
        if not folder_name:
            return
        main_window.labelInputFacesPath.setText(misc_helpers.truncate_text(folder_name))
        main_window.labelInputFacesPath.setToolTip(folder_name)
        main_window.last_input_media_folder_path = folder_name

    elif source_type == "files":
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        file_dir = misc_helpers.get_dir_of_file(files_list[0])
        main_window.labelInputFacesPath.setText(
            file_dir
        )  # Just a temp text until i think of something better
        main_window.labelInputFacesPath.setToolTip(file_dir)
        main_window.last_input_media_folder_path = file_dir

    clear_stop_loading_input_media(main_window)
    card_actions.clear_input_faces(main_window)
    main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(
        main_window=main_window, folder_name=folder_name, files_list=files_list
    )
    main_window.input_faces_loader_worker.thumbnail_ready.connect(
        partial(add_media_thumbnail_to_source_faces_list, main_window)
    )

    main_window.input_faces_loader_worker.start()


def set_up_list_widget_placeholder(
    main_window: "MainWindow", list_widget: QtWidgets.QListWidget
):
    # Placeholder label
    placeholder_label = QtWidgets.QLabel(list_widget)
    placeholder_label.setText(
        "<html><body style='text-align:center;'>"
        "<p>Drop Files</p>"
        "<p><b>or</b></p>"
        "<p>Click here to Select a Folder</p>"
        "</body></html>"
    )
    # placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    placeholder_label.setStyleSheet("color: gray; font-size: 15px; font-weight: bold;")

    # Center the label inside the QListWidget
    # placeholder_label.setGeometry(list_widget.rect())  # Match QListWidget's size
    placeholder_label.setAttribute(
        QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )  # Allow interactions to pass through
    placeholder_label.setVisible(not list_widget.count())  # Show if the list is empty

    # Use a QVBoxLayout to center the placeholder label
    layout = QtWidgets.QVBoxLayout(list_widget)
    layout.addWidget(placeholder_label)
    layout.setAlignment(
        QtCore.Qt.AlignmentFlag.AlignCenter
    )  # Center the label vertically and horizontally
    layout.setContentsMargins(0, 0, 0, 0)  # Remove margins to ensure full coverage

    # Keep a reference for toggling visibility later
    list_widget.placeholder_label = placeholder_label
    # Set default cursor as PointingHand
    list_widget.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)


def _open_existing_path_in_os_file_manager(path: str) -> None:
    """Open an existing file or directory in the OS file manager."""
    if not isinstance(path, str) or not path or not os.path.exists(path):
        return
    normalized_path = os.path.normpath(os.path.abspath(path))
    if sys.platform == "win32":
        try:
            subprocess.Popen(["explorer", normalized_path])
        except FileNotFoundError:
            subprocess.Popen([r"C:\Windows\explorer.exe", normalized_path])
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", path])
    else:
        directory = os.path.dirname(os.path.abspath(path))
        subprocess.run(["xdg-open", directory])


def open_target_media_folder(
    main_window: "MainWindow", folder_name: str | None = None
) -> None:
    if not folder_name:
        folder_name = getattr(main_window, "last_target_media_folder_path", "")
        if not (isinstance(folder_name, str) and folder_name.strip()):
            label = getattr(main_window, "labelTargetVideosPath", None)
            if label is not None:
                tip = label.toolTip()
                folder_name = tip if isinstance(tip, str) else ""
        if not (isinstance(folder_name, str) and folder_name.strip()):
            return
    _open_existing_path_in_os_file_manager(str(folder_name).strip())


def open_input_faces_folder(
    main_window: "MainWindow", folder_name: str | None = None
) -> None:
    if not folder_name:
        folder_name = getattr(main_window, "last_input_media_folder_path", "")
        if not (isinstance(folder_name, str) and folder_name.strip()):
            label = getattr(main_window, "labelInputFacesPath", None)
            if label is not None:
                tip = label.toolTip()
                folder_name = tip if isinstance(tip, str) else ""
        if not (isinstance(folder_name, str) and folder_name.strip()):
            return
    _open_existing_path_in_os_file_manager(str(folder_name).strip())


def select_output_media_folder(main_window: "MainWindow"):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory(main_window)
    if folder_name:
        main_window.outputFolderLineEdit.setText(folder_name)
        common_widget_actions.create_control(
            main_window, "OutputMediaFolder", folder_name
        )


def open_output_media_folder(main_window: "MainWindow", folder_name: str | None = None):
    if not folder_name:
        configured_folder = main_window.control.get("OutputMediaFolder")
        folder_name = configured_folder if isinstance(configured_folder, str) else None
    if isinstance(folder_name, str) and folder_name:
        _open_existing_path_in_os_file_manager(folder_name)


def show_shortcuts(main_window: "MainWindow"):
    # HTML formating
    shortcuts_text = (
        "<b><u>Actions:</u></b><br>"
        "<b>F11</b> : Fullscreen<br>"
        "<b>T</b> : Theatre Mode<br>"
        "<b>Space</b> : Play/Stop<br>"
        "<b>R</b> : Record start/stop<br>"
        "<b>S</b> : Swap face<br>"
        "<b>F5</b> : Pipeline profile overlay on/off<br>"
        "<b>F6</b> : Face restorer 1 on/off<br>"
        "<b>Shift+F6</b> : Face restorer 2 on/off<br>"
        "<b>F7</b> : Frame interpolation on/off<br>"
        "<br>"
        "<b><u>Seeking:</u></b><br>"
        "<b>V</b> : Advance 1 frame<br>"
        "<b>C</b> : Rewind 1 frame<br>"
        "<b>D</b> : Advance frames by slider value<br>"
        "<b>A</b> : Rewind frames by slider value<br>"
        "<b>Z</b> : Seek to start<br>"
        "<br>"
        "<b><u>Markers:</u></b><br>"
        "<b>F</b> : Add video marker<br>"
        "<b>ALT+F</b> : Remove video marker<br>"
        "<b>W</b> : Move to next marker<br>"
        "<b>Q</b> : Move to previous marker<br>"
        "<br>"
        "<b><u>Viewport:</u></b><br>"
        "<b>Ctrl+0</b> : Fit to View<br>"
        "<b>Ctrl+1</b> : 100% Zoom<br>"
        "<b>Middle Mouse Drag</b> : Pan view<br>"
        "<b>Right Click</b> : Viewport menu (Fit to View, 100% Zoom, Save Image)<br>"
        "<br>"
    )

    main_window.display_messagebox_signal.emit(
        "Shortcuts",
        shortcuts_text,
        main_window,
    )


def show_presets(main_window: "MainWindow"):
    # HTML formating
    presets_text = (
        "<b><u>What are Presets?</u></b><br>"
        "Presets are a functionality that allows saving and applying parameters on swapped faces.<br>"
        "Saved options come from the: 'Face Swap', 'Face Editor', 'Restorers', 'Denoiser', and 'Settings' tabs."
        "<br><br>"
        "<b><u>Option Categories</u></b><br>"
        "There are two distinct categories:"
        "<br><br>"
        "<b>1. Parameters (Applied <u>per face</u>)</b><br>"
        "Includes all options from:<br>"
        "&nbsp;&nbsp;&bull; 'Face Swap'<br>"
        "&nbsp;&nbsp;&bull; 'Face Editor'<br>"
        "&nbsp;&nbsp;&bull; 'Restorers'"
        "<br><br>"
        "<b>2. Controls (Applied <u>globally</u>)</b><br>"
        "Includes all options from:<br>"
        "&nbsp;&nbsp;&bull; 'Denoiser'<br>"
        "&nbsp;&nbsp;&bull; 'Settings'"
        "<br><br>"
        # Une couleur (ex: #FFCC00 pour jaune/orange) aide à attirer l'œil
        "<b><u><font color='#FFCC00'>IMPORTANT</font></u></b><br>"
        "To apply the <b>Controls</b> options (Denoiser/Settings), the "
        "<b>'Apply Settings'</b> button <u>must be checked</u> (it is OFF by default)."
    )

    main_window.display_messagebox_signal.emit(
        "Presets",
        presets_text,
        main_window,
    )


def _get_app_display_metadata(main_window: "MainWindow") -> AppDisplayMetadata:
    metadata = getattr(main_window, "app_display_metadata", None)
    if metadata is not None:
        return metadata

    base_title = getattr(main_window, "_base_window_title", main_window.windowTitle())
    return get_app_display_metadata(main_window.project_root_path, base_title)


def _selected_target_media_title_suffix(
    btn: widget_components.TargetMediaCardButton,
) -> str | None:
    mp = getattr(btn, "media_path", None)
    if mp is None or mp is False:
        return None
    name = os.path.basename(str(mp)).strip()
    if name:
        return name
    ft = getattr(btn, "file_type", None)
    if ft == "webcam":
        return f"Webcam ({int(getattr(btn, 'webcam_index', -1))})"
    if ft == "screen":
        return "Screen capture"
    return None


def apply_main_window_title_for_selected_media(main_window: "MainWindow") -> None:
    """Window title: app name (with optional git hash) plus selected target media filename."""
    meta = getattr(main_window, "app_display_metadata", None)
    if meta is None:
        base_title = getattr(main_window, "_base_window_title", main_window.windowTitle())
        meta = get_app_display_metadata(main_window.project_root_path, base_title)
        main_window.app_display_metadata = meta
    base = meta.window_title

    btn = getattr(main_window, "selected_video_button", None)
    if btn in (None, False):
        main_window.setWindowTitle(base)
        return
    if not isinstance(btn, widget_components.TargetMediaCardButton):
        main_window.setWindowTitle(base)
        return

    suffix = _selected_target_media_title_suffix(btn)
    if not suffix:
        main_window.setWindowTitle(base)
        return

    main_window.setWindowTitle(f"{base} — {suffix}")


def _open_about_link(main_window: "MainWindow", link_type: str):
    project_root = Path(main_window.project_root_path)
    local_links = {
        "quickstart": project_root / "docs" / "quickstart.md",
        "manual": project_root / "docs" / "user_manual.md",
    }
    remote_links = {
        "github": "https://github.com/VisoMasterFusion/VisoMaster-Fusion",
        "discord": "https://discord.gg/5rx4SQuDbp",
    }

    if link_type in local_links:
        target_path = local_links[link_type]
        if target_path.is_file():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(target_path.resolve()))
            )
        else:
            common_widget_actions.create_and_show_messagebox(
                main_window,
                "Document Not Found",
                f"Could not find:\n{target_path}",
                parent_widget=main_window,
            )
        return

    target_url = remote_links.get(link_type)
    if target_url:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(target_url))


def show_about(main_window: "MainWindow"):
    dialog = QtWidgets.QDialog(main_window)
    dialog.setWindowTitle("About")
    dialog.setModal(True)
    dialog.setMinimumWidth(420)

    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(12)

    title_label = QtWidgets.QLabel("VisoMaster Fusion", dialog)
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 2)
    title_font.setBold(True)
    title_label.setFont(title_font)

    version_label = QtWidgets.QLabel(
        _get_app_display_metadata(main_window).about_version_text, dialog
    )
    description_label = QtWidgets.QLabel(
        "Advanced image and video editing toolkit.\n"
        "See the User Manual for setup and usage guidance.",
        dialog,
    )
    description_label.setWordWrap(True)

    links_group = QtWidgets.QGroupBox("Quick Links", dialog)
    links_layout = QtWidgets.QVBoxLayout(links_group)
    links_layout.setContentsMargins(12, 12, 12, 12)
    links_layout.setSpacing(6)

    links_label = QtWidgets.QLabel(links_group)
    links_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
    links_label.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextBrowserInteraction
    )
    links_label.setOpenExternalLinks(False)
    links_label.setWordWrap(True)
    links_label.setText(
        '<a href="quickstart">Quick Start Guide</a><br>'
        '<a href="manual">User Manual</a><br>'
        '<a href="discord">Discord</a><br>'
        '<a href="github">GitHub</a>'
    )
    links_label.linkActivated.connect(
        lambda link_type: _open_about_link(main_window, link_type)
    )
    links_layout.addWidget(links_label)

    close_button = QtWidgets.QPushButton("Close", dialog)
    close_button.clicked.connect(dialog.accept)

    layout.addWidget(title_label)
    layout.addWidget(version_label)
    layout.addWidget(description_label)
    layout.addWidget(links_group)
    layout.addWidget(close_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

    dialog.exec()
