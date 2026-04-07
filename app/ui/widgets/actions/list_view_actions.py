from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Dict, Type
from pathlib import Path
import sys
import os
import uuid
import subprocess

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
_FACE_BUTTON_SIZE = (70, 70)
_EMBED_BUTTON_SIZE = (120, 25)
_EMBED_LIST_HEIGHT = 140
_THUMB_ZOOM_MIN = 0.5
_THUMB_ZOOM_MAX = 3.0


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


# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QImage, str, str)
def add_media_thumbnail_to_target_videos_list(
    main_window: "MainWindow", media_path, q_image, file_type, media_id
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
    add_media_thumbnail_button(
        main_window,
        widget_components.InputFaceCardButton,
        main_window.inputFacesList,
        main_window.input_faces,
        q_image,
        media_path=media_path,
        cropped_face=cropped_face,
        embedding_store=embedding_store,
        face_id=face_id,
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


def clear_stop_loading_target_media(main_window: "MainWindow"):
    if main_window.video_loader_worker is not None:
        worker = main_window.video_loader_worker
        worker._running = False
        worker.quit()
        if not worker.wait(_WORKER_STOP_TIMEOUT_MS):
            worker.terminate()
            worker.wait()
        main_window.video_loader_worker = None
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
        main_window=main_window, folder_name=folder_name, files_list=files_list
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

    if video_control_actions.is_issue_scan_active(main_window):
        video_control_actions._mark_pending_target_media_refresh(main_window)
        return
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


def clear_stop_loading_input_media(main_window: "MainWindow"):
    if main_window.input_faces_loader_worker is not None:
        worker = main_window.input_faces_loader_worker
        worker._running = False
        worker.quit()
        if not worker.wait(_WORKER_STOP_TIMEOUT_MS):
            worker.terminate()
            worker.wait()
        main_window.input_faces_loader_worker = None
        main_window.inputFacesList.clear()
        main_window.inputFacesFavoritesList.clear()


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
        if os.path.exists(folder_name):
            # Normalize path
            normalized_path = os.path.normpath(os.path.abspath(folder_name))

            if sys.platform == "win32":
                # Windows - use full path to explorer.exe to avoid PATH issues
                try:
                    # Method 1: Using subprocess without shell (more secure and reliable)
                    subprocess.Popen(["explorer", normalized_path])
                except FileNotFoundError:
                    # Fallback: Use full path to explorer.exe
                    subprocess.Popen([r"C:\Windows\explorer.exe", normalized_path])
            elif sys.platform == "darwin":
                # macOS
                subprocess.run(["open", "-R", folder_name])
            else:
                # Linux
                directory = os.path.dirname(os.path.abspath(folder_name))
                subprocess.run(["xdg-open", directory])


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
        main_window._apply_runtime_window_title()
        meta = main_window.app_display_metadata
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
