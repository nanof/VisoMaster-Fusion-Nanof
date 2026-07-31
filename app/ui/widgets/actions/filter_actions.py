from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def _get_target_video_filter_checked(
    main_window: "MainWindow", action_name: str, checkbox_name: str, default: bool
) -> bool:
    checkbox = getattr(main_window, checkbox_name, None)
    if checkbox is not None:
        return checkbox.isChecked()

    action = getattr(main_window, action_name, None)
    if action is not None:
        return action.isChecked()

    return default


def filter_target_videos(main_window: "MainWindow", *args):
    main_window.target_videos_filter_worker.stop_thread()

    # Capture all Qt widget data in the main thread before starting the worker
    search_text = main_window.targetVideosSearchBox.text().lower()

    include_file_types = []
    if _get_target_video_filter_checked(
        main_window,
        "targetVideosFilterImagesAction",
        "targetVideosFilterImagesCheckBox",
        True,
    ):
        include_file_types.append("image")
    if _get_target_video_filter_checked(
        main_window,
        "targetVideosFilterVideosAction",
        "targetVideosFilterVideosCheckBox",
        True,
    ):
        include_file_types.append("video")
    if _get_target_video_filter_checked(
        main_window,
        "targetVideosFilterWebcamsAction",
        "targetVideosFilterWebcamsCheckBox",
        False,
    ):
        include_file_types.append("webcam")
    if main_window.filterScreenCaptureCheckBox.isChecked():
        include_file_types.append("screen")

    min_width = 0
    min_height = 0
    min_w_spin = getattr(main_window, "targetMediaMinWidthSpinBox", None)
    min_h_spin = getattr(main_window, "targetMediaMinHeightSpinBox", None)
    if min_w_spin is not None:
        min_width = int(min_w_spin.value())
    if min_h_spin is not None:
        min_height = int(min_h_spin.value())

    items_snapshot = []
    for i in range(main_window.targetVideosList.count()):
        item = main_window.targetVideosList.item(i)
        item_widget = main_window.targetVideosList.itemWidget(item)
        if item_widget is not None:
            metadata = getattr(item_widget, "_media_metadata", None)
            width = int(getattr(metadata, "width", 0) or 0)
            height = int(getattr(metadata, "height", 0) or 0)
            items_snapshot.append(
                (i, item_widget.media_path, item_widget.file_type, width, height)
            )

    worker = main_window.target_videos_filter_worker
    worker.search_text = search_text
    worker.include_file_types = include_file_types
    worker.min_image_size = (min_width, min_height)
    worker.items_snapshot = items_snapshot
    worker.start()


def filter_input_faces(main_window: "MainWindow", *args):
    main_window.input_faces_filter_worker.stop_thread()

    # Capture all Qt widget data in the main thread before starting the worker
    search_text = main_window.inputFacesSearchBox.text().lower()

    items_snapshot = []
    for i in range(main_window.inputFacesList.count()):
        item = main_window.inputFacesList.item(i)
        item_widget = main_window.inputFacesList.itemWidget(item)
        if item_widget is not None:
            items_snapshot.append((i, item_widget.media_path))

    worker = main_window.input_faces_filter_worker
    worker.search_text = search_text
    worker.items_snapshot = items_snapshot
    worker.start()


def filter_input_faces_favorites(main_window: "MainWindow", *args):
    main_window.input_faces_favorites_filter_worker.stop_thread()

    search_text = main_window.inputFacesFavoritesSearchBox.text().lower()

    items_snapshot = []
    for i in range(main_window.inputFacesFavoritesList.count()):
        item = main_window.inputFacesFavoritesList.item(i)
        item_widget = main_window.inputFacesFavoritesList.itemWidget(item)
        if item_widget is not None:
            items_snapshot.append((i, item_widget.media_path))

    worker = main_window.input_faces_favorites_filter_worker
    worker.search_text = search_text
    worker.items_snapshot = items_snapshot
    worker.start()


def filter_merged_embeddings(main_window: "MainWindow", *args):
    main_window.merged_embeddings_filter_worker.stop_thread()

    # Capture all Qt widget data in the main thread before starting the worker
    search_text = main_window.inputEmbeddingsSearchBox.text().lower()

    items_snapshot = []
    for i in range(main_window.inputEmbeddingsList.count()):
        item = main_window.inputEmbeddingsList.item(i)
        item_widget = main_window.inputEmbeddingsList.itemWidget(item)
        if item_widget is not None:
            items_snapshot.append((i, item_widget.embedding_name))

    worker = main_window.merged_embeddings_filter_worker
    worker.search_text = search_text
    worker.items_snapshot = items_snapshot
    worker.start()


def update_filtered_list(
    main_window: "MainWindow",
    filter_list_widget: QtWidgets.QListWidget,
    visible_indices: list,
    snapshot_size: int = 0,
):
    # Defer hide/show work to the next event loop tick so pending paint events can
    # complete naturally without pumping the queue inside this function.
    sequence = getattr(filter_list_widget, "_pending_filter_sequence", 0) + 1
    filter_list_widget._pending_filter_sequence = sequence

    def apply_update():
        if getattr(filter_list_widget, "_pending_filter_sequence", None) != sequence:
            return

        filter_list_widget.setUpdatesEnabled(False)
        try:
            # Only manage items that existed at snapshot time; items added after the
            # snapshot was captured (index >= snapshot_size) are left visible so they
            # are not accidentally hidden by a stale filter result.
            limit = snapshot_size if snapshot_size > 0 else filter_list_widget.count()
            for i in range(min(limit, filter_list_widget.count())):
                filter_list_widget.item(i).setHidden(True)

            # Show only the items in the visible_indices list
            for i in visible_indices:
                if i < filter_list_widget.count():
                    filter_list_widget.item(i).setHidden(False)
        finally:
            filter_list_widget.setUpdatesEnabled(True)
            filter_list_widget.viewport().update()

    QtCore.QTimer.singleShot(0, apply_update)
