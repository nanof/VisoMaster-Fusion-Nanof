import time
from collections import deque

from PySide6 import QtWidgets, QtGui, QtCore
from typing import TYPE_CHECKING

import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_PREVIEW_FPS_WINDOW_SEC = 1.0
_PREVIEW_FPS_STALE_SEC = 0.85


# @misc_helpers.benchmark  (Keep this decorator if you have it)
def update_graphics_view(
    main_window: "MainWindow",
    pixmap: QtGui.QPixmap,
    current_frame_number: int,
    reset_fit: bool = False,
):
    # print('(update_graphics_view) current_frame_number', current_frame_number)

    # Update the video seek slider and line edit safely to avoid recursive signal firing
    if main_window.videoSeekSlider.value() != current_frame_number:
        main_window.videoSeekSlider.blockSignals(True)
        main_window.videoSeekSlider.setValue(current_frame_number)
        main_window.videoSeekSlider.blockSignals(False)

    current_text = main_window.videoSeekLineEdit.text()
    if current_text != str(current_frame_number):
        main_window.videoSeekLineEdit.setText(str(current_frame_number))

    # Safely find the QGraphicsPixmapItem in the scene, ignoring other overlays (rectangles, text, etc.)
    scene = main_window.graphicsViewFrame.scene()
    pixmap_item = None
    for item in scene.items():
        if isinstance(item, QtWidgets.QGraphicsPixmapItem):
            pixmap_item = item
            break

    # Resize the pixmap if necessary (e.g., face compare or mask compare mode)
    if pixmap_item:
        bounding_rect = pixmap_item.boundingRect()
        b_width = int(bounding_rect.width())  # Explicit cast to int for PySide6 safety
        b_height = int(
            bounding_rect.height()
        )  # Explicit cast to int for PySide6 safety

        # If the old pixmap bounding rect is larger than the new pixmap, scale the new one
        if b_width > pixmap.width() and b_height > pixmap.height():
            pixmap = pixmap.scaled(
                b_width,
                b_height,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,  # Added smooth filter
            )

    # Update or create pixmap item
    if pixmap_item:
        pixmap_item.setPixmap(pixmap)  # Update the pixmap of the existing item
    else:
        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        pixmap_item.setTransformationMode(
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        scene.addItem(pixmap_item)

    # Set the scene rectangle to the bounding rectangle of the pixmap
    scene_rect = pixmap_item.boundingRect()
    main_window.graphicsViewFrame.setSceneRect(scene_rect)

    # Reset the view or restore the previous transform
    if reset_fit:
        fit_image_to_view(main_window, pixmap_item, scene_rect)

    record_preview_frame_tick(main_window)


def zoom_andfit_image_to_view_onchange(main_window: "MainWindow", new_transform):
    """Restore the previous transform (zoom and pan state) and update the view."""
    main_window.graphicsViewFrame.setTransform(new_transform, combine=False)


def position_preview_overlay_labels(main_window: "MainWindow") -> None:
    """Coloca FPS y metadatos en la esquina superior izquierda del visor."""
    margin = 8
    gap = 4
    fps_lbl = main_window.previewFpsLabel
    meta_lbl = main_window.previewMediaMetaLabel
    fps_lbl.move(margin, margin)
    if meta_lbl.isVisible() and meta_lbl.text().strip():
        meta_lbl.move(margin, margin + fps_lbl.height() + gap)
        meta_lbl.raise_()
    fps_lbl.raise_()


def position_preview_fps_label(main_window: "MainWindow") -> None:
    """Compatibilidad: reposiciona la superposición del preview."""
    position_preview_overlay_labels(main_window)


def update_preview_media_metadata(main_window: "MainWindow") -> None:
    """Actualiza el bloque de metadatos (formato, resolución, códec, etc.)."""
    vp = main_window.video_processor
    wi = wb = None
    sb = getattr(main_window, "selected_video_button", None)
    if sb is not None and getattr(sb, "file_type", None) == "webcam":
        wi = sb.webcam_index
        wb = sb.webcam_backend
    text = misc_helpers.build_preview_media_metadata_text(
        file_type=vp.file_type,
        media_path=vp.media_path,
        fps=float(vp.fps or 0),
        max_frame_number=int(vp.max_frame_number),
        media_capture=vp.media_capture,
        frame=vp.current_frame,
        webcam_index=wi if wi is not None else -1,
        webcam_backend=wb if wb is not None else -1,
    )
    meta = main_window.previewMediaMetaLabel
    meta.setText(text)
    meta.setVisible(bool(text.strip()))
    meta.adjustSize()
    position_preview_overlay_labels(main_window)


def record_preview_frame_tick(main_window: "MainWindow") -> None:
    """
    Cuenta cuántos frames del preview se pintan por segundo (ventana móvil).
    Debe llamarse en el hilo de UI cada vez que se actualiza el pixmap del preview.
    """
    now = time.perf_counter()
    main_window._preview_fps_last_tick = now
    times: deque[float] = main_window._preview_fps_times
    times.append(now)
    cutoff = now - _PREVIEW_FPS_WINDOW_SEC
    while times and times[0] < cutoff:
        times.popleft()
    if len(times) < 2:
        main_window.previewFpsLabel.setText("— FPS")
        return
    span = now - times[0]
    if span <= 0:
        return
    fps = (len(times) - 1) / span
    main_window.previewFpsLabel.setText(f"{fps:.1f} FPS")


def refresh_preview_fps_stale(main_window: "MainWindow") -> None:
    """Si no hay frames nuevos, baja el contador a 0 para no dejar un valor obsoleto."""
    if main_window._preview_fps_last_tick == 0.0:
        return
    if time.perf_counter() - main_window._preview_fps_last_tick > _PREVIEW_FPS_STALE_SEC:
        main_window.previewFpsLabel.setText("0 FPS")


def fit_image_to_view(
    main_window: "MainWindow", pixmap_item: QtWidgets.QGraphicsPixmapItem, scene_rect
):
    """Reset the view and fit the image to the view, keeping the aspect ratio."""
    graphicsViewFrame = main_window.graphicsViewFrame
    # Reset the transform and set the scene rectangle
    graphicsViewFrame.resetTransform()
    graphicsViewFrame.setSceneRect(scene_rect)
    # Fit the image to the view, keeping the aspect ratio
    graphicsViewFrame.fitInView(pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
