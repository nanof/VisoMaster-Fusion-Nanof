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


def _instant_fps_text_from_deque(main_window: "MainWindow", now: float) -> str:
    times = main_window._preview_fps_times
    if len(times) < 2:
        return "— FPS"
    span = now - times[0]
    if span <= 0:
        return "— FPS"
    return f"{(len(times) - 1) / span:.1f} FPS"


def _session_fps_line(main_window: "MainWindow", now: float) -> str | None:
    """Second line: 'session: X.X' live while playing or frozen after Stop."""
    vp = main_window.video_processor
    frozen = getattr(main_window, "_preview_session_fps_frozen", None)
    active = getattr(main_window, "_playback_preview_fps_active", False)

    live_avg: float | None = None
    if active and vp.processing and vp.file_type in ("video", "webcam"):
        n = main_window._playback_preview_fps_frames
        elapsed = now - main_window._playback_preview_fps_start
        if elapsed >= 0.12 and n >= 2:
            live_avg = n / elapsed

    if live_avg is not None:
        return f"session: {live_avg:.1f}"
    if frozen is not None:
        return f"session: {frozen:.1f}"
    return None


def _set_preview_fps_label(main_window: "MainWindow", inst_text: str, now: float) -> None:
    line2 = _session_fps_line(main_window, now)
    if line2:
        main_window.previewFpsLabel.setText(f"{inst_text}\n{line2}")
    else:
        main_window.previewFpsLabel.setText(inst_text)
    _layout_preview_fps_label(main_window)


def _layout_preview_fps_label(main_window: "MainWindow") -> None:
    """Resize the FPS label so multi-line text (e.g. session line) is not clipped."""
    main_window.previewFpsLabel.adjustSize()
    position_preview_overlay_labels(main_window)


def position_preview_overlay_labels(main_window: "MainWindow") -> None:
    """Position FPS and metadata overlays at the top-left of the preview view."""
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
    """Backward-compatible alias: reposition preview overlays."""
    position_preview_overlay_labels(main_window)


def update_preview_media_metadata(main_window: "MainWindow") -> None:
    """Update the preview metadata block (container, resolution, codec, etc.)."""
    main_window._preview_session_fps_frozen = None
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
    now = time.perf_counter()
    _set_preview_fps_label(
        main_window, _instant_fps_text_from_deque(main_window, now), now
    )


def start_playback_fps_preview_session(main_window: "MainWindow") -> None:
    """Start session-average FPS measurement from Play (video or webcam)."""
    main_window._playback_preview_fps_active = True
    main_window._playback_preview_fps_start = time.perf_counter()
    main_window._playback_preview_fps_frames = 0


def reset_playback_fps_preview_session(main_window: "MainWindow") -> None:
    """On stop, freeze session-average FPS; value remains visible on the 'session:' line."""
    now = time.perf_counter()
    frozen_new: float | None = None
    if main_window._playback_preview_fps_active:
        n = main_window._playback_preview_fps_frames
        elapsed = now - main_window._playback_preview_fps_start
        if elapsed > 0 and n > 0:
            frozen_new = n / elapsed

    main_window._playback_preview_fps_active = False
    main_window._playback_preview_fps_start = 0.0
    main_window._playback_preview_fps_frames = 0

    if frozen_new is not None:
        main_window._preview_session_fps_frozen = frozen_new

    inst_text = _instant_fps_text_from_deque(main_window, now)
    _set_preview_fps_label(main_window, inst_text, now)


def record_preview_frame_tick(main_window: "MainWindow") -> None:
    """
    Track preview redraw rate with a ~1s sliding window for instant FPS.
    While playing, update the second line 'session:' live; after stop it shows the frozen average.
    """
    now = time.perf_counter()
    main_window._preview_fps_last_tick = now
    times: deque[float] = main_window._preview_fps_times
    times.append(now)
    cutoff = now - _PREVIEW_FPS_WINDOW_SEC
    while times and times[0] < cutoff:
        times.popleft()

    inst_text = _instant_fps_text_from_deque(main_window, now)

    vp = main_window.video_processor
    if (
        getattr(main_window, "_playback_preview_fps_active", False)
        and vp.processing
        and vp.file_type in ("video", "webcam")
    ):
        main_window._playback_preview_fps_frames += 1

    _set_preview_fps_label(main_window, inst_text, now)


def refresh_preview_fps_stale(main_window: "MainWindow") -> None:
    """If no new frames arrive, show 0 FPS for the instant line (session line may remain)."""
    if main_window._preview_fps_last_tick == 0.0:
        return
    if time.perf_counter() - main_window._preview_fps_last_tick > _PREVIEW_FPS_STALE_SEC:
        now = time.perf_counter()
        _set_preview_fps_label(main_window, "0 FPS", now)


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
