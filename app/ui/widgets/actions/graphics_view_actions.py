import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore

import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_PREVIEW_FPS_WINDOW_SEC = 1.0
_PREVIEW_FPS_STALE_SEC = 0.85


# Maps Settings → Linear blend quality (fragment shader u_mode).
_PREVIEW_LINEAR_BLEND_SHADER_MODE: dict[str, int] = {
    "sRGB mix": 0,
    "Linear (gamma)": 1,
    "Luma-weighted": 2,
    "Edge-aware": 3,
    "Pro (linear + luma + edges)": 4,
}


def preview_linear_blend_shader_mode_int(main_window: "MainWindow") -> int:
    sel = str(
        main_window.control.get("PreviewLinearBlendShaderSelection", "sRGB mix")
    )
    return _PREVIEW_LINEAR_BLEND_SHADER_MODE.get(sel, 0)


def is_linear_preview_interpolation_method(method: object) -> bool:
    """True for linear preview (current name or legacy workspace value)."""
    v = str(method).strip()
    return v in ("Linear (GPU)", "Linear (OpenGL)", "Linear (CPU)")


def ensure_video_preview_opengl_viewport(main_window: "MainWindow") -> bool:
    """
    Switch QGraphicsView to an OpenGL viewport once (for shader-based linear preview).
    Returns False if QtOpenGLWidgets is unavailable or setViewport fails.
    """
    if getattr(main_window, "_video_preview_opengl_viewport_active", False):
        try:
            from PySide6.QtOpenGLWidgets import QOpenGLWidget

            vport = main_window.graphicsViewFrame.viewport()
            if isinstance(vport, QOpenGLWidget) and not getattr(
                vport, "_visomaster_no_partial_done", False
            ):
                vport.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.NoPartialUpdate)
                setattr(vport, "_visomaster_no_partial_done", True)
        except Exception:
            pass
        return True
    try:
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
    except ImportError:
        print(
            "[WARN] Linear preview: PySide6.QtOpenGLWidgets not available; "
            "falling back to NumPy blend for interpolation ticks."
        )
        setattr(main_window, "_video_preview_opengl_viewport_failed", True)
        return False
    try:
        gl_vp = QOpenGLWidget()
        # PartialUpdate (default) can coalesce rapid repaints → ~half the visible refresh rate
        # (e.g. 30 logical updates / 15 compositor presents). Decoupled presenter needs each tick.
        gl_vp.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.NoPartialUpdate)
        main_window.graphicsViewFrame.setViewport(gl_vp)
        main_window._video_preview_opengl_viewport_active = True
        main_window._video_preview_opengl_viewport_failed = False
        return True
    except Exception as e:
        print(f"[WARN] Linear preview GPU: could not enable OpenGL viewport: {e}")
        main_window._video_preview_opengl_viewport_failed = True
        return False


def restore_video_preview_raster_viewport(main_window: "MainWindow") -> None:
    """Revert QGraphicsView to the default raster viewport (CPU preview path)."""
    if not getattr(main_window, "_video_preview_opengl_viewport_active", False):
        return
    try:
        main_window.graphicsViewFrame.setViewport(QtWidgets.QWidget())
    except Exception as e:
        print(f"[WARN] Could not restore raster preview viewport: {e}")
    main_window._video_preview_opengl_viewport_active = False
    item = getattr(main_window, "_video_preview_blend_gl_item", None)
    if item is not None:
        try:
            item.setVisible(False)
            item.reset_gl_state()
        except RuntimeError:
            invalidate_video_preview_blend_gl_item_ref(main_window)


def preview_linear_gpu_display_enabled(main_window: "MainWindow") -> bool:
    """Linear interpolation uses OpenGL in the main video preview (no separate CPU/GPU toggle)."""
    if main_window.video_processor.file_type != "video":
        return False
    if not main_window.control.get("PreviewFrameGenEnableToggle", False):
        return False
    m = main_window.control.get("FrameInterpolationMethodSelection", "Linear (GPU)")
    if not is_linear_preview_interpolation_method(m):
        return False
    if main_window.control.get("SendVirtCamFramesEnableToggle", False):
        return False
    return True


def invalidate_video_preview_blend_gl_item_ref(main_window: "MainWindow") -> None:
    """Call after scene.clear() (or similar): the C++ QGraphicsItem is destroyed but Python may still hold a wrapper."""
    main_window._video_preview_blend_gl_item = None
    setattr(main_window, "_gv_preview_pixmap_item", None)


def _cached_graphics_view_pixmap_item(
    main_window: "MainWindow", scene: QtWidgets.QGraphicsScene | None
) -> QtWidgets.QGraphicsPixmapItem | None:
    """Reuse the preview pixmap item ref; avoids O(n) scene.items() on every interpolation tick."""
    if scene is None:
        setattr(main_window, "_gv_preview_pixmap_item", None)
        return None
    cached = getattr(main_window, "_gv_preview_pixmap_item", None)
    if cached is not None:
        try:
            if (
                isinstance(cached, QtWidgets.QGraphicsPixmapItem)
                and cached.scene() is scene
            ):
                return cached
        except RuntimeError:
            pass
    for item in scene.items():
        if isinstance(item, QtWidgets.QGraphicsPixmapItem):
            main_window._gv_preview_pixmap_item = item
            return item
    main_window._gv_preview_pixmap_item = None
    return None


def _blend_gl_item_still_valid(
    item, scene: QtWidgets.QGraphicsScene
) -> bool:
    if item is None:
        return False
    try:
        sc = item.scene()
    except RuntimeError:
        return False
    return sc is scene


def bump_graphics_view_repaint(main_window: "MainWindow", *, sync: bool = False) -> None:
    """Nudge QGraphicsView + viewport so OpenGL repaints are not dropped (helps decoupled presenter / overlays)."""
    gv = main_window.graphicsViewFrame
    vp = gv.viewport()
    gl_vp = getattr(main_window, "_video_preview_opengl_viewport_active", False)
    if vp is not None:
        if sync and gl_vp:
            vp.repaint()
        else:
            vp.update()
    # With QOpenGLWidget viewport, updating the view as well doubles Qt's scheduling work.
    if not gl_vp:
        gv.update()


def _get_or_create_blend_gl_item(
    main_window: "MainWindow", scene: QtWidgets.QGraphicsScene
):
    from app.ui.widgets.video_preview_blend_gl_item import VideoBlendOpenGLItem

    item = getattr(main_window, "_video_preview_blend_gl_item", None)
    if not _blend_gl_item_still_valid(item, scene):
        invalidate_video_preview_blend_gl_item_ref(main_window)
        item = VideoBlendOpenGLItem()
        scene.addItem(item)
        main_window._video_preview_blend_gl_item = item
    return item


# @misc_helpers.benchmark  (Keep this decorator if you have it)
def update_graphics_view(
    main_window: "MainWindow",
    pixmap: QtGui.QPixmap,
    current_frame_number: int,
    reset_fit: bool = False,
    size_mode: str = "preserve_previous_pixmap_size",
    *,
    gpu_blend: tuple[np.ndarray, np.ndarray, float] | None = None,
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

    scene = main_window.graphicsViewFrame.scene()
    pixmap_item = _cached_graphics_view_pixmap_item(main_window, scene)

    blend_item = getattr(main_window, "_video_preview_blend_gl_item", None)
    if not _blend_gl_item_still_valid(blend_item, scene):
        if blend_item is not None:
            invalidate_video_preview_blend_gl_item_ref(main_window)
        blend_item = None

    if gpu_blend is not None and ensure_video_preview_opengl_viewport(main_window):
        prev_bgr, curr_bgr, bw = gpu_blend
        b_item = _get_or_create_blend_gl_item(main_window, scene)
        if getattr(b_item, "_gl_failed", False):
            b_item.reset_gl_state()
        b_item.set_blend_frames(
            prev_bgr,
            curr_bgr,
            float(bw),
            preview_linear_blend_shader_mode_int(main_window),
        )
        b_item.setVisible(True)
        if pixmap_item is None:
            pixmap_item = QtWidgets.QGraphicsPixmapItem()
            pixmap_item.setTransformationMode(
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            scene.addItem(pixmap_item)
            main_window._gv_preview_pixmap_item = pixmap_item
        pixmap_item.setVisible(False)
        scene_rect = b_item.boundingRect()
        gv = main_window.graphicsViewFrame
        prev = gv.sceneRect()
        if (
            abs(prev.width() - scene_rect.width()) > 0.5
            or abs(prev.height() - scene_rect.height()) > 0.5
        ):
            gv.setSceneRect(scene_rect)
        if reset_fit:
            fit_image_to_view(main_window, b_item, scene_rect)
        record_preview_frame_tick(main_window)
        bump_graphics_view_repaint(main_window)
        return

    if blend_item is not None and _blend_gl_item_still_valid(blend_item, scene):
        blend_item.setVisible(False)

    # Resize the pixmap if necessary (e.g., face compare or mask compare mode)
    if pixmap_item and size_mode == "preserve_previous_pixmap_size":
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
        pixmap_item.setVisible(True)
    else:
        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        pixmap_item.setTransformationMode(
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        scene.addItem(pixmap_item)
        main_window._gv_preview_pixmap_item = pixmap_item

    # Set the scene rectangle to the bounding rectangle of the pixmap
    scene_rect = pixmap_item.boundingRect()
    gv = main_window.graphicsViewFrame
    prev = gv.sceneRect()
    if (
        abs(prev.width() - scene_rect.width()) > 0.5
        or abs(prev.height() - scene_rect.height()) > 0.5
    ):
        gv.setSceneRect(scene_rect)

    # Reset the view or restore the previous transform
    if reset_fit:
        fit_image_to_view(main_window, pixmap_item, scene_rect)

    record_preview_frame_tick(main_window)
    bump_graphics_view_repaint(main_window)


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
    if active and vp.processing and vp.file_type in ("video", "webcam", "screen"):
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
    """Position FPS and metadata overlays at the top-left; pipeline profile top-right."""
    margin = 8
    gap = 4
    fps_lbl = main_window.previewFpsLabel
    meta_lbl = main_window.previewMediaMetaLabel
    prof_lbl = main_window.previewPipelineProfileLabel
    fps_lbl.move(margin, margin)
    if meta_lbl.isVisible() and meta_lbl.text().strip():
        meta_lbl.move(margin, margin + fps_lbl.height() + gap)
        meta_lbl.raise_()
    fps_lbl.raise_()
    vw = main_window.graphicsViewFrame.width()
    prof_lbl.adjustSize()
    if prof_lbl.isVisible():
        prof_lbl.move(max(margin, vw - prof_lbl.width() - margin), margin)
        prof_lbl.raise_()


def position_preview_fps_label(main_window: "MainWindow") -> None:
    """Backward-compatible alias: reposition preview overlays."""
    position_preview_overlay_labels(main_window)


def update_preview_media_metadata(main_window: "MainWindow") -> None:
    """Update the preview metadata block (container, resolution, codec, etc.)."""
    main_window._preview_session_fps_frozen = None
    vp = main_window.video_processor
    wi = wb = None
    sb = getattr(main_window, "selected_video_button", None)
    screen_mon = -1
    if sb is not None and getattr(sb, "file_type", None) == "webcam":
        wi = sb.webcam_index
        wb = sb.webcam_backend
    if vp.file_type == "screen":
        try:
            screen_mon = int(main_window.control.get("ScreenCaptureMonitorSelection", 1))
        except (TypeError, ValueError):
            screen_mon = 1
    text = misc_helpers.build_preview_media_metadata_text(
        file_type=vp.file_type,
        media_path=vp.media_path,
        fps=float(vp.fps or 0),
        max_frame_number=int(vp.max_frame_number),
        media_capture=vp.media_capture,
        frame=vp.current_frame,
        webcam_index=wi if wi is not None else -1,
        webcam_backend=wb if wb is not None else -1,
        screen_monitor_index=screen_mon,
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
    from app.ui.widgets.actions import pipeline_profile_actions as _ppa_sess

    main_window._preview_fps_label_last_ui_sec = 0.0
    main_window._playback_preview_fps_active = True
    main_window._playback_preview_fps_start = time.perf_counter()
    main_window._playback_preview_fps_frames = 0
    _ppa_sess.clear_pipeline_profile_session_samples(main_window)


def reset_playback_fps_preview_session(main_window: "MainWindow") -> None:
    """On stop, freeze session-average FPS; value remains visible on the 'session:' line."""
    main_window._preview_fps_label_last_ui_sec = 0.0
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

    from app.ui.widgets.actions import pipeline_profile_actions as ppa

    ppa.print_pipeline_profile_session_report(main_window)
    ppa.reset_pipeline_profile_state(main_window)
    update_pipeline_profile_overlay(main_window, None)

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
    active = (
        getattr(main_window, "_playback_preview_fps_active", False)
        and vp.processing
        and vp.file_type in ("video", "webcam", "screen")
    )
    if active:
        main_window._playback_preview_fps_frames += 1

    # Repositioning labels and setText every frame competes with the display metronome
    # (~30 Hz). Throttle overlay updates during active playback; deque/session counters
    # stay per-frame accurate.
    _throttle_sec = 0.1
    last_ui = getattr(main_window, "_preview_fps_label_last_ui_sec", 0.0)
    if active and (now - last_ui) < _throttle_sec:
        return
    main_window._preview_fps_label_last_ui_sec = now

    _set_preview_fps_label(main_window, inst_text, now)


def on_pipeline_profile_overlay_toggle(main_window: "MainWindow", _value: object) -> None:
    """Settings toggle: show/hide pipeline profile overlay immediately."""
    update_pipeline_profile_overlay(main_window, None)


def update_pipeline_profile_overlay(
    main_window: "MainWindow", profile_payload: object | None
) -> None:
    """Show aggregated pipeline timings when the settings toggle is on."""
    from app.ui.widgets.actions import pipeline_profile_actions as ppa

    lbl = main_window.previewPipelineProfileLabel
    if not main_window.control.get("PipelineProfileOverlayEnableToggle", False):
        lbl.setVisible(False)
        return
    lbl.setVisible(True)
    if profile_payload is None:
        vp = getattr(main_window, "video_processor", None)
        if vp is not None and vp.processing:
            return
        lbl.setText("Profile: —")
        lbl.adjustSize()
        position_preview_overlay_labels(main_window)
        return
    rows = ppa.flatten_pipeline_profile_payload(
        profile_payload if isinstance(profile_payload, dict) else None
    )
    wt = (
        profile_payload.get("worker_thread")
        if isinstance(profile_payload, dict)
        else None
    )
    header_lines: list[str] = []
    vp_ov = getattr(main_window, "video_processor", None)
    if vp_ov is not None:
        fq = getattr(vp_ov, "frame_queue", None)
        if fq is not None:
            try:
                header_lines.append(
                    f"Queue (live): {fq.qsize()}/{fq.maxsize}"
                )
            except (TypeError, ValueError, AttributeError):
                pass
    if isinstance(profile_payload, dict):
        qe = profile_payload.get("frame_queue_depth_at_emit")
        qm = profile_payload.get("frame_queue_max")
        if qe is not None and qm is not None:
            try:
                header_lines.append(
                    f"Queue (at emit): {int(qe)}/{int(qm)}"
                )
            except (TypeError, ValueError):
                pass
        fn = profile_payload.get("frame_number")
        if fn is not None and wt:
            header_lines.append(f"Profile frame: {fn} · {wt}")
    text = ppa.aggregate_rows_for_display(
        main_window, rows, wt, header_lines=header_lines or None
    )
    if rows and isinstance(profile_payload, dict):
        ppa.append_pipeline_profile_session_sample(main_window, profile_payload, rows)
    lbl.setText(text)
    lbl.adjustSize()
    position_preview_overlay_labels(main_window)


def refresh_preview_fps_stale(main_window: "MainWindow") -> None:
    """If no new frames arrive, show 0 FPS for the instant line (session line may remain)."""
    if main_window._preview_fps_last_tick == 0.0:
        return
    if time.perf_counter() - main_window._preview_fps_last_tick > _PREVIEW_FPS_STALE_SEC:
        now = time.perf_counter()
        _set_preview_fps_label(main_window, "0 FPS", now)


def fit_image_to_view(
    main_window: "MainWindow", scene_item: QtWidgets.QGraphicsItem, scene_rect
):
    """Reset the view and fit the image to the view, keeping the aspect ratio."""
    graphicsViewFrame = main_window.graphicsViewFrame
    # Reset the transform and set the scene rectangle
    graphicsViewFrame.resetTransform()
    graphicsViewFrame.setSceneRect(scene_rect)
    # Fit the image to the view, keeping the aspect ratio
    graphicsViewFrame.fitInView(scene_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
