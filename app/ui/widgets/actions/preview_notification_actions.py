"""Simple text toasts over the video preview (non-blocking, auto-hide)."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from PySide6 import QtCore

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_DEFAULT_DURATION_MS = 2800


def _hide_preview_notification(main_window: "MainWindow") -> None:
    lbl = getattr(main_window, "previewNotificationLabel", None)
    if lbl is not None:
        lbl.setVisible(False)


def _ensure_hide_timer(main_window: "MainWindow") -> QtCore.QTimer:
    t = getattr(main_window, "_preview_notification_timer", None)
    if t is None:
        t = QtCore.QTimer(main_window)
        t.setSingleShot(True)
        t.timeout.connect(partial(_hide_preview_notification, main_window))
        main_window._preview_notification_timer = t
    return t


def show_preview_notification(
    main_window: "MainWindow",
    message: str,
    duration_ms: int = _DEFAULT_DURATION_MS,
) -> None:
    """Show a short message on the preview; replaces any visible toast and resets the timer."""
    if not message or not str(message).strip():
        return
    if getattr(main_window, "_preview_notifications_suppressed", False):
        return
    lbl = getattr(main_window, "previewNotificationLabel", None)
    if lbl is None:
        return
    lbl.setText(str(message).strip())
    lbl.adjustSize()
    lbl.setVisible(True)
    lbl.raise_()
    from app.ui.widgets.actions import graphics_view_actions

    graphics_view_actions.position_preview_overlay_labels(main_window)
    timer = _ensure_hide_timer(main_window)
    timer.stop()
    timer.setInterval(max(400, int(duration_ms)))
    timer.start()


def show_swap_faces_state(main_window: "MainWindow", enabled: bool) -> None:
    show_preview_notification(
        main_window,
        "Swap faces on" if enabled else "Swap faces off",
    )


def show_frame_interpolation_state(main_window: "MainWindow", enabled: bool) -> None:
    show_preview_notification(
        main_window,
        "Frame interpolation on" if enabled else "Frame interpolation off",
    )


def show_face_restorer_slot_state(
    main_window: "MainWindow", slot: int, enabled: bool
) -> None:
    """slot 1 = restorer principal, 2 = restorer 2."""
    n = 1 if slot == 1 else 2
    state = "on" if enabled else "off"
    show_preview_notification(
        main_window,
        f"Face restorer {n} {state}",
    )
