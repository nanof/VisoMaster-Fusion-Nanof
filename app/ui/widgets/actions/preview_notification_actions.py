"""Simple text toasts over the video preview (non-blocking, auto-hide)."""

from __future__ import annotations

import logging
import os
import time
import warnings
from functools import partial
from typing import TYPE_CHECKING, Literal

from PySide6 import QtCore

from app.helpers.console_color import install_console_toast_tap

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_DEFAULT_DURATION_MS = 2800
_MAX_TOAST_CHARS = 620
_DEDUPE_WINDOW_SEC = 0.85

_TOAST_STYLES = {
    "info": (
        "QLabel { background-color: rgba(20, 20, 20, 220); color: #f0f0f0; "
        "padding: 10px 18px; border-radius: 6px; font-size: 13px; "
        "font-weight: 600; font-family: 'Segoe UI', 'Segoe UI Historic', sans-serif; }"
    ),
    "warn": (
        "QLabel { background-color: rgba(55, 48, 12, 235); color: #ffe08a; "
        "padding: 10px 18px; border-radius: 6px; font-size: 13px; "
        "font-weight: 600; font-family: 'Segoe UI', 'Segoe UI Historic', sans-serif; }"
    ),
    "error": (
        "QLabel { background-color: rgba(52, 18, 18, 240); color: #ffb4b4; "
        "padding: 10px 18px; border-radius: 6px; font-size: 13px; "
        "font-weight: 600; font-family: 'Segoe UI', 'Segoe UI Historic', sans-serif; }"
    ),
}

_orig_warnings_showwarning = warnings.showwarning
_console_toast_hooks_installed = False
_APP_LOG_HANDLER_NAME = "visomaster_preview_toast"


class _ConsoleToastBridge(QtCore.QObject):
    """Queued delivery of console / logging lines to the GUI thread."""

    toast_requested = QtCore.Signal(str, str)

    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__(main_window)
        self._main_window = main_window
        self.toast_requested.connect(self._deliver)

    @QtCore.Slot(str, str)
    def _deliver(self, level: str, message: str) -> None:
        lv: Literal["info", "warn", "error"] = (
            level if level in ("info", "warn", "error") else "info"
        )
        show_preview_notification_typed(self._main_window, message, lv)


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


def show_preview_notification_typed(
    main_window: "MainWindow",
    message: str,
    level: Literal["info", "warn", "error"] = "info",
    duration_ms: int | None = None,
) -> None:
    """Show a toast with style and duration matching severity."""
    if not message or not str(message).strip():
        return
    if getattr(main_window, "_preview_notifications_suppressed", False):
        return
    lbl = getattr(main_window, "previewNotificationLabel", None)
    if lbl is None:
        return
    text = str(message).strip()
    if len(text) > _MAX_TOAST_CHARS:
        text = text[: _MAX_TOAST_CHARS - 1] + "…"

    now = time.monotonic()
    last = getattr(main_window, "_preview_toast_dedupe", None)
    key = (level, text[:120])
    if last is not None:
        prev_key, prev_t = last
        if prev_key == key and (now - prev_t) < _DEDUPE_WINDOW_SEC:
            return
    main_window._preview_toast_dedupe = (key, now)

    style_level = level if level in _TOAST_STYLES else "info"
    lbl.setStyleSheet(_TOAST_STYLES[style_level])
    lbl.setText(text)
    lbl.adjustSize()
    lbl.setVisible(True)
    lbl.raise_()
    from app.ui.widgets.actions import graphics_view_actions

    graphics_view_actions.position_preview_overlay_labels(main_window)
    if duration_ms is None:
        if style_level == "error":
            duration_ms = 4500
        elif style_level == "warn":
            duration_ms = 3400
        else:
            duration_ms = _DEFAULT_DURATION_MS
    timer = _ensure_hide_timer(main_window)
    timer.stop()
    timer.setInterval(max(400, int(duration_ms)))
    timer.start()


def show_preview_notification(
    main_window: "MainWindow",
    message: str,
    duration_ms: int = _DEFAULT_DURATION_MS,
) -> None:
    """Show a short message on the preview; replaces any visible toast and resets the timer."""
    show_preview_notification_typed(
        main_window, message, "info", duration_ms=duration_ms
    )


def _install_warnings_hook(bridge: _ConsoleToastBridge) -> None:
    def _showwarning(
        message,
        category,
        filename,
        lineno,
        file=None,
        line=None,
    ):
        _orig_warnings_showwarning(
            message, category, filename, lineno, file=file, line=line
        )
        try:
            msg = warnings.formatwarning(message, category, filename, lineno, line)
        except Exception:
            msg = f"{category.__name__}: {message}"
        bridge.toast_requested.emit("warn", msg.rstrip())

    warnings.showwarning = _showwarning


class _AppToastLogHandler(logging.Handler):
    def __init__(self, bridge: _ConsoleToastBridge) -> None:
        super().__init__(level=logging.WARNING)
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        tag = "error" if record.levelno >= logging.ERROR else "warn"
        self._bridge.toast_requested.emit(tag, msg)


def setup_console_and_warning_toasts(main_window: "MainWindow") -> None:
    """
    Mirror console [WARN]/[ERROR] lines, warnings.warn, and app.* logging WARNING+
    into the preview toast (GUI thread).
    """
    global _console_toast_hooks_installed
    if os.environ.get("VISIOMASTER_DISABLE_CONSOLE_TOASTS", "").strip():
        return
    if getattr(main_window, "_console_toast_bridge", None) is not None:
        return

    bridge = _ConsoleToastBridge(main_window)
    main_window._console_toast_bridge = bridge

    def _on_tagged_line(tag: str, line: str) -> None:
        bridge.toast_requested.emit(tag, line)

    install_console_toast_tap(_on_tagged_line)

    if not _console_toast_hooks_installed:
        _install_warnings_hook(bridge)
        app_log = logging.getLogger("app")
        if not any(
            getattr(h, "name", None) == _APP_LOG_HANDLER_NAME
            for h in app_log.handlers
        ):
            h = _AppToastLogHandler(bridge)
            h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            h.name = _APP_LOG_HANDLER_NAME
            app_log.addHandler(h)
        _console_toast_hooks_installed = True


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
