"""Startup splash shown while PyTorch, the processors and the main window load.

Must stay importable with PySide6 + stdlib only: it is shown before torch is
imported. All functions are no-ops when no splash is active (tests, tools).
"""

from __future__ import annotations

import contextlib
import importlib.abc
import json
import sys
import time
from pathlib import Path
from typing import Iterator

from PySide6 import QtCore, QtGui, QtWidgets

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGO_PATH = _PROJECT_ROOT / "app" / "ui" / "core" / "media" / "visomaster_full.png"
_SPLASH_WIDTH = 520
_PUMP_INTERVAL_S = 0.1

_splash: "_StartupSplash | None" = None


def _version_text() -> str:
    try:
        data = json.loads((_PROJECT_ROOT / "version.json").read_text(encoding="utf-8"))
        version = str(data.get("version", "")).strip()
    except (OSError, ValueError, AttributeError):
        version = ""
    return f"VisoMaster Fusion {version}".strip()


class _StartupSplash(QtWidgets.QSplashScreen):
    def __init__(self, pixmap: QtGui.QPixmap):
        super().__init__(pixmap)
        self._title = _version_text()
        self._progress = 0.0

    def set_progress(self, fraction: float) -> None:
        self._progress = max(0.0, min(1.0, fraction))

    def drawContents(self, painter: QtGui.QPainter) -> None:
        rect = self.rect()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        title_font = QtGui.QFont(self.font())
        title_font.setPointSizeF(max(10.0, title_font.pointSizeF() * 1.25))
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#e8e4dc"))
        painter.drawText(
            rect.adjusted(0, 0, 0, -44),
            QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignHCenter,
            self._title,
        )

        msg_font = QtGui.QFont(self.font())
        painter.setFont(msg_font)
        painter.setPen(QtGui.QColor("#9c978f"))
        painter.drawText(
            rect.adjusted(16, 0, -16, -18),
            QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignHCenter,
            self.message(),
        )

        bar = QtCore.QRectF(0, rect.height() - 4, rect.width(), 4)
        painter.fillRect(bar, QtGui.QColor(255, 255, 255, 25))
        bar.setWidth(rect.width() * self._progress)
        painter.fillRect(bar, QtGui.QColor("#4090a3"))


def show(message: str = "Starting…") -> None:
    """Show the splash; call right after creating the QApplication."""
    global _splash
    app = QtWidgets.QApplication.instance()
    pixmap = QtGui.QPixmap(str(_LOGO_PATH))
    if app is None or pixmap.isNull():
        return
    screen = QtGui.QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    pixmap = pixmap.scaledToWidth(
        int(_SPLASH_WIDTH * dpr), QtCore.Qt.TransformationMode.SmoothTransformation
    )
    pixmap.setDevicePixelRatio(dpr)
    _splash = _StartupSplash(pixmap)
    _splash.show()
    set_status(message)
    app.processEvents()


def set_status(message: str, progress: float | None = None) -> None:
    """Update the splash text (and optional 0–1 progress); repaints synchronously.

    Does not pump the event loop, so queued timers/slots keep their usual order.
    """
    if _splash is None:
        return
    if progress is not None:
        _splash.set_progress(progress)
    _splash.showMessage(message)
    _splash.repaint()


def close() -> None:
    """Hide the splash now (e.g. before a modal startup dialog)."""
    global _splash
    if _splash is None:
        return
    _splash.close()
    _splash.deleteLater()
    _splash = None


def finish(window: QtWidgets.QWidget) -> None:
    """Close the splash once ``window`` is shown."""
    global _splash
    if _splash is None:
        return
    _splash.finish(window)
    _splash.deleteLater()
    _splash = None


class _EventPumpFinder(importlib.abc.MetaPathFinder):
    """Never finds anything; only lets the splash repaint during long import chains."""

    def __init__(self) -> None:
        self._last = 0.0

    def find_spec(self, fullname, path, target=None):  # noqa: ARG002
        now = time.monotonic()
        if _splash is not None and now - self._last >= _PUMP_INTERVAL_S:
            self._last = now
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
            )
        return None


@contextlib.contextmanager
def keep_responsive_during_imports() -> Iterator[None]:
    """Keep the splash painted (and Windows from marking it "Not Responding")
    while heavy modules import. Only use before any app logic exists: pumping
    events here would run queued slots early.
    """
    finder = _EventPumpFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)
