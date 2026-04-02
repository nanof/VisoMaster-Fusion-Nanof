"""
Fullscreen overlay to pick a screen rectangle (global pixel coordinates).

Used for desktop capture region selection. Coordinates match mss / Qt virtual desktop.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class ScreenRegionPickerOverlay(QtWidgets.QWidget):
    """Dim the desktop and drag a rectangle; emits global left, top, width, height."""

    region_chosen = QtCore.Signal(int, int, int, int)
    cancelled = QtCore.Signal()

    _MIN_SIDE = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._origin_global: QtCore.QPoint | None = None
        self._current_global: QtCore.QPoint | None = None
        self._dragging = False

        flags = (
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

        screens = QtGui.QGuiApplication.screens()
        if not screens:
            self._virtual = QtCore.QRect(0, 0, 1920, 1080)
        else:
            self._virtual = screens[0].geometry()
            for s in screens[1:]:
                self._virtual = self._virtual.united(s.geometry())

        self.setGeometry(self._virtual)

        self._hint = QtWidgets.QLabel(self)
        self._hint.setText(
            self.tr("Drag to select the capture area · Esc to cancel")
        )
        self._hint.setStyleSheet(
            "QLabel { background: rgba(0,0,0,180); color: white; padding: 8px 12px; "
            "border-radius: 4px; font-size: 13px; }"
        )
        self._hint.adjustSize()
        self._hint.move(12, 12)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        self.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self._origin_global = event.globalPosition().toPoint()
            self._current_global = self._origin_global
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            self._current_global = event.globalPosition().toPoint()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._current_global = event.globalPosition().toPoint()
            if self._origin_global is None:
                self.close()
                return
            gr = QtCore.QRect(self._origin_global, self._current_global).normalized()
            if gr.width() >= self._MIN_SIDE and gr.height() >= self._MIN_SIDE:
                self.region_chosen.emit(
                    gr.left(), gr.top(), gr.width(), gr.height()
                )
            else:
                self.cancelled.emit()
            self.close()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 110))

        if (
            self._dragging
            and self._origin_global is not None
            and self._current_global is not None
        ):
            g = QtCore.QRect(self._origin_global, self._current_global).normalized()
            local = QtCore.QRect(
                self.mapFromGlobal(g.topLeft()),
                self.mapFromGlobal(g.bottomRight()),
            ).normalized()
            painter.fillRect(local, QtGui.QColor(255, 255, 255, 35))
            pen = QtGui.QPen(QtGui.QColor(80, 220, 200), 2)
            painter.setPen(pen)
            painter.drawRect(local.adjusted(1, 1, -1, -1))
