"""VRAM progress bar with a vertical marker at the peak usage since last reset."""

from PySide6 import QtCore, QtGui, QtWidgets


class VramPeakProgressBar(QtWidgets.QProgressBar):
    """Shows peak GPU memory usage as a vertical line over the bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._peak_mb = 0

    def reset_peak(self) -> None:
        self._peak_mb = 0
        self.update()

    def note_used_mb(self, used_mb: int) -> None:
        if used_mb > self._peak_mb:
            self._peak_mb = used_mb
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        peak = self._peak_mb
        max_v = self.maximum()
        if peak <= 0 or max_v <= 0:
            return

        opt = QtWidgets.QStyleOptionProgressBar()
        self.initStyleOption(opt)
        cr = self.style().subElementRect(
            QtWidgets.QStyle.SubElement.SE_ProgressBarContents,
            opt,
            self,
        )
        if cr.width() <= 0:
            return

        capped = min(peak, max_v)
        x = cr.left() + int(round((capped / max_v) * cr.width()))
        x = min(max(cr.left() + 1, x), cr.right() - 1)

        # Ámbar: legible sobre el chunk azul y sobre el rojo de uso alto.
        line_color = QtGui.QColor("#e6a800")
        if self.palette().color(QtGui.QPalette.ColorRole.Window).lightness() > 128:
            line_color = QtGui.QColor("#b88600")

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        pen = QtGui.QPen(line_color, 2)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawLine(x, cr.top(), x, cr.bottom())
