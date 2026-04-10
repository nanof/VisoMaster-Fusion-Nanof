"""
QGraphicsView subclass for the video preview (swap-in at startup).

Linear GPU blend sigue en QGraphicsItem::paint(). FSR1 / NIS se componen aquí tras
super().paintEvent() porque Qt no invoca QOpenGLWidget.paintEvent en el viewport.
"""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets


class VisoMasterPreviewGraphicsView(QtWidgets.QGraphicsView):
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        mw = getattr(self, "_visomaster_main_window", None)
        if mw is None:
            return
        from app.ui.widgets.actions import graphics_view_actions as graphics_view_actions_mod

        graphics_view_actions_mod.composite_fsr_preview_overlay_if_needed(mw, self)
        graphics_view_actions_mod.composite_nis_preview_overlay_if_needed(mw, self)
