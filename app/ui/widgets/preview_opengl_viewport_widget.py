"""
QOpenGLWidget used as QGraphicsView viewport for FSR / linear-blend preview.

FSR1 se compone desde `VisoMasterPreviewGraphicsView.paintEvent` (tras pintar la
escena), vía `graphics_view_actions.composite_fsr_preview_overlay_if_needed` →
`VideoPreviewFsrGlItem.render_gl_in_viewport`. Qt no llama a
`QOpenGLWidget.paintEvent` cuando el viewport se pinta desde `QGraphicsView`.
El blend lineal GPU sigue en `VideoBlendOpenGLItem.paint()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtGui

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:  # pragma: no cover
    QOpenGLWidget = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

if QOpenGLWidget is None:  # pragma: no cover
    VisoMasterPreviewOpenGLViewport = None  # type: ignore[misc, assignment]
else:

    class VisoMasterPreviewOpenGLViewport(QOpenGLWidget):  # type: ignore[no-redef]
        """OpenGL viewport for QGraphicsView (scene + FSR overlay post-scene)."""

        def __init__(self, _main_window: "MainWindow") -> None:
            super().__init__()
            # Reservado por si hace falta contexto del MainWindow en el viewport;
            # FSR se compone desde VisoMasterPreviewGraphicsView.paintEvent.
            self._main_window: MainWindow | None = _main_window

        def paintEvent(self, event: QtGui.QPaintEvent) -> None:
            super().paintEvent(event)
