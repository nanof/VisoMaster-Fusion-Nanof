from PySide6 import QtWidgets
import json
import os
import sys

from app.helpers.console_color import (
    install_colored_console_streams,
    install_utf8_console_streams,
)
from app.ui import startup_splash
from app.ui.core.proxy_style import ProxyStyle
from app.ui.widgets.tooltip_utils import install_tooltip_vertical_wrap

# --- PyTorch VRAM Optimization ---
# MUST be set BEFORE importing torch (which occurs when main_ui is imported below).
# Left overridable so other values can be A/B'd against peak VRAM without a
# rebuild, e.g. PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")


def _initial_stylesheet() -> str:
    """Stylesheet the auto-loaded workspace will select, so it is applied only once.

    Applying a stylesheet after the main window is built repolishes every widget,
    which takes seconds; pre-applying the same one lets ``change_theme`` skip it.
    """
    import qdarktheme
    from app.ui.widgets.actions import control_actions

    try:
        with open("last_workspace.json", "r", encoding="utf-8") as f:
            control = json.load(f).get("control", {})
        if control.get("AutoLoadWorkspaceToggle", False):
            style = control_actions.build_theme_stylesheet(
                control.get("ThemeSelection", "")
            )
            if style:
                return style
    except (OSError, ValueError, AttributeError):
        pass
    with open("app/ui/styles/true_dark_styles.qss", "r") as f:
        _style = f.read()
    return (
        qdarktheme.load_stylesheet(theme="dark", custom_colors={"primary": "#4090a3"})
        + "\n"
        + _style
    )


if __name__ == "__main__":
    install_utf8_console_streams()
    install_colored_console_streams()
    app = QtWidgets.QApplication(sys.argv)
    startup_splash.show("Loading PyTorch…")

    # Heavy imports (torch, torchvision, kornia, ONNX Runtime, the whole UI)
    # happen after the splash is visible.
    with startup_splash.keep_responsive_during_imports():
        import torch

        torch.set_grad_enabled(False)
        startup_splash.set_status("Loading processing modules…", 0.2)
        from app.ui import main_ui

    startup_splash.set_status("Applying theme…", 0.45)
    app.setStyle(ProxyStyle())
    # Force long plain-text tooltips to wrap at a bounded width instead of
    # producing screen-wide single-line popups. Installed before the main
    # window so every widget's setToolTip() call is rewritten through the
    # QEvent.ToolTipChange notification.
    install_tooltip_vertical_wrap(app)
    app.setStyleSheet(_initial_stylesheet())
    window = main_ui.MainWindow()
    window.show()
    startup_splash.finish(window)
    app.exec()
