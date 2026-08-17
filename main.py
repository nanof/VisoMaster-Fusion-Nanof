from app.ui import main_ui
from PySide6 import QtWidgets
import os
import sys

import qdarktheme
from app.helpers.console_color import (
    install_colored_console_streams,
    install_utf8_console_streams,
)
from app.ui.core.proxy_style import ProxyStyle
from app.ui.widgets.tooltip_utils import install_tooltip_vertical_wrap

# --- PyTorch VRAM Optimization ---
# MUST be set BEFORE importing torch (which occurs during MainWindow setup).
# Left overridable so other values can be A/B'd against peak VRAM without a
# rebuild, e.g. PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import torch

torch.set_grad_enabled(False)

if __name__ == "__main__":
    install_utf8_console_streams()
    install_colored_console_streams()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(ProxyStyle())
    # Force long plain-text tooltips to wrap at a bounded width instead of
    # producing screen-wide single-line popups. Installed before the main
    # window so every widget's setToolTip() call is rewritten through the
    # QEvent.ToolTipChange notification.
    install_tooltip_vertical_wrap(app)
    with open("app/ui/styles/true_dark_styles.qss", "r") as f:
        _style = f.read()
        _style = (
            qdarktheme.load_stylesheet(
                theme="dark", custom_colors={"primary": "#4090a3"}
            )
            + "\n"
            + _style
        )
        app.setStyleSheet(_style)
    window = main_ui.MainWindow()
    window.show()
    app.exec()
