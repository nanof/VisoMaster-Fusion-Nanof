"""Settings UI: show detector internal size only when the active detector has a dynamic input."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# Matches ``detector_input_size_from_control`` clamp (128–640).
DETECTOR_INTERNAL_SIZE_OPTIONS: tuple[str, ...] = (
    "512",
    "416",
    "384",
    "320",
    "256",
    "224",
    "192",
    "160",
    "128",
)


def sync_detector_internal_size_combo(
    main_window: "MainWindow", detect_mode: str | None = None
) -> None:
    """
    Hide the combo when the detector uses a fixed letterbox (640 in code or fixed ONNX H=W).
    When visible, restrict items to ``DETECTOR_INTERNAL_SIZE_OPTIONS`` (all supported sizes).
    """
    w = main_window.parameter_widgets.get("DetectorInternalSizeSelection")
    if w is None:
        return

    mode = detect_mode
    if mode is None:
        mode = main_window.control.get("DetectorModelSelection", "RetinaFace")

    fixed = main_window.models_processor.face_detectors.get_declared_fixed_input_side(
        mode
    )

    lbl = getattr(w, "label_widget", None)
    rb = getattr(w, "reset_default_button", None)

    if fixed is not None:
        main_window.control["DetectorInternalSizeSelection"] = str(fixed)
        w.blockSignals(True)
        w.clear()
        w.addItems([str(fixed)])
        w.setCurrentText(str(fixed))
        w.blockSignals(False)
        w.hide()
        if lbl is not None:
            lbl.hide()
        if rb is not None:
            rb.hide()
        return

    opts = list(DETECTOR_INTERNAL_SIZE_OPTIONS)
    w.blockSignals(True)
    w.clear()
    w.addItems(opts)
    cur = str(main_window.control.get("DetectorInternalSizeSelection", "512"))
    if cur not in opts:
        cur = "512"
    w.setCurrentText(cur)
    w.blockSignals(False)
    main_window.control["DetectorInternalSizeSelection"] = cur
    w.show()
    if lbl is not None:
        lbl.show()
    if rb is not None:
        rb.show()
