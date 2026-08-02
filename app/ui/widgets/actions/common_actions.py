from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import cv2
import numpy as np
from pyqttoast import Toast, ToastPreset, ToastPosition
from PySide6 import QtWidgets, QtCore, QtGui

from app.ui.widgets import widget_components
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.denoiser_layout_data import DENOISER_LAYOUT_DATA
import app.helpers.miscellaneous as misc_helpers
from app.helpers.miscellaneous import get_video_rotation
from app.helpers.typing_helper import ControlTypes

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# PERF-01: Module-level constant built once from layout data, reused in set_control_widgets_values
_ALL_CONTROL_WIDGET_OPTIONS: dict = {}
for _layout_source in [
    SETTINGS_LAYOUT_DATA,
    COMMON_LAYOUT_DATA,
    DENOISER_LAYOUT_DATA,
]:
    for _group_data in _layout_source.values():
        for _widget_key, _widget_data in _group_data.items():
            _ALL_CONTROL_WIDGET_OPTIONS[_widget_key] = _widget_data


@QtCore.Slot(str, str, QtWidgets.QWidget)
def create_and_show_messagebox(
    main_window: "MainWindow",
    window_title: str,
    message: str,
    parent_widget: QtWidgets.QWidget,
):
    messagebox = QtWidgets.QMessageBox(parent_widget)
    messagebox.setWindowTitle(window_title)
    messagebox.setWindowIcon(QtGui.QIcon(":/media/media/visomaster_small.png"))

    messagebox.setText(message)
    messagebox.exec_()


def create_and_show_toast_message(
    main_window: "MainWindow", title: str, message: str, style_type="information"
):
    style_preset_map = {
        "success": ToastPreset.SUCCESS,
        "warning": ToastPreset.WARNING,
        "error": ToastPreset.ERROR,
        "information": ToastPreset.INFORMATION,
        "success_dark": ToastPreset.SUCCESS_DARK,
        "warning_dark": ToastPreset.WARNING_DARK,
        "error_dark": ToastPreset.ERROR_DARK,
        "information_dark": ToastPreset.INFORMATION_DARK,
    }
    toast = Toast(main_window)
    toast.setTitle(title)
    toast.setText(message)
    toast.setDuration(10000)
    toast.setPosition(ToastPosition.TOP_RIGHT)  # Default: ToastPosition.BOTTOM_RIGHT
    toast.applyPreset(style_preset_map[style_type])  # Apply style preset
    toast.show()


def create_control(main_window: "MainWindow", control_name, control_value):
    main_window.control[control_name] = control_value


def register_control_widget_mirror(
    main_window: "MainWindow", control_name: str, widget: QtWidgets.QWidget
) -> None:
    """Register a mirrored widget bound to ``control[control_name]``."""
    mirrors = getattr(main_window, "_control_widget_mirrors", None)
    if mirrors is None:
        mirrors = {}
        main_window._control_widget_mirrors = mirrors
    mirrors.setdefault(control_name, []).append(widget)


def sync_all_widgets_for_control_key(
    main_window: "MainWindow", control_name: str, control_value
) -> None:
    """Align the primary control and any mirrored widgets with ``control_value``."""
    mirrors = getattr(main_window, "_control_widget_mirrors", {}).get(control_name, ())
    primary = main_window.parameter_widgets.get(control_name)
    seen: set[int] = set()
    for w in (*([primary] if primary is not None else []), *mirrors):
        if w is None or id(w) in seen:
            continue
        seen.add(id(w))
        w.blockSignals(True)
        try:
            _set_single_widget_value(w, control_value)
        finally:
            w.blockSignals(False)


def sync_control_mirror_widgets_only(
    main_window: "MainWindow", control_name: str, control_value
) -> None:
    for w in getattr(main_window, "_control_widget_mirrors", {}).get(control_name, ()):
        w.blockSignals(True)
        try:
            _set_single_widget_value(w, control_value)
        finally:
            w.blockSignals(False)


def strip_control_backed_keys_from_parameters(
    main_window: "MainWindow", parameters: dict
) -> dict:
    """Drop keys owned by ``main_window.control`` from a parameter payload.

    Face-parameter panels and saved workspaces may still carry stale entries for
    widgets that were migrated to global controls (e.g. Swap all by index).
    """
    control_keys = main_window.control
    for key in list(parameters.keys()):
        if key in control_keys:
            parameters.pop(key, None)
    return parameters


def get_current_parameter_value(
    main_window: "MainWindow", parameter_name: str, default: Any
) -> Any:
    fid = main_window.selected_target_face_id
    if (
        fid
        and fid in main_window.parameters
        and parameter_name in main_window.parameters[fid]
    ):
        return main_window.parameters[fid][parameter_name]
    if (
        main_window.current_widget_parameters
        and parameter_name in main_window.current_widget_parameters
    ):
        return main_window.current_widget_parameters[parameter_name]
    return main_window.default_parameters.get(parameter_name, default)


def register_parameter_widget_mirror(
    main_window: "MainWindow", parameter_name: str, widget: QtWidgets.QWidget
) -> None:
    mirrors = getattr(main_window, "_parameter_widget_mirrors", None)
    if mirrors is None:
        mirrors = {}
        main_window._parameter_widget_mirrors = mirrors
    mirrors.setdefault(parameter_name, []).append(widget)


def sync_all_widgets_for_parameter_key(
    main_window: "MainWindow", parameter_name: str, parameter_value
) -> None:
    mirrors = getattr(main_window, "_parameter_widget_mirrors", {}).get(
        parameter_name, ()
    )
    primary = main_window.parameter_widgets.get(parameter_name)
    seen: set[int] = set()
    for w in (*([primary] if primary is not None else []), *mirrors):
        if w is None or id(w) in seen:
            continue
        seen.add(id(w))
        w.blockSignals(True)
        try:
            _set_single_widget_value(w, parameter_value)
        finally:
            w.blockSignals(False)


def sync_parameter_mirror_widgets_only(
    main_window: "MainWindow", parameter_name: str, parameter_value
) -> None:
    for w in getattr(main_window, "_parameter_widget_mirrors", {}).get(
        parameter_name, ()
    ):
        w.blockSignals(True)
        try:
            _set_single_widget_value(w, parameter_value)
        finally:
            w.blockSignals(False)


def update_control(
    main_window: "MainWindow",
    control_name,
    control_value,
    exec_function: Optional[Callable] = None,
    exec_function_args: Optional[list] = None,
):
    exec_function_args = exec_function_args or []
    current_position = main_window.videoSeekSlider.value()

    # Update marker control too — always persist the dict value (including exec_function
    # controls); skipping exec_function keys made marker state stale on seek.
    if main_window.markers.get(current_position):
        main_window.markers[current_position]["control"][control_name] = control_value

    if exec_function:
        # Only execute the function if the value is different from current
        if main_window.control[control_name] != control_value:
            # By default an exec function definition should have atleast one parameter : MainWindow
            exec_function_args = [main_window, control_value] + exec_function_args
            exec_function(*exec_function_args)
    main_window.control[control_name] = control_value
    if control_name == "ScreenCaptureRegionRectText":
        main_window.control["ScreenCaptureRegionRect"] = str(control_value)
    # Also update the feeder's state if it's running
    # BUG-16 / THREAD-03: feeder_control None check moved inside the lock to prevent TOCTOU race
    if hasattr(main_window, "video_processor") and main_window.video_processor:
        # --- DIRTY FLAG ---
        main_window.video_processor.ui_state_is_dirty = True
        with main_window.video_processor.state_lock:
            # Cast to ControlTypes to satisfy the type checker, as feeder_control is typed
            if main_window.video_processor.feeder_control and control_name in cast(
                ControlTypes, main_window.video_processor.feeder_control
            ):
                cast(ControlTypes, main_window.video_processor.feeder_control)[
                    control_name
                ] = control_value
    sync_all_widgets_for_control_key(main_window, control_name, control_value)
    refresh_frame(main_window)


def create_default_parameter(
    main_window: "MainWindow", parameter_name, parameter_value
):
    main_window.default_parameters[parameter_name] = parameter_value


def create_parameter_dict_for_face_id(main_window: "MainWindow", face_id: str):
    if not main_window.parameters.get(face_id):
        parameters = (
            main_window.parameters.get(main_window.selected_target_face_id)
            or main_window.current_widget_parameters
            or main_window.default_parameters
        )
        if isinstance(parameters, dict):
            parameters = misc_helpers.ParametersDict(
                parameters, main_window.default_parameters
            )
        main_window.parameters[face_id] = parameters.copy()
    # print("Created parameter_dict_for_face_id", face_id)


def update_parameter(
    main_window: "MainWindow",
    parameter_name,
    parameter_value,
    enable_refresh_frame=True,
    exec_function: Optional[Callable] = None,
    exec_function_args: Optional[list] = None,
):
    exec_function_args = exec_function_args or []
    current_position = main_window.videoSeekSlider.value()
    face_id = main_window.selected_target_face_id

    # Get the old value for comparison before any updates
    old_parameter_value = None
    if (
        main_window.target_faces
        and face_id
        and parameter_name in main_window.parameters[face_id]
    ):
        old_parameter_value = main_window.parameters[face_id][parameter_name]
    elif (
        main_window.current_widget_parameters
        and parameter_name in main_window.current_widget_parameters
    ):
        old_parameter_value = main_window.current_widget_parameters[parameter_name]

    # --- Update the data dictionaries ---

    # Update marker parameters if applicable
    if main_window.markers.get(current_position) and face_id:
        main_window.markers[current_position]["parameters"][face_id][parameter_name] = (
            parameter_value
        )

    # Update parameters for the selected face
    if main_window.target_faces and face_id:
        main_window.parameters[face_id][parameter_name] = parameter_value
        # Also update the feeder's state if it's running
        # BUG-16 / THREAD-03: feeder_parameters None check moved inside the lock to prevent TOCTOU race
        if hasattr(main_window, "video_processor") and main_window.video_processor:
            # --- DIRTY FLAG ---
            main_window.video_processor.ui_state_is_dirty = True
            with main_window.video_processor.state_lock:
                if (
                    main_window.video_processor.feeder_parameters
                    and face_id in main_window.video_processor.feeder_parameters
                ):
                    main_window.video_processor.feeder_parameters[face_id][
                        parameter_name
                    ] = parameter_value

    # Always update the current widget state
    if main_window.current_widget_parameters:
        main_window.current_widget_parameters[parameter_name] = parameter_value

    # --- Trigger actions ---

    # Refresh the frame if needed
    if enable_refresh_frame:
        refresh_frame(main_window)

    # Execute the associated function if the value has changed
    # This now runs even if no face is selected, fixing the unload issue.
    if exec_function and parameter_value != old_parameter_value:
        # The first argument is always the main_window, followed by the new value
        final_exec_args: list = [main_window, parameter_value] + exec_function_args
        exec_function(*final_exec_args)

    sync_all_widgets_for_parameter_key(main_window, parameter_name, parameter_value)


def refresh_frame(main_window: "MainWindow", synchronous: bool = False):
    # PERF-05: Skip frame refresh if a batch update is in progress
    if getattr(main_window, "_batch_update_in_progress", False):
        return
    video_processor = main_window.video_processor
    if not video_processor.processing:
        video_processor.process_current_frame(synchronous=synchronous)


def _resolve_target_face_id(
    main_window: "MainWindow", face_id: str | None = None
) -> str | None:
    resolved_face_id = face_id or main_window.selected_target_face_id
    if resolved_face_id and resolved_face_id in main_window.target_faces:
        return resolved_face_id
    return None


def _show_target_face_parameter_message(
    main_window: "MainWindow", title: str, message: str
):
    create_and_show_messagebox(
        main_window,
        title,
        message,
        parent_widget=main_window,
    )


def copy_selected_face_parameters(
    main_window: "MainWindow", face_id: str | None = None
) -> bool:
    face_id = _resolve_target_face_id(main_window, face_id)
    if not face_id:
        _show_target_face_parameter_message(
            main_window,
            "No target face selected",
            "Select a target face before copying parameters.",
        )
        return False

    face_parameters = main_window.parameters.get(face_id)
    if not face_parameters:
        _show_target_face_parameter_message(
            main_window,
            "No parameters found",
            "The selected target face has no parameters to copy.",
        )
        return False

    main_window.copied_parameters = face_parameters.copy()
    return True


def paste_selected_face_parameters(
    main_window: "MainWindow", face_id: str | None = None
) -> bool:
    from app.ui.widgets.actions import video_control_actions

    if video_control_actions.block_if_issue_scan_active(
        main_window, "apply copied parameters"
    ):
        return False

    face_id = _resolve_target_face_id(main_window, face_id)
    if not face_id:
        _show_target_face_parameter_message(
            main_window,
            "No target face selected",
            "Select a target face before pasting parameters.",
        )
        return False

    if not main_window.copied_parameters:
        _show_target_face_parameter_message(
            main_window,
            "No parameters found in Clipboard",
            "You need to copy parameters from any of the target face before pasting it!",
        )
        return False

    main_window.parameters[face_id] = main_window.copied_parameters.copy()
    # --- DIRTY FLAG ---
    if hasattr(main_window, "video_processor") and main_window.video_processor:
        if main_window.video_processor.processing:
            main_window.video_processor.ui_state_is_dirty = True
            with main_window.video_processor.state_lock:
                if (
                    main_window.video_processor.feeder_parameters
                    and face_id in main_window.video_processor.feeder_parameters
                ):
                    import copy

                    main_window.video_processor.feeder_parameters[face_id] = (
                        copy.deepcopy(main_window.parameters[face_id])
                    )
    set_widgets_values_using_face_id_parameters(main_window, face_id=face_id)
    return True


def reset_selected_face_parameters(
    main_window: "MainWindow", face_id: str | None = None
) -> bool:
    face_id = _resolve_target_face_id(main_window, face_id)
    if not face_id:
        _show_target_face_parameter_message(
            main_window,
            "No target face selected",
            "Select a target face before resetting parameters.",
        )
        return False

    main_window.parameters[face_id] = main_window.default_parameters.copy()
    # --- DIRTY FLAG ---
    if hasattr(main_window, "video_processor") and main_window.video_processor:
        if main_window.video_processor.processing:
            main_window.video_processor.ui_state_is_dirty = True
            with main_window.video_processor.state_lock:
                if (
                    main_window.video_processor.feeder_parameters
                    and face_id in main_window.video_processor.feeder_parameters
                ):
                    import copy

                    main_window.video_processor.feeder_parameters[face_id] = (
                        copy.deepcopy(main_window.parameters[face_id])
                    )
    set_widgets_values_using_face_id_parameters(main_window, face_id=face_id)
    return True


# Keep a parameter row container and its child widgets in sync when hiding/showing
# dependency-driven controls.
def set_parameter_row_visibility(current_widget, visible: bool):
    if hasattr(current_widget, "row_widget") and current_widget.row_widget:
        current_widget.row_widget.setVisible(visible)
    else:
        current_widget.setVisible(visible)
    if hasattr(current_widget, "below_row_widget") and current_widget.below_row_widget:
        current_widget.below_row_widget.setVisible(visible)

    # Keep the child widgets in sync so internal widget state matches the row state.
    current_widget.setVisible(visible)
    if hasattr(current_widget, "label_widget") and current_widget.label_widget:
        current_widget.label_widget.setVisible(visible)
    if (
        hasattr(current_widget, "reset_default_button")
        and current_widget.reset_default_button
    ):
        current_widget.reset_default_button.setVisible(visible)
    if hasattr(current_widget, "line_edit") and current_widget.line_edit:
        current_widget.line_edit.setVisible(visible)


# Function to Hide Elements conditionally from values in LayoutData (Currently supports using Selection box and Toggle button to hide other widgets)
def show_hide_related_widgets(
    main_window: "MainWindow",
    parent_widget,
    parent_widget_name: str,
    value1=False,
    value2=False,
):
    if main_window.parameter_widgets:
        group_layout_data = parent_widget.group_layout_data  # Dictionary contaning layout data of all elements in the group of the parent_widget

        # --- CASE 1: PARENT IS A SELECTION BOX (e.g., Simple/Advanced) ---
        if "Selection" in parent_widget_name:
            for widget_name in group_layout_data.keys():
                current_widget = main_window.parameter_widgets.get(widget_name)
                layout_info = group_layout_data[widget_name]

                # Only process widgets that depend on THIS selection box
                if (
                    layout_info.get("parentSelection", "") == parent_widget_name
                    and current_widget
                ):
                    # 1. Check Selection Condition
                    selection_condition_met = (
                        layout_info.get("requiredSelectionValue")
                        == parent_widget.currentText()
                    )

                    # 2. Check Toggle Condition (Cross-Check)
                    # Even if selection matches, we must check if the parent toggles are ON
                    toggle_condition_met = True
                    parentToggles = layout_info.get("parentToggle", "")

                    if parentToggles and selection_condition_met:
                        if "&" in parentToggles:
                            toggles = [t.strip() for t in parentToggles.split("&")]
                            for t_name in toggles:
                                t_widget = main_window.parameter_widgets.get(t_name)
                                if t_widget and not t_widget.isChecked():
                                    toggle_condition_met = False
                                    break
                        elif "|" in parentToggles:
                            toggle_condition_met = False
                            toggles = [t.strip() for t in parentToggles.split("|")]
                            for t_name in toggles:
                                t_widget = main_window.parameter_widgets.get(t_name)
                                if t_widget and t_widget.isChecked():
                                    toggle_condition_met = True
                                    break
                        else:
                            # Single toggle or simple logic
                            t_widget = main_window.parameter_widgets.get(parentToggles)
                            required_val = layout_info.get("requiredToggleValue", True)
                            if t_widget and t_widget.isChecked() != required_val:
                                toggle_condition_met = False

                    # Final Decision
                    if selection_condition_met and toggle_condition_met:
                        set_parameter_row_visibility(current_widget, True)
                    else:
                        set_parameter_row_visibility(current_widget, False)

        # --- CASE 2: PARENT IS A TOGGLE BUTTON ---
        elif "Toggle" in parent_widget_name:
            for widget_name in group_layout_data.keys():
                if widget_name not in main_window.parameter_widgets:
                    continue
                current_widget = main_window.parameter_widgets[widget_name]
                layout_info = group_layout_data[widget_name]

                parentToggles = layout_info.get("parentToggle", "")

                # Only process widgets that depend on THIS toggle (or have it in their chain)
                if parent_widget_name in parentToggles:
                    # 1. Check Selection Condition (Cross-Check)
                    # Before evaluating toggles, check if the parent Selection is valid
                    selection_condition_met = True
                    parentSelection = layout_info.get("parentSelection", "")
                    if parentSelection:
                        sel_widget = main_window.parameter_widgets.get(parentSelection)
                        if sel_widget and sel_widget.currentText() != layout_info.get(
                            "requiredSelectionValue"
                        ):
                            selection_condition_met = False

                    # 2. Check Toggle Condition
                    toggle_condition_met = False

                    # DEPRECATED: comma-separated toggle syntax; evaluates only the last toggle. Use '&' or '|' instead.
                    if "," in parentToggles:
                        # Legacy comma logic (iterative check)
                        result = [item.strip() for item in parentToggles.split(",")]
                        parentToggle_ischecked = False
                        for _, required_widget_name in enumerate(result):
                            w = main_window.parameter_widgets.get(required_widget_name)
                            if w:
                                parentToggle_ischecked = w.isChecked()

                        if (
                            layout_info.get("requiredToggleValue")
                            == parentToggle_ischecked
                        ):
                            toggle_condition_met = True

                    elif "|" in parentToggles:
                        # OR Logic
                        result = [item.strip() for item in parentToggles.split("|")]
                        any_checked = False
                        for required_widget_name in result:
                            w = main_window.parameter_widgets.get(required_widget_name)
                            if w and w.isChecked():
                                any_checked = True
                                break

                        if layout_info.get("requiredToggleValue") == any_checked:
                            toggle_condition_met = True

                    elif "&" in parentToggles:
                        # AND Logic
                        result = [item.strip() for item in parentToggles.split("&")]
                        all_checked = True
                        for required_widget_name in result:
                            w = main_window.parameter_widgets.get(required_widget_name)
                            if w and not w.isChecked():
                                all_checked = False
                                break

                        if layout_info.get("requiredToggleValue") == all_checked:
                            toggle_condition_met = True

                    else:
                        # Single Toggle
                        w = main_window.parameter_widgets.get(parentToggles)
                        parentToggle_ischecked = w.isChecked() if w else False
                        if (
                            layout_info.get("requiredToggleValue")
                            == parentToggle_ischecked
                        ):
                            toggle_condition_met = True

                    # Final Decision
                    if selection_condition_met and toggle_condition_met:
                        set_parameter_row_visibility(current_widget, True)
                    else:
                        set_parameter_row_visibility(current_widget, False)

            parent_widget.start_animation()


# @misc_helpers.benchmark
def get_pixmap_from_frame(main_window: "MainWindow", frame: np.ndarray):
    frame = np.ascontiguousarray(frame)
    height, width, channel = frame.shape
    # BUG-04: Grayscale check should be channel==1, not 2
    if channel == 1:
        # Frame in grayscale
        bytes_per_line = width
        q_img = QtGui.QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format.Format_Grayscale8,
        )
    else:
        # Frame in color
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(
            frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888
        ).rgbSwapped()
    pixmap = QtGui.QPixmap.fromImage(q_img)
    return pixmap


def update_gpu_memory_progressbar(main_window: "MainWindow"):
    rows = main_window.models_processor.get_all_gpus_memory_mb()
    main_window.gpu_memory_update_signal.emit(rows)


def _layout_and_stretch_index_for_widget(
    widget: QtWidgets.QWidget,
) -> tuple[QtWidgets.QLayout | None, int]:
    """Return the innermost layout that directly holds *widget* and its stretch index."""
    parent_widget = widget.parentWidget()
    if parent_widget is None:
        return None, -1
    top = parent_widget.layout()
    if top is None:
        return None, -1

    def walk(layout: QtWidgets.QLayout) -> tuple[QtWidgets.QLayout | None, int]:
        idx = layout.indexOf(widget)
        if idx >= 0:
            return layout, idx
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            sub = item.layout()
            if sub is not None:
                inner, j = walk(sub)
                if inner is not None and j >= 0:
                    return inner, j
        return None, -1

    return walk(top)


def _vram_physical_display_order(num_gpus: int, primary_phys: int) -> list[int]:
    """Bar slot 0 = primary CUDA device, then the rest (stable order)."""
    if num_gpus <= 0:
        return []
    p = max(0, min(int(primary_phys), num_gpus - 1))
    return [p] + [i for i in range(num_gpus) if i != p]


def _vram_bar_gpu_label(phys_idx: int, *, is_primary: bool) -> str:
    import torch

    tag = f"GPU {phys_idx}"
    if is_primary:
        tag += " · primary"
    if torch.cuda.is_available() and 0 <= phys_idx < int(torch.cuda.device_count()):
        try:
            name = torch.cuda.get_device_name(phys_idx)
            tag = f"GPU {phys_idx}: {name}"
            if is_primary:
                tag += " · primary"
        except Exception:
            pass
    return tag


def setup_vram_progress_bars_layout(main_window: "MainWindow") -> None:
    """Replace the single VRAM bar slot with a vertical stack of one bar per CUDA GPU."""
    import torch

    from app.ui.widgets.vram_progress_bar import VramPeakProgressBar

    bar0 = main_window.vramProgressBar
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) <= 1:
        main_window._vram_progress_bars = [bar0]
        return

    parent = bar0.parentWidget()
    hlay, idx = _layout_and_stretch_index_for_widget(bar0)
    if parent is None or hlay is None or idx < 0:
        main_window._vram_progress_bars = [bar0]
        return

    n = int(torch.cuda.device_count())
    container = QtWidgets.QWidget(parent)
    vlay = QtWidgets.QVBoxLayout(container)
    vlay.setContentsMargins(0, 0, 0, 0)
    vlay.setSpacing(3)

    hlay.removeWidget(bar0)
    bars: list = []
    for i in range(n):
        if i == 0:
            b = bar0
        else:
            b = VramPeakProgressBar(container)
            b.setMinimumHeight(max(18, bar0.minimumHeight()))
            mh = bar0.maximumHeight()
            if mh > 0:
                b.setMaximumHeight(mh)
            sp = bar0.sizePolicy()
            b.setSizePolicy(sp.horizontalPolicy(), sp.verticalPolicy())
        b.setObjectName(f"vramProgressBar_gpu{i}")
        b.setFont(bar0.font())
        bars.append(b)
        vlay.addWidget(b)

    hlay.insertWidget(idx, container)
    main_window._vram_progress_bars = bars


def _vram_bar_stylesheet_for_usage(
    memory_used: int, memory_total: int, bar: QtWidgets.QProgressBar
) -> str:
    palette = bar.palette()
    background_color = palette.color(QtGui.QPalette.ColorRole.Base).name()
    text_color = palette.color(QtGui.QPalette.ColorRole.Text).name()
    border_color = palette.color(QtGui.QPalette.ColorRole.Mid).name()

    base_style = f"""
        QProgressBar {{
            border: 1px solid {border_color};
            border-radius: 5px;
            text-align: center;
            background-color: {background_color};
            color: {text_color};
        }}
    """

    chunk_style_normal = """
        QProgressBar::chunk {
            background-color: #16759e;
            border-radius: 4px;
        }
    """

    chunk_style_high = """
        QProgressBar::chunk {
            background-color: #911414;
            border-radius: 4px;
        }
    """

    is_high = memory_total > 0 and (memory_used / memory_total) > 0.85
    return base_style + (chunk_style_high if is_high else chunk_style_normal)


@QtCore.Slot(object)
def set_gpu_memory_progressbars_values(main_window: "MainWindow", memory_rows):
    """*memory_rows*: list of ``(used_MB, total_MB)`` per CUDA ordinal (index = device id)."""
    if not isinstance(memory_rows, list):
        memory_rows = list(memory_rows) if memory_rows else []

    bars = getattr(main_window, "_vram_progress_bars", None) or [
        main_window.vramProgressBar
    ]
    mp = main_window.models_processor
    try:
        primary_phys = int(mp._primary_cuda_device_ordinal())
    except Exception:
        try:
            primary_phys = int(
                main_window.control.get("GpuPrimaryPhysicalIndex", 0)
            )
        except Exception:
            primary_phys = 0
    n_bars = len(bars)
    if n_bars > 0:
        primary_phys = max(0, min(primary_phys, n_bars - 1))

    order = _vram_physical_display_order(n_bars, primary_phys)
    for slot, phys_idx in enumerate(order):
        if slot >= n_bars:
            break
        bar = bars[slot]
        if phys_idx >= len(memory_rows):
            bar.setMaximum(1)
            bar.setValue(0)
            bar.setFormat(f"GPU {phys_idx}: —")
            continue
        memory_used, memory_total = memory_rows[phys_idx]
        bar.setMaximum(max(1, memory_total))
        bar.setValue(min(memory_used, memory_total))
        bar.note_used_mb(memory_used)
        tag = _vram_bar_gpu_label(phys_idx, is_primary=(phys_idx == primary_phys))
        bar.setFormat(
            f"{tag}: {round(memory_used / 1024, 2)} GB / "
            f"{round(memory_total / 1024, 2)} GB (%p%)"
        )
        st = _vram_bar_stylesheet_for_usage(memory_used, memory_total, bar)
        bar.setStyleSheet(st)
        bar.update()
    for slot in range(len(order), n_bars):
        bar = bars[slot]
        bar.setMaximum(1)
        bar.setValue(0)
        bar.setFormat("GPU: —")


def clear_gpu_memory(main_window: "MainWindow"):
    main_window.video_processor.stop_processing()
    main_window.models_processor.clear_gpu_memory()
    for _b in getattr(main_window, "_vram_progress_bars", [main_window.vramProgressBar]):
        _b.reset_peak()
    main_window.swapfacesButton.setChecked(False)
    main_window.editFacesButton.setChecked(False)
    from app.ui.widgets.actions import preview_notification_actions as _preview_notify

    _preview_notify.show_swap_faces_state(main_window, False)
    update_gpu_memory_progressbar(main_window)

    # main_window.videoSeekSlider.markers = set() # Comment this to keep markers visible after vram clear
    main_window.videoSeekSlider.update()


def extract_frame_as_image(
    main_window: "MainWindow",
    media_file_path,
    file_type,
    webcam_index=False,
    webcam_backend=False,
    cache_thumbnail=True,
):
    """
    Extracts a frame from a media file and converts it to a QImage for thumbnails.
    Returns a thread-safe QImage to avoid GUI/GPU crashes from background workers.
    It uses the ThumbnailManager to efficiently cache and retrieve thumbnails.
    """

    # This helper function converts a numpy frame to a scaled QImage safely.
    def convert_frame_to_image(frame):
        frame = np.ascontiguousarray(frame)
        height, width, _ = frame.shape
        bytes_per_line = 3 * width

        # Format_RGB888 strictly respects PySide6 enums
        q_img = QtGui.QImage(
            frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888
        ).rgbSwapped()

        # .scaled() returns a deep copy, completely decoupling from the numpy array memory!
        return q_img.scaled(70, 70, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    # For images and videos, first check for a cached thumbnail.
    if file_type in ["image", "video"]:
        # We use the thumbnail_manager instance from the main_window.
        thumbnail_path = main_window.thumbnail_manager.find_existing_thumbnail(
            media_file_path
        )

        # Freshness check: if the source file has been modified more recently
        # than the cached thumbnail (e.g., the user re-recorded over the same
        # filename), force regeneration. Without this guard a stale thumbnail
        # is served indefinitely.
        if thumbnail_path:
            try:
                import os as _os

                if _os.path.getmtime(thumbnail_path) < _os.path.getmtime(
                    media_file_path
                ):
                    thumbnail_path = None
            except OSError:
                pass  # source file missing — fall through to regeneration

        if thumbnail_path:
            frame = misc_helpers.read_image_file(thumbnail_path)
            if frame is not None:
                return convert_frame_to_image(frame)

    # If no cache is found, or for webcams, generate the frame from source.
    frame = None
    if file_type == "image":
        frame = misc_helpers.read_image_file(media_file_path)
    elif file_type == "video":
        # Get rotation for thumbnail
        rotation_angle = get_video_rotation(media_file_path)
        # Retry up to 3 times with a 500 ms backoff.  Freshly recorded output files
        # may still be held open by ffmpeg or the OS write cache when the media panel
        # first tries to generate a thumbnail, causing cap.isOpened() to fail or
        # CAP_PROP_FRAME_COUNT to return 0.  This is especially common on Windows 10.
        import time as _time

        for _attempt in range(3):
            cap = cv2.VideoCapture(media_file_path)
            if cap.isOpened():
                # Explicitly enable OpenCV's auto-rotation
                if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    middle_frame_no = total_frames // 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_no)
                    ret, frame = misc_helpers.read_frame(cap, rotation_angle)
                    cap.release()
                    if ret:
                        break  # success — exit retry loop
                    cap = None  # mark for next attempt
                else:
                    cap.release()
                    cap = None
            else:
                cap = None
            if _attempt < 2:
                _time.sleep(0.5)  # wait before retry (file may still be flushed)
    elif file_type == "webcam":
        camera = cv2.VideoCapture(webcam_index, webcam_backend)
        if camera.isOpened():
            # Pass 0 for webcam rotation
            ret, frame = misc_helpers.read_frame(camera, 0)
            camera.release()  # Release camera immediately after grabbing one frame

    if isinstance(frame, np.ndarray):
        # Create a new thumbnail in the cache for next time.
        if file_type != "webcam" and cache_thumbnail:
            main_window.thumbnail_manager.create_thumbnail(frame, media_file_path)

        # Return the generated thread-safe QImage.
        return convert_frame_to_image(frame)

    return None  # Return None if everything failed.


# QUAL-05: Helper to set a single widget's value based on its type, extracted from duplicate code
def _set_single_widget_value(widget, value):
    if isinstance(
        widget,
        (
            widget_components.ParameterLineEdit,
            widget_components.ParameterSlider,
        ),
    ):
        try:
            int_value = int(float(value))
            widget.set_value(int_value)
        except (ValueError, TypeError):
            pass
    elif isinstance(
        widget,
        (
            widget_components.ParameterLineDecimalEdit,
            widget_components.ParameterDecimalSlider,
        ),
    ):
        try:
            float_value = float(value)
            widget.set_value(float_value)
        except (ValueError, TypeError):
            pass
    elif isinstance(widget, widget_components.ToggleButton):
        widget.set_value(bool(value))
    elif isinstance(widget, widget_components.SelectionBox):
        widget.set_value(str(value))
    else:
        widget.set_value(value)


# QUAL-06: Changed face_id default from False to None; changed (face_id is False) to (not face_id)
def set_widgets_values_using_face_id_parameters(
    main_window: "MainWindow", face_id=None
):
    if not face_id or (not main_window.parameters.get(face_id)):
        # print("Set widgets values using default parameters")
        if main_window.current_widget_parameters:
            parameters = main_window.current_widget_parameters.copy()
        else:
            parameters = main_window.default_parameters
    else:
        # print(f"Set widgets values using face_id {face_id}")
        parameters = main_window.parameters[face_id].copy()
    parameter_widgets = main_window.parameter_widgets
    # Preserve outer suppression state so nested batch operations do not override it.
    previous_batch_flag = getattr(main_window, "_batch_update_in_progress", False)
    # PERF-05: Set batch update flag to suppress per-widget refresh_frame calls during the loop
    main_window._batch_update_in_progress = True
    try:
        for parameter_name, parameter_value in parameters.items():
            # Global controls (including Face Swap widgets with data_type "control")
            # must not be overwritten from face/current parameter dicts.
            if parameter_name in main_window.control:
                continue
            widget = parameter_widgets.get(parameter_name)
            if widget:
                # temporarily disable refreshing the frame to prevent slowing due to unnecessary processing
                widget.enable_refresh_frame = False
                # QUAL-05: Delegate to shared helper instead of inline isinstance chain
                _set_single_widget_value(widget, parameter_value)
                widget.enable_refresh_frame = True
    finally:
        main_window._batch_update_in_progress = previous_batch_flag
        # Trigger a single refresh only if this function owns the outermost batch scope.
        if not previous_batch_flag:
            refresh_frame(main_window)


def set_control_widgets_values(main_window: "MainWindow", enable_exec_func=True):
    """
    Set the values of control widgets based on the `control` data in the `main_window`.

    Temporarily disables frame refreshing while setting values to avoid unnecessary processing.
    """
    # Get control values and parameter widgets from the main window
    control = main_window.control.copy()
    parameter_widgets = main_window.parameter_widgets

    # PERF-01: Use the module-level pre-built constant instead of rebuilding the dict on every call
    all_widget_options = _ALL_CONTROL_WIDGET_OPTIONS

    # Iterate through control items and update widgets
    for control_name, control_value in control.items():
        widget = parameter_widgets.get(control_name)

        if widget:
            # Temporarily disable frame refresh
            widget.enable_refresh_frame = False

            # QUAL-05: Delegate to shared helper instead of inline isinstance chain
            _set_single_widget_value(widget, control_value)

            if enable_exec_func:
                # Execute any associated function, if defined
                widget_definition = all_widget_options.get(
                    control_name
                )  # Use .get() for safety
                if widget_definition:
                    exec_function_data = widget_definition.get("exec_function")
                    if exec_function_data:
                        # The functions in control_actions.py are typically (main_window, value, *additional_args)
                        exec_args_from_layout = widget_definition.get(
                            "exec_function_args", []
                        )
                        final_exec_args = [
                            main_window,
                            control_value,
                        ] + exec_args_from_layout
                        exec_function_data(*final_exec_args)

            # Re-enable frame refresh
            widget.enable_refresh_frame = True


@QtCore.Slot(QtWidgets.QListWidget, bool)
def update_placeholder_visibility(
    main_window: "MainWindow", list_widget: QtWidgets.QListWidget, default_hide
):
    # """Update the visibility of the placeholder text."""
    # """
    #     The default_hide parameter is used to Hide the placeholder text by default.
    #     If the default_hide is False, then the visibility of the placeholder text is set using the size of the list_widget
    # """
    if default_hide:
        is_visible = False
    else:
        is_visible = list_widget.count() == 0
    list_widget.placeholder_label.setVisible(is_visible)
    # Set Cursor on the List Widget
    if is_visible:
        list_widget.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    else:
        list_widget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
    # print("SetVisible", is_visible)
    # print("targetVideosList.count()", list_widget.count())


@QtCore.Slot()
def show_model_loading_dialog(main_window: "MainWindow"):
    # Debounce: Only show dialog if loading takes longer than 300ms
    if not hasattr(main_window, "_model_loading_timer"):
        main_window._model_loading_timer = QtCore.QTimer()
        main_window._model_loading_timer.setSingleShot(True)

        def show_dialog():
            if (
                not hasattr(main_window, "model_loading_dialog")
                or main_window.model_loading_dialog is None
            ):
                main_window.model_loading_dialog = widget_components.LoadingDialog()
            if not main_window.model_loading_dialog.isVisible():
                main_window.model_loading_dialog.show()
                # Excluding user input keeps a pending mouse release queued for
                # its original widget; a plain processEvents() here delivers it
                # early and leaves the video seek slider latched in drag state.
                QtWidgets.QApplication.processEvents(
                    QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                )

        main_window._model_loading_timer.timeout.connect(show_dialog)
    # Start or restart the timer
    main_window._model_loading_timer.start(300)


@QtCore.Slot()
def hide_model_loading_dialog(main_window: "MainWindow"):
    # Stop the timer if it's running
    if hasattr(main_window, "_model_loading_timer"):
        main_window._model_loading_timer.stop()
    # Only hide if dialog exists and is visible
    if (
        hasattr(main_window, "model_loading_dialog")
        and main_window.model_loading_dialog is not None
    ):
        if main_window.model_loading_dialog.isVisible():
            main_window.model_loading_dialog.hide()
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
            )
