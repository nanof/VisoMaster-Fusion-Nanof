from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import cv2
import numpy as np
from pyqttoast import Toast, ToastPreset, ToastPosition
from PySide6 import QtWidgets, QtCore, QtGui

from app.ui.widgets import widget_components
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
import app.helpers.miscellaneous as misc_helpers
from app.helpers.miscellaneous import get_video_rotation, THUMBNAIL_MAX_EDGE_PX
from app.helpers.typing_helper import ControlTypes

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


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
    toast.setDuration(1400)
    toast.setPosition(ToastPosition.TOP_RIGHT)  # Default: ToastPosition.BOTTOM_RIGHT
    toast.applyPreset(style_preset_map[style_type])  # Apply style preset
    toast.show()


def create_control(main_window: "MainWindow", control_name, control_value):
    main_window.control[control_name] = control_value


def _set_single_widget_value(widget: QtWidgets.QWidget, value) -> None:
    """Update a settings widget without :func:`update_control` (batch / preset paths)."""
    setter = getattr(widget, "set_value", None)
    if callable(setter):
        setter(value)
        return
    if isinstance(value, bool) and hasattr(widget, "setChecked"):
        widget.setChecked(value)
        return
    if hasattr(widget, "setCurrentText"):
        widget.setCurrentText(str(value))
        return
    if isinstance(widget, QtWidgets.QAbstractSlider):
        widget.setValue(int(value))


def update_control(
    main_window: "MainWindow",
    control_name,
    control_value,
    exec_function: Optional[Callable] = None,
    exec_function_args: Optional[list] = None,
):
    exec_function_args = exec_function_args or []
    current_position = main_window.videoSeekSlider.value()

    # Update marker control too
    # Do not update values of control with exec_function (like max threads count) as it would slow down the app heavily
    if main_window.markers.get(current_position) and not exec_function:
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
    if (
        main_window.video_processor.processing
        and main_window.video_processor.feeder_control
    ):
        with main_window.video_processor.state_lock:
            # Cast to ControlTypes to satisfy the type checker, as feeder_control is typed
            if control_name in cast(
                ControlTypes, main_window.video_processor.feeder_control
            ):
                cast(ControlTypes, main_window.video_processor.feeder_control)[
                    control_name
                ] = control_value
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
        if (
            main_window.video_processor.processing
            and main_window.video_processor.feeder_parameters
        ):
            with main_window.video_processor.state_lock:
                if face_id in main_window.video_processor.feeder_parameters:
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


def refresh_frame(main_window: "MainWindow", synchronous: bool = False):
    if getattr(main_window, "_loading_workspace", False):
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
    set_widgets_values_using_face_id_parameters(main_window, face_id=face_id)
    return True


# Function to Hide Elements conditionally from values in LayoutData (Currently supports using Selection box and Toggle button to hide other widgets)
def show_hide_related_widgets(
    main_window: "MainWindow",
    parent_widget,
    parent_widget_name: str,
    value1=False,
    value2=False,
):
    if main_window.parameter_widgets:
        group_layout_data = getattr(parent_widget, "group_layout_data", None)
        if not group_layout_data:
            return
        if "Selection" in parent_widget_name:
            # Loop through all widgets data in the parent widget's group layout data
            for widget_name in group_layout_data.keys():
                # Store the widget object (instance) from the parameters_widgets Dictionary
                current_widget = main_window.parameter_widgets.get(widget_name)
                # Check if the current_widget depends on the Parent Widget's (selection) value
                if (
                    group_layout_data[widget_name].get("parentSelection", "")
                    == parent_widget_name
                    and current_widget
                ):
                    # Check if the current_widget has the required value of Parent Widget's (selection) current value to hide/show the current_widget
                    if (
                        group_layout_data[widget_name].get("requiredSelectionValue")
                        != parent_widget.currentText()
                    ):
                        current_widget.hide()
                        current_widget.label_widget.hide()
                        current_widget.reset_default_button.hide()
                        if current_widget.line_edit:
                            current_widget.line_edit.hide()
                    else:
                        current_widget.show()
                        current_widget.label_widget.show()
                        current_widget.reset_default_button.show()
                        if current_widget.line_edit:
                            current_widget.line_edit.show()

        elif "Toggle" in parent_widget_name:
            # Loop through all widgets data in the parent widget's group layout data
            for widget_name in group_layout_data.keys():
                # Store the widget object (instance) from the parameters_widgets Dictionary
                if widget_name not in main_window.parameter_widgets:
                    continue
                current_widget = main_window.parameter_widgets[widget_name]
                # Check if the current_widget depends on the Parent Widget's (toggle) value
                parentToggles = group_layout_data[widget_name].get("parentToggle", "")
                if parent_widget_name in parentToggles:
                    if "," in parentToggles:
                        result = [item.strip() for item in parentToggles.split(",")]
                        parentToggle_ischecked = False
                        for _, required_widget_name in enumerate(result):
                            parentToggle_ischecked = main_window.parameter_widgets[
                                required_widget_name
                            ].isChecked()
                        # Check if the current_widget has the required toggle value of Parent Widget's (toggle) checked state to hide/show the current_widget
                        if (
                            group_layout_data[widget_name].get("requiredToggleValue")
                            != parentToggle_ischecked
                        ):
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()
                    elif "|" in parentToggles:
                        result = [item.strip() for item in parentToggles.split("|")]
                        parentToggle_ischecked = True
                        # Check if any of the parentToggles are checked
                        for _, required_widget_name in enumerate(result):
                            ischecked = main_window.parameter_widgets[
                                required_widget_name
                            ].isChecked()
                            if ischecked:
                                parentToggle_ischecked = True
                                break
                            else:
                                parentToggle_ischecked = False

                        # Check if the current_widget has the required toggle value of Parent Widget's (toggle) checked state to hide/show the current_widget
                        if (
                            group_layout_data[widget_name].get("requiredToggleValue")
                            != parentToggle_ischecked
                        ):
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()

                    elif "&" in parentToggles:
                        result = [item.strip() for item in parentToggles.split("&")]
                        parentToggle_ischecked = True
                        # Check if any of the parentToggles are checked
                        for _, required_widget_name in enumerate(result):
                            ischecked = main_window.parameter_widgets[
                                required_widget_name
                            ].isChecked()
                            parentToggle_ischecked = (
                                parentToggle_ischecked and ischecked
                            )

                        # Check if the current_widget has the required toggle value of Parent Widget's (toggle) checked state to hide/show the current_widget
                        if (
                            group_layout_data[widget_name].get("requiredToggleValue")
                            != parentToggle_ischecked
                        ):
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()

                    else:
                        parentToggle_ischecked = main_window.parameter_widgets[
                            parentToggles
                        ].isChecked()
                        if (
                            group_layout_data[widget_name].get("requiredToggleValue")
                            != parentToggle_ischecked
                        ):
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()

            parent_widget.start_animation()


# @misc_helpers.benchmark
def get_pixmap_from_frame(main_window: "MainWindow", frame: np.ndarray):
    frame = np.ascontiguousarray(frame)
    height, width, channel = frame.shape
    if channel == 2:
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
        # Frame in color — pipeline uses OpenCV BGR uint8. Format_BGR888 avoids a full
        # buffer copy from rgbSwapped() on every preview frame (major UI-thread win).
        bytes_per_line = 3 * width
        fmt_bgr = getattr(QtGui.QImage.Format, "Format_BGR888", None)
        if fmt_bgr is not None:
            q_img = QtGui.QImage(
                frame.data, width, height, bytes_per_line, fmt_bgr
            )
        else:
            q_img = QtGui.QImage(
                frame.data,
                width,
                height,
                bytes_per_line,
                QtGui.QImage.Format.Format_RGB888,
            ).rgbSwapped()
    pixmap = QtGui.QPixmap.fromImage(q_img)
    return pixmap


def update_gpu_memory_progressbar(main_window: "MainWindow"):
    _update_gpu_memory_progressbar(main_window)


def _update_gpu_memory_progressbar(main_window: "MainWindow"):
    memory_used, memory_total = main_window.models_processor.get_gpu_memory()
    main_window.gpu_memory_update_signal.emit(memory_used, memory_total)


@QtCore.Slot(int, int)
def set_gpu_memory_progressbar_value(
    main_window: "MainWindow", memory_used, memory_total
):
    main_window.vramProgressBar.setMaximum(memory_total)
    main_window.vramProgressBar.setValue(memory_used)
    main_window.vramProgressBar.note_used_mb(memory_used)
    main_window.vramProgressBar.setFormat(
        f"{round(memory_used / 1024, 2)} GB / {round(memory_total / 1024, 2)} GB (%p%)"
    )
    palette = main_window.vramProgressBar.palette()
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
            background-color: #911414; /* Red */
            border-radius: 4px;
        }
    """

    is_high = memory_total > 0 and (memory_used / memory_total) > 0.85
    was_high = getattr(main_window, "_vram_high_style_active", None)
    current_style = base_style + (chunk_style_high if is_high else chunk_style_normal)
    if (
        is_high != was_high
        or getattr(main_window, "_vram_progressbar_style", None) != current_style
    ):
        main_window._vram_high_style_active = is_high
        main_window._vram_progressbar_style = current_style
        main_window.vramProgressBar.setStyleSheet(current_style)

    main_window.vramProgressBar.update()


def clear_gpu_memory(main_window: "MainWindow"):
    main_window.video_processor.stop_processing()
    main_window.models_processor.clear_gpu_memory()
    main_window.vramProgressBar.reset_peak()
    main_window.swapfacesButton.setChecked(False)
    main_window.editFacesButton.setChecked(False)
    from app.ui.widgets.actions import preview_notification_actions as _preview_notify

    _preview_notify.show_swap_faces_state(main_window, False)
    update_gpu_memory_progressbar(main_window)

    # main_window.videoSeekSlider.markers = set() # Comment this to keep markers visible after vram clear
    main_window.videoSeekSlider.update()


def extract_frame_as_pixmap(
    main_window: "MainWindow",
    media_file_path,
    file_type,
    webcam_index=False,
    webcam_backend=False,
):
    """
    Extracts a frame from a media file and converts it to a QPixmap for thumbnails.
    It now uses the ThumbnailManager to efficiently cache and retrieve thumbnails.
    """

    # This helper function converts a numpy frame to a scaled QPixmap.
    def convert_frame_to_pixmap(frame):
        frame = np.ascontiguousarray(frame)
        height, width, _ = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(
            frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_img)
        return pixmap.scaled(
            THUMBNAIL_MAX_EDGE_PX,
            THUMBNAIL_MAX_EDGE_PX,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

    # For images and videos, first check for a cached thumbnail.
    if file_type in ["image", "video"]:
        # We use the thumbnail_manager instance from the main_window.
        thumbnail_path = main_window.thumbnail_manager.find_existing_thumbnail(
            media_file_path
        )

        if thumbnail_path:
            frame = misc_helpers.read_image_file(thumbnail_path)
            if frame is not None:
                return convert_frame_to_pixmap(frame)

    # If no cache is found, or for webcams, generate the frame from source.
    frame = None
    if file_type == "image":
        frame = misc_helpers.read_image_file(media_file_path)
    elif file_type == "video":
        # MODIFICATION: Get rotation for thumbnail
        rotation_angle = get_video_rotation(media_file_path)
        cap = cv2.VideoCapture(media_file_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame_no = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_no)
            ret, frame = misc_helpers.read_frame(cap, rotation_angle)
            cap.release()
    elif file_type == "webcam":
        camera = cv2.VideoCapture(webcam_index, webcam_backend)
        if camera.isOpened():
            # MODIFICATION: Pass 0 for webcam rotation
            ret, frame = misc_helpers.read_frame(camera, 0)
            camera.release()  # Release camera immediately after grabbing one frame
    elif file_type == "screen":
        from app.helpers.screen_capture import grab_one_frame_bgr, mss_available

        if mss_available():
            _ok, frame = grab_one_frame_bgr(main_window.control)
            if not _ok:
                frame = None

    if isinstance(frame, np.ndarray):
        # Create a new thumbnail in the cache for next time.
        if file_type not in ("webcam", "screen"):
            main_window.thumbnail_manager.create_thumbnail(frame, media_file_path)

        # Return the generated pixmap.
        return convert_frame_to_pixmap(frame)

    return None  # Return None if everything failed.


def extract_frame_as_image(
    main_window: "MainWindow",
    media_file_path,
    file_type,
    webcam_index=False,
    webcam_backend=False,
):
    """Like :func:`extract_frame_as_pixmap` but returns a ``QImage`` for thread thumbnail signals."""
    pix = extract_frame_as_pixmap(
        main_window,
        media_file_path,
        file_type,
        webcam_index=webcam_index,
        webcam_backend=webcam_backend,
    )
    if pix is None or pix.isNull():
        return None
    return pix.toImage()


def read_parameter_widget_value(widget) -> Any:
    """Current value from a parameter tab widget (for save / exec hooks)."""
    if isinstance(widget, widget_components.ToggleButton):
        return widget.isChecked()
    if isinstance(widget, widget_components.SelectionBox):
        return widget.currentText()
    if isinstance(widget, widget_components.ParameterDecimalSlider):
        return float(widget.value())
    if isinstance(widget, widget_components.ParameterSlider):
        return int(widget.value())
    if isinstance(widget, widget_components.ParameterText):
        return widget.text()
    raise TypeError(f"Unsupported parameter widget: {type(widget).__name__}")


def flush_parameter_widgets_into_storage(main_window: "MainWindow") -> None:
    """Copy visible parameter widget values into current_widget_parameters and the selected face.

    ParametersDict.data is sparse (only keys ever written by signals). Sliders can also
    lag behind the UI until debounce fires. Call this before saving workspace.
    """
    fid = main_window.selected_target_face_id
    for name in main_window.default_parameters:
        w = main_window.parameter_widgets.get(name)
        if w is None:
            continue
        try:
            val = read_parameter_widget_value(w)
        except TypeError:
            continue
        if main_window.current_widget_parameters is not None:
            main_window.current_widget_parameters[name] = val
        if fid and fid in main_window.parameters:
            main_window.parameters[fid][name] = val


def merged_parameter_dict_for_save(
    main_window: "MainWindow",
    source: misc_helpers.ParametersDict | dict | None,
) -> dict:
    """Full parameter map for JSON: layout defaults merged with explicit overrides."""
    base = dict(main_window.default_parameters)
    if source is None:
        return base
    if isinstance(source, misc_helpers.ParametersDict):
        base.update(source.data)
        return base
    base.update(source)
    return base


def set_widgets_values_using_face_id_parameters(
    main_window: "MainWindow", face_id=False
):
    if (face_id is False) or (not main_window.parameters.get(face_id)):
        # print("Set widgets values using default parameters")
        if main_window.current_widget_parameters:
            parameters = main_window.current_widget_parameters.copy()
        else:
            parameters = main_window.default_parameters
    else:
        # print(f"Set widgets values using face_id {face_id}")
        parameters = main_window.parameters[face_id].copy()
        # Restorer / swapper exec handlers read current_widget_parameters; sync the
        # full per-face dict before touching widgets so signal order cannot leave a
        # mix of stale defaults and new values (e.g. type selection vs enable toggle).
        main_window.current_widget_parameters = main_window.parameters[face_id]
    parameter_widgets = main_window.parameter_widgets
    for parameter_name, parameter_value in parameters.items():
        widget = parameter_widgets.get(parameter_name)
        if widget:
            # temporarily disable refreshing the frame to prevent slowing due to unnecessary processing
            widget.enable_refresh_frame = False
            if isinstance(
                widget,
                (
                    widget_components.ParameterLineEdit,
                    widget_components.ParameterSlider,
                ),
            ):
                try:
                    int_value = int(float(parameter_value))
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
                    float_value = float(parameter_value)
                    widget.set_value(float_value)
                except (ValueError, TypeError):
                    pass
            elif isinstance(widget, widget_components.ToggleButton):
                widget.set_value(bool(parameter_value))
            elif isinstance(widget, widget_components.SelectionBox):
                widget.set_value(str(parameter_value))
            else:
                widget.set_value(parameter_value)
            widget.enable_refresh_frame = True


def run_parameter_layout_exec_functions(main_window: "MainWindow") -> None:
    """Run exec_function hooks for Face Restorer, Face Swap, and Face Editor tabs.

    After a bulk restore (workspace load), Qt may not emit value-changed signals
    when the widget already matches the programmatic value, so model load/unload
    callbacks would be skipped. Call this once the parameter widgets reflect the
    saved data.
    """
    from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
    from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA

    for layout_data in (COMMON_LAYOUT_DATA, SWAPPER_LAYOUT_DATA, FACE_EDITOR_LAYOUT_DATA):
        for _category, widgets in layout_data.items():
            for widget_name, widget_data in widgets.items():
                exec_fn = widget_data.get("exec_function")
                if not exec_fn:
                    continue
                widget = main_window.parameter_widgets.get(widget_name)
                if not widget:
                    continue
                try:
                    value = read_parameter_widget_value(widget)
                except TypeError:
                    continue
                args = list(widget_data.get("exec_function_args", []))
                try:
                    exec_fn(main_window, value, *args)
                except Exception as e:
                    print(
                        f"[WARN] run_parameter_layout_exec_functions({widget_name}): {e}"
                    )


def migrate_interpolation_control_keys(control: dict) -> None:
    """Map removed/renamed settings keys (workspace JSON compatibility)."""
    fm = "FrameInterpolationMethodSelection"
    if str(control.get(fm, "")).strip() == "Linear (CPU)":
        control[fm] = "Linear (GPU)"
    control.pop("PreviewLinearInterpolationDisplaySelection", None)
    control.pop("PreviewSmoothDisplayDecoupledToggle", None)
    _migrate_preview_smooth_display_multiplier_keys(control)
    _migrate_unified_interpolation_steps_per_frame(control)


def _migrate_unified_interpolation_steps_per_frame(control: dict) -> None:
    """Merge legacy K + refresh multiplier into PreviewInterpolationStepsPerFrameSelection (2–6)."""
    newk = "PreviewInterpolationStepsPerFrameSelection"
    try:
        cur = int(float(str(control.get(newk, "")).strip()))
        if 2 <= cur <= 6:
            control.pop("PreviewFrameGenIntermediateCountSelection", None)
            control.pop("PreviewSmoothDisplayFpsMultiplierSelection", None)
            return
    except (TypeError, ValueError):
        pass

    k = 1
    try:
        kk = int(
            float(str(control.get("PreviewFrameGenIntermediateCountSelection", "1")).strip())
        )
        k = max(1, min(5, kk))
    except (TypeError, ValueError):
        k = 1

    m = 2
    raw_m = control.get("PreviewSmoothDisplayFpsMultiplierSelection")
    if raw_m is not None:
        try:
            mm = int(float(str(raw_m).strip()))
            m = max(1, min(3, mm))
        except (TypeError, ValueError):
            m = 2

    s = max(m, k + 1)
    s = max(2, min(6, s))
    control[newk] = str(s)
    control.pop("PreviewFrameGenIntermediateCountSelection", None)
    control.pop("PreviewSmoothDisplayFpsMultiplierSelection", None)


def _migrate_preview_smooth_display_multiplier_keys(control: dict) -> None:
    sel_k = "PreviewSmoothDisplayFpsMultiplierSelection"
    if sel_k in control:
        control.pop("PreviewSmoothDisplayHzSlider", None)
        control.pop("PreviewSmoothDisplayFpsMultiplierDecimalSlider", None)
        return

    old_dec = control.pop("PreviewSmoothDisplayFpsMultiplierDecimalSlider", None)
    if old_dec is not None:
        try:
            v = float(old_dec)
            m = int(round(v))
            m = max(1, min(3, m))
            control[sel_k] = str(m)
        except (TypeError, ValueError):
            control[sel_k] = "2"
        control.pop("PreviewSmoothDisplayHzSlider", None)
        return

    old_hz = control.pop("PreviewSmoothDisplayHzSlider", None)
    if old_hz is None:
        return
    try:
        hz = float(old_hz)
        m = int(round(max(1.0, min(3.0, hz / 30.0))))
        m = max(1, min(3, m))
        control[sel_k] = str(m)
    except (TypeError, ValueError):
        control[sel_k] = "2"


def set_control_widgets_values(main_window: "MainWindow", enable_exec_func=True):
    """
    Set the values of control widgets based on the `control` data in the `main_window`.

    Temporarily disables frame refreshing while setting values to avoid unnecessary processing.
    """
    setattr(main_window, "_preview_notifications_suppressed", True)
    try:
        migrate_interpolation_control_keys(main_window.control)
        # Get control values and parameter widgets from the main window
        control = main_window.control.copy()
        parameter_widgets = main_window.parameter_widgets

        # Prepare a dictionary of ALL widget options from layout data
        all_widget_options = {}
        for layout_data_source in [
            SETTINGS_LAYOUT_DATA,
            COMMON_LAYOUT_DATA,
        ]:  # Iterate over both
            for group_name, group_data in layout_data_source.items():
                for widget_key, widget_data in group_data.items():
                    all_widget_options[widget_key] = widget_data

        # Iterate through control items and update widgets
        for control_name, control_value in control.items():
            widget = parameter_widgets.get(control_name)

            if widget:
                # Temporarily disable frame refresh
                widget.enable_refresh_frame = False

                # Set the widget value
                if isinstance(
                    widget,
                    (
                        widget_components.ParameterLineEdit,
                        widget_components.ParameterSlider,
                    ),
                ):
                    try:
                        int_value = int(float(control_value))
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
                        float_value = float(control_value)
                        widget.set_value(float_value)
                    except (ValueError, TypeError):
                        pass
                elif isinstance(widget, widget_components.ToggleButton):
                    widget.set_value(bool(control_value))
                elif isinstance(widget, widget_components.SelectionBox):
                    widget.set_value(str(control_value))
                else:
                    widget.set_value(control_value)

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
        try:
            from app.helpers import detector_internal_size_ui

            detector_internal_size_ui.sync_detector_internal_size_combo(main_window)
        except Exception:
            pass
    finally:
        setattr(main_window, "_preview_notifications_suppressed", False)


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
                QtWidgets.QApplication.processEvents()

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
            QtWidgets.QApplication.processEvents()
