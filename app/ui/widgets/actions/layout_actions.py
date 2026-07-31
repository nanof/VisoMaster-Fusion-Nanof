from typing import TYPE_CHECKING, cast, Union, List, Callable
from functools import partial
import re

from PySide6 import QtWidgets, QtCore, QtGui

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import control_actions
from app.ui.widgets.actions import gpu_settings_actions
from app.ui.widgets import widget_components

# from app.UI.Widgets.WidgetComponents import *
from app.helpers.typing_helper import LayoutDictTypes


def add_widgets_to_tab_layout(
    main_window: "MainWindow",
    LAYOUT_DATA: LayoutDictTypes,
    layoutWidget: QtWidgets.QVBoxLayout,
    data_type="parameter",
    section_namespace: str = "default",
):
    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(6, 4, 10, 4)
    layout.setSpacing(8)
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll_content = QtWidgets.QWidget()
    scroll_content.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Preferred,
    )
    scroll_content.setLayout(layout)
    scroll_area.setWidget(scroll_content)
    scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

    def add_horizontal_layout_to_category(
        category_layout: QtWidgets.QFormLayout, *widgets
    ):
        row_widget = QtWidgets.QWidget()
        row_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        horizontal_layout = QtWidgets.QHBoxLayout(row_widget)

        for widget in widgets:
            horizontal_layout.addWidget(widget)
        category_layout.addRow(row_widget)
        return row_widget, horizontal_layout

    def create_layout_action_button(button_data: dict):
        action_button = QtWidgets.QPushButton(cast(str, button_data["label"]))
        action_button.setToolTip(cast(str, button_data.get("help", "")))
        if "fixed_width" in button_data:
            action_button.setFixedWidth(cast(int, button_data["fixed_width"]))
        else:
            action_button.setMaximumWidth(55)
        if "exec_function" in button_data:
            action_button.clicked.connect(
                partial(cast(Callable, button_data["exec_function"]), main_window)
            )
        return action_button

    def build_section_id(category_name: str) -> str:
        normalized_name = re.sub(r"[^a-z0-9]+", "_", category_name.lower()).strip("_")
        return f"{section_namespace}:{normalized_name}"

    for category, widgets in LAYOUT_DATA.items():
        section_id = build_section_id(category)
        display_title = (
            "Face Editor"
            if section_namespace == "face_editor" and not category
            else category
        )
        group_box = widget_components.CollapsibleSection(
            main_window,
            title=display_title,
            section_id=section_id,
            expanded=main_window.parameter_section_states.get(section_id, False),
        )
        category_layout = QtWidgets.QFormLayout()
        group_box.content_widget.setLayout(category_layout)
        main_window.register_parameter_section(section_id, group_box)

        for widget_name, widget_data in widgets.items():
            spacing_level = cast(int, widget_data["level"])
            label = QtWidgets.QLabel(cast(str, widget_data["label"]))
            label.setToolTip(cast(str, widget_data["help"]))
            # Per-widget data_type override: a widget can opt into global "control"
            # storage even when its tab layout is registered as "parameter" (and
            # vice-versa). Used e.g. for global swap-matching toggles that live in
            # the per-face Face Swap tab but must be read from main_window.control.
            widget_data_type = cast(str, widget_data.get("data_type", data_type))
            mirror_chk = cast(
                Union[str, None], widget_data.get("mirror_checkable_button")
            )
            if mirror_chk:
                # Default-arg binding: loop locals must not be captured by reference alone.
                mirror_button_attr = str(mirror_chk)
                widget = widget_components.ToggleButton(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    main_window=main_window,
                )
                target_btn = getattr(main_window, mirror_button_attr, None)
                init_swap = False
                if target_btn is not None and hasattr(target_btn, "isChecked"):
                    init_swap = bool(target_btn.isChecked())
                widget.setChecked(init_swap)
                stub_reset = QtWidgets.QWidget()
                stub_reset.setFixedSize(0, 0)
                widget.reset_default_button = stub_reset
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout, widget, label, stub_reset
                )

                def _mirror_to_toolbar_swap(
                    _checked,
                    _attr: str = mirror_button_attr,
                    _mw: "MainWindow" = main_window,
                    _tw: widget_components.ToggleButton = widget,
                ):
                    tb = getattr(_mw, _attr, None)
                    if tb is None or not hasattr(tb, "setChecked"):
                        return
                    tb.blockSignals(True)
                    tb.setChecked(_tw.isChecked())
                    tb.blockSignals(False)
                    video_control_actions.process_swap_faces(_mw)

                def _toolbar_to_mirror_swap(
                    checked,
                    _tw: widget_components.ToggleButton = widget,
                ):
                    _tw.blockSignals(True)
                    _tw.setChecked(bool(checked))
                    _tw.blockSignals(False)

                widget.toggled.connect(_mirror_to_toolbar_swap)
                if target_btn is not None and hasattr(target_btn, "toggled"):
                    target_btn.toggled.connect(_toolbar_to_mirror_swap)

                horizontal_layout.setContentsMargins(spacing_level * 10, 0, 0, 0)
                widget.row_widget = row_widget
                main_window.parameter_widgets[widget_name] = widget
                continue

            bind_control = cast(
                Union[str, None], widget_data.get("bind_control")
            )
            storage_key = bind_control or widget_name

            if "Toggle" in widget_name:
                init_toggle = cast(bool, widget_data["default"])
                if widget_data_type == "control" and (
                    bind_control or widget_data.get("data_type") == "control"
                ):
                    raw = main_window.control.get(storage_key, widget_data["default"])
                    init_toggle = bool(raw)
                elif bind_control and widget_data_type == "parameter":
                    init_toggle = bool(
                        common_widget_actions.get_current_parameter_value(
                            main_window, storage_key, widget_data["default"]
                        )
                    )
                widget = widget_components.ToggleButton(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    main_window=main_window,
                )
                widget.setChecked(init_toggle)
                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )

                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout, widget, label, widget.reset_default_button
                )

                if widget_data_type == "parameter":
                    if not bind_control:
                        common_widget_actions.create_default_parameter(
                            main_window, widget_name, cast(bool, widget_data["default"])
                        )
                elif not bind_control:
                    common_widget_actions.create_control(
                        main_window, widget_name, cast(bool, widget_data["default"])
                    )

                def onchange_toggle(
                    toggle_widget: widget_components.ToggleButton,
                    toggle_widget_name,
                    widget_data: dict,
                    *args,
                ):
                    toggle_state = toggle_widget.isChecked()
                    bc = cast(Union[str, None], widget_data.get("bind_control"))
                    control_key = bc or toggle_widget_name
                    _dt = cast(str, widget_data.get("data_type", data_type))
                    if _dt == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            control_key,
                            toggle_state,
                            enable_refresh_frame=toggle_widget.enable_refresh_frame,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                        if bc:
                            primary_toggle = main_window.parameter_widgets.get(bc)
                            if primary_toggle is not None:
                                common_widget_actions.show_hide_related_widgets(
                                    main_window,
                                    primary_toggle,
                                    bc,
                                )
                    elif _dt == "control":
                        common_widget_actions.update_control(
                            main_window,
                            control_key,
                            toggle_state,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                        if bc:
                            primary_toggle = main_window.parameter_widgets.get(bc)
                            if primary_toggle is not None:
                                common_widget_actions.show_hide_related_widgets(
                                    main_window,
                                    primary_toggle,
                                    bc,
                                )

                widget.toggled.connect(
                    partial(onchange_toggle, widget, widget_name, widget_data)
                )

            elif widget_name == "GpuPrimaryDeviceSelection":
                widget = widget_components.SelectionBox(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    main_window=main_window,
                    default_value=widget_data["default"],
                    selection_values=[],
                )
                gpu_settings_actions.fill_primary_gpu_combo(widget, main_window)
                if "GpuPrimaryPhysicalIndex" not in main_window.control:
                    common_widget_actions.create_control(
                        main_window,
                        "GpuPrimaryPhysicalIndex",
                        int(widget_data.get("default", 0)),
                    )
                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout, label, widget, widget.reset_default_button
                )
                # Must wire here: relying only on finalize_gpu_widgets + UniqueConnection
                # + lambda slots can fail to register in PySide6, so changing Primary GPU
                # would never call change_gpu_index / TRT cache paths.
                widget.currentIndexChanged.connect(
                    partial(
                        gpu_settings_actions.on_primary_gpu_combo_changed,
                        main_window,
                    )
                )

            elif "Selection" in widget_name:
                options = widget_data["options"]
                default = widget_data["default"]

                if callable(options):
                    options = options(main_window.dfm_model_manager)
                if callable(default):
                    default = default(main_window.dfm_model_manager)

                init_selection = default
                if bind_control and data_type == "control":
                    init_selection = main_window.control.get(storage_key, default)

                widget = widget_components.SelectionBox(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    main_window=main_window,
                    default_value=default,
                    selection_values=options,
                )

                widget.addItems(cast(List[str], options))
                widget.setCurrentText(str(init_selection))

                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout, label, widget, widget.reset_default_button
                )

                if data_type == "parameter":
                    common_widget_actions.create_default_parameter(
                        main_window, widget_name, default
                    )
                elif not bind_control:
                    common_widget_actions.create_control(
                        main_window, widget_name, default
                    )

                def onchange_selection(
                    selection_widget: widget_components.SelectionBox,
                    selection_widget_name,
                    widget_data: dict,
                    selected_value=False,
                ):
                    actual_value = selection_widget.currentData()
                    if actual_value is None:
                        actual_value = selected_value
                    bc_sel = cast(Union[str, None], widget_data.get("bind_control"))
                    control_key_sel = bc_sel or selection_widget_name
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            selection_widget_name,
                            actual_value,
                            enable_refresh_frame=selection_widget.enable_refresh_frame,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                    elif data_type == "control":
                        common_widget_actions.update_control(
                            main_window,
                            control_key_sel,
                            actual_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                        if bc_sel:
                            primary_sel = main_window.parameter_widgets.get(bc_sel)
                            if primary_sel is not None:
                                common_widget_actions.show_hide_related_widgets(
                                    main_window,
                                    primary_sel,
                                    bc_sel,
                                )

                widget.currentTextChanged.connect(
                    partial(onchange_selection, widget, widget_name, widget_data)
                )

            elif "DecimalSlider" in widget_name:
                widget = widget_components.ParameterDecimalSlider(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    min_value=float(
                        cast(Union[int, float, str], widget_data["min_value"])
                    ),
                    max_value=float(
                        cast(Union[int, float, str], widget_data["max_value"])
                    ),
                    default_value=float(
                        cast(Union[int, float, str], widget_data["default"])
                    ),
                    decimals=int(cast(Union[int, float, str], widget_data["decimals"])),
                    step_size=float(cast(Union[int, float, str], widget_data["step"])),
                    main_window=main_window,
                )
                widget.line_edit = widget_components.ParameterLineDecimalEdit(
                    min_value=float(
                        cast(Union[int, float, str], widget_data["min_value"])
                    ),
                    max_value=float(
                        cast(Union[int, float, str], widget_data["max_value"])
                    ),
                    default_value=str(widget_data["default"]),
                    decimals=int(cast(Union[int, float, str], widget_data["decimals"])),
                    step_size=float(cast(Union[int, float, str], widget_data["step"])),
                    fixed_width=48,
                    max_length=7
                    if int(cast(Union[int, float, str], widget_data["decimals"])) > 1
                    else 5,
                )
                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout,
                    label,
                    widget,
                    widget.line_edit,
                    widget.reset_default_button,
                )

                if data_type == "parameter":
                    common_widget_actions.create_default_parameter(
                        main_window,
                        widget_name,
                        float(cast(Union[int, float, str], widget_data["default"])),
                    )
                else:
                    common_widget_actions.create_control(
                        main_window,
                        widget_name,
                        float(cast(Union[int, float, str], widget_data["default"])),
                    )

                def onchange_decimal_slider(
                    slider_widget: widget_components.ParameterDecimalSlider,
                    slider_widget_name,
                    widget_data: dict,
                    new_value=False,
                ):
                    actual_value = slider_widget.value()
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            slider_widget_name,
                            actual_value,
                            enable_refresh_frame=slider_widget.enable_refresh_frame,
                        )
                    elif data_type == "control":
                        common_widget_actions.update_control(
                            main_window,
                            slider_widget_name,
                            actual_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                    slider_widget.line_edit.set_value(actual_value)

                widget.debounce_timer.timeout.connect(
                    partial(onchange_decimal_slider, widget, widget_name, widget_data)
                )

                def onchange_decimal_line_edit(
                    slider_widget: widget_components.ParameterDecimalSlider,
                    slider_widget_name: str,
                    widget_data: dict,
                    new_value=False,
                ):
                    if not new_value:
                        new_value = 0.0
                    try:
                        new_value = float(new_value)
                    except ValueError:
                        new_value = slider_widget.value()
                    if new_value > (
                        slider_widget.max_value / slider_widget.scale_factor
                    ):
                        new_value = slider_widget.max_value / slider_widget.scale_factor
                    elif new_value < (
                        slider_widget.min_value / slider_widget.scale_factor
                    ):
                        new_value = slider_widget.min_value / slider_widget.scale_factor
                    slider_widget.setValue(new_value)
                    slider_widget.line_edit.set_value(new_value)
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            slider_widget_name,
                            new_value,
                            enable_refresh_frame=slider_widget.enable_refresh_frame,
                        )
                    elif data_type == "control":
                        common_widget_actions.update_control(
                            main_window,
                            slider_widget_name,
                            new_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )

                widget.line_edit.textChanged.connect(
                    partial(
                        onchange_decimal_line_edit, widget, widget_name, widget_data
                    )
                )

            elif "Slider" in widget_name:
                widget = widget_components.ParameterSlider(
                    label=cast(str, widget_data["label"]),
                    widget_name=widget_name,
                    group_layout_data=widgets,
                    label_widget=label,
                    min_value=widget_data["min_value"],
                    max_value=widget_data["max_value"],
                    default_value=widget_data["default"],
                    step_size=widget_data["step"],
                    main_window=main_window,
                )
                if widget_data.get("enable_refresh_frame") is False:
                    widget.enable_refresh_frame = False
                widget.line_edit = widget_components.ParameterLineEdit(
                    min_value=int(
                        cast(Union[int, float, str], widget_data["min_value"])
                    ),
                    max_value=int(
                        cast(Union[int, float, str], widget_data["max_value"])
                    ),
                    default_value=str(widget_data["default"]),
                )
                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )
                _slider_row_widgets: list = [
                    label,
                    widget,
                    widget.line_edit,
                    widget.reset_default_button,
                ]
                if "action_button" in widget_data:
                    _ab_data: dict = cast(dict, widget_data["action_button"])
                    _action_btn = create_layout_action_button(_ab_data)
                    _slider_row_widgets.append(_action_btn)
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout,
                    *_slider_row_widgets,
                )
                if "below_row_button" in widget_data:
                    _below_ab_data: dict = cast(dict, widget_data["below_row_button"])
                    _below_action_btn = create_layout_action_button(_below_ab_data)
                    _below_spacer = QtWidgets.QWidget()
                    _below_spacer.setSizePolicy(
                        QtWidgets.QSizePolicy.Policy.Expanding,
                        QtWidgets.QSizePolicy.Policy.Maximum,
                    )
                    _below_row_widget, _below_horizontal_layout = (
                        add_horizontal_layout_to_category(
                            category_layout,
                            _below_action_btn,
                            _below_spacer,
                        )
                    )
                    widget.below_row_widget = _below_row_widget

                if widget_data_type == "control" and (
                    bind_control or widget_data.get("data_type") == "control"
                ):
                    try:
                        _iv = int(
                            float(
                                main_window.control.get(
                                    storage_key, widget_data["default"]
                                )
                            )
                        )
                    except (TypeError, ValueError):
                        _iv = int(cast(Union[int, float, str], widget_data["default"]))
                    widget.set_value(_iv)

                if widget_data_type == "parameter":
                    common_widget_actions.create_default_parameter(
                        main_window,
                        widget_name,
                        int(cast(Union[int, float, str], widget_data["default"])),
                    )
                elif not bind_control:
                    common_widget_actions.create_control(
                        main_window,
                        widget_name,
                        int(cast(Union[int, float, str], widget_data["default"])),
                    )

                def onchange_int_slider(
                    slider_widget: widget_components.ParameterSlider,
                    slider_widget_name,
                    widget_data: dict,
                    new_value=False,
                ):
                    bc_sl = cast(Union[str, None], widget_data.get("bind_control"))
                    control_key_sl = bc_sl or slider_widget_name
                    _dt = cast(str, widget_data.get("data_type", data_type))
                    if _dt == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            slider_widget_name,
                            new_value,
                            enable_refresh_frame=slider_widget.enable_refresh_frame,
                        )
                    elif _dt == "control":
                        common_widget_actions.update_control(
                            main_window,
                            control_key_sl,
                            new_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )
                    slider_widget.line_edit.setText(str(new_value))

                widget.debounce_timer.timeout.connect(
                    partial(onchange_int_slider, widget, widget_name, widget_data)
                )

                def onchange_int_line_edit(
                    slider_widget: widget_components.ParameterSlider,
                    slider_widget_name,
                    widget_data,
                    new_value=False,
                ):
                    if not new_value:
                        new_value = 0
                    try:
                        new_value = int(new_value)
                    except ValueError:
                        new_value = slider_widget.value()
                    if new_value > slider_widget.max_value:
                        new_value = slider_widget.max_value
                    elif new_value < slider_widget.min_value:
                        new_value = slider_widget.min_value
                    slider_widget.line_edit.set_value(new_value)
                    slider_widget.setValue(int(new_value))
                    bc_le = cast(Union[str, None], widget_data.get("bind_control"))
                    control_key_le = bc_le or slider_widget_name
                    _dt = cast(str, widget_data.get("data_type", data_type))
                    if _dt == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            slider_widget_name,
                            new_value,
                            enable_refresh_frame=slider_widget.enable_refresh_frame,
                        )
                    elif _dt == "control":
                        common_widget_actions.update_control(
                            main_window,
                            control_key_le,
                            new_value,
                            exec_function=widget_data.get("exec_function"),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )

                widget.line_edit.textChanged.connect(
                    partial(onchange_int_line_edit, widget, widget_name, widget_data)
                )

            elif "ScreenAction" in widget_name:
                btn = QtWidgets.QPushButton(widget_data["button_label"])
                btn.setToolTip(cast(str, widget_data.get("help", "")))
                btn.label_widget = label
                btn.group_layout_data = widgets
                btn.line_edit = None
                stub_reset = QtWidgets.QWidget()
                stub_reset.setFixedSize(0, 0)
                btn.reset_default_button = stub_reset
                btn.clicked.connect(
                    partial(
                        cast(Callable, widget_data["click_handler"]), main_window
                    )
                )
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout, label, btn
                )
                widget = btn

            elif "Text" in widget_name:

                def on_enter_pressed(
                    text_widget: widget_components.ParameterText,
                    text_widget_name,
                    widget_data,
                ):
                    new_value = text_widget.text()
                    if data_type == "parameter":
                        common_widget_actions.update_parameter(
                            main_window,
                            text_widget_name,
                            new_value,
                            enable_refresh_frame=text_widget.enable_refresh_frame,
                        )
                    else:
                        common_widget_actions.update_control(
                            main_window,
                            text_widget_name,
                            new_value,
                            exec_function=cast(
                                Callable, widget_data.get("exec_function")
                            ),
                            exec_function_args=cast(
                                list, widget_data.get("exec_function_args", [])
                            ),
                        )

                widget = widget_components.ParameterText(
                    default_value=cast(str, widget_data["default"]),
                    fixed_width=cast(int, widget_data["width"]),
                    max_length=256,
                )
                widget.returnPressed.connect(
                    partial(on_enter_pressed, widget, widget_name, widget_data)
                )
                widget.label_widget = label
                widget.widget_name = widget_name
                widget.group_layout_data = widgets
                widget.main_window = main_window
                widget.enable_refresh_frame = True

                widget.reset_default_button = (
                    widget_components.ParameterResetDefaultButton(related_widget=widget)
                )
                row_widget, horizontal_layout = add_horizontal_layout_to_category(
                    category_layout, label, widget, widget.reset_default_button
                )

                if data_type == "parameter":
                    common_widget_actions.create_default_parameter(
                        main_window, widget_name, cast(str, widget_data["default"])
                    )
                else:
                    common_widget_actions.create_control(
                        main_window, widget_name, cast(str, widget_data["default"])
                    )

                # The exec_function is now handled within the ParameterLineEdit itself
                # widget.returnPressed.connect(partial(on_enter_pressed, widget, widget_name))

            horizontal_layout.setContentsMargins(spacing_level * 10, 0, 0, 0)
            widget.row_widget = row_widget
            main_window.parameter_widgets[widget_name] = widget
            if bind_control:
                if data_type == "control":
                    common_widget_actions.register_control_widget_mirror(
                        main_window, bind_control, widget
                    )
                elif data_type == "parameter":
                    common_widget_actions.register_parameter_widget_mirror(
                        main_window, bind_control, widget
                    )

        category_layout.setVerticalSpacing(2)
        category_layout.setHorizontalSpacing(2)
        layout.addWidget(group_box)

    layout.addStretch(1)

    layoutWidget.addWidget(scroll_area)

    # Default show/hide widgets
    for category, widgets in LAYOUT_DATA.items():
        for widget_name, widget_data in widgets.items():
            widget = main_window.parameter_widgets[widget_name]
            common_widget_actions.show_hide_related_widgets(
                main_window, widget, widget_name
            )

    _bound_for_show_hide: set[str] = set()
    for _cat, _wmap in LAYOUT_DATA.items():
        for _wn, _wd in _wmap.items():
            _bc = _wd.get("bind_control")
            if _bc:
                _bound_for_show_hide.add(cast(str, _bc))
    for _bc in _bound_for_show_hide:
        _pw = main_window.parameter_widgets.get(_bc)
        if _pw is not None:
            common_widget_actions.show_hide_related_widgets(
                main_window, _pw, _bc
            )


def show_hide_faces_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.facesPanelGroupBox.show()
    else:
        main_window.facesPanelGroupBox.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_input_target_media_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.input_Target_DockWidget.show()
    else:
        main_window.input_Target_DockWidget.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_input_faces_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.input_Faces_DockWidget.show()
    else:
        main_window.input_Faces_DockWidget.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_input_jobs_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.jobManagerDockWidget.show()
    else:
        main_window.jobManagerDockWidget.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_parameters_panel(main_window: "MainWindow", checked):
    if checked:
        main_window.controlOptionsDockWidget.show()
    else:
        main_window.controlOptionsDockWidget.hide()
    fit_image_to_view_onchange(main_window)


def show_hide_theatre_mode_panels(main_window: "MainWindow", checked):
    # Collects the current state of all visible panel toggle actions/state
    def collect_states():
        return {
            "target_media": main_window.panel_visibility_state.get(
                "target_media", True
            ),
            "faces": main_window.panel_visibility_state.get("faces", True),
            "parameters": main_window.panel_visibility_state.get("parameters", True),
            "input_faces": main_window.panel_visibility_state.get("input_faces", True),
            "jobs": main_window.panel_visibility_state.get("jobs", True),
        }

    # Applies the saved states to the visible View-menu actions/state
    def apply_states(states):
        main_window._set_panel_visibility(
            "target_media", states.get("target_media", True)
        )
        main_window._set_panel_visibility("faces", states.get("faces", True))
        main_window._set_panel_visibility("parameters", states.get("parameters", True))
        main_window._set_panel_visibility(
            "input_faces", states.get("input_faces", True)
        )
        main_window._set_panel_visibility("jobs", states.get("jobs", True))

    if checked:
        # Entering Theatre Mode: Save normal states and apply theatre states (default all False/Hidden)
        main_window._theatre_normal_panel_states = collect_states()
        apply_states(
            main_window._theatre_mode_panel_states
            or {
                "target_media": False,
                "faces": False,
                "parameters": False,
                "input_faces": False,
                "jobs": False,
            }
        )
    else:
        # Exiting Theatre Mode: Save theatre states and restore normal states (default all True/Visible)
        main_window._theatre_mode_panel_states = collect_states()
        apply_states(
            main_window._theatre_normal_panel_states
            or {
                "target_media": True,
                "faces": True,
                "parameters": True,
                "input_faces": True,
                "jobs": True,
            }
        )
        main_window._theatre_normal_panel_states = None

    fit_image_to_view_onchange(main_window)


def fit_image_to_view_onchange(main_window: "MainWindow", *args):
    fit_item, scene_rect = graphics_view_actions.primary_preview_graphics_item_for_fit(
        main_window
    )
    if fit_item is not None and scene_rect is not None:
        QtCore.QTimer.singleShot(
            0,
            partial(
                graphics_view_actions.fit_image_to_view,
                main_window,
                fit_item,
                scene_rect,
            ),
        )


def set_up_menu_actions(main_window: "MainWindow"):
    if getattr(main_window, "_menu_actions_setup_installed", False):
        return

    if not hasattr(main_window, "actionEdit_CopyParameters"):
        main_window.actionEdit_CopyParameters = QtGui.QAction(
            "Copy Parameters from Selected Face", main_window
        )
        main_window.actionEdit_PasteParameters = QtGui.QAction(
            "Paste Parameters to Selected Face", main_window
        )
        main_window.actionEdit_ResetParameters = QtGui.QAction(
            "Reset Selected Face Parameters", main_window
        )
        main_window.menuEdit.clear()
        main_window.menuEdit.addAction(main_window.actionEdit_CopyParameters)
        main_window.menuEdit.addAction(main_window.actionEdit_PasteParameters)
        main_window.menuEdit.addSeparator()
        main_window.menuEdit.addAction(main_window.actionEdit_ResetParameters)

    if not hasattr(main_window, "actionHelp_QuickStartGuide"):
        main_window.actionHelp_QuickStartGuide = QtGui.QAction(
            "Quick Start Guide", main_window
        )
        main_window.actionHelp_UserManual = QtGui.QAction("User Manual", main_window)
        main_window.menuHelp.insertAction(
            main_window.actionView_Help_Shortcuts,
            main_window.actionHelp_QuickStartGuide,
        )
        main_window.menuHelp.insertAction(
            main_window.actionView_Help_Shortcuts,
            main_window.actionHelp_UserManual,
        )
        main_window.menuHelp.insertSeparator(main_window.actionView_Help_Shortcuts)

    if not hasattr(main_window, "actionHelp_About"):
        main_window.actionHelp_About = QtGui.QAction("About", main_window)
        main_window.menuHelp.addSeparator()
        main_window.menuHelp.addAction(main_window.actionHelp_About)

    # MainWindow.ui may omit these; layout_actions expects them since merge with dev.
    if not hasattr(main_window, "actionOpen_Target_Media_Folder"):
        main_window.actionOpen_Target_Media_Folder = QtGui.QAction(
            "Open Target Media Folder", main_window
        )
        main_window.actionOpen_Input_Faces_Folder = QtGui.QAction(
            "Open Input Faces Folder", main_window
        )
        main_window.actionOpen_Output_Folder = QtGui.QAction(
            "Open Output Folder", main_window
        )
        anchor = main_window.actionLoad_Embeddings
        main_window.menuFile.insertSeparator(anchor)
        main_window.menuFile.insertAction(
            anchor, main_window.actionOpen_Output_Folder
        )
        main_window.menuFile.insertAction(
            anchor, main_window.actionOpen_Input_Faces_Folder
        )
        main_window.menuFile.insertAction(
            anchor, main_window.actionOpen_Target_Media_Folder
        )

    main_window.actionLoad_SavedWorkspace.triggered.connect(
        partial(
            save_load_actions.load_saved_workspace,
            main_window,
        )
    )
    main_window.actionSave_CurrentWorkspace.triggered.connect(
        partial(
            save_load_actions.save_current_workspace,
            main_window,
        )
    )
    if hasattr(main_window, "actionReset_to_LastWorkspace"):
        main_window.actionReset_to_LastWorkspace.triggered.connect(
            partial(
                save_load_actions.load_saved_workspace,
                main_window,
                str(main_window.last_workspace_path),
            )
        )
    if hasattr(main_window, "actionSave_LastWorkspace"):
        main_window.actionSave_LastWorkspace.triggered.connect(
            partial(
                save_load_actions.save_current_workspace,
                main_window,
                str(main_window.last_workspace_path),
            )
        )

    def quit_without_saving(_checked=False):
        # close() rather than QCoreApplication.quit() so closeEvent still runs.
        main_window.quit_without_saving = True
        main_window.close()

    if hasattr(main_window, "actionQuit_WithoutSaving"):
        main_window.actionQuit_WithoutSaving.triggered.connect(quit_without_saving)
    if hasattr(main_window, "actionQuit"):
        main_window.actionQuit.triggered.connect(
            lambda _checked=False: main_window.close()
        )

    main_window.actionOpen_Videos_Folder.triggered.connect(
        partial(list_view_actions.select_target_medias, main_window, "folder")
    )
    main_window.actionOpen_Video_Files.triggered.connect(
        partial(list_view_actions.select_target_medias, main_window, "files")
    )
    main_window.actionLoad_Source_Image_Files.triggered.connect(
        partial(list_view_actions.select_input_face_images, main_window, "files")
    )
    main_window.actionLoad_Source_Images_Folder.triggered.connect(
        partial(list_view_actions.select_input_face_images, main_window, "folder")
    )
    main_window.actionOpen_Target_Media_Folder.triggered.connect(
        partial(list_view_actions.open_target_media_folder, main_window)
    )
    main_window.actionOpen_Input_Faces_Folder.triggered.connect(
        partial(list_view_actions.open_input_faces_folder, main_window)
    )
    main_window.actionOpen_Output_Folder.triggered.connect(
        partial(list_view_actions.open_output_media_folder, main_window)
    )
    main_window.actionLoad_Embeddings.triggered.connect(
        partial(save_load_actions.open_embeddings_from_file, main_window)
    )
    main_window.actionSave_Embeddings.triggered.connect(
        partial(save_load_actions.save_embeddings_to_file, main_window)
    )
    main_window.actionSave_Embeddings_As.triggered.connect(
        partial(save_load_actions.save_embeddings_to_file, main_window, True)
    )
    main_window.actionView_Fullscreen_F11.triggered.connect(
        partial(video_control_actions.view_fullscreen, main_window)
    )
    main_window.actionEdit_CopyParameters.triggered.connect(
        partial(common_widget_actions.copy_selected_face_parameters, main_window)
    )
    main_window.actionEdit_PasteParameters.triggered.connect(
        partial(common_widget_actions.paste_selected_face_parameters, main_window)
    )
    main_window.actionEdit_ResetParameters.triggered.connect(
        partial(common_widget_actions.reset_selected_face_parameters, main_window)
    )
    main_window.actionView_Help_Shortcuts.triggered.connect(
        partial(list_view_actions.show_shortcuts, main_window)
    )
    main_window.actionView_Help_Presets.triggered.connect(
        partial(list_view_actions.show_presets, main_window)
    )
    main_window.actionHelp_QuickStartGuide.triggered.connect(
        partial(list_view_actions._open_about_link, main_window, "quickstart")
    )
    main_window.actionHelp_UserManual.triggered.connect(
        partial(list_view_actions._open_about_link, main_window, "manual")
    )
    main_window.actionHelp_About.triggered.connect(
        partial(list_view_actions.show_about, main_window)
    )
    main_window._menu_actions_setup_installed = True


def set_all_parameters_and_control_widgets_enabled(
    main_window: "MainWindow", enabled: bool
):
    disabled = not enabled
    main_window.viewer_mode_actions_enabled = enabled

    # Bottom buttons
    main_window.saveImageButton.setDisabled(disabled)
    main_window.batchImageButton.setDisabled(disabled)
    main_window.batchallImageButton.setDisabled(disabled)
    main_window.findTargetFacesButton.setDisabled(disabled)
    main_window.clearTargetFacesButton.setDisabled(disabled)
    main_window.swapfacesButton.setDisabled(disabled)
    main_window.editFacesButton.setDisabled(disabled)
    main_window.openEmbeddingButton.setDisabled(disabled)
    main_window.saveEmbeddingButton.setDisabled(disabled)
    main_window.saveEmbeddingAsButton.setDisabled(disabled)

    # Video control buttons
    main_window.videoSeekSlider.setDisabled(disabled)
    main_window.addMarkerButton.setDisabled(disabled)
    main_window.removeMarkerButton.setDisabled(disabled)
    main_window.nextMarkerButton.setDisabled(disabled)
    main_window.previousMarkerButton.setDisabled(disabled)
    main_window.frameAdvanceButton.setDisabled(disabled)
    main_window.frameRewindButton.setDisabled(disabled)
    for attr_name in (
        "scanToolsToggleButton",
        "runScanButton",
        "clearScanResultsButton",
        "prevIssueButton",
        "nextIssueButton",
        "dropFrameButton",
        "dropAllIssueFramesButton",
        "clearDroppedFramesButton",
    ):
        if hasattr(main_window, attr_name):
            getattr(main_window, attr_name).setDisabled(disabled)

    # Compare/mask toolbar toggles
    if hasattr(main_window, "faceCompareToggleButton"):
        main_window.faceCompareToggleButton.setDisabled(disabled)
    if hasattr(main_window, "faceMaskToggleButton"):
        main_window.faceMaskToggleButton.setDisabled(disabled)

    # List items
    for _, embed_button in main_window.merged_embeddings.items():
        embed_button.setDisabled(disabled)
    for _, target_media_button in main_window.target_videos.items():
        target_media_button.setDisabled(disabled)
    for _, input_face_button in main_window.input_faces.items():
        input_face_button.setDisabled(disabled)
    for _, target_face_button in main_window.target_faces.items():
        target_face_button.setDisabled(disabled)

    # Parameters and controls dict widgets - SECURED
    for _, widget in main_window.parameter_widgets.items():
        if not widget:
            continue

        widget.setDisabled(disabled)

        # Check safely if the attributes exist before disabling them
        reset_btn = getattr(widget, "reset_default_button", None)
        if reset_btn:
            reset_btn.setDisabled(disabled)

        label_w = getattr(widget, "label_widget", None)
        if label_w:
            label_w.setDisabled(disabled)

        line_e = getattr(widget, "line_edit", None)
        if line_e:
            line_e.setDisabled(disabled)

def disable_all_parameters_and_control_widget(main_window: "MainWindow"):
    set_all_parameters_and_control_widgets_enabled(main_window, False)


def enable_all_parameters_and_control_widget(main_window: "MainWindow"):
    set_all_parameters_and_control_widgets_enabled(main_window, True)
