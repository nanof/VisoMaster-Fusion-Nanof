"""A Selection widget with per-widget ``data_type: control`` must land in main_window.control.

Regression: the Selection branch of ``add_widgets_to_tab_layout`` used the tab-level
``data_type`` instead of the per-widget override, so control-typed selection boxes in
the Face Swap tab (registered as "parameter") silently became per-face parameters and
``main_window.control[key]`` never existed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from app.ui.widgets.actions import layout_actions  # noqa: E402

pytestmark = pytest.mark.qt


class _MainWindowStub(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.control = {}
        self.default_parameters = {}
        self.current_widget_parameters = {}
        self.parameters = {}
        self.parameter_widgets = {}
        self.parameter_section_states = {}
        self.markers = {}
        self.selected_target_face_id = None
        self.videoSeekSlider = QtWidgets.QSlider()

    def register_parameter_section(self, section_id, group_box):
        self.parameter_section_states.setdefault(section_id, False)


LAYOUT = {
    "TestGroup": {
        "MyControlSelection": {
            "level": 1,
            "label": "Control selection",
            "options": ["All", "A", "B"],
            "default": "All",
            "data_type": "control",
            "help": "control-typed selection inside a parameter tab",
        },
        "MyParamSelection": {
            "level": 1,
            "label": "Param selection",
            "options": ["X", "Y"],
            "default": "X",
            "help": "plain parameter selection",
        },
    }
}


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture()
def built(qapp):
    mw = _MainWindowStub()
    # Host widget owns the layout so created widgets are not garbage-collected.
    host = QtWidgets.QWidget()
    container = QtWidgets.QVBoxLayout(host)
    mw._test_host = host
    layout_actions.add_widgets_to_tab_layout(
        mw,
        LAYOUT_DATA=LAYOUT,
        layoutWidget=container,
        data_type="parameter",
    )
    return mw


def test_control_typed_selection_goes_to_control(built):
    assert built.control.get("MyControlSelection") == "All"
    assert "MyControlSelection" not in built.default_parameters


def test_plain_selection_still_goes_to_parameters(built):
    assert built.default_parameters.get("MyParamSelection") == "X"
    assert "MyParamSelection" not in built.control


def test_changing_control_selection_updates_control(built):
    widget = built.parameter_widgets["MyControlSelection"]
    widget.setCurrentText("B")
    assert built.control["MyControlSelection"] == "B"


def test_swapper_gender_filter_is_control_typed():
    from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA

    entry = SWAPPER_LAYOUT_DATA["Swapper"]["GenderAppearanceFilterSelection"]
    assert entry["data_type"] == "control"
    assert entry["default"] == "All"
