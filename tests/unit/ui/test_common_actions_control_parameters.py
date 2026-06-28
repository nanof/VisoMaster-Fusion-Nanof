"""Tests for control vs face-parameter widget sync in common_actions."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _stub(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = None
    return m


_STUBS = [
    "PySide6",
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "pyqttoast",
    "qdarkstyle",
    "qdarktheme",
    "app.ui.widgets.widget_components",
    "app.ui.widgets.actions.control_actions",
    "app.ui.widgets.actions.card_actions",
    "app.ui.widgets.actions.list_view_actions",
    "app.ui.widgets.actions.video_control_actions",
    "app.ui.widgets.actions.layout_actions",
    "app.ui.widgets.actions.filter_actions",
    "app.ui.widgets.ui_workers",
]
for _s in _STUBS:
    if _s not in sys.modules:
        sys.modules[_s] = _stub(_s)

sys.modules.pop("app.ui.widgets.actions.common_actions", None)
sys.modules.pop("app.ui.widgets.settings_layout_data", None)

from app.ui.widgets.actions import common_actions  # noqa: E402


class _FakeToggle:
    def __init__(self):
        self.enable_refresh_frame = True
        self.checked = False
        self.set_calls: list[bool] = []

    def set_value(self, value):
        self.checked = bool(value)
        self.set_calls.append(self.checked)


def _fake_slider():
    slider = MagicMock()
    slider.enable_refresh_frame = True
    slider.set_value = MagicMock()
    return slider


@pytest.fixture
def main_window():
    mw = MagicMock()
    mw.control = {
        "SequentialTargetMatchEnableToggle": True,
    }
    mw.current_widget_parameters = {
        "SequentialTargetMatchEnableToggle": False,
        "SimilarityThresholdSlider": "75",
    }
    mw.default_parameters = {"SimilarityThresholdSlider": "60"}
    mw.parameters = {}
    mw.parameter_widgets = {
        "SequentialTargetMatchEnableToggle": _FakeToggle(),
        "SimilarityThresholdSlider": _fake_slider(),
    }
    mw._batch_update_in_progress = False
    return mw


def test_set_widgets_values_skips_control_backed_keys(main_window):
    applied: list[tuple[str, object]] = []

    def _record(widget, value):
        for name, w in main_window.parameter_widgets.items():
            if w is widget:
                applied.append((name, value))
                return
        applied.append(("?", value))

    with patch.object(common_actions, "refresh_frame"):
        with patch.object(
            common_actions, "_set_single_widget_value", side_effect=_record
        ):
            common_actions.set_widgets_values_using_face_id_parameters(
                main_window, face_id=None
            )

    assert ("SequentialTargetMatchEnableToggle", False) not in applied
    assert ("SimilarityThresholdSlider", "75") in applied


def test_strip_control_backed_keys_from_parameters(main_window):
    payload = {
        "SequentialTargetMatchEnableToggle": False,
        "SimilarityThresholdSlider": "80",
    }
    common_actions.strip_control_backed_keys_from_parameters(main_window, payload)
    assert payload == {"SimilarityThresholdSlider": "80"}
