from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.modules.pop("app.ui.widgets.actions.video_control_actions", None)


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
    "app.ui.widgets.widget_components",
    "app.ui.widgets.ui_workers",
    "app.ui.widgets.actions.common_actions",
    "app.ui.widgets.actions.card_actions",
    "app.ui.widgets.actions.graphics_view_actions",
    "app.ui.widgets.actions.layout_actions",
]
for _s in _STUBS:
    if _s not in sys.modules:
        sys.modules[_s] = _stub(_s)


from app.ui.widgets.actions import common_actions as common_widget_actions  # noqa: E402
from app.ui.widgets.actions import video_control_actions  # noqa: E402
from app.ui.widgets.actions.video_control_actions import (  # noqa: E402
    _disable_compare_preview_modes_for_recording,
    toggle_theatre_mode,
    view_fullscreen,
)


def test_view_fullscreen_temporarily_exits_and_reenters_theatre(monkeypatch):
    synced = []
    theatre_calls = []
    single_shot_calls = []

    def fake_toggle_theatre_mode(window):
        theatre_calls.append(window.is_theatre_mode)
        window.is_theatre_mode = not window.is_theatre_mode

    monkeypatch.setattr(
        video_control_actions, "toggle_theatre_mode", fake_toggle_theatre_mode
    )
    monkeypatch.setattr(
        video_control_actions.QtCore.QTimer,
        "singleShot",
        lambda delay, callback: (
            single_shot_calls.append(delay),
            callback(),
        )[-1],
    )

    main_window = SimpleNamespace(
        is_theatre_mode=True,
        is_full_screen=False,
        _fullscreen_theatre_transition_in_progress=False,
        showFullScreen=MagicMock(),
        showNormal=MagicMock(),
        isFullScreen=lambda: False,
        _sync_viewer_menu_actions=lambda: synced.append(True),
    )

    view_fullscreen(main_window)

    assert theatre_calls == [True, False]
    main_window.showFullScreen.assert_called_once()
    main_window.showNormal.assert_not_called()
    assert main_window.is_theatre_mode is True
    assert main_window.is_full_screen is True
    assert main_window._fullscreen_theatre_transition_in_progress is False
    assert single_shot_calls == [0]
    assert synced == [True]


def test_view_fullscreen_uses_real_window_transition_outside_theatre():
    synced = []
    main_window = SimpleNamespace(
        is_theatre_mode=False,
        is_full_screen=True,
        showFullScreen=MagicMock(),
        showNormal=MagicMock(),
        isFullScreen=lambda: False,
        _sync_viewer_menu_actions=lambda: synced.append(True),
    )

    view_fullscreen(main_window)

    main_window.showFullScreen.assert_called_once()
    main_window.showNormal.assert_not_called()
    assert synced == [True]


class _FakeWidget:
    def __init__(self, visible: bool = True):
        self._visible = visible
        self._minimum_height = None

    def isVisible(self):
        return self._visible

    def hide(self):
        self._visible = False

    def show(self):
        self._visible = True

    def sizeHint(self):
        return SimpleNamespace(height=lambda: 42)

    def setMinimumHeight(self, value):
        self._minimum_height = value


class _FakeMenuBar(_FakeWidget):
    pass


class _FakeLayout:
    def count(self):
        return 0

    def itemAt(self, _index):
        raise IndexError

    def takeAt(self, _index):
        raise IndexError

    def setContentsMargins(self, *_args):
        return None

    def contentsMargins(self):
        return (0, 0, 0, 0)

    def setSpacing(self, *_args):
        return None

    def spacing(self):
        return 0

    def insertItem(self, *_args):
        return None

    def invalidate(self):
        return None


class _FakeGraphicsViewFrame:
    def frameShape(self):
        return "box"

    def setFrameShape(self, *_args):
        return None

    def setStyleSheet(self, *_args):
        return None

    def setVerticalScrollBarPolicy(self, *_args):
        return None

    def setHorizontalScrollBarPolicy(self, *_args):
        return None


def _make_theatre_entry_window(*, is_fullscreen: bool, is_maximized: bool = False):
    menu_bar = _FakeMenuBar()
    return SimpleNamespace(
        is_theatre_mode=False,
        is_full_screen=False,
        _saved_window_state=None,
        input_Target_DockWidget=_FakeWidget(),
        input_Faces_DockWidget=_FakeWidget(),
        jobManagerDockWidget=_FakeWidget(),
        controlOptionsDockWidget=_FakeWidget(),
        facesPanelGroupBox=_FakeWidget(),
        menuBar=lambda: menu_bar,
        horizontalLayout=_FakeLayout(),
        verticalLayout=_FakeLayout(),
        verticalLayoutMediaControls=_FakeLayout(),
        panelVisibilityCheckBoxLayout=_FakeLayout(),
        graphicsViewFrame=_FakeGraphicsViewFrame(),
        saveState=lambda: "window-state",
        isMaximized=lambda: is_maximized,
        isFullScreen=lambda: is_fullscreen,
        normalGeometry=lambda: "normal-geometry",
        geometry=lambda: "live-geometry",
        setWindowState=MagicMock(),
        showFullScreen=MagicMock(),
    )


def test_toggle_theatre_mode_keeps_fullscreen_when_base_mode_is_fullscreen(monkeypatch):
    monkeypatch.setattr(
        video_control_actions, "_set_media_controls_visible", lambda *_args: None
    )
    video_control_actions.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(is_fullscreen=True)

    toggle_theatre_mode(main_window)

    assert main_window._was_custom_fullscreen is True
    assert main_window._was_normal_geometry == "normal-geometry"
    main_window.setWindowState.assert_called_once_with(
        video_control_actions.QtCore.Qt.WindowState.WindowFullScreen
    )
    main_window.showFullScreen.assert_called_once()
    assert main_window.is_full_screen is True


def test_toggle_theatre_mode_keeps_normal_window_when_base_mode_is_windowed(
    monkeypatch,
):
    monkeypatch.setattr(
        video_control_actions, "_set_media_controls_visible", lambda *_args: None
    )
    video_control_actions.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(is_fullscreen=False, is_maximized=False)

    toggle_theatre_mode(main_window)

    assert main_window._was_custom_fullscreen is False
    main_window.setWindowState.assert_not_called()
    main_window.showFullScreen.assert_not_called()
    assert main_window.is_full_screen is False


def test_toggle_theatre_mode_keeps_maximized_window_when_base_mode_is_maximized(
    monkeypatch,
):
    monkeypatch.setattr(
        video_control_actions, "_set_media_controls_visible", lambda *_args: None
    )
    video_control_actions.layout_actions.fit_image_to_view_onchange.reset_mock()
    main_window = _make_theatre_entry_window(is_fullscreen=False, is_maximized=True)

    toggle_theatre_mode(main_window)

    assert main_window._was_custom_fullscreen is False
    assert main_window._was_maximized is True
    main_window.setWindowState.assert_not_called()
    main_window.showFullScreen.assert_not_called()
    assert main_window.is_full_screen is False


def test_toggle_theatre_mode_restores_saved_normal_geometry_on_exit(monkeypatch):
    monkeypatch.setattr(
        video_control_actions, "_set_media_controls_visible", lambda *_args: None
    )
    video_control_actions.layout_actions.fit_image_to_view_onchange.reset_mock()

    menu_bar = _FakeMenuBar()
    saved_geometry = SimpleNamespace(
        x=lambda: 100,
        y=lambda: 200,
        width=lambda: 900,
        height=lambda: 600,
    )
    set_geometry_calls = []
    main_window = SimpleNamespace(
        is_theatre_mode=True,
        _was_custom_fullscreen=False,
        _was_maximized=False,
        _was_normal_geometry=saved_geometry,
        _saved_window_state="window-state",
        _saved_dock_states={},
        _saved_layout_props={},
        _main_v_spacers=[],
        _top_bar_spacers=[],
        _top_bar_widgets_state={},
        input_Target_DockWidget=_FakeWidget(False),
        input_Faces_DockWidget=_FakeWidget(False),
        jobManagerDockWidget=_FakeWidget(False),
        controlOptionsDockWidget=_FakeWidget(False),
        facesPanelGroupBox=_FakeWidget(False),
        menuBar=lambda: menu_bar,
        horizontalLayout=_FakeLayout(),
        verticalLayout=_FakeLayout(),
        verticalLayoutMediaControls=_FakeLayout(),
        panelVisibilityCheckBoxLayout=_FakeLayout(),
        graphicsViewFrame=_FakeGraphicsViewFrame(),
        isFullScreen=lambda: False,
        normalGeometry=lambda: saved_geometry,
        geometry=lambda: saved_geometry,
        showFullScreen=MagicMock(),
        showMaximized=MagicMock(),
        showNormal=MagicMock(),
        setGeometry=lambda geometry: set_geometry_calls.append(geometry),
        restoreState=MagicMock(),
        setUpdatesEnabled=MagicMock(),
    )

    toggle_theatre_mode(main_window)

    main_window.showNormal.assert_called_once()
    main_window.showFullScreen.assert_not_called()
    main_window.showMaximized.assert_not_called()
    assert set_geometry_calls == [saved_geometry]
    main_window.restoreState.assert_called_once_with("window-state")
    assert [call.args for call in main_window.setUpdatesEnabled.call_args_list] == [
        (False,),
        (True,),
    ]
    assert main_window.is_full_screen is False


def test_disable_compare_preview_modes_for_recording_disables_both_and_toasts():
    calls = []
    main_window = SimpleNamespace(
        view_face_compare_enabled=True,
        view_face_mask_enabled=True,
        _set_compare_mode=lambda mode, checked: calls.append((mode, checked)),
    )
    common_widget_actions.create_and_show_toast_message.reset_mock()

    _disable_compare_preview_modes_for_recording(main_window)

    assert calls == [("compare", False), ("mask", False)]
    common_widget_actions.create_and_show_toast_message.assert_called_once()


def test_disable_compare_preview_modes_for_recording_is_noop_when_already_off():
    calls = []
    main_window = SimpleNamespace(
        view_face_compare_enabled=False,
        view_face_mask_enabled=False,
        _set_compare_mode=lambda mode, checked: calls.append((mode, checked)),
    )
    common_widget_actions.create_and_show_toast_message.reset_mock()

    _disable_compare_preview_modes_for_recording(main_window)

    assert calls == []
    common_widget_actions.create_and_show_toast_message.assert_not_called()
