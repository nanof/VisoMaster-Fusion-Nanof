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
from app.ui.widgets.actions.video_control_actions import (  # noqa: E402
    _disable_compare_preview_modes_for_recording,
    view_fullscreen,
)


def test_view_fullscreen_flips_theatre_base_without_window_transition():
    synced = []
    main_window = SimpleNamespace(
        is_theatre_mode=True,
        _was_custom_fullscreen=False,
        showFullScreen=MagicMock(),
        showNormal=MagicMock(),
        isFullScreen=lambda: True,
        _sync_viewer_menu_actions=lambda: synced.append(True),
    )

    view_fullscreen(main_window)

    assert main_window._was_custom_fullscreen is True
    main_window.showFullScreen.assert_not_called()
    main_window.showNormal.assert_not_called()
    assert synced == [True]


def test_view_fullscreen_uses_real_window_transition_outside_theatre():
    synced = []
    main_window = SimpleNamespace(
        is_theatre_mode=False,
        showFullScreen=MagicMock(),
        showNormal=MagicMock(),
        isFullScreen=lambda: False,
        _sync_viewer_menu_actions=lambda: synced.append(True),
    )

    view_fullscreen(main_window)

    main_window.showFullScreen.assert_called_once()
    main_window.showNormal.assert_not_called()
    assert synced == [True]


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
