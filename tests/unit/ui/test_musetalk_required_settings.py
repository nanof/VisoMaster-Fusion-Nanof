"""Enabling lip-sync must switch on what its crop depends on.

MuseTalk frames the crop from the 68 face landmarks. With landmark detection off
the pipeline only produces 5 points, the crop silently falls back to the detector
box, and the model returns the small generic mouth the framing exists to avoid.
Nothing on screen says why, so the prerequisite is forced instead.
"""

from __future__ import annotations

import types

from app.ui.widgets.actions import common_actions as cwa  # noqa: F401  (import order)
from app.ui.widgets.actions import control_actions


def _window(control: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        control=control,
        markers={},
        videoSeekSlider=types.SimpleNamespace(value=lambda: 0),
        video_processor=None,
        parameter_widgets={},
        _parameter_widget_mirrors={},
    )


def test_the_required_settings_are_the_scheme_upstream_indexes() -> None:
    assert control_actions.MUSETALK_REQUIRED_CONTROL_SETTINGS == {
        "LandmarkDetectToggle": True,
        "LandmarkDetectModelSelection": "68",
    }


def test_landmarks_are_switched_on_when_they_are_off(monkeypatch) -> None:
    applied: list[tuple] = []
    monkeypatch.setattr(
        control_actions.common_widget_actions,
        "update_control",
        lambda mw, name, value, **kw: applied.append((name, value)),
    )
    win = _window(
        {"LandmarkDetectToggle": False, "LandmarkDetectModelSelection": "203"}
    )
    control_actions.apply_required_global_settings(
        win, control_actions.MUSETALK_REQUIRED_CONTROL_SETTINGS, "MuseTalk lip-sync"
    )
    assert ("LandmarkDetectToggle", True) in applied
    assert ("LandmarkDetectModelSelection", "68") in applied


def test_the_toggle_is_applied_before_the_model_choice(monkeypatch) -> None:
    """Selecting a model while detection is off would unload it again."""
    applied: list[str] = []
    monkeypatch.setattr(
        control_actions.common_widget_actions,
        "update_control",
        lambda mw, name, value, **kw: applied.append(name),
    )
    win = _window(
        {"LandmarkDetectToggle": False, "LandmarkDetectModelSelection": "203"}
    )
    control_actions.apply_required_global_settings(
        win, control_actions.MUSETALK_REQUIRED_CONTROL_SETTINGS, "MuseTalk lip-sync"
    )
    assert applied.index("LandmarkDetectToggle") < applied.index(
        "LandmarkDetectModelSelection"
    )


def test_settings_already_correct_are_left_alone(monkeypatch) -> None:
    """No spurious control writes, which would each trigger a re-render."""
    applied: list[str] = []
    monkeypatch.setattr(
        control_actions.common_widget_actions,
        "update_control",
        lambda mw, name, value, **kw: applied.append(name),
    )
    win = _window({"LandmarkDetectToggle": True, "LandmarkDetectModelSelection": "68"})
    control_actions.apply_required_global_settings(
        win, control_actions.MUSETALK_REQUIRED_CONTROL_SETTINGS, "MuseTalk lip-sync"
    )
    assert applied == []


def test_nothing_required_is_a_no_op(monkeypatch) -> None:
    monkeypatch.setattr(
        control_actions.common_widget_actions,
        "update_control",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    control_actions.apply_required_global_settings(_window({}), None, "x")
    control_actions.apply_required_global_settings(_window({}), {}, "x")
