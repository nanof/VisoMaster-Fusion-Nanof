"""Moving a lip-sync slider must repaint the frame you are looking at.

These sliders only pay off if their effect is visible while paused: judging a
mouth during playback is guesswork. The chain is update_control -> refresh_frame
-> process_current_frame, and it is easy to break by marking a widget as a
parameter or by adding an early return, so it is pinned here.
"""

from __future__ import annotations

import types

import pytest

from app.ui.widgets.actions import common_actions as cwa
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA

MASK_SLIDERS = [
    "MuseTalkBlendStrengthSlider",
    "MuseTalkMouthPaddingSlider",
    "MuseTalkMouthWidthSlider",
    "MuseTalkMouthHeightSlider",
    "MuseTalkMouthCentreSlider",
]

BYPASS = "MuseTalkBypassToggle"
MOUTH_ONLY = "MuseTalkMouthOnlyToggle"


class _Processor:
    def __init__(self, processing: bool = False) -> None:
        self.processing = processing
        self.ui_state_is_dirty = False
        self.feeder_control: dict = {}
        self.state_lock = __import__("threading").RLock()
        self.calls = 0

    def process_current_frame(self, synchronous: bool = False) -> None:
        self.calls += 1


def _window(processor: _Processor) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        control={k: 100 for k in MASK_SLIDERS},
        markers={},
        videoSeekSlider=types.SimpleNamespace(value=lambda: 0),
        video_processor=processor,
        parameter_widgets={},
        _parameter_widget_mirrors={},
    )


@pytest.mark.parametrize("slider", MASK_SLIDERS)
def test_changing_a_slider_repaints_the_paused_frame(slider, monkeypatch) -> None:
    monkeypatch.setattr(cwa, "sync_all_widgets_for_control_key", lambda *a, **k: None)
    proc = _Processor(processing=False)
    win = _window(proc)
    cwa.update_control(win, slider, 55)
    assert win.control[slider] == 55
    assert proc.calls == 1, "paused preview was not re-rendered"


def test_during_playback_the_feeder_is_updated_instead(monkeypatch) -> None:
    """No single-frame render while the pool is running; the feeder picks it up."""
    monkeypatch.setattr(cwa, "sync_all_widgets_for_control_key", lambda *a, **k: None)
    proc = _Processor(processing=True)
    proc.feeder_control = {"MuseTalkBlendStrengthSlider": 100}
    win = _window(proc)
    cwa.update_control(win, "MuseTalkBlendStrengthSlider", 40)
    assert proc.feeder_control["MuseTalkBlendStrengthSlider"] == 40
    assert proc.calls == 0


@pytest.mark.parametrize("slider", MASK_SLIDERS)
def test_the_sliders_are_controls_not_parameters(slider) -> None:
    """As parameters they would follow the selected face and skip the control path."""
    assert COMMON_LAYOUT_DATA["MuseTalk Lip-Sync"][slider]["data_type"] == "control"


def test_bypass_repaints_the_paused_frame(monkeypatch) -> None:
    """A/B comparison must update the exact frame currently being judged."""
    monkeypatch.setattr(cwa, "sync_all_widgets_for_control_key", lambda *a, **k: None)
    proc = _Processor(processing=False)
    win = _window(proc)
    win.control[BYPASS] = False
    cwa.update_control(win, BYPASS, True)
    assert win.control[BYPASS] is True
    assert proc.calls == 1


def test_bypass_is_a_control_not_a_parameter() -> None:
    layout = COMMON_LAYOUT_DATA["MuseTalk Lip-Sync"][BYPASS]
    assert layout["data_type"] == "control"
    assert layout["default"] is False


def test_local_mouth_repaint_defaults_on_and_hangs_off_face_parsing() -> None:
    """It is the default because it is what removes the doubled mouth and chin."""
    layout = COMMON_LAYOUT_DATA["MuseTalk Lip-Sync"][MOUTH_ONLY]
    assert layout["data_type"] == "control"
    assert layout["default"] is True
    assert layout["parentToggle"] == "MuseTalkFaceParsingToggle"


def test_the_padding_slider_follows_the_local_repaint_toggle() -> None:
    layout = COMMON_LAYOUT_DATA["MuseTalk Lip-Sync"]["MuseTalkMouthPaddingSlider"]
    assert layout["parentToggle"] == MOUTH_ONLY
    assert layout["default"] == "6"


def test_local_mouth_repaint_repaints_the_paused_frame(monkeypatch) -> None:
    monkeypatch.setattr(cwa, "sync_all_widgets_for_control_key", lambda *a, **k: None)
    proc = _Processor(processing=False)
    win = _window(proc)
    win.control[MOUTH_ONLY] = True
    cwa.update_control(win, MOUTH_ONLY, False)
    assert win.control[MOUTH_ONLY] is False
    assert proc.calls == 1
