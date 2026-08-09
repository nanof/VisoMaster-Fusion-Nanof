"""The MuseTalk toggle must act on the new value, not the stored one.

Regression: ``update_control`` runs exec functions *before* writing the new value
into ``main_window.control``, so a handler that read the control saw the previous
state and inverted the toggle (disabling loaded the engine and vice versa).
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("PySide6")

# Imported first: it pulls the UI action modules in an order that avoids the
# circular import triggered by importing control_actions directly.
from app.ui.widgets.actions import layout_actions  # noqa: F401,E402
from app.ui.widgets.actions import control_actions  # noqa: E402


class _ModelsProcessorStub:
    def __init__(self):
        self.calls: list[str] = []
        self.musetalk_engine = None

    def ensure_musetalk_loaded(self) -> bool:
        self.calls.append("load")
        return False  # stop before touching weights

    def unload_musetalk(self) -> None:
        self.calls.append("unload")


def _main_window(stored_value: bool):
    return types.SimpleNamespace(
        control={"MuseTalkEnableToggle": stored_value},
        models_processor=_ModelsProcessorStub(),
        video_processor=None,
    )


def test_enabling_loads_even_though_control_still_holds_false():
    main_window = _main_window(stored_value=False)
    # Mirrors update_control: exec_function(main_window, new_value, *exec_args)
    control_actions.handle_musetalk_toggle_change(
        main_window, True, "MuseTalkEnableToggle"
    )
    assert main_window.models_processor.calls == ["load"]


def test_disabling_unloads_even_though_control_still_holds_true():
    main_window = _main_window(stored_value=True)
    control_actions.handle_musetalk_toggle_change(
        main_window, False, "MuseTalkEnableToggle"
    )
    assert main_window.models_processor.calls == ["unload"]


def test_call_without_a_value_falls_back_to_the_control():
    main_window = _main_window(stored_value=True)
    control_actions.handle_musetalk_toggle_change(main_window)
    assert main_window.models_processor.calls == ["load"]


def test_audio_source_change_uses_the_new_selection(capsys):
    """'External file' arrives as new_value while the control still says 'Video track'."""
    main_window = _main_window(stored_value=True)
    control_actions.handle_musetalk_audio_change(
        main_window, "External file", "MuseTalkAudioSourceSelection"
    )
    # Engine never loads in this stub, so the run stops at ensure_musetalk_loaded.
    assert main_window.models_processor.calls == ["load"]
