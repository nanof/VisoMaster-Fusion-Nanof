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


class _CompiledEngine:
    def __init__(self, compiled: bool):
        self.is_loaded = True
        self._compiled = compiled


class _CompileModelsProcessorStub(_ModelsProcessorStub):
    def __init__(self, engine=None):
        super().__init__()
        self.musetalk_engine = engine
        self.compile_args: list[bool | None] = []

    def ensure_musetalk_loaded(self, compile: bool | None = None) -> bool:
        self.compile_args.append(compile)
        self.calls.append("load")
        return True

    def unload_musetalk(self) -> None:
        self.calls.append("unload")
        self.musetalk_engine = None


def test_compile_toggle_noop_when_lip_sync_disabled(monkeypatch):
    monkeypatch.delenv("VISOFUSION_MUSETALK_COMPILE", raising=False)
    main_window = types.SimpleNamespace(
        control={
            "MuseTalkEnableToggle": False,
            "MuseTalkCompileToggle": False,
        },
        models_processor=_CompileModelsProcessorStub(),
        video_processor=None,
    )
    control_actions.handle_musetalk_compile_change(
        main_window, True, "MuseTalkCompileToggle"
    )
    assert main_window.models_processor.calls == []


def test_compile_toggle_reloads_when_lip_sync_loaded(monkeypatch):
    monkeypatch.delenv("VISOFUSION_MUSETALK_COMPILE", raising=False)
    mp = _CompileModelsProcessorStub(engine=_CompiledEngine(compiled=False))
    main_window = types.SimpleNamespace(
        control={
            "MuseTalkEnableToggle": True,
            "MuseTalkCompileToggle": False,
        },
        models_processor=mp,
        video_processor=None,
    )

    def _fake_prepare(mw, **_):
        mp.calls.append("audio")

    monkeypatch.setattr(control_actions, "_prepare_musetalk_audio", _fake_prepare)
    control_actions.handle_musetalk_compile_change(
        main_window, True, "MuseTalkCompileToggle"
    )
    assert mp.calls == ["unload", "load", "audio"]
    assert mp.compile_args == [True]


def test_compile_toggle_skips_reload_when_already_matching(monkeypatch):
    monkeypatch.delenv("VISOFUSION_MUSETALK_COMPILE", raising=False)
    mp = _CompileModelsProcessorStub(engine=_CompiledEngine(compiled=True))
    main_window = types.SimpleNamespace(
        control={
            "MuseTalkEnableToggle": True,
            "MuseTalkCompileToggle": False,
        },
        models_processor=mp,
        video_processor=None,
    )
    control_actions.handle_musetalk_compile_change(
        main_window, True, "MuseTalkCompileToggle"
    )
    assert mp.calls == []


def test_compile_toggle_uses_new_value_not_stale_control(monkeypatch):
    """Regression: update_control runs exec before writing the control."""
    monkeypatch.delenv("VISOFUSION_MUSETALK_COMPILE", raising=False)
    mp = _CompileModelsProcessorStub(engine=_CompiledEngine(compiled=False))
    main_window = types.SimpleNamespace(
        control={
            "MuseTalkEnableToggle": True,
            "MuseTalkCompileToggle": False,  # stale
        },
        models_processor=mp,
        video_processor=None,
    )
    monkeypatch.setattr(
        control_actions, "_prepare_musetalk_audio", lambda *_a, **_k: None
    )
    control_actions.handle_musetalk_compile_change(
        main_window, True, "MuseTalkCompileToggle"
    )
    assert mp.compile_args == [True]