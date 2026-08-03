"""Unit tests for soft-retire / grace-period ONNX unload policy.

Mirrors ``ModelsProcessor.unload_model`` / pending-unload helpers without importing
the full processor (heavy ORT/TRT deps).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Dict, Optional


class _GraceUnloadHarness:
    """Minimal stand-in for the grace-period unload path in ModelsProcessor."""

    def __init__(self, *, grace: float = 60.0, keep_alive: bool = False) -> None:
        self.main_window = SimpleNamespace(
            control={
                "ModelUnloadGraceSecondsSlider": grace,
                "KeepModelsAliveToggle": keep_alive,
            }
        )
        self.force_unload_in_progress = False
        self.model_lock = threading.RLock()
        self.models: Dict[str, Optional[object]] = {
            "OldSwap": object(),
            "KeepMe": object(),
        }
        self._model_pending_unload_mono: Dict[str, float] = {}
        self._hard_unload_names: list[str] = []

    def _model_unload_grace_seconds(self) -> float:
        try:
            return float(
                self.main_window.control.get("ModelUnloadGraceSecondsSlider", 60) or 0
            )
        except (TypeError, ValueError):
            return 60.0

    def _cancel_pending_unload_locked(self, model_name: str) -> None:
        if not model_name:
            return
        prefix = f"{model_name}__cuda"
        for key in list(self._model_pending_unload_mono.keys()):
            if key == model_name or key.startswith(prefix):
                self._model_pending_unload_mono.pop(key, None)

    def touch_model_usage(self, model_name: str) -> None:
        with self.model_lock:
            self._cancel_pending_unload_locked(model_name)

    def unload_model(self, model_name_to_unload, force_immediate: bool = False):
        if not self.force_unload_in_progress:
            if self.main_window.control.get("KeepModelsAliveToggle", False):
                return
        grace = self._model_unload_grace_seconds()
        immediate = (
            bool(force_immediate)
            or self.force_unload_in_progress
            or grace <= 0.0
        )
        if not immediate:
            if not model_name_to_unload:
                return
            due = time.monotonic() + grace
            with self.model_lock:
                self._model_pending_unload_mono[model_name_to_unload] = due
            return
        with self.model_lock:
            self._cancel_pending_unload_locked(model_name_to_unload)
            if model_name_to_unload in self.models:
                self.models[model_name_to_unload] = None
            self._hard_unload_names.append(model_name_to_unload)

    def process_pending_model_unloads(self, *, force_all: bool = False) -> None:
        if self.main_window.control.get("KeepModelsAliveToggle", False):
            if not self.force_unload_in_progress:
                return
        now = time.monotonic()
        with self.model_lock:
            due = [
                name
                for name, deadline in list(self._model_pending_unload_mono.items())
                if force_all or now >= deadline
            ]
            for name in due:
                self._model_pending_unload_mono.pop(name, None)
        for name in due:
            self.unload_model(name, force_immediate=True)

    def flush_pending_model_unloads(self) -> None:
        self.process_pending_model_unloads(force_all=True)


def test_unload_schedules_when_grace_positive():
    mp = _GraceUnloadHarness(grace=60)
    mp.unload_model("OldSwap")
    assert mp.models["OldSwap"] is not None
    assert "OldSwap" in mp._model_pending_unload_mono
    assert mp._hard_unload_names == []


def test_unload_immediate_when_grace_zero():
    mp = _GraceUnloadHarness(grace=0)
    mp.unload_model("OldSwap")
    assert mp.models["OldSwap"] is None
    assert mp._hard_unload_names == ["OldSwap"]
    assert "OldSwap" not in mp._model_pending_unload_mono


def test_touch_cancels_pending_unload():
    mp = _GraceUnloadHarness(grace=60)
    mp.unload_model("OldSwap")
    assert "OldSwap" in mp._model_pending_unload_mono
    mp.touch_model_usage("OldSwap")
    assert "OldSwap" not in mp._model_pending_unload_mono


def test_process_pending_unloads_when_due():
    mp = _GraceUnloadHarness(grace=60)
    mp.unload_model("OldSwap")
    with mp.model_lock:
        mp._model_pending_unload_mono["OldSwap"] = time.monotonic() - 1.0
    mp.process_pending_model_unloads()
    assert mp.models["OldSwap"] is None
    assert mp._hard_unload_names == ["OldSwap"]
    assert "OldSwap" not in mp._model_pending_unload_mono


def test_process_pending_skips_not_yet_due():
    mp = _GraceUnloadHarness(grace=60)
    mp.unload_model("OldSwap")
    mp.process_pending_model_unloads()
    assert mp.models["OldSwap"] is not None
    assert mp._hard_unload_names == []


def test_flush_pending_forces_all():
    mp = _GraceUnloadHarness(grace=120)
    mp.unload_model("OldSwap")
    mp.unload_model("KeepMe")
    mp.flush_pending_model_unloads()
    assert set(mp._hard_unload_names) == {"OldSwap", "KeepMe"}
    assert not mp._model_pending_unload_mono


def test_keep_alive_blocks_schedule_and_hard():
    mp = _GraceUnloadHarness(grace=60, keep_alive=True)
    mp.unload_model("OldSwap")
    assert "OldSwap" not in mp._model_pending_unload_mono
    assert mp.models["OldSwap"] is not None
    mp.unload_model("OldSwap", force_immediate=True)
    assert mp.models["OldSwap"] is not None


def test_force_unload_bypasses_keep_alive():
    mp = _GraceUnloadHarness(grace=60, keep_alive=True)
    mp.force_unload_in_progress = True
    mp.unload_model("OldSwap")
    assert mp.models["OldSwap"] is None
    assert mp._hard_unload_names == ["OldSwap"]


def test_force_immediate_bypasses_grace():
    mp = _GraceUnloadHarness(grace=90)
    mp.unload_model("OldSwap", force_immediate=True)
    assert mp.models["OldSwap"] is None
    assert "OldSwap" not in mp._model_pending_unload_mono
