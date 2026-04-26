"""JSON persistence + ModelsProcessor sync for the new GPU advanced controls.

These tests exercise the ``gpu_settings_actions`` surface (the business
layer behind ``GpuWeightsEditor`` / ``GpuThreadsPerGpuEditor`` /
``GpuLoadBalancingModeSelection``) without instantiating any Qt widgets —
they are pure data round-trip tests.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Qt stubs (same approach used by test_settings_layout_data.py) so this file
# runs without PySide6 installed.
# ---------------------------------------------------------------------------


def _stub_module(name: str) -> MagicMock:
    mod = MagicMock()
    mod.__name__ = name
    mod.__spec__ = None
    return mod


import sys

for _n in (
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "qtpy",
    "qtpy.QtCore",
    "qtpy.QtGui",
    "qtpy.QtWidgets",
    "pyqttoast",
    "qdarktheme",
    # Stub the sibling UI modules that gpu_settings_actions imports to avoid
    # dragging the whole QtWidgets/SettingsLayout import chain.
    "app.ui.widgets.widget_components",
    "app.ui.widgets.actions.control_actions",
    "app.ui.widgets.actions.common_actions",
    "app.ui.widgets.actions.list_view_actions",
    "app.ui.widgets.actions.video_control_actions",
    "app.ui.widgets.actions.layout_actions",
    "app.ui.widgets.actions.save_load_actions",
    "app.ui.widgets.settings_layout_data",
):
    sys.modules.setdefault(_n, _stub_module(_n))


from app.processors.gpu_scheduler import WeightedScheduler  # noqa: E402
from app.ui.widgets.actions import gpu_settings_actions  # noqa: E402


def _make_main_window(targets: list[int] | None = None) -> SimpleNamespace:
    targets = targets or [0, 1]
    mp = SimpleNamespace(
        device="cuda",
        emulate_multi_gpu=False,
        gpu_index=0,
        emulated_gpu_count=len(targets),
        ui_multi_gpu_routing_enabled=True,
        load_balancing_mode="round_robin",
        gpu_weights={},
        threads_per_gpu={},
        gpu_auto_benchmark_on_start=False,
        gpu_auto_reweight_every_n_frames=0,
        get_ui_routing_targets_sorted=lambda tgts=targets: list(tgts),
        clamp_gpu_index=lambda idx: max(0, min(int(idx), len(targets) - 1)),
        resolve_effective_weights=None,  # filled below
    )

    def _resolve_effective_weights() -> dict[int, int]:
        mode = mp.load_balancing_mode
        if mode == "round_robin":
            return {int(t): 1 for t in targets}
        raw = mp.gpu_weights or {}
        return {int(t): int(raw.get(int(t), 1)) for t in targets}

    mp.resolve_effective_weights = _resolve_effective_weights

    vp = SimpleNamespace(scheduler=WeightedScheduler(targets=targets, weights={}))
    mw = SimpleNamespace(
        control={
            "GpuRoutingTargetsJson": json.dumps(targets),
            "MultiGpuRoutingEnableToggle": True,
            "GpuLoadBalancingModeSelection": "Round-Robin",
            "GpuWeightsJson": "{}",
            "GpuThreadsPerGpuJson": "{}",
            "GpuAutoBenchmarkOnStartToggle": False,
            "GpuAutoWeightsReweightEveryNFramesSlider": 0,
        },
        models_processor=mp,
        video_processor=vp,
        parameter_widgets={},
    )
    return mw


def test_on_gpu_weights_changed_round_trip_json_and_models_processor():
    mw = _make_main_window(targets=[0, 1])
    gpu_settings_actions.on_gpu_weights_changed(mw, {0: 3, 1: 1})
    stored = json.loads(mw.control["GpuWeightsJson"])
    assert stored == {"0": 3, "1": 1}
    # ModelsProcessor kept raw weights; the scheduler sees them only when the
    # mode is weighted (round_robin flattens them to 1).
    assert mw.models_processor.gpu_weights == {0: 3, 1: 1}
    assert mw.video_processor.scheduler.get_weights() == {0: 1, 1: 1}


def test_mode_selection_flips_to_weighted_manual_and_seeds_scheduler():
    mw = _make_main_window(targets=[0, 1])
    mw.control["GpuWeightsJson"] = json.dumps({0: 3, 1: 1})
    mw.control["GpuLoadBalancingModeSelection"] = "Weighted Manual"
    gpu_settings_actions.on_load_balancing_mode_changed(mw)
    assert mw.models_processor.load_balancing_mode == "weighted_manual"
    assert mw.video_processor.scheduler.get_weights() == {0: 3, 1: 1}


def test_threads_per_gpu_round_trip_and_zero_is_auto():
    mw = _make_main_window(targets=[0, 1])
    gpu_settings_actions.on_threads_per_gpu_changed(mw, {0: 3, 1: 0})
    stored = json.loads(mw.control["GpuThreadsPerGpuJson"])
    assert stored == {"0": 3, "1": 0}
    assert mw.models_processor.threads_per_gpu == {0: 3, 1: 0}


def test_apply_saved_gpu_settings_restores_mode_and_weights(monkeypatch):
    mw = _make_main_window(targets=[0, 1])
    mw.control["GpuLoadBalancingModeSelection"] = "Weighted Manual"
    mw.control["GpuWeightsJson"] = json.dumps({0: 5, 1: 2})
    mw.control["GpuThreadsPerGpuJson"] = json.dumps({0: 2, 1: 1})
    mw.control["GpuAutoBenchmarkOnStartToggle"] = True
    mw.control["GpuAutoWeightsReweightEveryNFramesSlider"] = 400

    monkeypatch.setattr(
        gpu_settings_actions,
        "sync_routing_json_to_models_processor",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        gpu_settings_actions,
        "migrate_legacy_gpu_slider_key",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        gpu_settings_actions,
        "_physical_cuda_count",
        lambda: 2,
    )
    mw.models_processor.set_gpu_index = lambda *_a, **_k: None

    gpu_settings_actions.apply_saved_gpu_settings(mw)

    assert mw.models_processor.load_balancing_mode == "weighted_manual"
    assert mw.models_processor.gpu_weights == {0: 5, 1: 2}
    assert mw.models_processor.threads_per_gpu == {0: 2, 1: 1}
    assert mw.models_processor.gpu_auto_benchmark_on_start is True
    assert mw.models_processor.gpu_auto_reweight_every_n_frames == 400
    assert mw.video_processor.scheduler.get_weights() == {0: 5, 1: 2}
