"""GPU device UI: primary device by name, multi-GPU routing toggle, optional targets + emulated slot."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List

import torch
from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt

from app.ui.widgets.actions import control_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def _physical_cuda_count() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return max(0, int(torch.cuda.device_count()))
    except Exception:
        return 0


def primary_gpu_combo_entries() -> list[tuple[int, str]]:
    """(physical_index, display_label) for each CUDA device."""
    n = _physical_cuda_count()
    if n <= 0:
        return [(0, "GPU 0 (CUDA not available)")]
    out: list[tuple[int, str]] = []
    for i in range(n):
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = "Unknown"
        out.append((i, f"{i}: {name}"))
    return out


def fill_primary_gpu_combo(combo: QtWidgets.QComboBox, main_window: MainWindow) -> None:
    combo.blockSignals(True)
    combo.clear()
    for phys, label in primary_gpu_combo_entries():
        combo.addItem(label, phys)
    idx = int(main_window.control.get("GpuPrimaryPhysicalIndex", 0))
    n = _physical_cuda_count()
    if n <= 0:
        combo.setCurrentIndex(0)
    else:
        idx = max(0, min(idx, n - 1))
        for j in range(combo.count()):
            if combo.itemData(j) == idx:
                combo.setCurrentIndex(j)
                break
        else:
            combo.setCurrentIndex(0)
    combo.blockSignals(False)


def apply_primary_gpu_physical_index(main_window: MainWindow, physical_index: int) -> None:
    """Apply primary CUDA device (global); same heavy path as legacy slider."""
    n = _physical_cuda_count()
    if n <= 0:
        physical_index = 0
    else:
        physical_index = max(0, min(int(physical_index), n - 1))
    control_actions.change_gpu_index(main_window, physical_index)


def on_primary_gpu_combo_changed(main_window: MainWindow, *_args) -> None:
    combo = main_window.parameter_widgets.get("GpuPrimaryDeviceSelection")
    if combo is None or not hasattr(combo, "currentData"):
        return
    data = combo.currentData()
    if data is None:
        return
    try:
        phys = int(data)
    except (TypeError, ValueError):
        return
    main_window.control["GpuPrimaryPhysicalIndex"] = phys
    apply_primary_gpu_physical_index(main_window, phys)
    picker = main_window.parameter_widgets.get("GpuRoutingTargetsPicker")
    if picker is not None and hasattr(picker, "rebuild_from_models"):
        picker.rebuild_from_models()


def on_multi_gpu_routing_toggle(main_window: MainWindow, enabled: bool) -> None:
    mp = main_window.models_processor
    mp.ui_multi_gpu_routing_enabled = bool(enabled)
    picker = main_window.parameter_widgets.get("GpuRoutingTargetsPicker")
    if picker is not None and hasattr(picker, "rebuild_from_models"):
        picker.rebuild_from_models()
    # When the set of routing targets changes, weight / thread editors must
    # refresh their row list; the GpuLiveMetricsPanel refreshes itself via
    # the GpuLoadMetrics signal once processing starts.
    for key in ("GpuWeightsEditor", "GpuThreadsPerGpuEditor"):
        w = main_window.parameter_widgets.get(key)
        if w is not None and hasattr(w, "rebuild_from_models"):
            w.rebuild_from_models()
    _sync_scheduler_config(main_window)


# ---------------------------------------------------------------------------
# Load-balancing mode / weights / threads-per-gpu plumbing
# ---------------------------------------------------------------------------


_MODE_LABEL_TO_KEY = {
    "Round-Robin": "round_robin",
    "Weighted Manual": "weighted_manual",
    "Weighted Auto": "weighted_auto",
    "Hybrid": "hybrid",
}


def _normalize_mode_label(label: object) -> str:
    s = str(label or "").strip()
    if s in _MODE_LABEL_TO_KEY:
        return _MODE_LABEL_TO_KEY[s]
    from app.processors.gpu_scheduler import normalize_mode

    return normalize_mode(s)


def _parse_int_dict(raw: object) -> dict[int, int]:
    try:
        data = raw if isinstance(raw, dict) else json.loads(str(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[int, int] = {}
    for k, v in data.items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _sync_scheduler_config(main_window: MainWindow) -> None:
    """Push mode / weights / threads into ModelsProcessor and VideoProcessor.

    Also re-seeds the live ``WeightedScheduler`` so that changes take effect
    immediately without having to stop and restart processing (except for the
    worker pool size, which is bound to the pool lifetime — that still needs
    a new run to apply).
    """
    mp = main_window.models_processor
    mp.load_balancing_mode = _normalize_mode_label(
        main_window.control.get("GpuLoadBalancingModeSelection", "Round-Robin")
    )
    mp.gpu_weights = _parse_int_dict(
        main_window.control.get("GpuWeightsJson", "{}")
    )
    mp.threads_per_gpu = _parse_int_dict(
        main_window.control.get("GpuThreadsPerGpuJson", "{}")
    )
    try:
        mp.gpu_auto_reweight_every_n_frames = int(
            main_window.control.get("GpuAutoWeightsReweightEveryNFramesSlider", 0)
        )
    except (TypeError, ValueError):
        mp.gpu_auto_reweight_every_n_frames = 0
    mp.gpu_auto_benchmark_on_start = bool(
        main_window.control.get("GpuAutoBenchmarkOnStartToggle", False)
    )
    vp = getattr(main_window, "video_processor", None)
    if vp is None:
        return
    try:
        targets = list(mp.get_ui_routing_targets_sorted())
        weights = mp.resolve_effective_weights()
        vp.scheduler.set_targets(targets, weights)
    except Exception:
        pass


def on_gpu_weights_changed(main_window: MainWindow, data: dict[int, int]) -> None:
    main_window.control["GpuWeightsJson"] = json.dumps(
        {int(k): int(v) for k, v in dict(data or {}).items()}
    )
    _sync_scheduler_config(main_window)


def on_threads_per_gpu_changed(main_window: MainWindow, data: dict[int, int]) -> None:
    main_window.control["GpuThreadsPerGpuJson"] = json.dumps(
        {int(k): int(v) for k, v in dict(data or {}).items()}
    )
    _sync_scheduler_config(main_window)


def on_load_balancing_mode_changed(main_window: MainWindow, *_args) -> None:
    _sync_scheduler_config(main_window)


def _parse_routing_json(raw: object) -> list[int]:
    if isinstance(raw, list):
        xs = [int(x) for x in raw]
    else:
        try:
            xs = json.loads(str(raw))
            if not isinstance(xs, list):
                return [0]
            xs = [int(x) for x in xs]
        except (json.JSONDecodeError, TypeError, ValueError):
            return [0]
    return xs


def sync_routing_json_to_models_processor(main_window: MainWindow) -> None:
    mp = main_window.models_processor
    raw = main_window.control.get("GpuRoutingTargetsJson", "[0]")
    xs = _parse_routing_json(raw)
    phy = max(0, _physical_cuda_count())
    prim = max(0, min(int(mp.gpu_index), max(0, phy - 1))) if phy > 0 else 0
    cpu_slot = (phy + 1) if phy > 0 else 1
    cleaned: list[int] = []
    for x in xs:
        if x < 0 or x > cpu_slot:
            continue
        cleaned.append(x)
    if not cleaned:
        cleaned = [prim]
    if prim not in cleaned:
        cleaned.append(prim)
    cleaned = sorted(set(cleaned))
    mp.ui_routing_targets = cleaned
    main_window.control["GpuRoutingTargetsJson"] = json.dumps(cleaned)


def apply_saved_gpu_settings(main_window: MainWindow) -> None:
    """Restore GPU controls from workspace (after widgets exist)."""
    migrate_legacy_gpu_slider_key(main_window.control)

    n = _physical_cuda_count()
    raw_pri = main_window.control.get("GpuPrimaryPhysicalIndex", 0)
    try:
        pri = int(raw_pri)
    except (TypeError, ValueError):
        pri = 0
    if n > 0:
        pri = max(0, min(pri, n - 1))
    main_window.control["GpuPrimaryPhysicalIndex"] = pri

    multi = bool(main_window.control.get("MultiGpuRoutingEnableToggle", False))
    main_window.models_processor.ui_multi_gpu_routing_enabled = multi

    sync_routing_json_to_models_processor(main_window)
    _sync_scheduler_config(main_window)

    combo = main_window.parameter_widgets.get("GpuPrimaryDeviceSelection")
    if combo is not None:
        combo.blockSignals(True)
        fill_primary_gpu_combo(combo, main_window)
        combo.blockSignals(False)

    toggle = main_window.parameter_widgets.get("MultiGpuRoutingEnableToggle")
    if toggle is not None and hasattr(toggle, "setChecked"):
        toggle.blockSignals(True)
        toggle.setChecked(multi)
        toggle.blockSignals(False)

    picker = main_window.parameter_widgets.get("GpuRoutingTargetsPicker")
    if picker is not None and hasattr(picker, "rebuild_from_models"):
        picker.blockSignals(True)
        picker.rebuild_from_models()
        picker.blockSignals(False)

    for key in ("GpuWeightsEditor", "GpuThreadsPerGpuEditor"):
        w = main_window.parameter_widgets.get(key)
        if w is not None and hasattr(w, "rebuild_from_models"):
            w.blockSignals(True)
            w.rebuild_from_models()
            w.blockSignals(False)

    main_window.models_processor.set_gpu_index(pri)


def migrate_legacy_gpu_slider_key(control: dict) -> None:
    if "GpuPrimaryPhysicalIndex" in control:
        return
    legacy = control.pop("GpuDeviceIndexSlider", None)
    if legacy is None:
        return
    try:
        control["GpuPrimaryPhysicalIndex"] = int(float(str(legacy).strip()))
    except (TypeError, ValueError):
        control["GpuPrimaryPhysicalIndex"] = 0


def finalize_gpu_widgets_after_settings_layout(main_window: MainWindow) -> None:
    """Populate dynamic GPU names and sync ModelsProcessor once."""
    migrate_legacy_gpu_slider_key(main_window.control)
    if "GpuPrimaryPhysicalIndex" not in main_window.control:
        main_window.control["GpuPrimaryPhysicalIndex"] = 0
    if "GpuRoutingTargetsJson" not in main_window.control:
        main_window.control["GpuRoutingTargetsJson"] = json.dumps([0])
    if "MultiGpuRoutingEnableToggle" not in main_window.control:
        main_window.control["MultiGpuRoutingEnableToggle"] = False
    if "MultiGpuAdvancedToggle" not in main_window.control:
        main_window.control["MultiGpuAdvancedToggle"] = False
    if "GpuLoadBalancingModeSelection" not in main_window.control:
        main_window.control["GpuLoadBalancingModeSelection"] = "Round-Robin"
    if "GpuWeightsJson" not in main_window.control:
        main_window.control["GpuWeightsJson"] = "{}"
    if "GpuThreadsPerGpuJson" not in main_window.control:
        main_window.control["GpuThreadsPerGpuJson"] = "{}"
    if "GpuAutoBenchmarkOnStartToggle" not in main_window.control:
        main_window.control["GpuAutoBenchmarkOnStartToggle"] = False
    if "GpuAutoWeightsReweightEveryNFramesSlider" not in main_window.control:
        main_window.control["GpuAutoWeightsReweightEveryNFramesSlider"] = 600
    if "GpuLiveMetricsOverlayToggle" not in main_window.control:
        main_window.control["GpuLiveMetricsOverlayToggle"] = False

    combo = main_window.parameter_widgets.get("GpuPrimaryDeviceSelection")
    if combo is not None:
        fill_primary_gpu_combo(combo, main_window)
        combo.currentIndexChanged.connect(
            lambda *_: on_primary_gpu_combo_changed(main_window),
            type=Qt.ConnectionType.UniqueConnection,
        )

    apply_saved_gpu_settings(main_window)
