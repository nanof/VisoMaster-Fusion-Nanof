"""GPU device UI: primary CUDA device selection by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from PySide6 import QtWidgets

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


def _primary_combo_physical_index(combo: QtWidgets.QComboBox) -> int | None:
    """Resolve CUDA ordinal from combo userData (handles Qt roles where currentData() is None)."""
    for getter in (
        lambda: combo.currentData(),
        lambda: combo.itemData(combo.currentIndex()),
    ):
        try:
            data = getter()
        except Exception:
            data = None
        if data is None:
            continue
        try:
            return int(data)
        except (TypeError, ValueError):
            continue
    return None


def on_primary_gpu_combo_changed(main_window: MainWindow, *_args) -> None:
    combo = main_window.parameter_widgets.get("GpuPrimaryDeviceSelection")
    if combo is None:
        return
    phys = _primary_combo_physical_index(combo)
    if phys is None:
        print(
            "[WARN] Primary GPU: could not read CUDA index from combo (no itemData).",
            flush=True,
        )
        return
    main_window.control["GpuPrimaryPhysicalIndex"] = phys
    apply_primary_gpu_physical_index(main_window, phys)


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

    combo = main_window.parameter_widgets.get("GpuPrimaryDeviceSelection")
    if combo is not None:
        combo.blockSignals(True)
        fill_primary_gpu_combo(combo, main_window)
        combo.blockSignals(False)

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

    combo = main_window.parameter_widgets.get("GpuPrimaryDeviceSelection")
    if combo is not None:
        fill_primary_gpu_combo(combo, main_window)

    apply_saved_gpu_settings(main_window)
