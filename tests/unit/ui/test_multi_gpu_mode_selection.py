from __future__ import annotations

from app.processors.gpu_mode import resolve_multi_gpu_mode_key


def test_resolve_multi_gpu_mode_key_from_selection():
    control = {"MultiGpuModeSelection": "Stage offload"}
    assert resolve_multi_gpu_mode_key(control) == "stage"
    control = {"MultiGpuModeSelection": "Frame routing (legacy)"}
    assert resolve_multi_gpu_mode_key(control) == "frame"


def test_resolve_multi_gpu_mode_key_migrates_legacy_toggle():
    control = {"MultiGpuRoutingEnableToggle": True}
    assert resolve_multi_gpu_mode_key(control) == "frame"
    control = {"MultiGpuRoutingEnableToggle": False}
    assert resolve_multi_gpu_mode_key(control) == "off"
