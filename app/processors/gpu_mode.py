"""Multi-GPU mode resolution from workspace control dict (no Qt imports)."""

from __future__ import annotations

from typing import Any

MULTI_GPU_MODE_OFF = "off"
MULTI_GPU_MODE_STAGE = "stage"
MULTI_GPU_MODE_FRAME = "frame"

MULTI_GPU_MODE_LABEL_TO_KEY = {
    "Off": MULTI_GPU_MODE_OFF,
    "Stage offload": MULTI_GPU_MODE_STAGE,
    "Frame routing (legacy)": MULTI_GPU_MODE_FRAME,
}


def resolve_multi_gpu_mode_key(control: dict[str, Any]) -> str:
    label = control.get("MultiGpuModeSelection")
    if label is not None:
        key = MULTI_GPU_MODE_LABEL_TO_KEY.get(str(label).strip())
        if key:
            return key
    if bool(control.get("MultiGpuRoutingEnableToggle", False)):
        return MULTI_GPU_MODE_FRAME
    return MULTI_GPU_MODE_OFF
