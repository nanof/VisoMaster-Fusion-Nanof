"""Integer output scale for the last-stage pipeline upscaler (no ML imports)."""

from __future__ import annotations

from typing import Mapping, Any

PIPELINE_UPSCALER_TYPE_HYPIR = "HYPIR SD2"
PIPELINE_UPSCALER_TYPE_PMRF = "PMRF 2x"
PIPELINE_UPSCALER_TYPE_FLASHVSR_TINY = "FlashVSR Tiny"
PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL = "FlashVSR Full"
PIPELINE_UPSCALER_TYPE_SEEDVR2_3B = "SeedVR2 3B"
PIPELINE_UPSCALER_TYPE_SEEDVR2_7B = "SeedVR2 7B"

PIPELINE_UPSCALER_TYPES = (
    PIPELINE_UPSCALER_TYPE_HYPIR,
    PIPELINE_UPSCALER_TYPE_PMRF,
    PIPELINE_UPSCALER_TYPE_FLASHVSR_TINY,
    PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL,
    PIPELINE_UPSCALER_TYPE_SEEDVR2_3B,
    PIPELINE_UPSCALER_TYPE_SEEDVR2_7B,
)

HYPIR_TYPES = frozenset({PIPELINE_UPSCALER_TYPE_HYPIR})
PMRF_TYPES = frozenset({PIPELINE_UPSCALER_TYPE_PMRF})
FLASHVSR_TYPES = frozenset(
    {
        PIPELINE_UPSCALER_TYPE_FLASHVSR_TINY,
        PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL,
    }
)
SEEDVR2_TYPES = frozenset(
    {
        PIPELINE_UPSCALER_TYPE_SEEDVR2_3B,
        PIPELINE_UPSCALER_TYPE_SEEDVR2_7B,
    }
)

_DEFAULT_SCALE = {
    PIPELINE_UPSCALER_TYPE_HYPIR: 1,
    PIPELINE_UPSCALER_TYPE_PMRF: 2,
    PIPELINE_UPSCALER_TYPE_FLASHVSR_TINY: 4,
    PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL: 4,
    PIPELINE_UPSCALER_TYPE_SEEDVR2_3B: 4,
    PIPELINE_UPSCALER_TYPE_SEEDVR2_7B: 4,
}


def pipeline_upscaler_enabled(control: Mapping[str, Any] | None) -> bool:
    if not control:
        return False
    return bool(control.get("PipelineUpscalerEnableToggle", False))


def pipeline_upscaler_type(control: Mapping[str, Any] | None) -> str:
    if not control:
        return PIPELINE_UPSCALER_TYPE_HYPIR
    raw = str(
        control.get("PipelineUpscalerTypeSelection", PIPELINE_UPSCALER_TYPE_HYPIR)
        or PIPELINE_UPSCALER_TYPE_HYPIR
    )
    if raw in PIPELINE_UPSCALER_TYPES:
        return raw
    return PIPELINE_UPSCALER_TYPE_HYPIR


def pipeline_upscaler_scale_factor(control: Mapping[str, Any] | None) -> int:
    """Spatial multiplier applied after the rest of the pipeline (1 if disabled)."""
    if not pipeline_upscaler_enabled(control):
        return 1
    kind = pipeline_upscaler_type(control)
    raw = str(control.get("PipelineUpscalerScaleSelection") or "")
    stripped = raw.strip()
    if stripped.startswith("1"):
        return 1
    if stripped.startswith("2"):
        return 2
    if stripped.startswith("4"):
        return 4
    return int(_DEFAULT_SCALE.get(kind, 4))


def pipeline_upscaler_output_dims(
    control: Mapping[str, Any] | None, *, frame_height: int, frame_width: int
) -> tuple[int, int]:
    """Height/width after the last-stage upscaler (even, for video codecs)."""
    scale = pipeline_upscaler_scale_factor(control)
    if scale <= 1:
        return int(frame_height), int(frame_width)
    h = int(frame_height) * scale
    w = int(frame_width) * scale
    return h & ~1, w & ~1
