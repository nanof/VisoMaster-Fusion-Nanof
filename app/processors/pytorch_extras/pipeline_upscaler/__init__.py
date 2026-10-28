"""Last-stage HYPIR / PMRF / FlashVSR / SeedVR2 upscaler (optional PyTorch extras)."""

from __future__ import annotations

from app.processors.pytorch_extras.pipeline_upscaler.engine import PipelineUpscalerEngine
from app.processors.pytorch_extras.pipeline_upscaler.paths import (
    assets_error_message,
    assets_ready,
    deps_error_message,
    format_load_error,
    missing_python_deps,
)
from app.processors.pytorch_extras.pipeline_upscaler.scale import (
    PIPELINE_UPSCALER_TYPES,
    pipeline_upscaler_enabled,
    pipeline_upscaler_output_dims,
    pipeline_upscaler_scale_factor,
    pipeline_upscaler_type,
)

__all__ = [
    "PipelineUpscalerEngine",
    "PIPELINE_UPSCALER_TYPES",
    "assets_error_message",
    "assets_ready",
    "deps_error_message",
    "format_load_error",
    "missing_python_deps",
    "pipeline_upscaler_enabled",
    "pipeline_upscaler_output_dims",
    "pipeline_upscaler_scale_factor",
    "pipeline_upscaler_type",
]
