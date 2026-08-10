"""
MuseTalk 1.5 lip-sync (optional PyTorch extras).

Enabled by the UI toggle 'MuseTalk Lip-Sync'. Does not load weights or import
torch/diffusers/transformers until ``MuseTalkEngine.load()`` is called.
"""

from __future__ import annotations

from app.processors.pytorch_extras.musetalk.paths import (
    format_musetalk_load_error,
    musetalk_assets_error_message,
    musetalk_assets_ready,
    musetalk_deps_error_message,
    musetalk_missing_assets,
    musetalk_missing_python_deps,
    musetalk_root,
    prepare_transformers_env,
)

# Must run before anything imports transformers; only this package does.
prepare_transformers_env()

from app.processors.pytorch_extras.musetalk.engine import (  # noqa: E402
    MuseTalkEngine,
    musetalk_compile_enabled,
    musetalk_perf_enabled,
)

__all__ = [
    "MuseTalkEngine",
    "format_musetalk_load_error",
    "musetalk_assets_error_message",
    "musetalk_assets_ready",
    "musetalk_compile_enabled",
    "musetalk_deps_error_message",
    "musetalk_missing_assets",
    "musetalk_missing_python_deps",
    "musetalk_perf_enabled",
    "musetalk_root",
]
