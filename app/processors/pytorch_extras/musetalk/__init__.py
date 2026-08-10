"""
MuseTalk 1.5 lip-sync (optional PyTorch extras).

Enabled by the UI toggle 'MuseTalk Lip-Sync'. Does not load weights or import
torch/diffusers/transformers until ``MuseTalkEngine.load()`` is called.
"""

from __future__ import annotations

from app.processors.pytorch_extras.musetalk.paths import (
    musetalk_assets_ready,
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
    "musetalk_assets_ready",
    "musetalk_compile_enabled",
    "musetalk_perf_enabled",
    "musetalk_root",
]
