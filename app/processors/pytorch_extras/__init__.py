"""
Extensiones PyTorch opcionales (InstantID, retalking, etc.).

Activación: variable de entorno VISOFUSION_PYTORCH_EXTRAS=1 (o "true").
Sin activar, la app no importa estos módulos en el arranque.
"""

from __future__ import annotations

import os


def is_pytorch_extras_enabled() -> bool:
    v = os.environ.get("VISOFUSION_PYTORCH_EXTRAS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


__all__ = ["is_pytorch_extras_enabled"]
