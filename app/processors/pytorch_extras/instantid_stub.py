"""
Stub para InstantID / identidad generativa en PyTorch.

Implementación completa: cargar diffusers + checkpoints y exponer una API estable
desde el worker o un subproceso. Este módulo solo documenta el hook.
"""

from __future__ import annotations

from app.processors.pytorch_extras import is_pytorch_extras_enabled


def run_instantid_placeholder(*_args, **_kwargs) -> None:
    if not is_pytorch_extras_enabled():
        print(
            "[INFO] InstantID: stub gated by VISOFUSION_PYTORCH_EXTRAS=1 "
            "(deps already in requirements_cu13.txt via launcher)."
        )
        return
    print(
        "[INFO] InstantID: stub — conectar checkpoints y orquestación en "
        "una iteración posterior."
    )
