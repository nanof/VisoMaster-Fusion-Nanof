"""
Retalking / audio-driven lip-sync entrypoint.

Preferred backend: MuseTalk 1.5 (see ``pytorch_extras.musetalk``).
Activation is via the UI toggle 'MuseTalk Lip-Sync'; this helper only reports
readiness. Dependencies and weights are installed by the launcher
("Check / Update Dependencies" + "Check / Update Models").
"""

from __future__ import annotations

from typing import Any


def run_retalking_placeholder(*_args: Any, **_kwargs: Any) -> None:
    from app.processors.pytorch_extras.musetalk.paths import musetalk_assets_ready

    if not musetalk_assets_ready():
        print(
            "[INFO] MuseTalk: pesos no encontrados. Ejecute en el launcher "
            "'Check / Update Dependencies' y 'Check / Update Models'."
        )
        return
    print(
        "[INFO] MuseTalk listo: active el toggle 'MuseTalk Lip-Sync' en la UI "
        "(Common) tras cargar un vídeo con audio."
    )
