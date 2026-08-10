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
    from app.processors.pytorch_extras.musetalk.paths import (
        musetalk_assets_error_message,
        musetalk_assets_ready,
        musetalk_deps_error_message,
        musetalk_missing_python_deps,
    )

    if not musetalk_assets_ready():
        print(f"[INFO] {musetalk_assets_error_message()}")
        return
    missing_deps = musetalk_missing_python_deps()
    if missing_deps:
        print(f"[INFO] {musetalk_deps_error_message(missing_deps)}")
        return
    print(
        "[INFO] MuseTalk ready: enable 'MuseTalk Lip-Sync' in Common after "
        "loading a video with audio."
    )
