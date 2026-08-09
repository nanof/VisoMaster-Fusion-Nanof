"""Download VisoMaster model assets, including MuseTalk lip-sync weights.

MuseTalk (~4 GB) is part of the default set so the launcher's
"Check / Update Models" installs it. Use ``--skip-musetalk`` (or
``VISOFUSION_SKIP_MUSETALK=1``) to opt out.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.helpers.downloader import download_file
from app.processors.models_data import (
    models_list,
    musetalk_assets_list,
    pytorch_assets_list,
)

# When USE_OPTIMIZED_MODELS=true is set in portable.cfg (written by the
# launcher after running "Optimize Models (onnxsim)"), skip the hash check
# for existing files so optimized models are not re-downloaded.
_cfg_path = Path(__file__).resolve().parent.parent / "portable.cfg"
_skip_hash = False
if _cfg_path.is_file():
    for _line in _cfg_path.read_text(encoding="utf-8").splitlines():
        if _line.strip().upper() == "USE_OPTIMIZED_MODELS=TRUE":
            _skip_hash = True
            break


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download VisoMaster model assets")
    p.add_argument(
        "--skip-musetalk",
        action="store_true",
        help="Do not download MuseTalk 1.5 + VAE + Whisper (~4 GB)",
    )
    # Accepted for backwards compatibility: MuseTalk is now downloaded by default.
    p.add_argument("--musetalk", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    skip_musetalk = bool(args.skip_musetalk) or _env_truthy("VISOFUSION_SKIP_MUSETALK")

    for model_data in models_list + list(pytorch_assets_list):
        download_file(
            model_data["model_name"],
            model_data["local_path"],
            model_data["hash"],
            model_data["url"],
            skip_hash_check=_skip_hash,
        )

    if skip_musetalk:
        print(
            "\n[INFO] Skipping MuseTalk weights (--skip-musetalk / "
            "VISOFUSION_SKIP_MUSETALK). Lip-Sync will stay unavailable."
        )
        return

    print("\n[INFO] Downloading MuseTalk lip-sync assets (~4 GB on first run)...")
    for model_data in musetalk_assets_list:
        download_file(
            model_data["model_name"],
            model_data["local_path"],
            model_data["hash"],
            model_data["url"],
            skip_hash_check=_skip_hash,
        )


if __name__ == "__main__":
    main()
