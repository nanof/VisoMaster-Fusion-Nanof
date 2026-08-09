"""Filesystem layout for MuseTalk weights under ``model_assets/musetalk/``."""

from __future__ import annotations

import os
from pathlib import Path

from app.processors.models_data import models_dir

_PKG_DIR = Path(__file__).resolve().parent


def prepare_transformers_env() -> None:
    """Pin transformers to the torch backend before it is first imported.

    TensorFlow is installed for ``mouth_action_detector``, so transformers would
    otherwise import it and crash: TF pulls in ``astunparse`` → ``six.moves``,
    which the PySide6 import hook inspects and trips over
    ("'_SixMetaPathImporter' object has no attribute '_path'"). Values are only
    defaults so an explicit user setting still wins.
    """
    for var, value in (("USE_TORCH", "1"), ("USE_TF", "0"), ("USE_FLAX", "0")):
        os.environ.setdefault(var, value)


def musetalk_root() -> Path:
    return Path(models_dir) / "musetalk"


def unet_weights_path() -> Path:
    return musetalk_root() / "musetalkV15" / "unet.pth"


def unet_config_path() -> Path:
    """Prefer downloaded config; fall back to the small JSON shipped with the package."""
    downloaded = musetalk_root() / "musetalkV15" / "musetalk.json"
    if downloaded.is_file():
        return downloaded
    return _PKG_DIR / "musetalk_v15.json"


def vae_dir() -> Path:
    return musetalk_root() / "sd-vae"


def whisper_dir() -> Path:
    return musetalk_root() / "whisper"


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 8:
            return candidate
    return None


def musetalk_assets_ready() -> bool:
    """True when the minimum inference weights exist (VAE + UNet + Whisper)."""
    # Either weight format is fine; the download ships .bin but a safetensors
    # copy dropped in by hand works just as well.
    vae_weights = _first_existing(
        vae_dir() / "diffusion_pytorch_model.bin",
        vae_dir() / "diffusion_pytorch_model.safetensors",
    )
    whisper_weights = _first_existing(
        whisper_dir() / "pytorch_model.bin",
        whisper_dir() / "model.safetensors",
    )
    if vae_weights is None or whisper_weights is None:
        return False
    required = (
        unet_weights_path(),
        unet_config_path(),
        vae_dir() / "config.json",
        whisper_dir() / "config.json",
        whisper_dir() / "preprocessor_config.json",
    )
    return all(p.is_file() and p.stat().st_size > 8 for p in required)
