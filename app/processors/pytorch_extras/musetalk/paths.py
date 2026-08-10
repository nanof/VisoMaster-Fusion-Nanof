"""Filesystem layout for MuseTalk weights under ``model_assets/musetalk/``."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from app.processors.models_data import models_dir

_PKG_DIR = Path(__file__).resolve().parent

# Import names MuseTalk load() needs. Hints point at the launcher, not pip.
_MUSETALK_PYTHON_DEPS: tuple[tuple[str, str], ...] = (
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
)

_FIX_MODELS = (
    "Fix: Launcher → Check / Update Models, or run: python download_models.py"
)
_FIX_DEPS = (
    "Fix: Launcher → Check / Update Dependencies "
    "(packages are in requirements_cu13.txt)."
)


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


def _is_usable_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 8


def musetalk_missing_assets() -> list[str]:
    """Human labels for required weight/config files that are missing or empty.

    Labels are relative to ``musetalk_root()`` when possible so the message stays
    short. Either ``.bin`` or ``.safetensors`` satisfies VAE / Whisper weights.
    """
    root = musetalk_root()
    missing: list[str] = []

    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            return str(path)

    if not _is_usable_file(unet_weights_path()):
        missing.append(_rel(unet_weights_path()))

    # Packaged JSON fallback counts; only complain when neither exists.
    downloaded_cfg = root / "musetalkV15" / "musetalk.json"
    if not _is_usable_file(downloaded_cfg) and not _is_usable_file(
        _PKG_DIR / "musetalk_v15.json"
    ):
        missing.append(_rel(downloaded_cfg))

    vae_weights = _first_existing(
        vae_dir() / "diffusion_pytorch_model.bin",
        vae_dir() / "diffusion_pytorch_model.safetensors",
    )
    if vae_weights is None:
        missing.append("sd-vae/diffusion_pytorch_model.bin|.safetensors")
    if not _is_usable_file(vae_dir() / "config.json"):
        missing.append("sd-vae/config.json")

    whisper_weights = _first_existing(
        whisper_dir() / "pytorch_model.bin",
        whisper_dir() / "model.safetensors",
    )
    if whisper_weights is None:
        missing.append("whisper/pytorch_model.bin|model.safetensors")
    if not _is_usable_file(whisper_dir() / "config.json"):
        missing.append("whisper/config.json")
    if not _is_usable_file(whisper_dir() / "preprocessor_config.json"):
        missing.append("whisper/preprocessor_config.json")

    return missing


def musetalk_assets_ready() -> bool:
    """True when the minimum inference weights exist (VAE + UNet + Whisper)."""
    return not musetalk_missing_assets()


def musetalk_assets_error_message() -> str:
    """Actionable message listing missing MuseTalk weight/config files."""
    missing = musetalk_missing_assets()
    if not missing:
        return ""
    lines = ", ".join(missing)
    return (
        f"MuseTalk weights incomplete under {musetalk_root()}. "
        f"Missing: {lines}. {_FIX_MODELS}"
    )


def musetalk_missing_python_deps() -> list[str]:
    """Import names required by MuseTalk that are not installed (no import side effects)."""
    missing: list[str] = []
    for import_name, _label in _MUSETALK_PYTHON_DEPS:
        if importlib.util.find_spec(import_name) is None:
            missing.append(import_name)
    return missing


def musetalk_deps_error_message(missing: list[str] | None = None) -> str:
    """Actionable message for missing MuseTalk Python packages."""
    names = list(missing) if missing is not None else musetalk_missing_python_deps()
    if not names:
        return ""
    labels = []
    by_import = {name: label for name, label in _MUSETALK_PYTHON_DEPS}
    for name in names:
        labels.append(by_import.get(name, name))
    return (
        "MuseTalk Python dependencies missing: "
        + ", ".join(labels)
        + f". {_FIX_DEPS}"
    )


def format_musetalk_import_error(exc: BaseException) -> str | None:
    """Map an ImportError to a deps message; return None if not an import failure."""
    if not isinstance(exc, ImportError):
        return None
    name = getattr(exc, "name", None) or ""
    # "No module named 'diffusers'" / nested "diffusers.models..."
    if not name:
        text = str(exc)
        for import_name, _label in _MUSETALK_PYTHON_DEPS:
            if import_name in text:
                name = import_name
                break
    if name:
        root = name.split(".", 1)[0]
        return musetalk_deps_error_message([root])
    return f"MuseTalk dependency import failed: {exc}. {_FIX_DEPS}"


def format_musetalk_load_error(exc: BaseException) -> str:
    """Turn a load-time exception into a short actionable message."""
    mapped = format_musetalk_import_error(exc)
    if mapped:
        return mapped
    text = str(exc).strip() or exc.__class__.__name__
    lower = text.lower()
    if "cuda" in lower and (
        "out of memory" in lower or "oom" in lower or "ENOMEM" in text
    ):
        return (
            f"MuseTalk ran out of GPU memory while loading ({text}). "
            "Close other GPU apps, lower batch size, or disable torch.compile."
        )
    if "no such file" in lower or "filenotfound" in lower:
        return f"MuseTalk load failed (file missing): {text}. {_FIX_MODELS}"
    return f"MuseTalk load failed: {text}"
