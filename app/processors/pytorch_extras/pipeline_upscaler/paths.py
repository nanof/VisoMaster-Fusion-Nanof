"""Filesystem layout for last-stage pipeline upscaler weights."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from app.processors.models_data import models_dir
from app.processors.pytorch_extras.pipeline_upscaler.scale import (
    FLASHVSR_TYPES,
    HYPIR_TYPES,
    PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL,
    PIPELINE_UPSCALER_TYPE_SEEDVR2_7B,
    PMRF_TYPES,
    SEEDVR2_TYPES,
    pipeline_upscaler_type,
)

_FIX_MODELS = (
    "Fix: enable Pipeline Upscaler (weights download on first use), "
    "or place files under model_assets/hypir, model_assets/pmrf, "
    "model_assets/flashvsr, or model_assets/seedvr2."
)
_FIX_DEPS = (
    "Fix: Launcher → Check / Update Dependencies "
    "(packages are in requirements_cu13.txt)."
)

_FLASHVSR_PYTHON_DEPS: tuple[tuple[str, str], ...] = (
    ("torch", "PyTorch"),
    ("einops", "einops"),
    ("safetensors", "safetensors"),
    ("huggingface_hub", "huggingface-hub"),
)
_SEEDVR2_PYTHON_DEPS: tuple[tuple[str, str], ...] = (
    ("torch", "PyTorch"),
    ("einops", "einops"),
    ("omegaconf", "omegaconf"),
    ("safetensors", "safetensors"),
    ("huggingface_hub", "huggingface-hub"),
    ("rotary_embedding_torch", "rotary-embedding-torch"),
)
_HYPIR_PYTHON_DEPS: tuple[tuple[str, str], ...] = (
    ("torch", "PyTorch"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("huggingface_hub", "huggingface-hub"),
)
_PMRF_PYTHON_DEPS: tuple[tuple[str, str], ...] = (
    ("torch", "PyTorch"),
    ("einops", "einops"),
    ("huggingface_hub", "huggingface-hub"),
    ("pytorch_lightning", "pytorch-lightning"),
)


def flashvsr_root() -> Path:
    return Path(models_dir) / "flashvsr"


def seedvr2_root() -> Path:
    return Path(models_dir) / "seedvr2"


def hypir_root() -> Path:
    return Path(models_dir) / "hypir"


def hypir_sd2_dir() -> Path:
    return hypir_root() / "sd2"


def hypir_weight_path() -> Path:
    return hypir_root() / "HYPIR_sd2.pth"


def pmrf_root() -> Path:
    return Path(models_dir) / "pmrf"


def vendor_flashvsr_root() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "flashvsr"


def vendor_seedvr2_root() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "seedvr2"


def vendor_hypir_root() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "hypir"


def vendor_pmrf_root() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "pmrf"


def _is_usable_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 8


def flashvsr_dit_path() -> Path:
    return flashvsr_root() / "diffusion_pytorch_model_streaming_dmd.safetensors"


def flashvsr_lq_proj_path() -> Path:
    return flashvsr_root() / "LQ_proj_in.ckpt"


def flashvsr_tcdecoder_path() -> Path:
    return flashvsr_root() / "TCDecoder.ckpt"


def flashvsr_vae_path() -> Path:
    return flashvsr_root() / "Wan2.1_VAE.pth"


def flashvsr_prompt_path() -> Path:
    return flashvsr_root() / "posi_prompt.pth"


def seedvr2_dit_path(model_type: str) -> Path:
    if model_type == PIPELINE_UPSCALER_TYPE_SEEDVR2_7B:
        return seedvr2_root() / "seedvr2_ema_7b_fp16.safetensors"
    return seedvr2_root() / "seedvr2_ema_3b_fp8_e4m3fn.safetensors"


def seedvr2_vae_path() -> Path:
    return seedvr2_root() / "ema_vae_fp16.safetensors"


def seedvr2_dit_filename(model_type: str) -> str:
    return seedvr2_dit_path(model_type).name


def seedvr2_vae_filename() -> str:
    return seedvr2_vae_path().name


def _flashvsr_required(model_type: str) -> list[Path]:
    paths = [
        flashvsr_dit_path(),
        flashvsr_lq_proj_path(),
        flashvsr_prompt_path(),
    ]
    if model_type == PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL:
        paths.append(flashvsr_vae_path())
    else:
        paths.append(flashvsr_tcdecoder_path())
    return paths


def _hypir_required() -> list[Path]:
    base = hypir_sd2_dir()
    return [
        hypir_weight_path(),
        base / "model_index.json",
        base / "unet" / "config.json",
        base / "vae" / "config.json",
        base / "scheduler" / "scheduler_config.json",
        base / "tokenizer" / "vocab.json",
        base / "text_encoder" / "config.json",
    ]


def _pmrf_weight_candidates() -> list[Path]:
    root = pmrf_root()
    names = (
        "model.safetensors",
        "pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
    )
    found = [root / name for name in names]
    if root.is_dir():
        found.extend(sorted(root.glob("*.safetensors")))
    return found


def _pmrf_required() -> list[Path]:
    root = pmrf_root()
    weight = next((p for p in _pmrf_weight_candidates() if _is_usable_file(p)), None)
    required = [root / "config.json"]
    if weight is None:
        required.append(root / "model.safetensors")
    return required


def missing_assets(model_type: str) -> list[str]:
    missing: list[str] = []
    if model_type in FLASHVSR_TYPES:
        root = flashvsr_root()
        required = _flashvsr_required(model_type)
    elif model_type in SEEDVR2_TYPES:
        root = seedvr2_root()
        required = [seedvr2_dit_path(model_type), seedvr2_vae_path()]
    elif model_type in HYPIR_TYPES:
        root = hypir_root()
        required = _hypir_required()
    elif model_type in PMRF_TYPES:
        root = pmrf_root()
        required = _pmrf_required()
    else:
        return [f"unknown type: {model_type}"]
    for path in required:
        if not _is_usable_file(path):
            try:
                missing.append(str(path.relative_to(root)).replace("\\", "/"))
            except ValueError:
                missing.append(str(path))
    return missing


def assets_ready(model_type: str) -> bool:
    return not missing_assets(model_type)


def assets_error_message(model_type: str) -> str:
    missing = missing_assets(model_type)
    if not missing:
        return ""
    if model_type in FLASHVSR_TYPES:
        root = flashvsr_root()
    elif model_type in SEEDVR2_TYPES:
        root = seedvr2_root()
    elif model_type in HYPIR_TYPES:
        root = hypir_root()
    else:
        root = pmrf_root()
    return (
        f"Pipeline upscaler weights incomplete under {root}. "
        f"Missing: {', '.join(missing)}. {_FIX_MODELS}"
    )


def _deps_for_type(model_type: str) -> tuple[tuple[str, str], ...]:
    if model_type in FLASHVSR_TYPES:
        return _FLASHVSR_PYTHON_DEPS
    if model_type in SEEDVR2_TYPES:
        return _SEEDVR2_PYTHON_DEPS
    if model_type in HYPIR_TYPES:
        return _HYPIR_PYTHON_DEPS
    if model_type in PMRF_TYPES:
        return _PMRF_PYTHON_DEPS
    return ()


def missing_python_deps(model_type: str) -> list[str]:
    missing: list[str] = []
    for import_name, _label in _deps_for_type(model_type):
        if importlib.util.find_spec(import_name) is None:
            missing.append(import_name)
    return missing


def deps_error_message(model_type: str, missing: list[str] | None = None) -> str:
    miss = missing if missing is not None else missing_python_deps(model_type)
    if not miss:
        return ""
    return f"Pipeline upscaler Python deps missing: {', '.join(miss)}. {_FIX_DEPS}"


def format_load_error(exc: BaseException, model_type: str | None = None) -> str:
    kind = (
        pipeline_upscaler_type({"PipelineUpscalerTypeSelection": model_type})
        if model_type
        else "Pipeline upscaler"
    )
    return f"{kind} could not load: {exc}"
