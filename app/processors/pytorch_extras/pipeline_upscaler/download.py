"""On-demand weight download for the last-stage pipeline upscaler."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

from app.processors.pytorch_extras.pipeline_upscaler.paths import (
    assets_ready,
    flashvsr_dit_path,
    flashvsr_lq_proj_path,
    flashvsr_prompt_path,
    flashvsr_root,
    flashvsr_tcdecoder_path,
    flashvsr_vae_path,
    hypir_root,
    hypir_sd2_dir,
    hypir_weight_path,
    pmrf_root,
    seedvr2_root,
)
from app.processors.pytorch_extras.pipeline_upscaler.scale import (
    FLASHVSR_TYPES,
    HYPIR_TYPES,
    PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL,
    PIPELINE_UPSCALER_TYPE_SEEDVR2_7B,
    PMRF_TYPES,
    SEEDVR2_TYPES,
)

_HF_FLASHVSR = "JunhaoZhuang/FlashVSR-v1.1"
_HF_SEEDVR2 = "numz/SeedVR2_comfyUI"
_HF_HYPIR = "lxq007/HYPIR"
# stabilityai/stable-diffusion-2-1-base is gated/removed; use public 512-base mirrors.
_HF_SD2_REPOS: tuple[str, ...] = (
    "Manojb/stable-diffusion-2-1-base",
    "omeregev/sd-2-1-base-mirror",
    "SagiPolaczek/stable-diffusion-2-1-base",
)
_HF_PMRF = "ohayonguy/PMRF_blind_face_image_restoration"
_PROMPT_URL = (
    "https://github.com/OpenImagingLab/FlashVSR/raw/main/"
    "examples/WanVSR/prompt_tensor/posi_prompt.pth"
)
_SD2_ALLOW = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/*",
    "scheduler/scheduler_config.json",
]
_SD2_IGNORE = ["*.bin", "*.ckpt", "*fp16*"]


def _hf_download(repo_id: str, filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 8:
        return
    import inspect

    from huggingface_hub import hf_hub_download

    print(f"[INFO] Downloading {filename} from {repo_id} ...")
    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": str(dest.parent),
    }
    # huggingface-hub <0.24 still accepts this; newer versions dropped it.
    if "local_dir_use_symlinks" in inspect.signature(hf_hub_download).parameters:
        kwargs["local_dir_use_symlinks"] = False
    cached = hf_hub_download(**kwargs)
    cached_path = Path(cached)
    if cached_path.resolve() != dest.resolve() and cached_path.is_file():
        if dest.exists():
            dest.unlink()
        os.replace(cached_path, dest)


def _hf_snapshot(
    repo_id: str,
    dest: Path,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    import inspect

    from huggingface_hub import snapshot_download

    print(f"[INFO] Downloading {repo_id} -> {dest} ...")
    kwargs: dict = {
        "repo_id": repo_id,
        "local_dir": str(dest),
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns
    params = inspect.signature(snapshot_download).parameters
    if "local_dir_use_symlinks" in params:
        kwargs["local_dir_use_symlinks"] = False
    snapshot_download(**kwargs)


def _hf_snapshot_first_ok(
    repo_ids: tuple[str, ...],
    dest: Path,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    last: BaseException | None = None
    for repo_id in repo_ids:
        try:
            _hf_snapshot(
                repo_id,
                dest,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            return
        except Exception as e:
            last = e
            print(f"[WARN] Hugging Face download failed for {repo_id}: {e}")
    tried = ", ".join(repo_ids)
    raise RuntimeError(
        f"Could not download weights. Tried: {tried}. Last error: {last}"
    ) from last


def _url_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 8:
        return
    print(f"[INFO] Downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(url, tmp)
    os.replace(tmp, dest)


def ensure_weights(model_type: str) -> None:
    """Download missing weights for ``model_type``. Raises on failure."""
    if model_type in FLASHVSR_TYPES:
        flashvsr_root().mkdir(parents=True, exist_ok=True)
        _hf_download(
            _HF_FLASHVSR,
            "diffusion_pytorch_model_streaming_dmd.safetensors",
            flashvsr_dit_path(),
        )
        _hf_download(_HF_FLASHVSR, "LQ_proj_in.ckpt", flashvsr_lq_proj_path())
        _url_download(_PROMPT_URL, flashvsr_prompt_path())
        if model_type == PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL:
            _hf_download(_HF_FLASHVSR, "Wan2.1_VAE.pth", flashvsr_vae_path())
        else:
            _hf_download(_HF_FLASHVSR, "TCDecoder.ckpt", flashvsr_tcdecoder_path())
        return
    if model_type in SEEDVR2_TYPES:
        seedvr2_root().mkdir(parents=True, exist_ok=True)
        dit_name = (
            "seedvr2_ema_7b_fp16.safetensors"
            if model_type == PIPELINE_UPSCALER_TYPE_SEEDVR2_7B
            else "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
        )
        _hf_download(_HF_SEEDVR2, dit_name, seedvr2_root() / dit_name)
        _hf_download(
            _HF_SEEDVR2, "ema_vae_fp16.safetensors", seedvr2_root() / "ema_vae_fp16.safetensors"
        )
        return
    if model_type in HYPIR_TYPES:
        hypir_root().mkdir(parents=True, exist_ok=True)
        _hf_download(_HF_HYPIR, "HYPIR_sd2.pth", hypir_weight_path())
        if not assets_ready(model_type):
            _hf_snapshot_first_ok(
                _HF_SD2_REPOS,
                hypir_sd2_dir(),
                allow_patterns=_SD2_ALLOW,
                ignore_patterns=_SD2_IGNORE,
            )
        return
    if model_type in PMRF_TYPES:
        _hf_snapshot(_HF_PMRF, pmrf_root())
        return
    raise ValueError(f"Unknown pipeline upscaler type: {model_type}")


def ensure_weights_for_types(types: Iterable[str]) -> None:
    for model_type in types:
        ensure_weights(model_type)
