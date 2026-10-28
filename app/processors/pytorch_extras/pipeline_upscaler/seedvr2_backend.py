"""SeedVR2 3B / 7B backend (ComfyUI-independent generation phases)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from app.processors.pytorch_extras.pipeline_upscaler.paths import (
    seedvr2_dit_filename,
    seedvr2_root,
    seedvr2_vae_filename,
    vendor_seedvr2_root,
)

_SEEDVR2_SRC = "app.processors.external.seedvr2.src"
_SEEDVR2_VAE_TILE = 512
_SEEDVR2_VAE_OVERLAP = 64


def _tuple_hw(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return default
    if isinstance(value, int) and value > 0:
        return int(value), int(value)
    return default


def _ensure_vae_tile_sizes(runner: Any, enabled: bool) -> None:
    """Tiled VAE encode/decode needs (H, W) tuples; None crashes in attn_video_vae."""
    size = (_SEEDVR2_VAE_TILE, _SEEDVR2_VAE_TILE)
    overlap = (_SEEDVR2_VAE_OVERLAP, _SEEDVR2_VAE_OVERLAP)
    runner.encode_tiled = bool(enabled)
    runner.decode_tiled = bool(enabled)
    runner.encode_tile_size = _tuple_hw(getattr(runner, "encode_tile_size", None), size)
    runner.decode_tile_size = _tuple_hw(getattr(runner, "decode_tile_size", None), size)
    runner.encode_tile_overlap = _tuple_hw(
        getattr(runner, "encode_tile_overlap", None), overlap
    )
    runner.decode_tile_overlap = _tuple_hw(
        getattr(runner, "decode_tile_overlap", None), overlap
    )


def _purge_module_tree(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


def _drop_polluted_top_level_src() -> None:
    """Previous builds inserted seedvr2 on sys.path so `import src` shadowed HF/six."""
    mod = sys.modules.get("src")
    if mod is None:
        return
    blob = str(getattr(mod, "__file__", "") or "")
    paths = getattr(mod, "__path__", None)
    if paths:
        blob += " " + " ".join(str(p) for p in paths)
    if "seedvr2" in blob.replace("\\", "/").lower():
        _purge_module_tree("src")


def _remove_vendor_from_sys_path() -> None:
    root = vendor_seedvr2_root().resolve()
    kept: list[str] = []
    for item in sys.path:
        try:
            if Path(item).resolve() == root:
                continue
        except (OSError, RuntimeError, ValueError):
            pass
        kept.append(item)
    sys.path[:] = kept


def _patch_six_meta_importers() -> None:
    """Python 3.12 / HuggingFace read ``finder._path``; six.moves still needs the importer."""
    classes: list[type] = []
    for importer in sys.meta_path:
        cls = type(importer)
        if cls.__name__ != "_SixMetaPathImporter":
            continue
        if cls not in classes:
            classes.append(cls)
        importer._path = []
    for name in ("six", "setuptools._vendor.six", "pkg_resources._vendor.six"):
        mod = sys.modules.get(name)
        cls = getattr(mod, "_SixMetaPathImporter", None) if mod is not None else None
        if cls is not None and cls not in classes:
            classes.append(cls)
    for cls in classes:
        cls._path = []
        if getattr(cls, "_visofusion_path_patched", False):
            continue
        orig_init = cls.__init__

        def _init(self, *args, __orig=orig_init, **kwargs):
            __orig(self, *args, **kwargs)
            self._path = []

        cls.__init__ = _init  # type: ignore[method-assign]
        cls._visofusion_path_patched = True


def _ensure_huggingface_imports() -> None:
    """Import HF with six.moves intact; retry after patching ``_path``."""
    _patch_six_meta_importers()
    import six.moves.urllib  # noqa: F401

    _patch_six_meta_importers()
    import diffusers  # noqa: F401
    import transformers.modeling_utils  # noqa: F401


def _prepare_import_env() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    _patch_six_meta_importers()
    _drop_polluted_top_level_src()
    _remove_vendor_from_sys_path()
    fa = sys.modules.get("flash_attn")
    if fa is not None and not callable(getattr(fa, "flash_attn_func", None)):
        sys.modules.pop("flash_attn", None)
    bnb = sys.modules.get("bitsandbytes")
    if bnb is not None and getattr(bnb, "__file__", None) is None:
        sys.modules.pop("bitsandbytes", None)
    root = vendor_seedvr2_root()
    os.environ["VISOFUSION_SEEDVR2_ROOT"] = str(root)
    os.environ["VISOFUSION_SEEDVR2_CACHE"] = str(seedvr2_root())


def _import_seedvr2(module: str):
    """Load vendored SeedVR2 as ``app.processors.external.seedvr2.src.*`` (not top-level ``src``)."""
    _patch_six_meta_importers()
    return __import__(f"{_SEEDVR2_SRC}.{module}", fromlist=["*"])


class SeedVR2Backend:
    def __init__(self) -> None:
        self.runner: Any = None
        self.ctx: Any = None
        self.debug: Any = None
        self.model_type: str | None = None
        self.device: str = "cuda"
        self._last_error: str | None = None
        self._dit_name: str | None = None
        self._vae_name: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.runner is not None

    def unload(self) -> None:
        try:
            if self.runner is not None:
                memory_manager = _import_seedvr2("optimization.memory_manager")
                memory_manager.complete_cleanup(runner=self.runner, debug=self.debug)
        except Exception as e:
            print(f"[WARN] SeedVR2 cleanup: {e}")
        self.runner = None
        self.ctx = None
        self.debug = None
        self.model_type = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_type: str, device: str = "cuda") -> bool:
        self._last_error = None
        try:
            _prepare_import_env()
            _ensure_huggingface_imports()
            seedvr2_root().mkdir(parents=True, exist_ok=True)
            generation_utils = _import_seedvr2("core.generation_utils")
            debug_mod = _import_seedvr2("utils.debug")
            downloads = _import_seedvr2("utils.downloads")
            _ensure_huggingface_imports()

            use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
            self.device = "cuda" if use_cuda else "cpu"
            dit_dev = torch.device(self.device)
            vae_dev = torch.device(self.device)
            dit_name = seedvr2_dit_filename(model_type)
            vae_name = seedvr2_vae_filename()
            debug = debug_mod.Debug(enabled=False)
            if not downloads.download_weight(
                dit_model=dit_name, vae_model=vae_name, model_dir=str(seedvr2_root()), debug=debug
            ):
                raise RuntimeError(
                    f"Failed to download SeedVR2 weights ({dit_name}, {vae_name})"
                )
            ctx = generation_utils.setup_generation_context(
                dit_device=dit_dev,
                vae_device=vae_dev,
                dit_offload_device=None,
                vae_offload_device=None,
                tensor_offload_device=None,
                debug=debug,
            )
            runner, cache_context = generation_utils.prepare_runner(
                dit_model=dit_name,
                vae_model=vae_name,
                model_dir=str(seedvr2_root()),
                debug=debug,
                ctx=ctx,
                dit_cache=True,
                vae_cache=True,
                dit_id="visofusion-seedvr2-dit",
                vae_id="visofusion-seedvr2-vae",
                attention_mode="sdpa",
                encode_tiled=True,
                encode_tile_size=(_SEEDVR2_VAE_TILE, _SEEDVR2_VAE_TILE),
                encode_tile_overlap=(_SEEDVR2_VAE_OVERLAP, _SEEDVR2_VAE_OVERLAP),
                decode_tiled=True,
                decode_tile_size=(_SEEDVR2_VAE_TILE, _SEEDVR2_VAE_TILE),
                decode_tile_overlap=(_SEEDVR2_VAE_OVERLAP, _SEEDVR2_VAE_OVERLAP),
            )
            ctx["cache_context"] = cache_context
            ctx["text_embeds"] = generation_utils.load_text_embeddings(
                generation_utils.script_directory,
                ctx["dit_device"],
                ctx["compute_dtype"],
                debug,
            )
            self.runner = runner
            self.ctx = ctx
            self.debug = debug
            self.model_type = model_type
            self._dit_name = dit_name
            self._vae_name = vae_name
            print(f"[INFO] SeedVR2 loaded ({model_type}) on {self.device}")
            return True
        except Exception as e:
            self._last_error = str(e)
            print(f"[WARN] SeedVR2 load failed: {e}")
            self.unload()
            return False

    @torch.no_grad()
    def upscale_bgr(
        self,
        frame_bgr: np.ndarray,
        *,
        scale: int = 4,
        color_fix: bool = True,
        tiled_vae: bool = True,
        seed: int = 42,
    ) -> np.ndarray:
        if self.runner is None or self.ctx is None:
            return frame_bgr
        _ensure_vae_tile_sizes(self.runner, tiled_vae)
        phases = _import_seedvr2("core.generation_phases")
        generation_utils = _import_seedvr2("core.generation_utils")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
        image = image.unsqueeze(0)  # N H W C
        h0, w0 = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        shortest = min(h0, w0)
        resolution = max(16, int(shortest * scale))
        color_correction = "lab" if color_fix else "none"
        ctx = dict(self.ctx)
        image, _gen_info = generation_utils.compute_generation_info(
            ctx=ctx,
            images=image,
            resolution=resolution,
            max_resolution=0,
            batch_size=1,
            uniform_batch_size=False,
            seed=int(seed),
            prepend_frames=0,
            temporal_overlap=0,
            debug=self.debug,
        )
        ctx = phases.encode_all_batches(
            self.runner,
            ctx=ctx,
            images=image,
            debug=self.debug,
            batch_size=1,
            uniform_batch_size=False,
            seed=int(seed),
            progress_callback=None,
            temporal_overlap=0,
            resolution=resolution,
            max_resolution=0,
            input_noise_scale=0.0,
            color_correction=color_correction,
        )
        ctx = phases.upscale_all_batches(
            self.runner,
            ctx=ctx,
            debug=self.debug,
            progress_callback=None,
            seed=int(seed),
            latent_noise_scale=0.0,
            cache_model=True,
        )
        ctx = phases.decode_all_batches(
            self.runner,
            ctx=ctx,
            debug=self.debug,
            progress_callback=None,
            cache_model=True,
        )
        ctx = phases.postprocess_all_batches(
            ctx=ctx,
            debug=self.debug,
            progress_callback=None,
            color_correction=color_correction,
            prepend_frames=0,
            temporal_overlap=0,
            batch_size=1,
        )
        sample = ctx.get("final_video")
        if not torch.is_tensor(sample):
            return frame_bgr
        if sample.ndim == 4:
            frame = sample[0]
        else:
            frame = sample
        if frame.shape[-1] not in (3, 4):
            frame = frame.permute(1, 2, 0)
        rgb_out = (frame[..., :3].float().clamp(0, 1) * 255.0).cpu().numpy().astype(np.uint8)
        target_h = h0 * scale
        target_w = w0 * scale
        if rgb_out.shape[0] != target_h or rgb_out.shape[1] != target_w:
            rgb_out = cv2.resize(rgb_out, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        bgr = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr)
