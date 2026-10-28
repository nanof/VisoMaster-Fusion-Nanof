"""HYPIR SD2.1 one-step enhancer (play / preview)."""

from __future__ import annotations

import sys
from typing import Any

import cv2
import numpy as np
import torch

from app.processors.pytorch_extras.pipeline_upscaler.paths import (
    hypir_sd2_dir,
    hypir_weight_path,
    vendor_hypir_root,
)

_HYPIR_LORA_MODULES = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "conv_out",
    "proj_in",
    "proj_out",
    "ff.net.2",
    "ff.net.0.proj",
]
_PATCH_SIZE = 512
_STRIDE = 256


def _ensure_vendor_on_path() -> None:
    root = str(vendor_hypir_root())
    if root not in sys.path:
        sys.path.insert(0, root)


class HypirBackend:
    def __init__(self) -> None:
        self.enhancer: Any = None
        self.device: str = "cuda"
        self._last_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.enhancer is not None

    def unload(self) -> None:
        self.enhancer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_type: str, device: str = "cuda") -> bool:
        self._last_error = None
        try:
            from app.processors.pytorch_extras.pipeline_upscaler.seedvr2_backend import (
                _ensure_huggingface_imports,
            )

            _ensure_huggingface_imports()
            _ensure_vendor_on_path()
            from HYPIR.enhancer.sd2 import SD2Enhancer

            use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
            self.device = "cuda" if use_cuda else "cpu"
            enhancer = SD2Enhancer(
                base_model_path=str(hypir_sd2_dir()),
                weight_path=str(hypir_weight_path()),
                lora_modules=_HYPIR_LORA_MODULES,
                lora_rank=256,
                model_t=200,
                coeff_t=200,
                device=self.device,
            )
            if not (use_cuda and torch.cuda.is_bf16_supported()):
                enhancer.weight_dtype = torch.float16
            enhancer.init_models()
            self.enhancer = enhancer
            print(f"[INFO] HYPIR loaded ({model_type}) on {self.device}")
            return True
        except Exception as e:
            self._last_error = str(e)
            print(f"[WARN] HYPIR load failed: {e}")
            self.unload()
            return False

    @torch.no_grad()
    def upscale_bgr(
        self,
        frame_bgr: np.ndarray,
        *,
        scale: int = 1,
        color_fix: bool = True,
        seed: int = 0,
    ) -> np.ndarray:
        if self.enhancer is None:
            return frame_bgr
        del color_fix, seed
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        out = self.enhancer.enhance(
            lq=tensor,
            prompt="",
            upscale=max(1, int(scale)),
            patch_size=_PATCH_SIZE,
            stride=_STRIDE,
            return_type="pt",
        )
        arr = (
            (out[0].permute(1, 2, 0).numpy() * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
