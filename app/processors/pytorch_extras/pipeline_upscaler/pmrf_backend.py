"""PMRF blind restoration, tiled, then resized to the selected scale (export)."""

from __future__ import annotations

import gc
from typing import Any

import cv2
import numpy as np
import torch

from app.processors.pytorch_extras.pipeline_upscaler.paths import pmrf_root
from app.processors.pytorch_extras.pipeline_upscaler.tiles import (
    overlap_weights,
    tile_boxes,
)

_PMRF_TILE = 512
_PMRF_OVERLAP = 64
_PMRF_FLOW_STEPS = 25
_PMRF_ALIGN = 64


def _pad_to_tile(rgb: np.ndarray, size: int = _PMRF_TILE) -> tuple[np.ndarray, int, int]:
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    ph = max(0, size - h)
    pw = max(0, size - w)
    if ph or pw:
        rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return rgb, h, w


class PmrfBackend:
    def __init__(self) -> None:
        self.model: Any = None
        self.device: str = "cuda"
        self._last_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_type: str, device: str = "cuda") -> bool:
        self._last_error = None
        try:
            from app.processors.pytorch_extras.pipeline_upscaler.seedvr2_backend import (
                _ensure_huggingface_imports,
            )

            _ensure_huggingface_imports()
            from app.processors.external.pmrf.lightning_models.mmse_rectified_flow import (
                MMSERectifiedFlow,
            )

            use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
            self.device = "cuda" if use_cuda else "cpu"
            model = MMSERectifiedFlow.from_pretrained(str(pmrf_root()))
            model.freeze()
            model.eval()
            model.to(self.device)
            self.model = model
            print(f"[INFO] PMRF loaded ({model_type}) on {self.device}")
            return True
        except Exception as e:
            self._last_error = str(e)
            print(f"[WARN] PMRF load failed: {e}")
            self.unload()
            return False

    def _restore_nchw(self, rgb_u8: np.ndarray) -> np.ndarray:
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb_u8))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        ).to(self.device)
        xhat, _, _ = self.model.generate_reconstructions(
            tensor, tensor, None, _PMRF_FLOW_STEPS, torch.device("cpu")
        )
        arr = (
            (xhat[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )
        del tensor, xhat
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return arr

    def _restore_tile(self, rgb: np.ndarray) -> np.ndarray:
        h0, w0 = int(rgb.shape[0]), int(rgb.shape[1])
        padded, _, _ = _pad_to_tile(rgb, _PMRF_TILE)
        out = self._restore_nchw(padded)
        return out[:h0, :w0]

    def _restore_full(self, rgb: np.ndarray) -> np.ndarray:
        h0, w0 = int(rgb.shape[0]), int(rgb.shape[1])
        if h0 <= _PMRF_TILE and w0 <= _PMRF_TILE:
            return self._restore_tile(rgb)
        overlap = min(_PMRF_OVERLAP, max(h0 // 4, 1), max(w0 // 4, 1))
        boxes = tile_boxes(h0, w0, _PMRF_TILE, _PMRF_TILE, overlap)
        acc = np.zeros((h0, w0, 3), dtype=np.float32)
        wsum = np.zeros((h0, w0, 1), dtype=np.float32)
        for y0, y1, x0, x1 in boxes:
            crop = rgb[y0:y1, x0:x1]
            restored = self._restore_tile(crop)
            if restored.shape[0] != (y1 - y0) or restored.shape[1] != (x1 - x0):
                restored = cv2.resize(
                    restored, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC
                )
            weight = overlap_weights(y1 - y0, x1 - x0, y0, x0, h0, w0, overlap)[
                ..., None
            ]
            acc[y0:y1, x0:x1] += restored.astype(np.float32) * weight
            wsum[y0:y1, x0:x1] += weight
        merged = acc / np.maximum(wsum, 1e-6)
        return np.clip(merged, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def upscale_bgr(
        self,
        frame_bgr: np.ndarray,
        *,
        scale: int = 2,
        color_fix: bool = True,
        seed: int = 0,
    ) -> np.ndarray:
        if self.model is None:
            return frame_bgr
        del color_fix, seed
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        restored = self._restore_full(rgb)
        bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        if int(scale) <= 1:
            return np.ascontiguousarray(bgr)
        h, w = int(bgr.shape[0]) * int(scale), int(bgr.shape[1]) * int(scale)
        return np.ascontiguousarray(
            cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        )

    def __del__(self) -> None:
        try:
            gc.collect()
        except Exception:
            pass
