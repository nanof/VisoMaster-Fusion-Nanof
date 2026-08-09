"""
Minimal MuseTalk VAE / UNet wrappers (adapted from TMElyralab/MuseTalk, MIT).

Lazy-imports torch/diffusers so the rest of the app can import this package
without installing pytorch-extras.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _build_positional_encoding(d_model: int = 384, max_len: int = 5000) -> Any:
    import torch
    import torch.nn as nn

    class _PE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: Any) -> Any:
            # Match x's dtype: a float32 buffer would promote a half input and
            # break the fp16 UNet's linear layers.
            pe = self.pe[:, : x.size(1), :]
            return x + pe.to(device=x.device, dtype=x.dtype)

    return _PE()


class MuseTalkVAE:
    def __init__(
        self,
        model_path: str | Path,
        resized_img: int = 256,
        use_float16: bool = True,
        device: Any = None,
    ) -> None:
        import torch
        from diffusers import AutoencoderKL
        from torchvision import transforms

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Say up front which weight format is on disk. Left to guess, diffusers
        # probes for safetensors first and prints a fetch error before falling
        # back to the .bin we ship.
        has_safetensors = (
            Path(model_path) / "diffusion_pytorch_model.safetensors"
        ).is_file()
        self.vae = AutoencoderKL.from_pretrained(
            str(model_path), use_safetensors=has_safetensors
        )
        self.vae.to(self.device)
        self._use_float16 = bool(use_float16) and self.device.type == "cuda"
        if self._use_float16:
            self.vae = self.vae.half()
        self.scaling_factor = self.vae.config.scaling_factor
        self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self._resized_img = int(resized_img)
        self._rebuild_mask()

    def _rebuild_mask(self) -> None:
        """Keep the top half of the crop, zero the half the model must repaint.

        The split is exactly half and stays that way. Upstream's ``bbox_shift``
        moves the crop instead (see ``framing.py``): the mask edge is a hard step
        in the model's input, so it has to stay where the blend fades in, or the
        step becomes a visible line across the face.
        """
        import torch

        side = self._resized_img
        mask = torch.zeros((side, side), device=self.device)
        mask[: side // 2, :] = 1.0
        self._mask_tensor = mask

    def preprocess_batch(
        self, imgs_bgr: list[np.ndarray], half_mask: bool = False
    ) -> Any:
        import torch

        side = self._resized_img
        prepared = []
        for img_bgr in imgs_bgr:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img.shape[0] != side or img.shape[1] != side:
                img = cv2.resize(img, (side, side), interpolation=cv2.INTER_LANCZOS4)
            prepared.append(img)
        x = np.asarray(prepared, dtype=np.float32) / 255.0
        x = np.transpose(x, (0, 3, 1, 2))  # NCHW
        x = torch.from_numpy(x).to(self.vae.device)
        if half_mask:
            x = x * (self._mask_tensor > 0.5)
        return self.transform(x)

    def get_latents_for_unet_batch(self, imgs_bgr: list[np.ndarray]) -> Any:
        """Masked+reference latents for N crops, using one VAE encode pass."""
        import torch

        masked = self.preprocess_batch(imgs_bgr, half_mask=True)
        reference = self.preprocess_batch(imgs_bgr, half_mask=False)
        encoded = self.encode_latents(torch.cat([masked, reference], dim=0))
        n = len(imgs_bgr)
        return torch.cat([encoded[:n], encoded[n:]], dim=1)

    def preprocess_img(self, img_bgr: np.ndarray, half_mask: bool = False) -> Any:
        import torch

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            (self._resized_img, self._resized_img),
            interpolation=cv2.INTER_LANCZOS4,
        )
        x = np.asarray([img], dtype=np.float32) / 255.0
        x = np.transpose(x, (0, 3, 1, 2))  # NCHW
        x = torch.from_numpy(x).squeeze(0).to(self.vae.device)
        if half_mask:
            x = x * (self._mask_tensor > 0.5)
        x = self.transform(x).unsqueeze(0)
        return x

    def encode_latents(self, image: Any) -> Any:
        import torch

        with torch.no_grad():
            dist = self.vae.encode(image.to(self.vae.dtype)).latent_dist
            return self.scaling_factor * dist.sample()

    def decode_latents(self, latents: Any) -> np.ndarray:
        import torch

        with torch.no_grad():
            latents = (1 / self.scaling_factor) * latents
            image = self.vae.decode(latents.to(self.vae.dtype)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            return image[..., ::-1]  # RGB → BGR

    def get_latents_for_unet(self, img_bgr: np.ndarray) -> Any:
        import torch

        masked = self.encode_latents(self.preprocess_img(img_bgr, half_mask=True))
        ref = self.encode_latents(self.preprocess_img(img_bgr, half_mask=False))
        return torch.cat([masked, ref], dim=1)


class MuseTalkUNet:
    def __init__(
        self,
        unet_config: str | Path,
        model_path: str | Path,
        use_float16: bool = True,
        device: Any = None,
    ) -> None:
        import torch
        from diffusers import UNet2DConditionModel

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        with open(unet_config, encoding="utf-8") as f:
            cfg = json.load(f)
        self.model = UNet2DConditionModel(**cfg)
        weights = torch.load(
            str(model_path), map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(weights)
        self._use_float16 = bool(use_float16) and self.device.type == "cuda"
        if self._use_float16:
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()


# Back-compat name used by engine.py
PositionalEncoding = _build_positional_encoding
