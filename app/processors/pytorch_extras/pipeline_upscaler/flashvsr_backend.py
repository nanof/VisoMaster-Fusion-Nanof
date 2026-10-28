"""FlashVSR v1.1 backend (tiny / full)."""

from __future__ import annotations

import collections
import gc
import sys
from typing import Any

import cv2
import numpy as np
import torch

from app.processors.pytorch_extras.pipeline_upscaler.paths import (
    flashvsr_dit_path,
    flashvsr_lq_proj_path,
    flashvsr_prompt_path,
    flashvsr_tcdecoder_path,
    flashvsr_vae_path,
    vendor_flashvsr_root,
)
from app.processors.pytorch_extras.pipeline_upscaler.scale import (
    PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL,
)

_WORKING_CANDIDATES = (
    (640, 768),
    (768, 640),
    (640, 640),
    (512, 640),
    (640, 512),
    (512, 512),
    (384, 512),
    (512, 384),
    (384, 384),
)

_FLASHVSR_MIN_FRAMES = 25  # TinyLong: process_total_num = (N-1)//8 - 2 needs N>=25
_FLASHVSR_HR_MULTIPLE = 128  # DiT + PixelShuffle 16 (LQ HxW must equal pipe H/W)
_OOM_MIN_HR_PIXELS = 384 * 384
_MIN_FREE_GB_TO_INFER = 1.0


def _drop_unusable_flash_attn() -> None:
    """SeedVR2 used to stub flash_attn with flash_attn_func=None; drop that fake module."""
    fa = sys.modules.get("flash_attn")
    if fa is not None and not callable(getattr(fa, "flash_attn_func", None)):
        sys.modules.pop("flash_attn", None)
    dit_mod = sys.modules.get("diffsynth.models.wan_video_dit")
    if dit_mod is None:
        return
    fa = sys.modules.get("flash_attn")
    dit_mod.flash_attn = fa
    dit_mod.FLASH_ATTN_2_AVAILABLE = bool(
        fa is not None and callable(getattr(fa, "flash_attn_func", None))
    )


def _ensure_vendor_on_path() -> Path:
    _drop_unusable_flash_attn()
    root = vendor_flashvsr_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def _pad_to_multiple(h: int, w: int, multiple: int = 128) -> tuple[int, int]:
    th = ((int(h) + multiple - 1) // multiple) * multiple
    tw = ((int(w) + multiple - 1) // multiple) * multiple
    return max(th, multiple), max(tw, multiple)


def flashvsr_aligned_hw(tile_h: int, tile_w: int) -> tuple[int, int]:
    """LQ spatial size and pipe height/width must be this same aligned pair."""
    return _pad_to_multiple(tile_h, tile_w, _FLASHVSR_HR_MULTIPLE)


def _bgr_to_lq_nchw(frame_bgr: np.ndarray, t_h: int, t_w: int, dtype, device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]
    if (h0, w0) != (t_h, t_w):
        rgb = cv2.resize(rgb, (t_w, t_h), interpolation=cv2.INTER_CUBIC)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


class FlashVSRBackend:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_type: str | None = None
        self.device: str = "cuda"
        self.dtype = torch.bfloat16
        self._last_error: str | None = None
        self._clip: collections.deque[torch.Tensor] = collections.deque(
            maxlen=_FLASHVSR_MIN_FRAMES
        )
        self._clip_hw: tuple[int, int] | None = None

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    def unload(self) -> None:
        self.pipe = None
        self.model_type = None
        self._clip.clear()
        self._clip_hw = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_type: str, device: str = "cuda") -> bool:
        self._last_error = None
        try:
            _ensure_vendor_on_path()
            from diffsynth import FlashVSRFullPipeline, FlashVSRTinyLongPipeline, ModelManager
            from wanvsr_utils import Causal_LQ4x_Proj, build_tcdecoder
            _drop_unusable_flash_attn()

            use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
            self.device = "cuda" if use_cuda else "cpu"
            self.dtype = (
                torch.bfloat16
                if use_cuda and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            use_full = model_type == PIPELINE_UPSCALER_TYPE_FLASHVSR_FULL
            mm = ModelManager(torch_dtype=self.dtype, device="cpu")
            files = [str(flashvsr_dit_path())]
            if use_full:
                files.append(str(flashvsr_vae_path()))
            mm.load_models(files)
            pipe_cls = FlashVSRFullPipeline if use_full else FlashVSRTinyLongPipeline
            pipe = pipe_cls.from_model_manager(mm, device=self.device)
            dit = pipe.denoising_model()
            dit.LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
                self.device, dtype=self.dtype
            )
            lq_path = flashvsr_lq_proj_path()
            dit.LQ_proj_in.load_state_dict(
                torch.load(str(lq_path), map_location="cpu", weights_only=False),
                strict=True,
            )
            dit.LQ_proj_in.to(self.device)
            if not use_full:
                pipe.TCDecoder = build_tcdecoder(
                    new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768
                )
                pipe.TCDecoder.load_state_dict(
                    torch.load(
                        str(flashvsr_tcdecoder_path()),
                        map_location="cpu",
                        weights_only=False,
                    ),
                    strict=False,
                )
            pipe.to(self.device)
            try:
                pipe.enable_vram_management(num_persistent_param_in_dit=None)
            except Exception as e:
                print(f"[WARN] FlashVSR VRAM management skipped: {e}")
            ctx = torch.load(str(flashvsr_prompt_path()), map_location=self.device, weights_only=False)
            pipe.init_cross_kv(context_tensor=ctx)
            pipe.load_models_to_device(["dit", "vae"] if use_full else ["dit"])
            self.pipe = pipe
            self.model_type = model_type
            print(f"[INFO] FlashVSR loaded ({model_type}) on {self.device}")
            return True
        except Exception as e:
            self._last_error = str(e)
            print(f"[WARN] FlashVSR load failed: {e}")
            self.unload()
            return False

    def _cuda_free_gb(self) -> float:
        """Driver-visible free VRAM. Do not empty_cache first — that over-sizes tiles."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return 0.0
        try:
            free, _total = torch.cuda.mem_get_info()
            return float(free) / float(1024**3)
        except Exception:
            return 0.0

    def _max_hr_pixels(self) -> int:
        """Same-size LQ/HR tiles. Full 4K DiT OOMs on 16 GB; 640×768 is the 16 GB default."""
        free_gb = self._cuda_free_gb()
        if self.device != "cuda":
            return 640 * 768
        if free_gb < 1.2:
            return 384 * 512
        if free_gb < 2.5:
            return 512 * 640
        if free_gb < 6.0:
            return 640 * 768
        return 768 * 1024

    def _flush_cuda(self) -> None:
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        text = str(exc).lower()
        return "out of memory" in text or "cuda oom" in text

    def _working_hw(self, h0: int, w0: int, max_pixels: int) -> tuple[int, int]:
        """One LQ/HR canvas. Shrink to a single 128-aligned size that fits VRAM."""
        aligned = flashvsr_aligned_hw(h0, w0)
        if aligned[0] * aligned[1] <= max_pixels:
            return aligned
        aspect = w0 / max(h0, 1)
        best = (384, 384)
        best_score: float | None = None
        for th, tw in _WORKING_CANDIDATES:
            if th * tw > max_pixels:
                continue
            score = -abs((tw / max(th, 1)) - aspect)
            if best_score is None or score > best_score:
                best_score = score
                best = (th, tw)
        return best

    def _infer_streaming(
        self,
        work_bgr: np.ndarray,
        *,
        color_fix: bool,
        seed: int,
    ) -> np.ndarray:
        """One DiT pass at working LQ size; LQ HxW must equal pipe height/width."""
        t_h, t_w = int(work_bgr.shape[0]), int(work_bgr.shape[1])
        if self._clip_hw != (t_h, t_w):
            self._clip.clear()
            self._clip_hw = (t_h, t_w)
        lq = _bgr_to_lq_nchw(work_bgr, t_h, t_w, self.dtype, self.device)
        self._clip.append(lq)
        while len(self._clip) < _FLASHVSR_MIN_FRAMES:
            self._clip.appendleft(self._clip[0].clone())
        frames = torch.stack(list(self._clip), dim=1)
        lq_vid = frames.unsqueeze(0)
        topk = 2.0 * 640 * 768 / float(max(t_h * t_w, 1))
        video = None
        rgb_np = None
        try:
            video = self.pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=int(seed),
                LQ_video=lq_vid,
                num_frames=_FLASHVSR_MIN_FRAMES,
                height=t_h,
                width=t_w,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=topk,
                kv_ratio=3.0,
                local_range=11,
                color_fix=bool(color_fix),
            )
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            if not isinstance(video, torch.Tensor):
                raise RuntimeError("FlashVSR returned a non-tensor")
            out = video[0] if video.ndim == 5 else video
            frame = out[:, -1] if out.ndim == 4 else out
            if frame.ndim == 3 and frame.shape[0] in (3, 4):
                pass
            elif frame.ndim == 3 and frame.shape[-1] in (3, 4):
                frame = frame.permute(2, 0, 1)
            rgb = ((frame[:3].float() + 1.0) * 127.5).clamp(0, 255)
            rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        finally:
            del frames, lq_vid, video
            self._flush_cuda()
        if rgb_np is None:
            raise RuntimeError("FlashVSR produced no RGB frame")
        if rgb_np.shape[0] != t_h or rgb_np.shape[1] != t_w:
            rgb_np = cv2.resize(rgb_np, (t_w, t_h), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    @torch.no_grad()
    def upscale_bgr(
        self,
        frame_bgr: np.ndarray,
        *,
        scale: int = 4,
        color_fix: bool = True,
        seed: int = 0,
    ) -> np.ndarray:
        if self.pipe is None:
            return frame_bgr
        h0, w0 = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        s_h = max(1, int(round(h0 * max(1, int(scale)))))
        s_w = max(1, int(round(w0 * max(1, int(scale)))))
        bicubic = cv2.resize(frame_bgr, (s_w, s_h), interpolation=cv2.INTER_CUBIC)
        if self.device == "cuda" and self._cuda_free_gb() < _MIN_FREE_GB_TO_INFER:
            print(
                "[WARN] FlashVSR: less than 1 GB free VRAM (TensorRT + weights). "
                "Returning bicubic so playback can start."
            )
            return np.ascontiguousarray(bicubic)
        budgets: list[int] = []
        start = self._max_hr_pixels()
        for value in (start, 640 * 768, 512 * 640, _OOM_MIN_HR_PIXELS):
            if value <= start and value not in budgets:
                budgets.append(value)
        last_oom: BaseException | None = None
        for max_pixels in budgets:
            try:
                work_h, work_w = self._working_hw(h0, w0, max_pixels)
                work = frame_bgr
                if (h0, w0) != (work_h, work_w):
                    work = cv2.resize(
                        frame_bgr, (work_w, work_h), interpolation=cv2.INTER_CUBIC
                    )
                hr = self._infer_streaming(work, color_fix=color_fix, seed=seed)
                if (hr.shape[0], hr.shape[1]) != (s_h, s_w):
                    hr = cv2.resize(hr, (s_w, s_h), interpolation=cv2.INTER_CUBIC)
                return np.ascontiguousarray(hr)
            except Exception as e:
                if not self._is_cuda_oom(e):
                    raise
                last_oom = e
                self._clip.clear()
                self._clip_hw = None
                self._flush_cuda()
                print(
                    f"[WARN] FlashVSR: CUDA OOM at working budget {max_pixels} px; "
                    "trying a smaller canvas (or bicubic)."
                )
        print(
            "[WARN] FlashVSR: CUDA out of memory; returning bicubic upscale "
            f"so playback can continue ({last_oom})."
        )
        return np.ascontiguousarray(bicubic)
