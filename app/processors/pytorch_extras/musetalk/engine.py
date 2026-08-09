"""
MuseTalk engine: load/unload, audio prep, per-frame realtime apply.

Isolated from the ONNX swap hot path. Failures never raise into callers of
``apply_frame_bgr`` (returns the input frame unchanged).

The frame pipeline calls ``apply_frame_bgr`` from several worker threads at once,
so GPU work is funnelled to one batching thread that coalesces the concurrent
requests into a single VAE/UNet pass. Batch size is capped by
``VISOFUSION_MUSETALK_BATCH`` (default 8).
"""

from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from app.processors.pytorch_extras.musetalk.audio import (
    MuseTalkAudioProcessor,
    extract_wav_from_media,
)
from app.processors.pytorch_extras.musetalk.blending import (
    blend_face_region,
    expand_bbox,
)
from app.processors.pytorch_extras.musetalk.framing import landmark_crop_bbox
from app.processors.pytorch_extras.musetalk.parsing import (
    ParsedMasks,
    crop_mouth_mask,
    mouth_only_blend_mask,
    parsed_masks,
)
from app.processors.pytorch_extras.musetalk.paths import (
    musetalk_assets_ready,
    musetalk_root,
    prepare_transformers_env,
    unet_config_path,
    unet_weights_path,
    vae_dir,
    whisper_dir,
)


def _batch_limit() -> int:
    try:
        return max(1, int(os.environ.get("VISOFUSION_MUSETALK_BATCH", "8")))
    except (TypeError, ValueError):
        return 8


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_flag(name: str, default: bool) -> bool:
    """Read a 0/1 env flag; missing/empty keeps ``default``."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def musetalk_compile_enabled() -> bool:
    """True when VISOFUSION_MUSETALK_COMPILE asks for torch.compile (default on)."""
    return _env_flag("VISOFUSION_MUSETALK_COMPILE", True)


def musetalk_debug_enabled() -> bool:
    """True when VISOFUSION_MUSETALK_DEBUG asks for lip-sync tracing."""
    return _env_truthy("VISOFUSION_MUSETALK_DEBUG")


# Whisper-tiny encoder: embed + 4 layers → 5; window = 2*(2+2+1) = 10 → 50 tokens.
_MUSETALK_AUDIO_TOKENS = 50
_MUSETALK_AUDIO_DIM = 384
_MUSETALK_LATENT_HW = 32


class _EffectProbe:
    """Measures what lip-sync actually changes on screen.

    Answers two questions that eyeballing a video cannot: whether the blend
    changes any pixels at all, and whether the mouth it paints moves over time
    (audio-driven) instead of being a constant overlay.
    """

    REPORT_EVERY = 25

    def __init__(self, dump_dir: Path | None, dump_frames: int) -> None:
        self._dump_dir = dump_dir
        self._dumps_left = dump_frames
        self._deltas: list[float] = []
        self._coverage: list[float] = []
        self._peak = 0.0
        self._mouth_means: list[float] = []
        self._skips: dict[str, int] = {}
        self._geometry: tuple[int, int] | None = None

    def record_skip(self, reason: str) -> None:
        """Count a frame the engine left untouched, so silence has an explanation."""
        self._skips[reason] = self._skips.get(reason, 0) + 1
        total = sum(self._skips.values())
        if total % self.REPORT_EVERY == 0:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self._skips.items()))
            print(
                f"[MUSETALK-PROBE] {total} frames left untouched ({detail})", flush=True
            )

    def record_geometry(
        self, frame_shape: tuple[int, ...], box: tuple[int, int, int, int]
    ) -> None:
        """Log the crop geometry on the first frame, then only on real changes.

        A tracked face jitters by a pixel or two every frame, so logging any
        difference floods the console; only a size shift worth acting on is news.
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        last = self._geometry
        if last is not None:
            last_w, last_h = last
            if abs(w - last_w) <= max(8, last_w * 0.25) and abs(h - last_h) <= max(
                8, last_h * 0.25
            ):
                return
        self._geometry = (w, h)
        print(
            f"[MUSETALK-PROBE] geometry frame={frame_shape[1]}x{frame_shape[0]} "
            f"crop=({x1},{y1})-({x2},{y2}) size={w}x{h}",
            flush=True,
        )

    def record(
        self,
        before: np.ndarray,
        after: np.ndarray,
        box: tuple[int, int, int, int],
        frame_index: int,
    ) -> None:
        x1, y1, x2, y2 = box
        roi_before = np.ascontiguousarray(before[y1:y2, x1:x2]).astype(np.int16)
        roi_after = np.ascontiguousarray(after[y1:y2, x1:x2]).astype(np.int16)
        if roi_before.size == 0:
            return
        diff = np.abs(roi_after - roi_before)
        # Averaged over the repainted pixels, not the whole crop. Diluting by crop
        # area makes the number depend on how tight the mask is and on the frame
        # resolution: the anatomical mask covers ~5% of the crop where the old
        # ellipse covered far more, so the same repaint read as "no change".
        touched = diff.max(axis=2) > 2
        coverage = float(touched.mean())
        self._deltas.append(float(diff[touched].mean()) if coverage > 0.0 else 0.0)
        self._coverage.append(coverage)
        self._peak = max(self._peak, float(diff.max()))

        # Mouth band only, so head motion elsewhere does not mask the signal.
        h = roi_after.shape[0]
        mouth = roi_after[int(h * 0.55) : int(h * 0.95)]
        if mouth.size:
            self._mouth_means.append(float(mouth.mean()))

        self._maybe_dump(roi_before, roi_after, diff, frame_index)
        if len(self._deltas) >= self.REPORT_EVERY:
            self._report()

    def _maybe_dump(
        self,
        roi_before: np.ndarray,
        roi_after: np.ndarray,
        diff: np.ndarray,
        frame_index: int,
    ) -> None:
        if self._dump_dir is None or self._dumps_left <= 0:
            return
        self._dumps_left -= 1
        try:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            stem = f"frame_{frame_index:06d}"
            cv2.imwrite(
                str(self._dump_dir / f"{stem}_a_off.png"), roi_before.astype(np.uint8)
            )
            cv2.imwrite(
                str(self._dump_dir / f"{stem}_b_on.png"), roi_after.astype(np.uint8)
            )
            amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
            cv2.imwrite(str(self._dump_dir / f"{stem}_c_diff_x5.png"), amplified)
        except Exception as e:
            print(f"[MUSETALK-PROBE] could not write dump: {e}")
            self._dump_dir = None

    # Below this the blend reached essentially no pixels, whatever the mask shape.
    _COVERAGE_FLOOR = 0.002

    def _report(self) -> None:
        mean_delta = sum(self._deltas) / len(self._deltas)
        coverage = sum(self._coverage) / len(self._coverage) if self._coverage else 0.0
        motion = float(np.std(self._mouth_means)) if self._mouth_means else 0.0
        # Whether anything was repainted is a question about area, not magnitude,
        # so it is answered without a brightness threshold to get wrong.
        verdict = "NO CHANGE — lip-sync is not touching the frame"
        if coverage >= self._COVERAGE_FLOOR:
            verdict = (
                "mouth is moving over time"
                if motion >= 0.5
                else "pixels change but the mouth looks static"
            )
        print(
            f"[MUSETALK-PROBE] {len(self._deltas)} frames | "
            f"avg change {mean_delta:.2f}/255 inside the repaint "
            f"({coverage * 100:.1f}% of the crop), peak {self._peak:.0f}/255 | "
            f"mouth variation over time {motion:.2f} -> {verdict}",
            flush=True,
        )
        self._deltas.clear()
        self._coverage.clear()
        self._mouth_means.clear()
        self._peak = 0.0


@dataclass
class _CropRequest:
    """One frame's crop waiting for its turn in a batched forward pass."""

    crop: np.ndarray
    audio_index: int
    done: threading.Event = field(default_factory=threading.Event)
    recon: np.ndarray | None = None


class MuseTalkEngine:
    # How long a worker waits for its batched result before giving up on the frame.
    _REQUEST_TIMEOUT_S = 20.0
    # Window the batcher keeps open for sibling workers to join the same pass.
    _GATHER_S = 0.004

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._batch_queue: queue.Queue | None = None
        self._batch_thread: threading.Thread | None = None
        self._batch_stop = threading.Event()
        self._max_batch = _batch_limit()
        self._probe: _EffectProbe | None = None
        self._probe_lock = threading.Lock()
        self._loaded = False
        self._vae: Any = None
        self._unet: Any = None
        self._pe: Any = None
        self._whisper: Any = None
        self._audio_proc: MuseTalkAudioProcessor | None = None
        self._device: Any = None
        self._dtype: Any = None
        self._timesteps: Any = None
        self._whisper_chunks: Any = None
        self._audio_key: str | None = None
        self._audio_fps: float = 25.0
        self._last_error: str | None = None
        self._audio_error: str | None = None
        self._bbox_shift: int = 0
        # Set by ModelsProcessor: the app-wide CUDA-graph capture lock. Our batch
        # thread must not touch the GPU while another model is capturing.
        self.gpu_capture_lock: Any = None
        self._warn_once: set[str] = set()
        self._compiled = False
        self._channels_last = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_audio(self) -> bool:
        return self._whisper_chunks is not None and len(self._whisper_chunks) > 0

    def is_ready_for_frame(self) -> bool:
        return self._loaded and self.has_audio

    def log_not_ready_once(self) -> None:
        """Explain once why an enabled lip-sync leaves frames untouched."""
        if "not_ready" in self._warn_once:
            return
        self._warn_once.add("not_ready")
        if not self._loaded:
            print(
                "[WARN] MuseTalk is enabled but not loaded. Toggle it off and on "
                "again, then check the load error in the log."
            )
        elif self._audio_error:
            # Preparing the track was attempted and failed, so advice about picking a
            # file would send the user looking in the wrong place entirely.
            print(
                "[WARN] MuseTalk is enabled but preparing the audio failed, so "
                f"lip-sync is skipped: {self._audio_error}"
            )
        else:
            print(
                "[WARN] MuseTalk is enabled but has no audio: load a video with an "
                "audio track, or set Audio Source to 'External file' and pick a "
                "WAV/MP3. Lip-sync is skipped until then."
            )

    def load(self, device: str | None = None, use_float16: bool = True) -> bool:
        """Load UNet + VAE + Whisper. Returns False if assets/deps missing.

        Activation is controlled by the UI toggle; the pytorch-extras env var is
        only an optional override and is not required here.
        """
        if not musetalk_assets_ready():
            self._last_error = (
                f"Weights missing under {musetalk_root()}. "
                "Run the launcher's 'Check / Update Models' (or python download_models.py)."
            )
            print(f"[WARN] MuseTalk: {self._last_error}")
            return False
        with self._lock:
            if self._loaded:
                return True
            try:
                prepare_transformers_env()
                import torch
                from transformers import WhisperModel

                from app.processors.pytorch_extras.musetalk.models import (
                    MuseTalkUNet,
                    MuseTalkVAE,
                    PositionalEncoding,
                )

                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                self._device = torch.device(device)
                use_fp16 = bool(use_float16) and self._device.type == "cuda"

                t0 = time.perf_counter()
                self._vae = MuseTalkVAE(
                    vae_dir(), use_float16=use_fp16, device=self._device
                )
                self._unet = MuseTalkUNet(
                    unet_config_path(),
                    unet_weights_path(),
                    use_float16=use_fp16,
                    device=self._device,
                )
                self._pe = PositionalEncoding(d_model=384)
                self._pe.to(self._device)

                self._audio_proc = MuseTalkAudioProcessor(whisper_dir())
                # Whisper only runs during prepare_audio; keep it on CPU so
                # realtime lip-sync VRAM is UNet+VAE only.
                self._whisper = WhisperModel.from_pretrained(str(whisper_dir()))
                self._dtype = self._unet.model.dtype
                self._whisper = self._whisper.to(
                    device="cpu", dtype=torch.float32
                ).eval()
                self._whisper.requires_grad_(False)
                self._timesteps = torch.tensor([0], device=self._device)
                if self._device.type == "cuda":
                    self._prefer_channels_last()
                    if musetalk_compile_enabled():
                        self._try_torch_compile()
                self._probe = self._build_probe()
                self._start_batcher()
                self._loaded = True
                self._last_error = None
                self._warn_once.clear()
                print(
                    f"[INFO] MuseTalk loaded on {self._device} "
                    f"in {(time.perf_counter() - t0) * 1000:.0f} ms "
                    f"(max batch {self._max_batch}, whisper=cpu"
                    f"{', compile=on' if self._compiled else ''})"
                )
                return True
            except Exception as e:
                self._last_error = str(e)
                print(f"[ERROR] MuseTalk load failed: {e}")
                traceback.print_exc()
                self.unload()
                return False

    @staticmethod
    def _build_probe() -> _EffectProbe | None:
        """Create the effect probe when VISOFUSION_MUSETALK_DEBUG is set."""
        if not _env_truthy("VISOFUSION_MUSETALK_DEBUG"):
            return None
        try:
            frames = int(os.environ.get("VISOFUSION_MUSETALK_DEBUG_FRAMES", "6"))
        except (TypeError, ValueError):
            frames = 6
        dump_dir = musetalk_root() / "debug" if frames > 0 else None
        print(
            "[MUSETALK-PROBE] enabled: reporting how much lip-sync changes the frame"
            + (f"; first {frames} crops go to {dump_dir}" if dump_dir else "")
        )
        return _EffectProbe(dump_dir, max(0, frames))

    def _prefer_channels_last(self) -> None:
        """Use NHWC on CUDA for UNet/VAE convolutions when supported."""
        import torch

        try:
            if self._unet is not None:
                self._unet.model = self._unet.model.to(
                    memory_format=torch.channels_last
                )
            if self._vae is not None:
                self._vae.vae = self._vae.vae.to(memory_format=torch.channels_last)
                if hasattr(self._vae, "set_channels_last"):
                    self._vae.set_channels_last(True)
            self._channels_last = True
        except Exception as e:
            self._channels_last = False
            print(f"[WARN] MuseTalk channels_last skipped: {e}")

    def _try_torch_compile(self) -> None:
        """Compile UNet (and VAE if stable). Failures stay on eager path."""
        import torch

        if self._unet is None or self._device is None or self._device.type != "cuda":
            return
        try:
            from custom_kernels.compile_utils import (
                _skip_torch_compile_cuda_inductor_reason,
                setup_compile_env,
            )
        except Exception as e:
            print(f"[WARN] MuseTalk torch.compile unavailable: {e}")
            return

        n = max(1, int(self._max_batch))
        dtype = self._dtype
        device = self._device
        sample = torch.zeros(
            (n, 8, _MUSETALK_LATENT_HW, _MUSETALK_LATENT_HW),
            device=device,
            dtype=dtype,
        )
        if self._channels_last:
            sample = sample.to(memory_format=torch.channels_last)
        skip = _skip_torch_compile_cuda_inductor_reason(sample)
        if skip is not None:
            print(f"[INFO] MuseTalk torch.compile skipped ({skip})")
            return

        setup_compile_env(compile_mode="default")
        # Specialize on batch 1 and max_batch (dynamic=True thrashes sympy on this
        # UNet and has hung/crashed the process on Windows + triton-windows).
        audio_full = torch.zeros(
            (n, _MUSETALK_AUDIO_TOKENS, _MUSETALK_AUDIO_DIM),
            device=device,
            dtype=dtype,
        )
        timesteps = self._timesteps

        try:
            compiled_unet = torch.compile(
                self._unet.model, mode="default", fullgraph=False, dynamic=False
            )
            with torch.inference_mode():
                for size in sorted({1, n}):
                    s = sample[:size]
                    a = audio_full[:size]
                    if self._channels_last:
                        s = s.to(memory_format=torch.channels_last)
                    for _ in range(2):
                        compiled_unet(s, timesteps, encoder_hidden_states=a)
            if device.type == "cuda":
                torch.cuda.synchronize()
            self._unet.model = compiled_unet
            self._compiled = True
            print("[INFO] MuseTalk UNet torch.compile ready", flush=True)
        except Exception as e:
            print(f"[WARN] MuseTalk UNet torch.compile failed: {e}", flush=True)
            return

        if self._vae is None:
            return
        try:
            compiled_vae = torch.compile(
                self._vae.vae, mode="default", fullgraph=False, dynamic=False
            )
            img_full = torch.zeros((n, 3, 256, 256), device=device, dtype=dtype)
            with torch.inference_mode():
                for size in sorted({1, n}):
                    img = img_full[:size]
                    if self._channels_last:
                        img = img.to(memory_format=torch.channels_last)
                    for _ in range(2):
                        lat = compiled_vae.encode(img).latent_dist.sample()
                        _ = compiled_vae.decode(lat).sample
            if device.type == "cuda":
                torch.cuda.synchronize()
            self._vae.vae = compiled_vae
            print("[INFO] MuseTalk VAE torch.compile ready", flush=True)
        except Exception as e:
            print(
                f"[WARN] MuseTalk VAE torch.compile failed (UNet still compiled): {e}",
                flush=True,
            )
    def _start_batcher(self) -> None:
        self._batch_stop.clear()
        self._batch_queue = queue.Queue()
        self._batch_thread = threading.Thread(
            target=self._batch_loop, name="MuseTalkBatcher", daemon=True
        )
        self._batch_thread.start()

    def _stop_batcher(self) -> None:
        """Stop the batching thread and release anyone still waiting on it."""
        self._batch_stop.set()
        pending = self._batch_queue
        thread = self._batch_thread
        self._batch_queue = None
        self._batch_thread = None
        if pending is not None:
            pending.put(None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        while pending is not None:
            try:
                request = pending.get_nowait()
            except queue.Empty:
                break
            if request is not None:
                request.done.set()

    def _batch_loop(self) -> None:
        """Coalesce concurrent crop requests into one forward pass each round."""
        while not self._batch_stop.is_set():
            pending = self._batch_queue
            if pending is None:
                return
            try:
                first = pending.get(timeout=0.2)
            except queue.Empty:
                continue
            if first is None:
                return
            batch = [first]
            deadline = time.perf_counter() + self._GATHER_S
            while len(batch) < self._max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    nxt = pending.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self._batch_stop.set()
                    break
                batch.append(nxt)
            try:
                self._infer_batch(batch)
            except Exception as e:
                if "batch" not in self._warn_once:
                    print(f"[WARN] MuseTalk batch inference failed: {e}")
                    self._warn_once.add("batch")
            finally:
                for request in batch:
                    request.done.set()

    def _infer_batch(self, batch: list[_CropRequest]) -> None:
        import contextlib

        import torch

        vae, unet, pe = self._vae, self._unet, self._pe
        chunks = self._whisper_chunks
        if vae is None or unet is None or pe is None or chunks is None:
            return
        unet_dtype = unet.model.dtype
        # Uncontended except while another model builds its CUDA graph, which is a
        # one-off per model, so this costs a lock acquire per batch.
        gate = self.gpu_capture_lock or contextlib.nullcontext()
        with gate, torch.inference_mode():
            # The VAE mask stays the exact lower half. bbox_shift belongs to the
            # crop geometry (see framing.py); moving this line instead detaches it
            # from the blend boundary and shows up as a seam across the nose.
            latents = vae.get_latents_for_unet_batch([r.crop for r in batch])
            if getattr(self, "_channels_last", False) and getattr(
                latents, "is_cuda", False
            ):
                latents = latents.to(memory_format=torch.channels_last)
            audio = torch.cat(
                [chunks[r.audio_index : r.audio_index + 1] for r in batch], dim=0
            )
            audio_feat = pe(audio.to(device=self._device, dtype=unet_dtype)).to(
                unet_dtype
            )
            pred = unet.model(
                latents.to(device=self._device, dtype=unet_dtype),
                self._timesteps,
                encoder_hidden_states=audio_feat,
            ).sample
            recon = vae.decode_latents(pred.to(dtype=vae.vae.dtype))
        for request, image in zip(batch, recon):
            request.recon = image

    def unload(self) -> None:
        self._stop_batcher()
        with self._lock:
            self._vae = None
            self._unet = None
            self._pe = None
            self._whisper = None
            self._audio_proc = None
            self._whisper_chunks = None
            self._audio_key = None
            self._timesteps = None
            self._probe = None
            self._loaded = False
            self._compiled = False
            self._channels_last = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print("[INFO] MuseTalk unloaded.")

    @staticmethod
    def _to_host(chunks: Any) -> Any:
        """Copy audio chunks to CPU and release the device copy."""
        try:
            import torch

            if getattr(chunks, "is_cuda", False):
                chunks = chunks.to("cpu")
                torch.cuda.empty_cache()
        except Exception:
            pass
        return chunks

    def clear_audio(self) -> None:
        with self._lock:
            self._whisper_chunks = None
            self._audio_key = None
            self._warn_once.discard("not_ready")

    def prepare_audio(
        self,
        audio_or_media_path: str | Path,
        fps: float,
        *,
        is_video_container: bool = False,
    ) -> bool:
        """Build per-frame Whisper chunks for realtime apply."""
        if not self._loaded or self._audio_proc is None or self._whisper is None:
            if not self.load():
                return False
        path = Path(audio_or_media_path)
        if not path.is_file():
            self._last_error = f"Audio/media not found: {path}"
            print(f"[WARN] MuseTalk: {self._last_error}")
            return False

        key = f"{path.resolve()}|{path.stat().st_mtime_ns}|{float(fps):.4f}"
        key_hash = hashlib.sha1(key.encode("utf-8")).hexdigest()
        with self._lock:
            if self._audio_key == key_hash and self.has_audio:
                return True
            # Drop the previous track now: a failure below must not leave the old
            # media's audio driving the new one.
            self.clear_audio()

        wav_path = path
        tmp_wav: Path | None = None
        try:
            import torch

            if is_video_container or path.suffix.lower() in {
                ".mp4",
                ".mkv",
                ".mov",
                ".avi",
                ".webm",
                ".m4v",
            }:
                tmp_wav = musetalk_root() / "cache" / f"audio_{key_hash[:16]}.wav"
                if not tmp_wav.is_file() or tmp_wav.stat().st_size < 64:
                    extract_wav_from_media(path, tmp_wav)
                wav_path = tmp_wav

            feats, librosa_len = self._audio_proc.get_audio_feature(
                wav_path, weight_dtype=torch.float32
            )
            if feats is None or librosa_len <= 0:
                self._last_error = "Failed to extract audio features"
                print(f"[WARN] MuseTalk: {self._last_error}")
                return False

            whisper_device = next(self._whisper.parameters()).device
            with self._lock:
                chunks = self._audio_proc.get_whisper_chunk(
                    feats,
                    whisper_device,
                    torch.float32,
                    self._whisper,
                    librosa_len,
                    fps=float(fps),
                )
                # Park the whole track in system RAM: at ~37 KB per frame it would
                # cost GBs of VRAM on a long video, and apply_frame_bgr only ever
                # needs one frame's slice on the device.
                self._whisper_chunks = self._to_host(chunks)
                self._audio_error = None
                self._audio_key = key_hash
                self._audio_fps = float(fps)
                n = 0 if chunks is None else int(chunks.shape[0])
                print(f"[INFO] MuseTalk audio ready: {n} frames @ {fps:.3f} fps")
                return n > 0
        except Exception as e:
            self._last_error = str(e)
            self._audio_error = str(e)
            self._warn_once.discard("not_ready")
            print(f"[WARN] MuseTalk audio unavailable: {e}")
            return False

    def apply_frame_bgr(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        bboxes: Sequence[Any] | None = None,
        *,
        extra_margin: int = 10,
        face_index: int = 0,
        mask_options: dict | None = None,
        parse_labels: Callable[[np.ndarray], Any] | None = None,
        landmarks: Sequence[Any] | None = None,
        kpss_5: Sequence[Any] | None = None,
        restore_crop: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Lip-sync one BGR frame. On any failure returns ``frame_bgr`` unchanged."""
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            self._note_skip("bad_frame")
            return frame_bgr
        if not self.is_ready_for_frame():
            self._note_skip("not_loaded" if not self._loaded else "no_audio")
            return frame_bgr
        try:
            return self._apply_frame_bgr_unlocked(
                frame_bgr,
                int(frame_index),
                bboxes,
                extra_margin=int(extra_margin),
                face_index=int(face_index),
                mask_options=mask_options,
                parse_labels=parse_labels,
                landmarks=landmarks,
                kpss_5=kpss_5,
                restore_crop=restore_crop,
            )
        except Exception as e:
            self._note_skip("exception")
            if "apply" not in self._warn_once:
                print(f"[WARN] MuseTalk apply_frame failed: {e}")
                self._warn_once.add("apply")
            return frame_bgr

    @staticmethod
    def _pick(items: Sequence[Any] | None, face_index: int):
        """One face's entry, tolerating numpy arrays and short sequences."""
        try:
            n = 0 if items is None else len(items)
        except TypeError:
            return None
        if n == 0:
            return None
        return items[min(max(int(face_index), 0), n - 1)]

    _ELLIPSE_KEYS = ("strength", "radius_x", "radius_y", "centre_y")

    @staticmethod
    def _ellipse_options(mask_options: dict | None) -> dict:
        """Only the keys the geometric fallback understands."""
        opts = mask_options or {}
        return {k: opts[k] for k in MuseTalkEngine._ELLIPSE_KEYS if k in opts}

    def _blend_masks(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        parse_labels: Callable[[np.ndarray], Any] | None,
        mask_options: dict | None,
    ) -> ParsedMasks:
        """Every parsed mask, or all-None to fall back to the ellipse.

        Never fatal: without the parser model the ellipse still produces a usable
        frame, so a missing or failing parser degrades quality instead of
        dropping lip-sync entirely. All the masks come from one parse.
        """
        if parse_labels is None:
            return ParsedMasks(None, None, None)
        opts = mask_options or {}
        try:
            return parsed_masks(
                frame_bgr,
                bbox,
                parse_labels,
                upper_boundary_ratio=float(opts.get("upper_boundary_ratio", 0.5)),
                left_cheek_width=int(opts.get("left_cheek_width", 90)),
                right_cheek_width=int(opts.get("right_cheek_width", 90)),
                strength=float(opts.get("strength", 1.0)),
            )
        except Exception as e:
            if "parse" not in self._warn_once:
                print(f"[WARN] MuseTalk face parsing failed, using the soft mask: {e}")
                self._warn_once.add("parse")
            return ParsedMasks(None, None, None)

    def _mouth_only_mask(
        self,
        original_mouth: np.ndarray | None,
        recon_bgr: np.ndarray,
        parse_labels: Callable[[np.ndarray], Any] | None,
        region_shape: tuple[int, int],
        mask_options: dict | None,
    ) -> np.ndarray | None:
        """The union of both mouth poses, or None to keep the jaw mask.

        Repainting the whole jaw is what puts a second chin and a second mouth in
        the frame whenever the composite is not fully opaque, and a fully opaque
        jaw repaint is what costs the identity. Restricting the paste to where the
        mouth is or ends up keeps the swapped jawline untouched, so full opacity
        stops being a trade-off.
        """
        opts = mask_options or {}
        if not opts.get("mouth_only") or parse_labels is None:
            return None
        # Zero strength still means "leave the frame alone"; letting it through
        # would repaint at full opacity, the opposite of what the slider says.
        if float(opts.get("strength", 1.0)) <= 0.0:
            return None
        try:
            generated = crop_mouth_mask(recon_bgr, parse_labels, region_shape)
            padding = int(opts.get("mouth_padding", 6))
            mask = mouth_only_blend_mask(
                original_mouth,
                generated,
                padding_px=padding,
                # A hard edge in skin reads as a scar; the falloff scales with the
                # padding so widening the region also softens its border.
                feather_px=max(padding, 3) + 3,
            )
            # Full opacity is what stops the doubled mouth when the two poses differ.
            # The hybrid after-pass is the one exception: its input already carries
            # the mouth pose from the pre-swap pass, so both mouths are aligned and a
            # partial alpha only re-sharpens the same shape instead of ghosting.
            alpha = float(opts.get("mouth_alpha", 1.0))
            if mask is not None and alpha < 1.0:
                mask = mask * max(alpha, 0.0)
            return mask
        except Exception as e:
            if "mouth_only" not in self._warn_once:
                print(
                    f"[WARN] MuseTalk mouth-only mask failed, using the jaw mask: {e}"
                )
                self._warn_once.add("mouth_only")
            return None

    def _blend_mask(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        parse_labels: Callable[[np.ndarray], Any] | None,
        mask_options: dict | None,
    ) -> np.ndarray | None:
        """The repaint mask alone; kept for callers that do not need the lip mask."""
        return self._blend_masks(frame_bgr, bbox, parse_labels, mask_options)[0]

    def _note_skip(self, reason: str) -> None:
        probe = self._probe
        if probe is not None:
            with self._probe_lock:
                probe.record_skip(reason)

    def _apply_frame_bgr_unlocked(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        bboxes: Sequence[Any] | None,
        *,
        extra_margin: int,
        face_index: int,
        mask_options: dict | None = None,
        parse_labels: Callable[[np.ndarray], Any] | None = None,
        landmarks: Sequence[Any] | None = None,
        kpss_5: Sequence[Any] | None = None,
        restore_crop: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        chunks = self._whisper_chunks
        if chunks is None or chunks.shape[0] == 0:
            self._note_skip("no_chunks")
            return frame_bgr
        self._bbox_shift = int((mask_options or {}).get("bbox_shift", 0))
        idx = int(frame_index) % int(chunks.shape[0])

        # Length, never truthiness: the frame worker hands over a numpy array of
        # boxes, and ``if bboxes:`` raises "truth value of an array is ambiguous"
        # on it, which silently cost every single frame.
        bbox_src = None
        try:
            n_bboxes = 0 if bboxes is None else len(bboxes)
        except TypeError:
            n_bboxes = 0
        if n_bboxes > 0:
            bbox_src = bboxes[min(max(face_index, 0), n_bboxes - 1)]
        if bbox_src is None:
            # No detection means no face to drive: a centred guess used to paint a
            # generated mouth across ~700x640 px of frame, which is far worse than
            # leaving the frame alone.
            self._note_skip("no_bbox")
            return frame_bgr

        # Landmarks first: that is the window MuseTalk was trained on. The
        # detector box only approximates it, and no amount of nudging fixes the
        # width, which the landmark extent gets right by construction.
        framed = landmark_crop_bbox(
            self._pick(landmarks, face_index),
            self._pick(kpss_5, face_index),
            frame_bgr.shape,
            extra_margin=extra_margin,
            reference_bbox=bbox_src,
            bbox_shift=self._bbox_shift,
        )
        if framed is not None:
            x1, y1, x2, y2 = framed
        else:
            x1, y1, x2, y2 = expand_bbox(
                bbox_src, frame_bgr.shape, extra_margin=extra_margin
            )
        if self._probe is not None:
            with self._probe_lock:
                self._probe.record_geometry(frame_bgr.shape, (x1, y1, x2, y2))

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            self._note_skip("empty_crop")
            return frame_bgr
        crop256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        pending = self._batch_queue
        if pending is None:
            self._note_skip("no_batcher")
            return frame_bgr
        request = _CropRequest(crop256, idx)
        pending.put(request)
        if not request.done.wait(timeout=self._REQUEST_TIMEOUT_S):
            self._note_skip("batch_timeout")
            if "timeout" not in self._warn_once:
                print("[WARN] MuseTalk frame timed out waiting for the batcher.")
                self._warn_once.add("timeout")
            return frame_bgr
        if request.recon is None:
            self._note_skip("no_recon")
            return frame_bgr
        # Add texture back to the 256-capped mouth before blending. The model gives
        # a smooth, generic mouth; a restorer (GFPGAN/CodeFormer) hallucinates the
        # missing high frequencies. Non-fatal: a failing restorer just leaves the
        # softer mouth rather than dropping the frame.
        if restore_crop is not None:
            try:
                restored = restore_crop(request.recon)
                if isinstance(restored, np.ndarray) and restored.size:
                    request.recon = restored
            except Exception as e:
                if "restore" not in self._warn_once:
                    print(f"[WARN] MuseTalk mouth restore failed, skipping it: {e}")
                    self._warn_once.add("restore")
        # Blending stays on the calling worker so the batch thread only does GPU work.
        masks = self._blend_masks(
            frame_bgr, (x1, y1, x2, y2), parse_labels, mask_options
        )
        repaint = self._mouth_only_mask(
            masks.mouth,
            request.recon,
            parse_labels,
            (y2 - y1, x2 - x1),
            mask_options,
        )
        blended = blend_face_region(
            frame_bgr,
            request.recon,
            (x1, y1, x2, y2),
            mask=masks.jaw if repaint is None else repaint,
            lip_mask=masks.lip,
            lip_color_strength=float(
                (mask_options or {}).get("lip_color_strength", 0.0)
            ),
            mask_options=self._ellipse_options(mask_options),
        )
        if self._probe is not None:
            with self._probe_lock:
                self._probe.record(
                    frame_bgr, blended, (x1, y1, x2, y2), int(frame_index)
                )
        return blended
