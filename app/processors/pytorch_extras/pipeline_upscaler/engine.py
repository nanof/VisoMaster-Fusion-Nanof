"""Last-stage pipeline upscaler (HYPIR / PMRF / FlashVSR / SeedVR2).

Runs after swap, restore, frame enhancer, and MuseTalk. Failures return the
input frame unchanged. GPU work is serialized with a process-wide lock because
these models keep temporal / DiT state.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any, Mapping

import numpy as np

from app.processors.pytorch_extras.pipeline_upscaler.paths import (
    assets_error_message,
    assets_ready,
    deps_error_message,
    format_load_error,
    missing_python_deps,
)
from app.processors.pytorch_extras.pipeline_upscaler.scale import (
    FLASHVSR_TYPES,
    HYPIR_TYPES,
    PMRF_TYPES,
    SEEDVR2_TYPES,
    pipeline_upscaler_enabled,
    pipeline_upscaler_scale_factor,
    pipeline_upscaler_type,
)


class PipelineUpscalerEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: Any = None
        self._backend_kind: str | None = None
        self._loaded_type: str | None = None
        self._last_error: str | None = None
        self.gpu_capture_lock: threading.Lock | None = None

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None and bool(getattr(self._backend, "is_loaded", False))

    def unload(self) -> None:
        with self._lock:
            backend = self._backend
            self._backend = None
            self._backend_kind = None
            self._loaded_type = None
            if backend is not None:
                try:
                    backend.unload()
                except Exception as e:
                    print(f"[WARN] Pipeline upscaler unload: {e}")

    def load(self, model_type: str, device: str = "cuda") -> bool:
        """Download (if needed) and load ``model_type``. Never raises."""
        try:
            missing = missing_python_deps(model_type)
            if missing:
                self._last_error = deps_error_message(model_type, missing)
                print(f"[WARN] {self._last_error}")
                return False
            from app.processors.pytorch_extras.pipeline_upscaler.download import (
                ensure_weights,
            )

            ensure_weights(model_type)
            if not assets_ready(model_type):
                self._last_error = assets_error_message(model_type)
                print(f"[WARN] {self._last_error}")
                return False
            with self._lock:
                if self._loaded_type == model_type and self.is_loaded:
                    return True
                if self._backend is not None:
                    try:
                        self._backend.unload()
                    except Exception:
                        pass
                    self._backend = None
                if model_type in FLASHVSR_TYPES:
                    from app.processors.pytorch_extras.pipeline_upscaler.flashvsr_backend import (
                        FlashVSRBackend,
                    )

                    backend = FlashVSRBackend()
                    kind = "flashvsr"
                elif model_type in SEEDVR2_TYPES:
                    from app.processors.pytorch_extras.pipeline_upscaler.seedvr2_backend import (
                        SeedVR2Backend,
                    )

                    backend = SeedVR2Backend()
                    kind = "seedvr2"
                elif model_type in HYPIR_TYPES:
                    from app.processors.pytorch_extras.pipeline_upscaler.hypir_backend import (
                        HypirBackend,
                    )

                    backend = HypirBackend()
                    kind = "hypir"
                elif model_type in PMRF_TYPES:
                    from app.processors.pytorch_extras.pipeline_upscaler.pmrf_backend import (
                        PmrfBackend,
                    )

                    backend = PmrfBackend()
                    kind = "pmrf"
                else:
                    self._last_error = f"Unknown pipeline upscaler: {model_type}"
                    print(f"[WARN] {self._last_error}")
                    return False
                ok = bool(backend.load(model_type, device=device))
                if not ok:
                    self._last_error = getattr(backend, "_last_error", None) or (
                        f"{model_type} failed to load"
                    )
                    return False
                self._backend = backend
                self._backend_kind = kind
                self._loaded_type = model_type
                self._last_error = None
                return True
        except Exception as e:
            self._last_error = format_load_error(e, model_type)
            print(f"[WARN] {self._last_error}")
            traceback.print_exc()
            return False

    def apply_bgr(
        self,
        frame_bgr: np.ndarray,
        control: Mapping[str, Any] | None,
        *,
        frame_number: int | None = None,
    ) -> np.ndarray:
        """Upscale a BGR uint8 frame. Returns ``frame_bgr`` unchanged on failure."""
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return frame_bgr
        if not pipeline_upscaler_enabled(control):
            return frame_bgr
        model_type = pipeline_upscaler_type(control)
        if not self.is_loaded or self._loaded_type != model_type:
            return frame_bgr
        scale = pipeline_upscaler_scale_factor(control)
        color_fix = bool((control or {}).get("PipelineUpscalerColorFixToggle", True))
        tiled = bool((control or {}).get("PipelineUpscalerTiledVaeToggle", True))
        seed = 0
        try:
            seed = int((control or {}).get("DenoiserBaseSeedSlider", 0) or 0)
        except (TypeError, ValueError):
            seed = 0
        cap = self.gpu_capture_lock
        try:
            if cap is not None:
                cap.acquire()
            with self._lock:
                backend = self._backend
                if backend is None:
                    return frame_bgr
                if self._backend_kind == "seedvr2":
                    return backend.upscale_bgr(
                        frame_bgr,
                        scale=scale,
                        color_fix=color_fix,
                        tiled_vae=tiled,
                        seed=seed,
                    )
                return backend.upscale_bgr(
                    frame_bgr, scale=scale, color_fix=color_fix, seed=seed
                )
        except Exception as e:
            is_oom = "out of memory" in str(e).lower()
            if not getattr(self, "_apply_err_logged", False) or is_oom:
                print(f"[WARN] Pipeline upscaler apply failed: {e}")
                if not is_oom:
                    traceback.print_exc()
                self._apply_err_logged = True
            # Keep encoder/preview size at the selected scale so playback can start.
            if scale > 1:
                try:
                    import cv2

                    h, w = int(frame_bgr.shape[0]) * scale, int(frame_bgr.shape[1]) * scale
                    return np.ascontiguousarray(
                        cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
                    )
                except Exception:
                    return frame_bgr
            return frame_bgr
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
