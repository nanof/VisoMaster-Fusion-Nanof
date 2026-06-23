import contextlib
import gc
import os
import threading
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import numpy as np
from torchvision.transforms import v2
from skimage import transform as trans
import kornia.geometry.transform as kgm

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class _CodeFormerDirectRunner:
    """Runs CodeFormerTorch without CUDA graph (same interface as CUDAGraphRunner)."""

    __slots__ = ("_model", "_fw")

    def __init__(self, model, fidelity_weight: float):
        self._model = model
        self._fw = float(fidelity_weight)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x, fidelity_weight=self._fw)


class FaceRestorers:
    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.active_model_slot1: Optional[str] = None
        self.active_model_slot2: Optional[str] = None
        self._warned_models: set[str] = set()  # To track warnings
        self._gfpgan_torch: Optional[object] = None  # GFPGANTorch (v1.4)
        self._gfpgan1024_torch: Optional[object] = None  # GFPGANTorch (1024)
        self._gfpgan_runner: Optional[object] = None  # CUDA graph runner (v1.4)
        self._gfpgan1024_runner: Optional[object] = None  # CUDA graph runner (1024)
        # Key (size, variant): variant "std" = GPEN-BFR-{size}.onnx; "fp16hf" = HF FP16 256 only.
        self._gpen_torch: Dict[Tuple[int, str], Optional[object]] = {}
        self._gpen_runner: Dict[Tuple[int, str], Optional[object]] = {}
        self._ref_ldm_encoder_torch: Optional[object] = None  # RefLDMEncoderTorch
        self._ref_ldm_encoder_runner: Optional[object] = None  # CUDA graph runner
        self._ref_ldm_decoder_torch: Optional[object] = None  # RefLDMDecoderTorch
        self._ref_ldm_decoder_runner: Optional[object] = None  # CUDA graph runner
        self._ref_ldm_unet_torch: Optional[object] = None  # RefLDMUNetTorch
        self._ref_ldm_unet_runner: dict = {}  # {use_exclusive: UNetCUDAGraphRunner}
        self._codeformer_torch: Optional[object] = None  # CodeFormerTorch
        self._codeformer_runner: Optional[object] = None  # CUDA graph runner
        self._codeformer_runner_w: Optional[float] = (
            None  # fidelity weight baked into runner
        )
        self._restoreformer_torch: Optional[object] = None  # RestoreFormerPlusPlusTorch
        self._restoreformer_runner: Optional[object] = None  # CUDA graph runner
        self._custom_inference_lock = threading.Lock()
        self._runner_locks: Dict[int, threading.Lock] = {}
        self._custom_init_lock = threading.Lock()  # serialises Custom-kernel lazy inits
        self._dmdnet_model: Optional[torch.nn.Module] = None
        self._dmdnet_lock = threading.Lock()
        self._restorer_ort_batch_session_disabled = False
        self._restorer_ort_batch_fail_logged = False
        self.model_map = {
            "GFPGAN-v1.4": "GFPGANv1.4",
            "GFPGAN-1024": "GFPGAN1024",
            "CodeFormer": "CodeFormer",
            "GPEN-256": "GPENBFR256",
            "GPEN-256 FP16 (HF)": "GPENBFR256FP16",
            "GPEN-256 Fast (128→256)": "GPENBFR256",
            "GPEN-256 Fast FP16 (128→256)": "GPENBFR256FP16",
            "GPEN-512": "GPENBFR512",
            "GPEN-1024": "GPENBFR1024",
            "GPEN-2048": "GPENBFR2048",
            "RestoreFormer++": "RestoreFormerPlusPlus",
            "RestoreFormer": "RestoreFormerFP16",
            "VQFR-v2": "VQFRv2",
            "DMDNet": "DMDNetTorch",
            "DMDNet FP16": "DMDNetTorch",
        }

    def unload_dmdnet(self) -> None:
        with self._custom_init_lock:
            self._dmdnet_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_dmdnet_loaded(self) -> Optional[torch.nn.Module]:
        if self.models_processor.device != "cuda":
            if "DMDNetCuda" not in self._warned_models:
                print("[WARN] DMDNet is only supported on CUDA; skipping load.")
                self._warned_models.add("DMDNetCuda")
            return None
        with self._custom_init_lock:
            if self._dmdnet_model is not None:
                return self._dmdnet_model
            from pathlib import Path

            from app.processors.external.dmdnet_arch import DMDNet
            from app.processors.models_data import models_dir

            pth = Path(models_dir) / "pytorch_weights" / "DMDNet.pth"
            if not pth.is_file():
                if "DMDNetPath" not in self._warned_models:
                    print(
                        f"[WARN] DMDNet weights not found at {pth}. "
                        "Run download_models.py (includes pytorch_weights/DMDNet.pth)."
                    )
                    self._warned_models.add("DMDNetPath")
                return None
            try:
                net = DMDNet().to("cuda")
                state = torch.load(
                    str(pth), map_location="cuda", weights_only=False
                )
                net.load_state_dict(state, strict=True)
                net.eval()
                self._dmdnet_model = net
                print("[INFO] DMDNet (PyTorch) loaded.")
                return net
            except Exception as e:
                print(f"[ERROR] Failed to load DMDNet: {e}")
                self._dmdnet_model = None
                return None

    def run_dmdnet(
        self,
        temp_nchw: torch.Tensor,
        lm68_xy: np.ndarray,
        output: torch.Tensor,
        sp_bchw_normalized: Optional[torch.Tensor] = None,
        sp_lm68_xy: Optional[np.ndarray] = None,
        use_half_autocast: bool = False,
    ) -> bool:
        """Run DMDNet on ``temp_nchw`` (B×3×512×512, normalized [-1,1]).

        Optional **specific** path: high-quality reference crop ``sp_bchw_normalized``
        with 68 landmarks ``sp_lm68_xy`` in the **same 512×512** space (runs ``memorize``
        then uses the specific branch in ``forward``). When omitted, uses generic memory only.

        ``use_half_autocast``: CUDA AMP (FP16) for ``memorize``/``forward``; weights stay FP32.
        Attention readout and AdaIN run in FP32 inside the model to avoid NaN/black ROI tiles.
        """
        model = self.ensure_dmdnet_loaded()
        if model is None:
            return False
        from app.processors.dmdnet_landmarks import get_component_location_tensor

        dev = next(model.parameters()).device
        dtype = torch.float32
        lq = temp_nchw.to(device=dev, dtype=dtype).contiguous()
        loc_t = get_component_location_tensor(lm68_xy, dev).unsqueeze(0).to(
            device=dev, dtype=dtype
        )

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_half_autocast
            else contextlib.nullcontext()
        )

        sp_256 = sp_128 = sp_64 = None
        if sp_bchw_normalized is not None and sp_lm68_xy is not None:
            try:
                sp = sp_bchw_normalized.to(device=dev, dtype=dtype).contiguous()
                if sp.dim() != 4 or sp.shape[0] != 1 or sp.shape[1] != 3:
                    raise ValueError(f"expected sp (1,3,H,W), got {tuple(sp.shape)}")
                if sp.shape[2] != 512 or sp.shape[3] != 512:
                    sp = v2.functional.resize(sp, [512, 512], antialias=True)
                lm_sp = np.asarray(sp_lm68_xy, dtype=np.float32).reshape(68, 2)
                loc_sp = get_component_location_tensor(lm_sp, dev).unsqueeze(0).to(
                    device=dev, dtype=dtype
                )
                with torch.no_grad(), amp_ctx:
                    with self._dmdnet_lock:
                        sp_256, sp_128, sp_64 = model.memorize(sp, loc_sp)
            except Exception as e:
                if "DMDNetSpWarn" not in self._warned_models:
                    print(
                        f"[WARN] DMDNet specific-memory path failed ({e}); using generic memory only."
                    )
                    self._warned_models.add("DMDNetSpWarn")
                sp_256 = sp_128 = sp_64 = None

        try:
            with torch.no_grad(), amp_ctx:
                with self._dmdnet_lock:
                    ge, gs = model(
                        lq=lq,
                        loc=loc_t,
                        sp_256=sp_256,
                        sp_128=sp_128,
                        sp_64=sp_64,
                    )
            # Drop the unused head tensor immediately so peak VRAM does not hold both
            # GeOut and GSOut until this function returns (specific path = two full outputs).
            if gs is not None:
                output.copy_(gs.to(device=output.device, dtype=output.dtype))
                del ge, gs
            else:
                output.copy_(ge.to(device=output.device, dtype=output.dtype))
                del ge
            del lq, loc_t
            if sp_256 is not None:
                del sp_256, sp_128, sp_64
            return True
        except Exception as e:
            print(f"[ERROR] DMDNet forward failed: {e}")
            return False

    def unload_models(self):
        """Unloads the restorer models held in both slots and resets state."""
        if self.active_model_slot1:
            self.models_processor.unload_model(self.active_model_slot1)
            self.active_model_slot1 = None
        if self.active_model_slot2:
            self.models_processor.unload_model(self.active_model_slot2)
            self.active_model_slot2 = None
        self.unload_dmdnet()
        self._unload_custom_torch_kernels()

    def _unload_custom_torch_kernels(self) -> None:
        """Release Custom-provider PyTorch restorers (GPEN/GFPGAN/etc.) held outside ORT."""
        with self._custom_init_lock:
            self._gfpgan_torch = None
            self._gfpgan1024_torch = None
            self._gfpgan_runner = None
            self._gfpgan1024_runner = None
            self._gpen_torch.clear()
            self._gpen_runner.clear()
            self._ref_ldm_encoder_torch = None
            self._ref_ldm_encoder_runner = None
            self._ref_ldm_decoder_torch = None
            self._ref_ldm_decoder_runner = None
            self._ref_ldm_unet_torch = None
            self._ref_ldm_unet_runner.clear()
            self._codeformer_torch = None
            self._codeformer_runner = None
            self._codeformer_runner_w = None
            self._restoreformer_torch = None
            self._restoreformer_runner = None

    def _get_model_session(self, model_name: str):
        """
        Gets the model session by calling the centralized, provider-aware loader
        in ModelsProcessor. This ensures correct logging, caching, and provider handling.
        """
        # All complex logic is now delegated to the main loader.
        ort_session = self.models_processor.load_model(model_name)

        if not ort_session:
            if model_name not in self._warned_models:
                print(
                    f"[WARN] Model '{model_name}' failed to load or is not available. This operation will be skipped."
                )
                self._warned_models.add(model_name)
            return None
        return ort_session

    def _ort_output_dtype(self, model_name: str, output_name: str) -> torch.dtype:
        """Declared ONNX output dtype, or float32 if the session is unavailable."""
        if self.models_processor.get_onnx_session(model_name) is None:
            if self._get_model_session(model_name) is None:
                return torch.float32
        return self.models_processor.get_ort_io_torch_dtype(
            model_name, output_name, is_output=True
        )

    def _gfpgan_prefers_torch(self) -> bool:
        return self.models_processor.device == "cuda" and torch.cuda.is_available()

    def _gfpgan_output_dtype(self, is_1024: bool) -> torch.dtype:
        if self._gfpgan_prefers_torch():
            return torch.float32
        model_name = "GFPGAN1024" if is_1024 else "GFPGANv1.4"
        return self._ort_output_dtype(model_name, "output")

    def _get_gfpgan_runner(self, is_1024: bool = False):
        """Lazily load GFPGANTorch + CUDA graph runner (FP16 PyTorch kernel)."""
        model_attr = "_gfpgan1024_torch" if is_1024 else "_gfpgan_torch"
        runner_attr = "_gfpgan1024_runner" if is_1024 else "_gfpgan_runner"
        label = "GFPGAN-1024" if is_1024 else "GFPGAN v1.4"

        runner = getattr(self, runner_attr)
        if runner is not None:
            return runner
        self.models_processor.show_build_dialog.emit(
            "Loading face restorer",
            f"Compiling & capturing CUDA graph for {label}…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                runner = getattr(self, runner_attr)
                if runner is not None:
                    return runner  # built by another thread while we waited
                model = getattr(self, model_attr)
                if model is None:
                    try:
                        import pathlib
                        from custom_kernels.gfpgan_v1_4.gfpgan_torch import GFPGANTorch

                        onnx_name = "gfpgan-1024.onnx" if is_1024 else "GFPGANv1.4.onnx"
                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / onnx_name
                        )
                        print(f"[GFPGANTorch] Loading PyTorch {label} model...")
                        m = GFPGANTorch.from_onnx(onnx_path).cuda().eval()
                        setattr(self, model_attr, m)
                        model = m
                    except Exception as e:
                        print(
                            f"[GFPGANTorch] Failed to load {label} custom kernel: {e}"
                        )
                        return None
                # Build CUDA graph runner
                try:
                    from custom_kernels.gfpgan_v1_4.gfpgan_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        compile_1024 = os.environ.get(
                            "GFPGAN1024_TORCH_COMPILE", ""
                        ).strip().lower() in ("1", "true", "yes", "on")
                        use_compile = (not is_1024) or compile_1024
                        r = build_cuda_graph_runner(
                            model,
                            inp_shape=(1, 3, 512, 512),
                            torch_compile=use_compile,
                        )
                    setattr(self, runner_attr, r)
                    runner = r
                except Exception as e:
                    print(
                        f"[GFPGANTorch] CUDA graph build failed for {label}, "
                        f"using direct inference: {e}"
                    )
                    setattr(self, runner_attr, model)
                    runner = model
        finally:
            self.models_processor.hide_build_dialog.emit()
        return runner

    def _get_gpen_runner(self, size: int, variant: str = "std"):
        """Lazily load GPENTorch + CUDA graph runner for GPEN-BFR-{size} (optional HF FP16 file)."""
        cache_key = (size, variant)
        if cache_key in self._gpen_runner:
            return self._gpen_runner.get(cache_key)
        label = f"{size}" + (" HF-FP16" if variant == "fp16hf" else "")
        _dlg_msg = (
            f"Loading GPEN-BFR-{label} (direct inference)…"
            if variant == "fp16hf"
            else f"Capturing CUDA graph for GPEN-BFR-{label}…"
        )
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            _dlg_msg,
        )
        try:
            with self._custom_init_lock:
                if cache_key in self._gpen_runner:
                    return self._gpen_runner.get(cache_key)
                model = self._gpen_torch.get(cache_key)
                if model is None:
                    try:
                        from custom_kernels.gpen_bfr.gpen_torch import GPENTorch
                        import pathlib

                        if size == 256 and variant == "fp16hf":
                            basename = "GPEN-BFR-256.fp16.onnx"
                        else:
                            basename = f"GPEN-BFR-{size}.onnx"
                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / basename
                        )
                        print(f"[GPENTorch] Loading {basename}…")
                        model = (
                            GPENTorch.from_onnx(onnx_path, compute_dtype=torch.float16)
                            .cuda()
                            .eval()
                        )
                        self._gpen_torch[cache_key] = model
                    except Exception as e:
                        print(f"[GPENTorch] Failed to load GPEN-BFR-{label}: {e}")
                        self._gpen_torch[cache_key] = None
                        self._gpen_runner[cache_key] = None
                        return None
                # FP16 HF weights: CUDA graph capture has been observed to hang or freeze
                # the app on some Windows + driver stacks; direct GPENTorch.__call__ is stable.
                if variant == "fp16hf":
                    print(
                        "[GPENTorch] FP16 HF: direct inference (CUDA graph capture skipped)."
                    )
                    self._gpen_runner[cache_key] = model
                else:
                    try:
                        from custom_kernels.gpen_bfr.gpen_torch import (
                            build_cuda_graph_runner,
                        )

                        inp_hw = model.in_size  # type: ignore[attr-defined]
                        with self.models_processor.cuda_graph_capture_lock:
                            runner = build_cuda_graph_runner(
                                model,  # type: ignore[arg-type]
                                inp_shape=(1, 3, inp_hw, inp_hw),
                            )
                        self._gpen_runner[cache_key] = runner
                    except Exception as e:
                        print(
                            f"[GPENTorch] CUDA graph build failed for GPEN-{label}, using direct inference: {e}"
                        )
                        self._gpen_runner[cache_key] = model  # fallback: direct model call
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._gpen_runner.get(cache_key)

    def _run_gfpgan_custom(
        self,
        image: torch.Tensor,
        output: torch.Tensor,
        *,
        is_1024: bool = False,
    ) -> bool:
        """Run GFPGAN via GFPGANTorch + CUDA graph (CUDA devices)."""
        runner = self._get_gfpgan_runner(is_1024=is_1024)
        if runner is None:
            return False
        with torch.no_grad():
            with self._custom_inference_lock:
                result = runner(image)
        output.copy_(result)
        return True

    def _prefer_custom_gpen(self) -> bool:
        """Whether GPEN should run via the custom torch CUDA-graph kernel.

        GPEN is in ``ONNX_MODELS_SKIP_TENSORRT_EP`` — TensorRT cannot run its graph,
        so under any GPU provider ORT falls back to the **CUDA EP**, which leaks
        ~0.5-1GB of intermediate activations on EVERY inference (confirmed via
        VRAM tracing) until the device OOMs; only destroying the session reclaims it.
        The custom torch CUDA-graph kernel is leak-free (torch-managed, reused
        memory) and faster, so prefer it on CUDA. Set VISIOMASTER_GPEN_FORCE_ORT=1
        to force the legacy (leaky) ORT path.
        """
        if self.models_processor.provider_name == "Custom":
            return True
        if os.environ.get("VISIOMASTER_GPEN_FORCE_ORT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return False
        return self.models_processor.device == "cuda" and torch.cuda.is_available()

    def _run_gpen_custom(
        self,
        size: int,
        image: torch.Tensor,
        output: torch.Tensor,
        variant: str = "std",
    ):
        """Run GPEN via Custom provider (GPENTorch + CUDA graph)."""
        runner = self._get_gpen_runner(size, variant)
        if runner is None:
            return False
        with torch.no_grad():
            with self._custom_inference_lock:
                result = runner(image)
        output.copy_(result)
        return True

    def _get_ref_ldm_encoder_runner(self):
        """Lazily load RefLDMEncoderTorch + CUDA graph runner."""
        if self._ref_ldm_encoder_runner is not None:
            return self._ref_ldm_encoder_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for RefLDM VAE Encoder…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._ref_ldm_encoder_runner is not None:
                    return self._ref_ldm_encoder_runner
                if self._ref_ldm_encoder_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.ref_ldm.ref_ldm_torch import (
                            RefLDMEncoderTorch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "ref_ldm_vae_encoder.onnx"
                        )
                        print("[RefLDMTorch] Loading VAE encoder...")
                        m = RefLDMEncoderTorch.from_onnx(onnx_path).cuda().eval()
                        self._ref_ldm_encoder_torch = m
                    except Exception as e:
                        print(f"[RefLDMTorch] Failed to load VAE encoder: {e}")
                        return None
                try:
                    from custom_kernels.ref_ldm.ref_ldm_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        runner = build_cuda_graph_runner(
                            self._ref_ldm_encoder_torch,
                            inp_shape=(1, 3, 512, 512),
                            torch_compile=True,
                        )
                    self._ref_ldm_encoder_runner = runner
                except Exception as e:
                    print(
                        f"[RefLDMTorch] CUDA graph build failed for encoder, using direct inference: {e}"
                    )
                    self._ref_ldm_encoder_runner = self._ref_ldm_encoder_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._ref_ldm_encoder_runner

    def _get_ref_ldm_decoder_runner(self):
        """Lazily load RefLDMDecoderTorch + CUDA graph runner."""
        if self._ref_ldm_decoder_runner is not None:
            return self._ref_ldm_decoder_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for RefLDM VAE Decoder…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._ref_ldm_decoder_runner is not None:
                    return self._ref_ldm_decoder_runner
                if self._ref_ldm_decoder_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.ref_ldm.ref_ldm_torch import (
                            RefLDMDecoderTorch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "ref_ldm_vae_decoder.onnx"
                        )
                        print("[RefLDMTorch] Loading VAE decoder...")
                        m = RefLDMDecoderTorch.from_onnx(onnx_path).cuda().eval()
                        self._ref_ldm_decoder_torch = m
                    except Exception as e:
                        print(f"[RefLDMTorch] Failed to load VAE decoder: {e}")
                        return None
                try:
                    from custom_kernels.ref_ldm.ref_ldm_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        runner = build_cuda_graph_runner(
                            self._ref_ldm_decoder_torch,
                            inp_shape=(1, 8, 64, 64),
                            torch_compile=True,
                        )
                    self._ref_ldm_decoder_runner = runner
                except Exception as e:
                    print(
                        f"[RefLDMTorch] CUDA graph build failed for decoder, using direct inference: {e}"
                    )
                    self._ref_ldm_decoder_runner = self._ref_ldm_decoder_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._ref_ldm_decoder_runner

    def _get_ref_ldm_unet_torch(self):
        """Lazily load RefLDMUNetTorch."""
        if self._ref_ldm_unet_torch is not None:
            return self._ref_ldm_unet_torch
        with self._custom_init_lock:
            if self._ref_ldm_unet_torch is None:
                try:
                    import pathlib
                    from custom_kernels.ref_ldm.ref_ldm_torch import RefLDMUNetTorch

                    onnx_path = str(
                        pathlib.Path(__file__).parent.parent.parent
                        / "model_assets"
                        / "ref_ldm_unet_external_kv.onnx"
                    )
                    print("[RefLDMTorch] Loading UNet...")
                    m = RefLDMUNetTorch.from_onnx(onnx_path).cuda().eval()
                    self._ref_ldm_unet_torch = m
                except Exception as e:
                    print(f"[RefLDMTorch] Failed to load UNet: {e}")
        return self._ref_ldm_unet_torch

    def _get_restoreformer_runner(self):
        """Lazily load RestoreFormerPlusPlusTorch + CUDA graph runner."""
        if self._restoreformer_runner is not None:
            return self._restoreformer_runner
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Compiling & capturing CUDA graph for RestoreFormer++…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                if self._restoreformer_runner is not None:
                    return self._restoreformer_runner
                if self._restoreformer_torch is None:
                    try:
                        import pathlib
                        from custom_kernels.restoreformer.restoreformer_torch import (
                            RestoreFormerPlusPlusTorch,
                        )

                        onnx_path = str(
                            pathlib.Path(__file__).parent.parent.parent
                            / "model_assets"
                            / "RestoreFormerPlusPlus.fp16.onnx"
                        )
                        print("[RFP++Torch] Loading RestoreFormerPlusPlus model...")
                        m = (
                            RestoreFormerPlusPlusTorch.from_onnx(onnx_path)
                            .to(self.models_processor.get_effective_torch_device())
                            .eval()
                        )
                        self._restoreformer_torch = m
                    except Exception as e:
                        print(f"[RFP++Torch] Failed to load model: {e}")
                        return None
                try:
                    from custom_kernels.restoreformer.restoreformer_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        runner = build_cuda_graph_runner(
                            self._restoreformer_torch,
                            inp_shape=(1, 3, 512, 512),
                            torch_compile=True,
                        )
                    self._restoreformer_runner = runner
                except Exception as e:
                    print(
                        f"[RFP++Torch] CUDA graph build failed, using direct inference: {e}"
                    )
                    self._restoreformer_runner = self._restoreformer_torch
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._restoreformer_runner

    def _get_codeformer_torch(self):
        """Lazily load CodeFormerTorch (FP16 PyTorch kernel)."""
        if self._codeformer_torch is not None:
            return self._codeformer_torch
        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            "Loading CodeFormer model (Custom provider)…\nThis only happens once.",
        )
        try:
            with self._custom_init_lock:
                if self._codeformer_torch is not None:
                    return self._codeformer_torch
                try:
                    import pathlib
                    from custom_kernels.codeformer.codeformer_torch import (
                        CodeFormerTorch,
                    )

                    onnx_path = str(
                        pathlib.Path(__file__).parent.parent.parent
                        / "model_assets"
                        / "codeformer_fp16.onnx"
                    )
                    self._codeformer_torch = (
                        CodeFormerTorch.from_onnx(onnx_path)
                        .to(self.models_processor.get_effective_torch_device())
                        .eval()
                    )
                except Exception as e:
                    print(f"[Custom] CodeFormerTorch load failed: {e}")
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._codeformer_torch

    def _get_codeformer_runner(self, model, fidelity_weight: float):
        """
        Return a CUDA-graph runner for the given fidelity_weight.

        The fidelity weight is baked into the graph at capture time.
        If the weight changes, the old runner is invalidated and a new one is
        built on the next call (one direct-inference frame in between).
        """
        w = round(float(fidelity_weight), 3)  # quantise to 3dp to reduce rebuilds
        if (
            self._codeformer_runner is not None
            and self._codeformer_runner_w is not None
            and abs(self._codeformer_runner_w - w) < 0.005
        ):
            return self._codeformer_runner

        # Invalidate stale runner — caller will use direct inference this frame.
        self._codeformer_runner = None
        self._codeformer_runner_w = None

        self.models_processor.show_build_dialog.emit(
            "Finalizing Custom Provider",
            f"Compiling & capturing CUDA graph for CodeFormer (w={w:.2f})…\nFirst run only — future sessions load instantly from cache.",
        )
        try:
            with self._custom_init_lock:
                # Re-check under lock in case another thread just built it.
                if (
                    self._codeformer_runner is not None
                    and self._codeformer_runner_w is not None
                    and abs(self._codeformer_runner_w - w) < 0.005
                ):
                    return self._codeformer_runner
                try:
                    from custom_kernels.codeformer.codeformer_torch import (
                        build_cuda_graph_runner,
                    )

                    with self.models_processor.cuda_graph_capture_lock:
                        # FP16 + Triton + CUDA graph (Tier 3). torch.compile (Tier 6) is
                        # skipped: Inductor subprocess often fails on Windows; it only
                        # added ~15% speed when it worked and blocked first-run for minutes.
                        runner = build_cuda_graph_runner(
                            model,
                            inp_shape=(1, 3, 512, 512),
                            fidelity_weight=w,
                            torch_compile=False,
                        )
                    self._codeformer_runner = runner
                    self._codeformer_runner_w = w
                except Exception as e:
                    print(f"[Custom] CodeFormer CUDA graph build failed: {e}")
                    self._codeformer_runner = _CodeFormerDirectRunner(model, w)
                    self._codeformer_runner_w = w
        finally:
            self.models_processor.hide_build_dialog.emit()
        return self._codeformer_runner

    def _run_model_with_lazy_build_check(
        self, model_name: str, ort_session, io_binding
    ):
        """
        Runs the ONNX session with IOBinding, handling TensorRT lazy build dialogs.
        This centralizes the try/finally logic for showing/hiding the build progress dialog
        and synchronizes CUDA before inference so PyTorch-prepared input buffers are
        visible to ORT. Set ``VISIOMASTER_ORT_IOBINDING_POST_SYNC=1`` to restore the
        old post-sync if needed.

        Args:
            model_name (str): The name of the model being run.
            ort_session: The ONNX Runtime session instance.
            io_binding: The pre-configured IOBinding object.

        Returns:
            list: The network outputs from copy_outputs_to_cpu().
        """
        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        net_outs: list = []
        try:
            if self.models_processor.uses_cuda_ep_for_thread():
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()

            self.models_processor.run_session_with_iobinding(ort_session, io_binding)

            if os.environ.get("VISIOMASTER_ORT_IOBINDING_POST_SYNC", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                if self.models_processor.uses_cuda_ep_for_thread():
                    torch.cuda.current_stream().synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()

            net_outs = io_binding.copy_outputs_to_cpu()

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

        return net_outs

    @torch.inference_mode()
    def apply_facerestorer(
        self,
        swapped_face_upscaled,
        restorer_det_type,
        restorer_type,
        restorer_blend,
        fidelity_weight,
        detect_score,
        target_kps=None,
        slot_id: int = 1,
        dmd_landmarks_68_crop: Optional[np.ndarray] = None,
    ):
        model_name_to_load = self.model_map.get(restorer_type)
        if not model_name_to_load:
            return swapped_face_upscaled

        if restorer_type in ("DMDNet", "DMDNet FP16") and self.models_processor.device != "cuda":
            return swapped_face_upscaled

        # If using a separate detection mode
        if restorer_det_type in ["Blend", "Reference"]:
            if restorer_det_type == "Blend":
                # Set up Transformation
                dst = self.models_processor.arcface_dst * 4.0
                dst[:, 0] += 32.0

            elif restorer_det_type == "Reference":
                # Instead of re-detecting landmarks, use the target_kps passed to the function.
                if target_kps is None or len(target_kps) == 0:
                    print(
                        "[WARN] 'Reference' alignment selected, but no target landmarks (target_kps) were provided. Skipping restoration."
                    )
                    return swapped_face_upscaled
                dst = target_kps

            try:
                # Use from_estimate constructor instead of .estimate()
                if hasattr(trans.SimilarityTransform, "from_estimate"):
                    tform = trans.SimilarityTransform.from_estimate(
                        dst, self.models_processor.FFHQ_kps
                    )
                else:
                    tform = trans.SimilarityTransform()
                    tform.estimate(dst, self.models_processor.FFHQ_kps)
            except Exception:
                return swapped_face_upscaled

            # OPTIMIZED: Direct GPU Affine Warp with Kornia, skipping torchvision crop/affine
            M_tensor = (
                torch.from_numpy(tform.params[0:2])
                .float()
                .unsqueeze(0)
                .to(swapped_face_upscaled.device)
            )
            img_b = (
                swapped_face_upscaled.unsqueeze(0)
                if swapped_face_upscaled.dim() == 3
                else swapped_face_upscaled
            )

            # Kornia allocates a new tensor here, so we own this memory space.
            temp = kgm.warp_affine(
                img_b.float(),
                M_tensor,
                dsize=(512, 512),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            # Safe to perform math operations since 'temp' is a brand new tensor
            temp = temp.float() / 255.0

        else:
            # If we did not warp the image, we MUST clone the original tensor
            # before applying division. Using .div_(255.0) on the original reference corrupts
            # memory for other threads (Race Condition).
            temp = swapped_face_upscaled.clone().float() / 255.0

        # Now safe to use inplace normalization as we definitely own the 'temp' memory footprint
        temp = v2.functional.normalize(
            temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
        )

        if restorer_type in (
            "GPEN-256 Fast (128→256)",
            "GPEN-256 Fast FP16 (128→256)",
        ):
            # Low-res bottleneck before the fixed 256×256 GPEN ONNX (preview / small-face path).
            temp = v2.functional.resize(temp, [128, 128], antialias=True)
            temp = v2.functional.resize(temp, [256, 256], antialias=True)
        elif restorer_type in ("GPEN-256", "GPEN-256 FP16 (HF)"):
            temp = v2.functional.resize(temp, [256, 256], antialias=False)

        temp = torch.unsqueeze(temp, 0).contiguous()

        # Bindings
        # FR-ROBUST-04: removed default 512x512 pre-allocation; each branch allocates at correct size
        outpred = None

        dev = self.models_processor.get_effective_torch_device()

        if restorer_type == "GFPGAN-v1.4":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=self._gfpgan_output_dtype(is_1024=False),
                device=dev,
            ).contiguous()
            self.run_GFPGAN(temp, outpred)

        elif restorer_type == "GFPGAN-1024":
            outpred = torch.empty(
                (1, 3, 1024, 1024),
                dtype=self._gfpgan_output_dtype(is_1024=True),
                device=dev,
            ).contiguous()
            self.run_GFPGAN1024(temp, outpred)

        elif restorer_type == "CodeFormer":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=self._ort_output_dtype("CodeFormer", "y"),
                device=dev,
            ).contiguous()
            self.run_codeformer(temp, outpred, fidelity_weight)

        elif restorer_type == "GPEN-256":
            outpred = torch.empty(
                (1, 3, 256, 256),
                dtype=self._ort_output_dtype("GPENBFR256", "output"),
                device=dev,
            ).contiguous()
            self.run_GPEN_256(temp, outpred)

        elif restorer_type == "GPEN-256 FP16 (HF)":
            outpred = torch.empty(
                (1, 3, 256, 256),
                dtype=self._ort_output_dtype("GPENBFR256FP16", "output"),
                device=dev,
            ).contiguous()
            self.run_GPEN_256(
                temp, outpred, model_name="GPENBFR256FP16", gpen_variant="fp16hf"
            )

        elif restorer_type == "GPEN-256 Fast (128→256)":
            outpred = torch.empty(
                (1, 3, 256, 256),
                dtype=self._ort_output_dtype("GPENBFR256", "output"),
                device=dev,
            ).contiguous()
            self.run_GPEN_256(temp, outpred)

        elif restorer_type == "GPEN-256 Fast FP16 (128→256)":
            outpred = torch.empty(
                (1, 3, 256, 256),
                dtype=self._ort_output_dtype("GPENBFR256FP16", "output"),
                device=dev,
            ).contiguous()
            self.run_GPEN_256(
                temp, outpred, model_name="GPENBFR256FP16", gpen_variant="fp16hf"
            )

        elif restorer_type == "GPEN-512":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=self._ort_output_dtype("GPENBFR512", "output"),
                device=dev,
            ).contiguous()
            self.run_GPEN_512(temp, outpred)

        elif restorer_type == "GPEN-1024":
            temp = v2.functional.resize(temp, [1024, 1024], antialias=False)
            outpred = torch.empty(
                (1, 3, 1024, 1024),
                dtype=self._ort_output_dtype("GPENBFR1024", "output"),
                device=dev,
            ).contiguous()
            self.run_GPEN_1024(temp, outpred)

        elif restorer_type == "GPEN-2048":
            temp = v2.functional.resize(temp, [2048, 2048], antialias=False)
            outpred = torch.empty(
                (1, 3, 2048, 2048),
                dtype=self._ort_output_dtype("GPENBFR2048", "output"),
                device=dev,
            ).contiguous()
            self.run_GPEN_2048(temp, outpred)

        elif restorer_type == "RestoreFormer++":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=self._ort_output_dtype("RestoreFormerPlusPlus", "2359"),
                device=dev,
            ).contiguous()
            self.run_RestoreFormerPlusPlus(temp, outpred)

        elif restorer_type == "RestoreFormer":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=self._ort_output_dtype("RestoreFormerFP16", "output"),
                device=dev,
            ).contiguous()
            self.run_RestoreFormer(temp, outpred)

        elif restorer_type == "VQFR-v2":
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=self._ort_output_dtype("VQFRv2", "main_dec"),
                device=dev,
            ).contiguous()
            self.run_VQFR_v2(temp, outpred, fidelity_weight)

        elif restorer_type in ("DMDNet", "DMDNet FP16"):
            outpred = torch.empty(
                (1, 3, 512, 512),
                dtype=torch.float32,
                device=dev,
            ).contiguous()
            if dmd_landmarks_68_crop is None or np.asarray(dmd_landmarks_68_crop).size < 136:
                if "DMDNetLm" not in self._warned_models:
                    print(
                        "[WARN] DMDNet: need target landmarks (106-point mode recommended). "
                        "Skipping restoration for this face."
                    )
                    self._warned_models.add("DMDNetLm")
                return swapped_face_upscaled
            lm68 = np.asarray(dmd_landmarks_68_crop, dtype=np.float32).reshape(68, 2)
            if restorer_det_type in ("Blend", "Reference"):
                lm68 = np.array(tform(lm68), dtype=np.float32)
            _dmd_amp = restorer_type == "DMDNet FP16"
            if not self.run_dmdnet(
                temp, lm68, outpred, use_half_autocast=_dmd_amp
            ):
                return swapped_face_upscaled

        if outpred is None:
            return swapped_face_upscaled

        # Post-ONNX path (warp/resize) expects float32 activations.
        outpred = outpred.float()

        # OPTIMIZED: Fused in-place math operations to save VRAM allocations.
        # Math: ((x clamped [-1, 1]) + 1.0) * 127.5 is equivalent to /2 * 255.
        outpred = outpred.squeeze(0).clamp_(-1.0, 1.0).add_(1.0).mul_(127.5)

        if restorer_type in [
            "GPEN-256",
            "GPEN-256 FP16 (HF)",
            "GPEN-256 Fast (128→256)",
            "GPEN-256 Fast FP16 (128→256)",
            "GPEN-1024",
            "GPEN-2048",
            "GFPGAN-1024",
        ]:
            outpred = v2.functional.resize(outpred, [512, 512], antialias=True)

        # Invert Transform
        if restorer_det_type in ["Blend", "Reference"]:
            # OPTIMIZED: Direct Inverse GPU Affine Warp with Kornia
            M_inv_tensor = (
                torch.from_numpy(tform.inverse.params[0:2])
                .float()
                .unsqueeze(0)
                .to(outpred.device)
            )
            out_b = outpred.unsqueeze(0) if outpred.dim() == 3 else outpred
            dsize = (swapped_face_upscaled.shape[1], swapped_face_upscaled.shape[2])

            outpred = kgm.warp_affine(
                out_b,
                M_inv_tensor,
                dsize=(dsize[0], dsize[1]),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)

        # Blend (Disabled by default as in original code)
        # alpha = float(restorer_blend)/100.0
        # outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred

    def run_vae_encoder(
        self, image_input_tensor: torch.Tensor, output_latent_tensor: torch.Tensor
    ):
        """
        Runs the VAE encoder model.
        image_input_tensor: Batch x 3 x Height x Width, float32, normalized to [-1, 1]
        output_latent_tensor: Placeholder for Batch x 8 x LatentH x LatentW, float32
        """
        model_name = "RefLDMVAEEncoder"
        # FR-BUG-04: use .get() to avoid KeyError when model is not yet loaded
        ort_session = self.models_processor.get_onnx_session(model_name)
        if ort_session is None:
            # Lazy reload in case clear_gpu_memory() cleared the session after a provider switch.
            self.models_processor.ensure_denoiser_models_loaded()
            ort_session = self.models_processor.get_onnx_session(model_name)
        if ort_session is None:
            error_msg = f"[ERROR] VAE Encoder model '{model_name}' not loaded when run_vae_encoder was called. This model should be loaded by ModelsProcessor.ensure_denoiser_models_loaded()."
            print(error_msg)
            raise RuntimeError(error_msg)

        input_name = (
            ort_session.get_inputs()[0].name
            if ort_session.get_inputs()
            else "image_input"
        )
        output_name = (
            ort_session.get_outputs()[0].name
            if ort_session.get_outputs()
            else "latent_pre_quant_unscaled"
        )

        io_binding = ort_session.io_binding()
        image_input_tensor = self.models_processor.bind_ort_io_input(
            io_binding, model_name, input_name, image_input_tensor
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, output_name, output_latent_tensor
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_vae_decoder(
        self, latent_input_tensor: torch.Tensor, output_image_tensor: torch.Tensor
    ):
        """
        Runs the VAE decoder model.
        latent_input_tensor: Batch x 8 x LatentH x LatentW, float32
        output_image_tensor: Placeholder for Batch x 3 x H x W, float32, normalized to [-1, 1]
        """
        model_name = "RefLDMVAEDecoder"
        # FR-BUG-04: use .get() to avoid KeyError when model is not yet loaded
        ort_session = self.models_processor.get_onnx_session(model_name)
        if ort_session is None:
            # Lazy reload in case clear_gpu_memory() cleared the session after a provider switch.
            self.models_processor.ensure_denoiser_models_loaded()
            ort_session = self.models_processor.get_onnx_session(model_name)
        if ort_session is None:
            error_msg = f"[ERROR] VAE Decoder model '{model_name}' not loaded when run_vae_decoder was called. This model should be loaded by ModelsProcessor.ensure_denoiser_models_loaded()."
            print(error_msg)
            raise RuntimeError(error_msg)

        input_name = (
            ort_session.get_inputs()[0].name
            if ort_session.get_inputs()
            else "scaled_latent_input"
        )
        output_name = (
            ort_session.get_outputs()[0].name
            if ort_session.get_outputs()
            else "image_output"
        )

        io_binding = ort_session.io_binding()
        latent_input_tensor = self.models_processor.bind_ort_io_input(
            io_binding, model_name, input_name, latent_input_tensor
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, output_name, output_image_tensor
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_ref_ldm_unet(
        self,
        x_noisy_plus_lq_latent: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        is_ref_flag_tensor: torch.Tensor,
        use_reference_exclusive_path_globally_tensor: torch.Tensor,
        kv_tensor_map: Optional[Dict[str, Dict[str, torch.Tensor]]],
        output_unet_tensor: torch.Tensor,
    ):
        """
        Runs the UNet denoiser model with external K/V inputs.
        """
        model_name = self.models_processor.main_window.fixed_unet_model_name
        ort_session = self.models_processor.get_onnx_session(model_name)

        if not ort_session:
            # Enhanced error reporting
            error_messages = [
                f"[ERROR] UNet model '{model_name}' not loaded when run_ref_ldm_unet was called.",
                "  This model should be loaded by ModelsProcessor.apply_denoiser_unet or a similar setup routine.",
            ]
            print("\n".join(error_messages))
            return

        onnx_output_name = "unet_output"

        io_binding = ort_session.io_binding()
        bind_device_type = self.models_processor.get_ort_bind_device_type()
        bind_device_id = self.models_processor.get_ort_bind_input_cuda_device_id()

        x_noisy_plus_lq_latent = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "x_noisy_plus_lq_latent",
            x_noisy_plus_lq_latent,
            device_type=bind_device_type,
            device_id=bind_device_id,
        )
        timesteps_tensor = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "timesteps",
            timesteps_tensor,
            device_type=bind_device_type,
            device_id=bind_device_id,
        )
        is_ref_flag_tensor = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "is_ref_flag_input",
            is_ref_flag_tensor,
            device_type=bind_device_type,
            device_id=bind_device_id,
        )
        use_reference_exclusive_path_globally_tensor = (
            self.models_processor.bind_ort_io_input(
                io_binding,
                model_name,
                "use_reference_exclusive_path_globally_input",
                use_reference_exclusive_path_globally_tensor,
                device_type=bind_device_type,
                device_id=bind_device_id,
            )
        )

        onnx_model_inputs = ort_session.get_inputs()
        onnx_kv_input_names_to_shape: Dict[str, tuple] = {
            inp.name: tuple(
                dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape
            )
            for inp in onnx_model_inputs
            if inp.name.endswith("_k_ext") or inp.name.endswith("_v_ext")
        }

        actual_kv_tensors_for_binding: Dict[str, torch.Tensor] = {}
        if kv_tensor_map:
            for pt_module_name, kv_pair in kv_tensor_map.items():
                onnx_base_name = pt_module_name.replace(".", "_")
                k_name_onnx = f"{onnx_base_name}_k_ext"
                v_name_onnx = f"{onnx_base_name}_v_ext"

                k_tensor_original = kv_pair.get("k")
                v_tensor_original = kv_pair.get("v")

                if (
                    k_tensor_original is not None
                    and k_name_onnx in onnx_kv_input_names_to_shape
                ):
                    td_k = self.models_processor.get_ort_io_torch_dtype(
                        model_name, k_name_onnx, is_output=False
                    )
                    actual_kv_tensors_for_binding[k_name_onnx] = (
                        k_tensor_original.unsqueeze(0)
                        .to(
                            device=self.models_processor.get_effective_torch_device(),
                            dtype=td_k,
                        )
                        .contiguous()
                    )

                if (
                    v_tensor_original is not None
                    and v_name_onnx in onnx_kv_input_names_to_shape
                ):
                    td_v = self.models_processor.get_ort_io_torch_dtype(
                        model_name, v_name_onnx, is_output=False
                    )
                    actual_kv_tensors_for_binding[v_name_onnx] = (
                        v_tensor_original.unsqueeze(0)
                        .to(
                            device=self.models_processor.get_effective_torch_device(),
                            dtype=td_v,
                        )
                        .contiguous()
                    )

        # IMPORTANT: Keep references to temporary zero tensors to prevent GC
        keep_alive_tensors: list = []
        # FS-MEM-01: also keep actual KV tensors alive to prevent premature GC
        keep_alive_tensors.extend(actual_kv_tensors_for_binding.values())

        for onnx_kv_name, expected_shape in onnx_kv_input_names_to_shape.items():
            tensor_to_bind = actual_kv_tensors_for_binding.get(onnx_kv_name)

            if tensor_to_bind is None:
                td_z = self.models_processor.get_ort_io_torch_dtype(
                    model_name, onnx_kv_name, is_output=False
                )
                tensor_to_bind = torch.zeros(
                    expected_shape,
                    dtype=td_z,
                    device=self.models_processor.get_effective_torch_device(),
                ).contiguous()
                # We MUST store this tensor in a list that persists for the function scope
                # Otherwise, it might be garbage collected before .run() is called
                keep_alive_tensors.append(tensor_to_bind)

            tensor_to_bind = self.models_processor.bind_ort_io_input(
                io_binding,
                model_name,
                onnx_kv_name,
                tensor_to_bind,
                device_type=bind_device_type,
                device_id=bind_device_id,
            )

        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            onnx_output_name,
            output_unet_tensor,
            device_type=bind_device_type,
            device_id=bind_device_id,
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def restorer_ort_batched_attempt_enabled(self) -> bool:
        """PERF-008: mirror VISIOMASTER_INSWAPPER_ORT_BATCH semantics for restorer B>1."""
        if self.models_processor.provider_name == "Custom":
            return False
        if self._restorer_ort_batch_session_disabled:
            return False
        v_raw = os.environ.get("VISIOMASTER_RESTORER_ORT_BATCH", "").strip()
        vl = v_raw.lower()
        if vl in ("0", "false", "no", "off"):
            return False
        if self.models_processor.provider_name == "TensorRT-Engine":
            if vl not in ("1", "true", "yes", "on"):
                return False
        return True

    @torch.inference_mode()
    def try_apply_facerestorer_batched_original_stack(
        self,
        swaps_bchw_0_255: torch.Tensor,
        restorer_effective_type: str,
        fidelity_weight: float,
    ) -> torch.Tensor | None:
        """
        One ORT forward for B>=2 faces (``FaceRestorerDetTypeSelection`` == Original only).
        Input: ``(B,3,H,W)`` float in ~0..255; output same spatial size per row.
        """
        if not self.restorer_ort_batched_attempt_enabled():
            return None
        if swaps_bchw_0_255.dim() != 4 or swaps_bchw_0_255.shape[1] != 3:
            return None
        b = int(swaps_bchw_0_255.shape[0])
        if b < 2:
            return None
        if restorer_effective_type not in ("GFPGAN-v1.4", "CodeFormer"):
            return None
        dev = self.models_processor.get_effective_torch_device()
        x = swaps_bchw_0_255.to(device=dev, dtype=torch.float32).contiguous()
        _, _, h0, w0 = x.shape
        temp = (x / 255.0).clamp(0.0, 1.0)
        temp = v2.functional.normalize(
            temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
        )
        try:
            if restorer_effective_type == "GFPGAN-v1.4":
                mn = "GFPGANv1.4"
                if temp.shape[-2] != 512 or temp.shape[-1] != 512:
                    temp = v2.functional.resize(temp, [512, 512], antialias=True)
                outpred = torch.empty(
                    (b, 3, 512, 512),
                    dtype=self._ort_output_dtype(mn, "output"),
                    device=dev,
                ).contiguous()
                self.run_GFPGAN(temp, outpred)
                out = outpred.float().clamp_(-1.0, 1.0).add_(1.0).mul_(127.5)
            else:
                mn = "CodeFormer"
                if temp.shape[-2] != 512 or temp.shape[-1] != 512:
                    temp = v2.functional.resize(temp, [512, 512], antialias=True)
                outpred = torch.empty(
                    (b, 3, 512, 512),
                    dtype=self._ort_output_dtype(mn, "y"),
                    device=dev,
                ).contiguous()
                self.run_codeformer(temp, outpred, float(fidelity_weight))
                out = outpred.float().clamp_(-1.0, 1.0).add_(1.0).mul_(127.5)
        except Exception as e:
            self._restorer_ort_batch_session_disabled = True
            if not self._restorer_ort_batch_fail_logged:
                self._restorer_ort_batch_fail_logged = True
                print(
                    f"[WARN] Batched face restorer ORT failed (B={b}); falling back per-face. "
                    f"First error: {e}. For TensorRT-Engine, set VISIOMASTER_RESTORER_ORT_BATCH=1 "
                    "only if the engine supports dynamic batch.",
                    flush=True,
                )
            return None
        if out.shape[-2] != h0 or out.shape[-1] != w0:
            out = v2.functional.resize(out, [h0, w0], antialias=True)
        return out.contiguous()

    def run_GFPGAN(self, image, output):
        if self._gfpgan_prefers_torch():
            if self._run_gfpgan_custom(image, output, is_1024=False):
                return

        model_name = "GFPGANv1.4"

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip if model failed to load

        io_binding = ort_session.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "input", image
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "output", output
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GFPGAN1024(self, image, output):
        if self._gfpgan_prefers_torch():
            if self._run_gfpgan_custom(image, output, is_1024=True):
                return

        model_name = "GFPGAN1024"

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "input", image
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "output", output
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_256(
        self,
        image,
        output,
        *,
        model_name: str = "GPENBFR256",
        gpen_variant: str = "std",
    ):
        if self._prefer_custom_gpen():
            if self._run_gpen_custom(256, image, output, variant=gpen_variant):
                return

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        td_out = self.models_processor.get_ort_io_torch_dtype(
            model_name, "output", is_output=True
        )
        out_bind = (
            output
            if output.dtype == td_out
            else torch.empty(
                (1, 3, 256, 256), dtype=td_out, device=output.device
            ).contiguous()
        )
        image = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "input", image
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "output", out_bind
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)
        if out_bind is not output:
            output.copy_(out_bind.to(dtype=output.dtype))

    def run_GPEN_512(self, image, output):
        model_name = "GPENBFR512"

        if self._prefer_custom_gpen():
            if self._run_gpen_custom(512, image, output, variant="std"):
                return

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "input", image
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "output", output
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_1024(self, image, output):
        model_name = "GPENBFR1024"

        if self._prefer_custom_gpen():
            if self._run_gpen_custom(1024, image, output, variant="std"):
                return

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "input", image
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "output", output
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_GPEN_2048(self, image, output):
        model_name = "GPENBFR2048"

        if self._prefer_custom_gpen():
            if self._run_gpen_custom(2048, image, output, variant="std"):
                return

        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "input", image
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "output", output
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_codeformer(self, image, output, fidelity_weight_value=0.9):
        if self.models_processor.provider_name == "Custom":
            model = self._get_codeformer_torch()
            if model is not None:
                w = float(fidelity_weight_value)
                dev = next(model.parameters()).device
                image = image.to(device=dev, dtype=torch.float32).contiguous()
                runner = self._get_codeformer_runner(model, w)
                with torch.no_grad():
                    with self._get_runner_lock(model):
                        try:
                            if runner is not None:
                                result = runner(image)
                            else:
                                result = model(image, fidelity_weight=w)
                        except Exception as e:
                            print(
                                f"[Custom] CodeFormer runner failed, using direct "
                                f"inference: {e}"
                            )
                            with self._custom_init_lock:
                                self._codeformer_runner = _CodeFormerDirectRunner(
                                    model, w
                                )
                                self._codeformer_runner_w = w
                            result = model(image, fidelity_weight=w)
                output.copy_(result)
                return

        model_name = "CodeFormer"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "x", image
        )
        w = np.array([fidelity_weight_value], dtype=np.double)
        io_binding.bind_cpu_input("w", w)
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "y", output
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_VQFR_v2(self, image, output, fidelity_ratio_value):
        model_name = "VQFRv2"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        # FR-ROBUST-05: replace assert with an explicit ValueError so it is never silenced by -O flag
        if not (0.0 <= fidelity_ratio_value <= 1.0):
            raise ValueError(
                f"fidelity_ratio_value must be in [0,1], got {fidelity_ratio_value}"
            )
        fidelity_ratio = torch.tensor(fidelity_ratio_value, dtype=torch.float32).to(
            self.models_processor.get_effective_torch_device()
        )

        io_binding = ort_session.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "x_lq", image
        )
        fidelity_ratio = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "fidelity_ratio", fidelity_ratio
        )
        self.models_processor.bind_ort_output_dynamic(io_binding, "enc_feat")
        self.models_processor.bind_ort_output_dynamic(io_binding, "quant_logit")
        self.models_processor.bind_ort_output_dynamic(io_binding, "texture_dec")
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "main_dec", output
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_RestoreFormer(self, image_fp32_nchw, output_fp32_nchw):
        """RestoreFormer base (FP16 ONNX): entrada/salida NCHW normalizada [-1,1]."""
        model_name = "RestoreFormerFP16"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return
        td_in = self.models_processor.get_ort_io_torch_dtype(
            model_name, "input", is_output=False
        )
        td_out = self.models_processor.get_ort_io_torch_dtype(
            model_name, "output", is_output=True
        )
        x_in = image_fp32_nchw.to(dtype=td_in).contiguous()
        out16 = torch.empty(
            (1, 3, 512, 512),
            dtype=td_out,
            device=self.models_processor.get_effective_torch_device(),
        ).contiguous()
        io_binding = ort_session.io_binding()
        x_in = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "input", x_in
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "output", out16
        )
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)
        output_fp32_nchw.copy_(out16.float())

    def run_RestoreFormerPlusPlus(self, image, output):
        model_name = "RestoreFormerPlusPlus"
        ort_session = self._get_model_session(model_name)
        if not ort_session:
            return  # Silently skip

        io_binding = ort_session.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding, model_name, "input", image
        )
        self.models_processor.bind_ort_io_output(
            io_binding, model_name, "2359", output
        )
        self.models_processor.bind_ort_output_dynamic(io_binding, "1228")
        self.models_processor.bind_ort_output_dynamic(io_binding, "1238")
        self.models_processor.bind_ort_output_dynamic(io_binding, "onnx::MatMul_1198")
        self.models_processor.bind_ort_output_dynamic(io_binding, "onnx::Shape_1184")
        self.models_processor.bind_ort_output_dynamic(io_binding, "onnx::ArgMin_1182")
        self.models_processor.bind_ort_output_dynamic(io_binding, "input.1")
        self.models_processor.bind_ort_output_dynamic(io_binding, "x")
        self.models_processor.bind_ort_output_dynamic(io_binding, "x.3")
        self.models_processor.bind_ort_output_dynamic(io_binding, "x.7")
        self.models_processor.bind_ort_output_dynamic(io_binding, "x.11")
        self.models_processor.bind_ort_output_dynamic(io_binding, "x.15")
        self.models_processor.bind_ort_output_dynamic(io_binding, "input.252")
        self.models_processor.bind_ort_output_dynamic(io_binding, "input.280")
        self.models_processor.bind_ort_output_dynamic(io_binding, "input.288")

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)
