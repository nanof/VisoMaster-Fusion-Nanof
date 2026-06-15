"""PerformRecast — expression-only "Recast" backend for the Face Expression Restorer.

PerformRecast (CVPR 2026) transfers the *expression* of a driving face onto a
source face while preserving the source's identity and head pose. Inside
VisoMaster the driving face is the original pre-swap face and the source is the
swapped face, mirroring the role of the LivePortrait-based expression restorer.

It uses four exported ONNX sub-networks (see ``models_data.py``):

    F  PerformRecastAppearanceFeatureExtractor  (B,3,512,512)[0,1] -> feature_3d (B,32,16,64,64)
    M  PerformRecastMotionExtractor             (B,3,256,256)[0,1] -> pitch,yaw,roll,t,exp,scale,kp
    W  PerformRecastWarpingModule               feature_3d,kp_driving,kp_source -> occlusion_map,deformation,out (B,256,64,64)
    G  PerformRecastSpadeGenerator              feature (B,256,64,64) -> out (B,3,512,512)[0,1]

The model uses 49 implicit keypoints (NUM_KP = 49) — not LivePortrait's 21 — and
a head-pose binning convention of ``softmax * 3 - 99`` (HopeNet), which differs
from LivePortrait's ``- 97.5``. The keypoint transform and the two composition
modes are ported verbatim from the reference ``src/pipeline.py`` /
``scripts/onnx_inference.py`` so behaviour matches the upstream model.

W and G have no fp16 kernel (5-D grid_sample / SPADE), so they always run fp32;
all graph I/O is float32 regardless of internal precision.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

# Implicit keypoint count of the PerformRecast motion model
# (performrecast_models.yaml -> common_params.num_kp).
NUM_KP = 49

# ONNX model registry keys for the four sub-networks.
APPEARANCE_MODEL = "PerformRecastAppearanceFeatureExtractor"
MOTION_MODEL = "PerformRecastMotionExtractor"
WARPING_MODEL = "PerformRecastWarpingModule"
SPADE_MODEL = "PerformRecastSpadeGenerator"

# Mapping of the user-facing mode label -> upstream inference_mode integer.
MODE_REPLACEMENT = "Replacement"  # inference_mode 1
MODE_ENHANCEMENT = "Enhancement"  # inference_mode 2


class PerformRecast:
    """onnxruntime-backed runner for the four PerformRecast sub-networks.

    Mirrors the structure of :class:`app.processors.face_editors.FaceEditors`:
    models are lazily loaded through ``models_processor.load_model`` on first
    use and can be released together via :meth:`unload_models`.
    """

    # All four ONNX models, grouped so the UI can load/unload them together.
    model_group = [APPEARANCE_MODEL, MOTION_MODEL, WARPING_MODEL, SPADE_MODEL]

    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        # Single, app-lifetime worker thread that owns *every* PerformRecast ORT
        # inference call (see ``_run``). The four sub-networks run on the CUDA EP
        # (W/G have no TensorRT-friendly kernels), and the CUDA EP allocates a
        # per-thread execution context (cuDNN/cuBLAS handles, compute stream and
        # staging/workspace buffers) for each thread that calls ``Run``. The
        # video pipeline spawns fresh worker threads on every play and a fresh
        # single-frame worker on every slider reprocess, and ORT does not reuse
        # those per-thread contexts once the threads die — so VRAM grew by
        # several GB per play/reprocess until it OOMed. Funnelling all inference
        # through one persistent thread means ORT ever sees a single thread, so
        # exactly one per-thread context is created for the whole app lifetime.
        self._ort_executor: "ThreadPoolExecutor | None" = None
        self._ort_executor_lock = threading.Lock()

    def _get_ort_executor(self) -> "ThreadPoolExecutor":
        if self._ort_executor is None:
            with self._ort_executor_lock:
                if self._ort_executor is None:
                    self._ort_executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="PerformRecast-ORT"
                    )
        return self._ort_executor

    def shutdown_executor(self) -> None:
        """Tear down the inference thread (frees its ORT per-thread context)."""
        with self._ort_executor_lock:
            ex = self._ort_executor
            self._ort_executor = None
        if ex is not None:
            ex.shutdown(wait=True)

    # ------------------------------------------------------------------ #
    # Low-level ONNX runner
    # ------------------------------------------------------------------ #
    def _session(self, model_name: str):
        """Return a loaded InferenceSession for ``model_name`` (lazy load).

        Always routes through ``load_model`` (thread-safe + refreshes the
        model's last-used timestamp) instead of reading ``models`` directly,
        so the idle-evictor never treats an actively-used PerformRecast model
        as stale and unloads/reloads it (which would churn ORT arenas).
        """
        session = self.models_processor.load_model(model_name)
        if session is None:
            raise RuntimeError(f"Model {model_name} could not be loaded")
        return session

    def _run(self, model_name: str, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run a model and return ``{output_name: np.ndarray}``.

        The exported graphs always take/return float32 (``keep_io_types=True``
        for the fp16 variants), so every feed is coerced to contiguous float32.
        Feeds and results cross the thread boundary as numpy (host) arrays, so
        there are no cross-thread CUDA tensor hazards: all device work happens
        on the single inference thread (see ``__init__``).

        TensorRT "lazy build" models compile their engine on the first real
        inference (not during the isolated probe). We surface the build dialog
        around that first run so the UI does not appear frozen.
        """
        session = self._session(model_name)
        feeds = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in feeds.items()}
        names = [o.name for o in session.get_outputs()]

        mp = self.models_processor
        is_lazy_build = False
        check_pending = getattr(mp, "check_and_clear_pending_build", None)
        if callable(check_pending):
            is_lazy_build = check_pending(model_name)
        if is_lazy_build:
            mp.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\n"
                f"This may take several minutes.",
            )
        try:
            if getattr(mp, "device", "cpu") == "cuda":
                # Run on the dedicated inference thread so the CUDA EP only ever
                # creates one per-thread context (no per-play / per-reprocess
                # VRAM growth).
                outs = self._get_ort_executor().submit(
                    session.run, None, feeds
                ).result()
            else:
                outs = session.run(None, feeds)
        finally:
            if is_lazy_build:
                mp.hide_build_dialog.emit()

        return dict(zip(names, outs))

    def unload_models(self):
        """Release all four PerformRecast models from VRAM."""
        for model_name in self.model_group:
            self.models_processor.unload_model(model_name)

    # ------------------------------------------------------------------ #
    # Model-level API (mirrors scripts/onnx_inference.py)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_input(img: torch.Tensor) -> np.ndarray:
        """(C,H,W) uint8/float [0,255] -> (1,3,H,W) float32 [0,1] numpy."""
        x = img.detach().to(torch.float32) / 255.0
        x = torch.clamp(x, 0.0, 1.0)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x.contiguous().cpu().numpy()

    def extract_appearance(self, img512: torch.Tensor) -> np.ndarray:
        """F: source crop (C,512,512)[0,255] -> feature_3d (1,32,16,64,64)."""
        feeds = {"source_image": self._to_input(img512)}
        return self._run(APPEARANCE_MODEL, feeds)["feature_3d"]

    def motion(self, img256: torch.Tensor) -> Dict[str, torch.Tensor]:
        """M: crop (C,256,256)[0,255] -> raw motion dict as torch tensors.

        Keys: pitch, yaw, roll (1,66); t (1,3); scale (1,1);
        exp, kp reshaped to (1, NUM_KP, 3).
        """
        out = self._run(MOTION_MODEL, {"image": self._to_input(img256)})
        device = self.models_processor.device
        kp_info = {
            k: torch.from_numpy(out[k]).to(device)
            for k in ("pitch", "yaw", "roll", "t", "exp", "scale", "kp")
        }
        kp_info["exp"] = kp_info["exp"].reshape(1, -1, 3)
        kp_info["kp"] = kp_info["kp"].reshape(1, -1, 3)
        return kp_info

    def warp_decode(
        self, feature_3d: np.ndarray, kp_source: torch.Tensor, kp_driving: torch.Tensor
    ) -> torch.Tensor:
        """W + G: D(W(f_s; x_s, x_d)) -> RGB image (1,3,512,512) torch [0,1].

        Note the upstream input ordering: the warping module receives
        ``kp_driving`` first and ``kp_source`` second.
        """
        warped = self._run(
            WARPING_MODEL,
            {
                "feature_3d": feature_3d,
                "kp_driving": kp_driving.detach().cpu().numpy(),
                "kp_source": kp_source.detach().cpu().numpy(),
            },
        )["out"]
        out = self._run(SPADE_MODEL, {"feature": warped})["out"]
        return torch.from_numpy(out).to(self.models_processor.device)

    # ------------------------------------------------------------------ #
    # Geometry — ported verbatim from src/pipeline.py (HopeNet -99 binning)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _headpose_to_degree(pred: torch.Tensor) -> torch.Tensor:
        """(bs,66) pose logits -> (bs,) degrees. PerformRecast uses *3 - 99."""
        idx = torch.arange(66, dtype=torch.float32, device=pred.device)
        pred = F.softmax(pred, dim=1)
        return torch.sum(pred * idx, dim=1) * 3 - 99

    @staticmethod
    def _rotation_matrix(
        yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor
    ) -> torch.Tensor:
        """(bs,) degree angles -> (bs,3,3) rotation matrix (pitch·yaw·roll)."""
        deg2rad = 3.14 / 180  # match upstream's literal 3.14 constant exactly
        yaw = (yaw * deg2rad).unsqueeze(1)
        pitch = (pitch * deg2rad).unsqueeze(1)
        roll = (roll * deg2rad).unsqueeze(1)

        zeros = torch.zeros_like(pitch)
        ones = torch.ones_like(pitch)

        pitch_mat = torch.cat(
            [ones, zeros, zeros,
             zeros, torch.cos(pitch), -torch.sin(pitch),
             zeros, torch.sin(pitch), torch.cos(pitch)], dim=1
        ).view(-1, 3, 3)
        yaw_mat = torch.cat(
            [torch.cos(yaw), zeros, torch.sin(yaw),
             zeros, ones, zeros,
             -torch.sin(yaw), zeros, torch.cos(yaw)], dim=1
        ).view(-1, 3, 3)
        roll_mat = torch.cat(
            [torch.cos(roll), -torch.sin(roll), zeros,
             torch.sin(roll), torch.cos(roll), zeros,
             zeros, zeros, ones], dim=1
        ).view(-1, 3, 3)

        return torch.einsum("bij,bjk,bkm->bim", pitch_mat, yaw_mat, roll_mat)

    def build_source_info(self, kp_info: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the per-frame source descriptor used during animation.

        Returns a dict with the canonical keypoints ``kp`` (1,N,3), expression
        ``exp`` (1,N,3), rotation ``R`` (1,3,3), ``scale`` (1,1), translation
        ``t`` (1,3, tz zeroed) and the transformed source keypoints ``x_s``.
        """
        kp = kp_info["kp"]
        exp = kp_info["exp"]
        yaw = self._headpose_to_degree(kp_info["yaw"])
        pitch = self._headpose_to_degree(kp_info["pitch"])
        roll = self._headpose_to_degree(kp_info["roll"])
        R = self._rotation_matrix(yaw, pitch, roll)
        scale = kp_info["scale"]
        t = kp_info["t"].clone()
        t[..., 2] = 0  # zero tz

        kp_e = kp + exp
        kp_rotated = torch.einsum("bmp,bkp->bkm", R, kp_e)
        kp_rotated = kp_rotated * scale.unsqueeze(1)
        x_s = kp_rotated + t.unsqueeze(1)

        return {"kp": kp, "exp": exp, "R": R, "scale": scale, "t": t, "x_s": x_s}

    # ------------------------------------------------------------------ #
    # Expression composition — ported from src/pipeline.py animate loop
    # ------------------------------------------------------------------ #
    # Implicit-keypoint channel groups referenced by the upstream
    # "Replacement" modulation. Used here for optional region gating.
    EYE_INDICES = list(range(31, 39))    # 31..38 (eye channels)
    MOUTH_INDICES = list(range(44, 47))  # 44..46 (jaw / lip-contour channels)

    def compose_driven_keypoints(
        self,
        source_info: Dict[str, torch.Tensor],
        exp_d: torch.Tensor,
        mode: str = MODE_ENHANCEMENT,
        factor: float = 1.0,
        region: str = "all",
    ) -> torch.Tensor:
        """Build the driven keypoints ``x_d_i`` fed to the warping module.

        Args:
            source_info: output of :meth:`build_source_info` (the swapped face).
            exp_d: driving expression (1,N,3) from the original face's motion.
            mode: ``"Replacement"`` (upstream mode 1) or ``"Enhancement"`` (mode 2).
            factor: expression strength. 0 keeps the source expression, 1 applies
                the full transfer; values >1 exaggerate it.
            region: ``"all"`` | ``"eyes"`` | ``"lips"`` — restrict where the
                driving expression is applied (source expression kept elsewhere).

        Per-frame note: the upstream video pipeline uses the driving video's
        first frame as the neutral reference for the relative delta. VisoMaster
        processes frames independently, so the source (swapped) face's own
        expression is used as that reference — making ``factor`` a clean
        interpolation between "keep source" and "full driving expression".
        """
        x_s_c = source_info["kp"]
        exp_s = source_info["exp"]
        R = source_info["R"]
        scale = source_info["scale"]
        t = source_info["t"]

        if mode == MODE_REPLACEMENT:
            # Start from the absolute driving expression and blend back
            # source-side eye / lip / jaw channels (identity micro-cues).
            modulated = exp_d.clone()
            modulated[:, 31:34, 2] = exp_s[:, 31:34, 2]
            modulated[:, 36:39, 2] = exp_s[:, 36:39, 2]
            modulated[:, 44:47, 2] = exp_s[:, 44:47, 2]
            modulated[:, 44:47, 0] = exp_s[:, 44:47, 0]
            modulated[:, 44:47, 1] = exp_s[:, 44:47, 1] * 0.2 + exp_d[:, 44:47, 1] * 0.8
            modulated[:, 31:34, :2] = exp_s[:, 31:34, :2] * 0.3 + exp_d[:, 31:34, :2] * 0.7
            modulated[:, 36:39, :2] = exp_s[:, 36:39, :2] * 0.3 + exp_d[:, 36:39, :2] * 0.7
            # Interpolate from source toward the modulated target by `factor`.
            new_exp = exp_s + factor * (modulated - exp_s)
        elif mode == MODE_ENHANCEMENT:
            # Add the driving expression delta (relative to source) on top of
            # the source expression, scaled by `factor`.
            new_exp = exp_s + factor * (exp_d - exp_s)
        else:
            raise ValueError(f"Unsupported recast mode: {mode!r}")

        # Optional region gating: keep the source expression outside the region.
        if region != "all":
            indices = self.EYE_INDICES if region == "eyes" else self.MOUTH_INDICES
            gated = exp_s.clone()
            gated[:, indices, :] = new_exp[:, indices, :]
            new_exp = gated

        kp_e = x_s_c + new_exp
        kp_rotated = torch.einsum("bmp,bkp->bkm", R, kp_e)
        kp_rotated = kp_rotated * scale.unsqueeze(1)
        x_d_i = kp_rotated + t.unsqueeze(1)
        return x_d_i
