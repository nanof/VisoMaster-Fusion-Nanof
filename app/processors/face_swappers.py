import os
import torch
import threading
from collections import OrderedDict
from torchvision.transforms import v2
from app.processors.utils import faceutil
from app.processors.models_data import models_dir
import numpy as np
from numpy.linalg import norm as l2norm
from typing import TYPE_CHECKING, List, Optional, Union
import kornia.geometry.transform as kgm

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FaceSwappers:
    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        # Dual-swapper pipelines keep up to two ONNX swappers resident so
        # primary/secondary (or a UI model switch) does not ping-pong VRAM.
        self._active_swapper_models: OrderedDict[str, bool] = OrderedDict()
        self._MAX_ACTIVE_SWAPPERS: int = 2
        self.current_arcface_model = None
        # FS-PERF-02: cache input/output names keyed by (model_name, id(session)).
        # The model name is part of the key because CPython reuses id() after a session
        # is unloaded, which otherwise leaks names across models (e.g. HyperSwap "source"
        # bound on an ArcFace session).
        self._session_io_name_cache: dict = {}
        self._io_cache_lock = threading.Lock()
        self._inswapper_init_lock = threading.Lock()
        self._w600k_lock = threading.Lock()
        self._inswapper_b1_lock = threading.Lock()
        self._inswapper_ort_batch_fail_logged = False
        # After first batched ORT/TRT failure, skip further attempts this session (avoid B=16 errors every frame).
        self._inswapper_ort_batch_session_disabled = False
        self._hyperswap_ort_batch_fail_logged = False
        self._hyperswap_ort_batch_session_disabled = False
        # HyperSwap ONNX exports a second output ("mask") with the region the generator
        # considers valid. Disabled for the session if the export lacks it or binding fails.
        self._hyperswap_native_mask_disabled = False
        self._inswapper_torch = None  # InSwapperTorch instance
        self._inswapper_runner_b1: Optional[object] = None  # CUDA graph runner for B=1
        self._w600k_torch: Optional[object] = None  # IResNet50Torch
        self._w600k_runner: Optional[object] = None  # _CapturedGraph or eager model
        self.resize_112 = v2.Resize(
            (112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )
        self.swapper_models = [
            "Inswapper128",
            "AlphaFace",
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
            "SimSwap512",
            "SimSwap512-CrossFace",
            "GhostFacev1",
            "GhostFacev2",
            "GhostFacev3",
            "HyperSwapv1",
            "HyperSwapv2",
            "HyperSwapv3",
            "ReHiFaceS",
            "CSCS",
            "BlendSwap256",
            "UniFace256",
        ]
        # ONNX used only with ReHiFace-S (FaceFusion crossface_hififace); unload with swapper cleanup
        self._crossface_aux_model_names = ("CrossFaceHiFaceS", "CrossFaceSimSwap")
        self.arcface_models = [
            "Inswapper128ArcFace",
            "HyperSwapArcFace",
            "SimSwapArcFace",
            "GhostArcFace",
            "CSCSArcFace",
            "CSCSIDArcFace",
        ]
        # AlphaFace ships its own ArcFace->ID projection matrix; loaded on first use.
        self._alphaface_emap: np.ndarray | None = None

    @property
    def current_swapper_model(self) -> str | None:
        """Most recently activated swapper (backward-compatible single-slot view)."""
        with self.models_processor.model_lock:
            if self._active_swapper_models:
                return next(reversed(self._active_swapper_models))
            return None

    @current_swapper_model.setter
    def current_swapper_model(self, value: str | None) -> None:
        with self.models_processor.model_lock:
            if value is None:
                return
            self._active_swapper_models[value] = True
            self._active_swapper_models.move_to_end(value)

    def _reset_hyperswap_session_flags(self) -> None:
        self._hyperswap_ort_batch_session_disabled = False
        self._hyperswap_ort_batch_fail_logged = False
        self._hyperswap_native_mask_disabled = False

    def _evict_swapper_model(self, model_name: str) -> None:
        self.models_processor.unload_model(model_name)
        if str(model_name).startswith("HyperSwap"):
            self._reset_hyperswap_session_flags()

    def unload_models(self):
        with self.models_processor.model_lock:
            self._active_swapper_models.clear()
            for model_name in (
                *self.swapper_models,
                *self._crossface_aux_model_names,
            ):
                self.models_processor.unload_model(model_name)
            _unloaded_arc: set[str] = set()
            for model_name in self.arcface_models:
                ort_name = self._arcface_ort_session_name(model_name)
                if ort_name in _unloaded_arc:
                    continue
                _unloaded_arc.add(ort_name)
                self.models_processor.unload_model(ort_name)
        # Allow a fresh HyperSwap/TRT session to retry batch + native mask after unload.
        self._reset_hyperswap_session_flags()

    def _manage_model(self, new_model_name):
        # FS-RACE-01: protect read-modify-write of the active-swapper LRU with lock
        with self.models_processor.model_lock:
            # HyperSwap variants share instance-level ORT session flags; keep only one.
            if str(new_model_name).startswith("HyperSwap"):
                for name in list(self._active_swapper_models):
                    if name != new_model_name and str(name).startswith("HyperSwap"):
                        self._active_swapper_models.pop(name, None)
                        self._evict_swapper_model(name)

            if new_model_name in self._active_swapper_models:
                self._active_swapper_models.move_to_end(new_model_name)
                return

            while len(self._active_swapper_models) >= self._MAX_ACTIVE_SWAPPERS:
                oldest_model, _ = self._active_swapper_models.popitem(last=False)
                if oldest_model != new_model_name:
                    self._evict_swapper_model(oldest_model)
            # FS-BUG-07: current_swapper_model is committed only after load confirmation (see _load_swapper_model)

    def _load_swapper_model(self, model_name):
        """Handles loading and swapping of swapper models."""
        self._manage_model(model_name)
        model = self.models_processor.get_onnx_session(model_name)
        if not model:
            model = self.models_processor.load_model(model_name)
        # FS-BUG-07: only commit state after load is confirmed non-None
        if model is not None:
            with self.models_processor.model_lock:
                self._active_swapper_models[model_name] = True
                self._active_swapper_models.move_to_end(model_name)
        return model

    def _run_model_with_lazy_build_check(
        self, model_name: str, ort_session, io_binding
    ):
        """
        Runs the ONNX session with IOBinding, handling TensorRT lazy build dialogs.
        This centralizes the try/finally logic for showing/hiding the build progress dialog
        and includes the critical synchronization step for CUDA or other devices.

        Args:
            model_name (str): The name of the model being run.
            ort_session: The ONNX Runtime session instance.
            io_binding: The pre-configured IOBinding object.
        """
        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            # ⚠️ This is a critical synchronization point.
            # PRE-INFERENCE SYNC
            if self.models_processor.uses_cuda_ep_for_thread():
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                # This handles synchronization for other execution providers (e.g., DirectML)
                self.models_processor.syncvec.cpu()

            self.models_processor.run_session_with_iobinding(ort_session, io_binding)

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def _session_io_names(self, model_name: str, ort_session) -> dict:
        """Cached ``{"input", "outputs"}`` ONNX names for this (model, session) pair."""
        cache_key = (str(model_name), id(ort_session))
        with self._io_cache_lock:
            names = self._session_io_name_cache.get(cache_key)
            if names is None:
                names = {
                    "input": ort_session.get_inputs()[0].name,
                    "outputs": [o.name for o in ort_session.get_outputs()],
                }
                self._session_io_name_cache[cache_key] = names
            return names

    @staticmethod
    def _arcface_ort_session_name(arcface_model: str) -> str:
        """HyperSwapArcFace reuses the w600k Inswapper128ArcFace ONNX session."""
        if arcface_model == "HyperSwapArcFace":
            return "Inswapper128ArcFace"
        return arcface_model

    def _align_hyperswap_ff_arcface_112(
        self, img: torch.Tensor, face_kps: np.ndarray
    ) -> torch.Tensor:
        """HyperSwap source crop: landmark warp to ``arcface_112_v2`` (FaceFusion 3.3).

        FaceFusion computes the HyperSwap identity with ``face_recognizer.calc_embedding``
        (template ``arcface_112_v2``, w600k), never the pose-aware ``arcfacemap`` fallback.
        ``faceutil`` mode ``arcface112`` is that same template (``arcface_src`` / 112).
        The Labs ``arcface_128_to_arcface_112_v2`` convert only applies to training, where
        inputs are already aligned crops; applying it here double-warps and corrupts identity.
        """
        crop, _ = faceutil.warp_face_by_face_landmark_5(
            img,
            face_kps,
            image_size=112,
            mode="arcface112",
            interpolation=v2.InterpolationMode.BILINEAR,
        )
        return crop

    def run_recognize_direct(
        self, img, kps, similarity_type="Auto", arcface_model="Inswapper128ArcFace"
    ):
        ort_name = self._arcface_ort_session_name(arcface_model)
        # FS-RACE-01: protect read-modify-write of current_arcface_model with lock
        with self.models_processor.model_lock:
            if self.current_arcface_model:
                prev_ort = self._arcface_ort_session_name(self.current_arcface_model)
                if prev_ort != ort_name:
                    self.models_processor.unload_model(prev_ort)
            self.current_arcface_model = ort_name

        ort_session = self.models_processor.get_onnx_session(ort_name)
        if not ort_session:
            ort_session = self.models_processor.load_model(ort_name)

        if not ort_session:
            print(
                f"[WARN] ArcFace model '{arcface_model}' (ort={ort_name}) failed to load. "
                "Skipping recognition."
            )
            return None, None

        if arcface_model == "CSCSArcFace":
            embedding, cropped_image = self.recognize_cscs(img, kps)
        else:
            embedding, cropped_image = self.recognize(
                arcface_model, img, kps, similarity_type=similarity_type
            )

        return embedding, cropped_image

    def _chw112_for_inswapper_arcface(
        self, img: torch.Tensor, face_kps: np.ndarray, similarity_type: str
    ) -> torch.Tensor:
        """Aligned 3×112 face crop (Pearl/Opal paths) before Inswapper128ArcFace normalization."""
        if similarity_type == "Pearl":
            dst = self.models_processor.arcface_dst.copy()
            dst[:, 0] += 8.0
            tform = faceutil.similarity_transform_from_correspondences(face_kps, dst)
            M_tensor = (
                torch.from_numpy(tform.params[0:2]).float().unsqueeze(0).to(img.device)
            )
            img_b = img.unsqueeze(0) if img.dim() == 3 else img
            out = kgm.warp_affine(
                img_b.float(),
                M_tensor,
                dsize=(128, 128),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            return v2.functional.resize(out, [112, 112], antialias=True)

        tform = faceutil.similarity_transform_from_correspondences(
            face_kps, self.models_processor.arcface_dst
        )
        M_tensor = (
            torch.from_numpy(tform.params[0:2]).float().unsqueeze(0).to(img.device)
        )
        img_b = img.unsqueeze(0) if img.dim() == 3 else img
        return kgm.warp_affine(
            img_b.float(),
            M_tensor,
            dsize=(112, 112),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

    def run_recognize_direct_batch(
        self,
        img: Union[torch.Tensor, List[torch.Tensor]],
        kps_list: List[np.ndarray],
        similarity_type: str,
        arcface_model: str,
    ) -> Optional[List[Optional[np.ndarray]]]:
        """
        One ORT inference for B>=1 faces (Inswapper128ArcFace + Opal/Pearl, non-Custom).
        Pass a single CHW frame tensor plus multiple kps (same image), or a list of CHW
        crops with one kps per crop (e.g. VR180 per-face crops).
        Returns None to fall back to per-face run_recognize_direct.
        """
        if len(kps_list) < 1:
            return None
        if arcface_model != "Inswapper128ArcFace":
            return None
        # "Auto" / "Optimal" use per-face pose-aware alignment; batch path is Opal/Pearl only.
        if similarity_type not in ("Opal", "Pearl"):
            return None
        if self.models_processor.provider_name == "Custom":
            return None

        with self.models_processor.model_lock:
            if (
                self.current_arcface_model
                and self.current_arcface_model != arcface_model
            ):
                self.models_processor.unload_model(self.current_arcface_model)
            self.current_arcface_model = arcface_model

        ort_session = self.models_processor.get_onnx_session(arcface_model)
        if not ort_session:
            ort_session = self.models_processor.load_model(arcface_model)
        if not ort_session:
            return None

        try:
            if isinstance(img, torch.Tensor):
                crops = [
                    self._chw112_for_inswapper_arcface(img, kps, similarity_type)
                    for kps in kps_list
                ]
            else:
                if len(img) != len(kps_list):
                    return None
                crops = [
                    self._chw112_for_inswapper_arcface(
                        img[i], kps_list[i], similarity_type
                    )
                    for i in range(len(kps_list))
                ]
            batch = torch.stack(crops, dim=0).float().clone()
            if batch.max() <= 1.0:
                batch = batch * 255.0
            batch.sub_(127.5).div_(127.5)

            _names = self._session_io_names(arcface_model, ort_session)
            input_name = _names["input"]
            output_names = _names["outputs"]

            io_binding = ort_session.io_binding()
            batch = self.models_processor.bind_ort_io_input(
                io_binding,
                arcface_model,
                input_name,
                batch,
                session=ort_session,
            )
            for name in output_names:
                self.models_processor.bind_ort_output_dynamic(io_binding, name)

            self._run_model_with_lazy_build_check(
                arcface_model, ort_session, io_binding
            )
            outs = io_binding.copy_outputs_to_cpu()
            emb_arr = np.array(outs[0])
            if emb_arr.ndim == 1:
                dim = emb_arr.size // len(kps_list)
                if dim * len(kps_list) != emb_arr.size:
                    return None
                emb_arr = emb_arr.reshape(len(kps_list), dim)
            elif emb_arr.ndim == 2 and emb_arr.shape[0] == len(kps_list):
                pass
            else:
                return None
            return [
                emb_arr[i].flatten().astype(np.float32, copy=False)
                for i in range(len(kps_list))
            ]
        except Exception as e:
            print(f"[WARN] ArcFace batch inference failed, falling back per-face: {e}")
            return None

    def run_recognize(
        self, img, kps, similarity_type="Auto", face_swapper_model="Inswapper128"
    ):
        arcface_model = self.models_processor.get_arcface_model(face_swapper_model)
        return self.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def recognize(self, arcface_model, img, face_kps, similarity_type=None):
        """
        ArcFace embedding with pose-aware alignment (dynamic Optimal vs frontal arcface112).

        ``HyperSwapArcFace`` uses a fixed FaceFusion ``arcface_112_v2`` landmark warp
        instead of pose-aware Optimal/Opal, matching HyperSwap inference identity.

        The ``similarity_type`` argument is kept for API compatibility but alignment follows
        yaw/pitch so ArcFace avoids brittle manual modes (e.g. Pearl) on challenging poses
        (non-HyperSwap paths).
        """
        ort_name = self._arcface_ort_session_name(arcface_model)
        ort_session = self.models_processor.get_onnx_session(ort_name)
        if not ort_session:
            return None, None

        if arcface_model == "HyperSwapArcFace":
            img = self._align_hyperswap_ff_arcface_112(img, face_kps)
        else:
            yaw, pitch = faceutil.calc_face_yaw_pitch(face_kps)
            if abs(yaw) > 30.0 or abs(pitch) > 30.0:
                actual_mode = "Optimal"
            else:
                actual_mode = "Opal"

            if actual_mode == "Optimal":
                img, _ = faceutil.warp_face_by_face_landmark_5(
                    img,
                    face_kps,
                    image_size=112,
                    mode="arcfacemap",
                    interpolation=v2.InterpolationMode.BILINEAR,
                )
            else:
                img, _ = faceutil.warp_face_by_face_landmark_5(
                    img,
                    face_kps,
                    image_size=112,
                    mode="arcface112",
                    interpolation=v2.InterpolationMode.BILINEAR,
                )

        # --- NORMALIZATION & PRE-PROCESSING ---
        cropped_image = img.permute(1, 2, 0).clone()  # Store for display/debug (H,W,3)

        if img.dtype == torch.uint8:
            img = img.float()

        img = img.clone()

        # OPTIMIZED: In-Place math operations (.sub_ and .div_) to save VRAM fragmentation
        if arcface_model in ("Inswapper128ArcFace", "HyperSwapArcFace"):
            # FS-BUG-03: ensure input is in [0, 255] before normalizing
            if img.max() <= 1.0:
                img = img * 255.0
            img.sub_(127.5).div_(127.5)

        elif arcface_model == "SimSwapArcFace":
            img.div_(255.0)
            v2.functional.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True
            )

        else:
            # GhostArcFace, CSCSArcFace, etc.
            img.div_(127.5).sub_(1.0)

        # --- INFERENCE ---
        # Prepare data (N, C, H, W)
        img = torch.unsqueeze(img, 0).contiguous()

        _names = self._session_io_names(ort_name, ort_session)
        input_name = _names["input"]
        output_names = _names["outputs"]

        io_binding = ort_session.io_binding()
        img = self.models_processor.bind_ort_io_input(
            io_binding,
            ort_name,
            input_name,
            img,
            session=ort_session,
        )

        for name in output_names:
            self.models_processor.bind_ort_output_dynamic(io_binding, name)

        # Run the model with lazy build handling (TensorRT safety)
        self._run_model_with_lazy_build_check(ort_name, ort_session, io_binding)

        # Return embedding (flattened) and the cropped image for visualization
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image

    def preprocess_image_cscs(self, img, face_kps):
        tform = faceutil.similarity_transform_from_correspondences(
            face_kps, self.models_processor.FFHQ_kps
        )

        # GPU Accelerated Affine Transformation (img is already a GPU Tensor here)
        # We preserve the exact center=(0,0) geometry required by CSCS models.
        temp = v2.functional.affine(
            img,
            angle=tform.rotation * 57.2958,  # Rad to Deg
            translate=(tform.translation[0], tform.translation[1]),
            scale=tform.scale,
            shear=0.0,
            center=(0, 0),
        )

        # Fast GPU Crop and Resize
        temp = v2.functional.crop(temp, top=0, left=0, height=512, width=512)
        image = self.resize_112(temp)

        cropped_image = image.permute(1, 2, 0).clone()

        if image.dtype == torch.uint8:
            image = image.float()

        # CLONE: Prevent cross-thread race conditions before in-place math
        image = image.clone()

        # OPTIMIZED: In-place division and normalization for CSCS [-1.0, 1.0] standard
        image.div_(255.0)
        v2.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

        return torch.unsqueeze(image, 0).contiguous(), cropped_image

    def recognize_cscs(self, img, face_kps):
        img, cropped_image = self.preprocess_image_cscs(img, face_kps)

        model_name = "CSCSArcFace"
        model = self.models_processor.get_onnx_session(model_name)
        if not model:
            print("[ERROR] CSCSArcFace model not loaded in recognize_cscs.")
            return None, None

        io_binding = model.io_binding()

        # SAFETY: Clear bindings to prevent thread caching errors
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        img = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "input",
            img,
            session=model,
        )
        self.models_processor.bind_ort_output_dynamic(io_binding, "output")

        self._run_model_with_lazy_build_check(model_name, model, io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]

        # Exact p=2 normalization math required by CSCS
        embedding = torch.from_numpy(output).to("cpu")
        embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)
        embedding = embedding.numpy().flatten()

        embedding_id = self.recognize_cscs_id_adapter(img, None)

        if embedding_id.size == embedding.size:
            embedding = embedding + embedding_id

        return embedding, cropped_image

    def recognize_cscs_id_adapter(self, img, face_kps):
        model_name = "CSCSIDArcFace"
        model = self.models_processor.get_onnx_session(model_name)
        if not model:
            model = self.models_processor.load_model(model_name)

        if not model:
            print(f"[WARN] {model_name} model not loaded.")
            return np.array([])

        if face_kps is not None:
            img, _ = self.preprocess_image_cscs(img, face_kps)

        io_binding = model.io_binding()

        # SAFETY: Clear bindings
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        img = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "input",
            img,
            session=model,
        )
        self.models_processor.bind_ort_output_dynamic(io_binding, "output")

        self._run_model_with_lazy_build_check(model_name, model, io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]

        # Exact p=2 normalization math required by CSCS
        embedding_id = torch.from_numpy(output).to("cpu")
        embedding_id = torch.nn.functional.normalize(embedding_id, dim=-1, p=2)

        return embedding_id.numpy().flatten()

    def calc_swapper_latent_cscs(self, source_embedding):
        latent = source_embedding.reshape((1, -1))
        return latent

    def run_swapper_cscs(self, image, embedding, output):
        model_name = "CSCS"
        model = self._load_swapper_model(model_name)
        if not model:
            print("[ERROR] CSCS model not loaded.")
            return

        # SAFETY: Contiguous memory blocks required by TensorRT
        if not image.is_contiguous():
            image = image.contiguous()
        if not embedding.is_contiguous():
            embedding = embedding.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()

        io_binding = model.io_binding()

        # SAFETY: Clear bindings
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        # Hardcoded IO names validated by standard CSCS export
        image = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "input_1",
            image,
            session=model,
        )
        embedding = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "input_2",
            embedding,
            session=model,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            "output",
            output,
            session=model,
        )

        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def _calc_emap_latent(self, source_embedding):
        """FS-PERF-05: shared emap-based latent computation extracted from
        calc_inswapper_latent and calc_swapper_latent_iss."""
        if source_embedding is None:
            return None
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1, -1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def _ensure_emap(self):
        """Ensures emap is loaded; returns True if available, False otherwise."""
        if (
            not hasattr(self.models_processor, "emap")
            or not isinstance(self.models_processor.emap, np.ndarray)
            or self.models_processor.emap.size == 0
        ):
            self.models_processor.load_model("Inswapper128")

        return (
            hasattr(self.models_processor, "emap")
            and isinstance(self.models_processor.emap, np.ndarray)
            and self.models_processor.emap.size > 0
        )

    def calc_inswapper_latent(self, source_embedding):
        if source_embedding is None:
            return None
        if not self._ensure_emap():
            print("[ERROR] Emap could not be loaded for latent calculation.")
            # FS-ROBUST-01: return None so callers can detect and handle the failure
            return None

        return self._calc_emap_latent(source_embedding)

    def run_inswapper(self, image, embedding, output):
        model_name = "Inswapper128"

        # ORT-based inference
        model = self._load_swapper_model(model_name)
        if not model:
            print("[ERROR] Inswapper128 model not loaded.")
            return

        # FORCE CONTIGUOUS: Essential safety check.
        # Ensures that the memory pointer passed to TensorRT is valid and linear.
        if not image.is_contiguous():
            image = image.contiguous()
        if not embedding.is_contiguous():
            embedding = embedding.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()

        io_binding = model.io_binding()

        # Clear previous bindings to avoid pointer caching issues
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        image = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "target",
            image,
            session=model,
        )
        embedding = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "source",
            embedding,
            session=model,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            "output",
            output,
            session=model,
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def run_inswapper_ort_batched(
        self, images: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> bool:
        """
        Single ORT/TRT Inswapper128 forward for B>1 tiles (pixel-shift dim>1).

        Returns False if disabled, B<2, or the session rejects the batched shape
        (e.g. TensorRT engine fixed at batch 1) — caller falls back per-tile.
        """
        model_name = "Inswapper128"
        if self._inswapper_ort_batch_session_disabled:
            return False
        if self.models_processor.provider_name == "Custom":
            return False

        v_raw = os.environ.get("VISIOMASTER_INSWAPPER_ORT_BATCH", "").strip()
        if v_raw:
            vl = v_raw.lower()
            if vl in ("0", "false", "no", "off"):
                return False
        else:
            # Native .engine builds are almost always batch 1; skip batched path unless user forces it.
            if self.models_processor.provider_name == "TensorRT-Engine":
                return False

        if images.dim() != 4 or images.shape[1] != 3:
            return False
        B = int(images.shape[0])
        if B < 2:
            return False
        if tuple(images.shape[2:]) != (128, 128):
            return False
        if output.shape != images.shape:
            return False

        model = self._load_swapper_model(model_name)
        if not model:
            return False

        inp = images if images.is_contiguous() else images.contiguous()
        out = output if output.is_contiguous() else output.contiguous()

        emb = embedding
        td_src = self.models_processor.get_ort_io_torch_dtype(
            model_name, "source", is_output=False
        )
        if emb.dtype != td_src:
            emb = emb.to(dtype=td_src)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if emb.shape[-1] != 512:
            return False
        if emb.shape[0] == 1:
            emb_b = emb.repeat(B, 1).contiguous()
        elif emb.shape[0] == B:
            emb_b = emb.contiguous()
        else:
            return False

        try:
            io_binding = model.io_binding()
            io_binding.clear_binding_inputs()
            io_binding.clear_binding_outputs()

            inp = self.models_processor.bind_ort_io_input(
                io_binding,
                model_name,
                "target",
                inp,
                session=model,
            )
            emb_b = self.models_processor.bind_ort_io_input(
                io_binding,
                model_name,
                "source",
                emb_b,
                session=model,
            )
            self.models_processor.bind_ort_io_output(
                io_binding,
                model_name,
                "output",
                out,
                session=model,
            )
            self._run_model_with_lazy_build_check(model_name, model, io_binding)
            return True
        except Exception as e:
            self._inswapper_ort_batch_session_disabled = True
            if not self._inswapper_ort_batch_fail_logged:
                self._inswapper_ort_batch_fail_logged = True
                print(
                    f"[WARN] Inswapper128 ORT/TRT batched inference failed (B={B}); "
                    f"using per-tile for this session. First error: {e}. "
                    "Set VISIOMASTER_INSWAPPER_ORT_BATCH=0 to skip the attempt, or leave unset with "
                    "TensorRT-Engine (batched path is off by default for .engine). "
                    "Set VISIOMASTER_INSWAPPER_ORT_BATCH=1 to force batched tries (needs a dynamic-batch engine).",
                    flush=True,
                )
            return False

    def run_inswapper_batched(
        self, images: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Batched InSwapper inference for pixel-shift resolution mode."""

        torch_model = self._get_inswapper_torch()
        with torch.no_grad():
            # Same lock as B=1 CUDA-graph path: one shared InSwapperTorch instance.
            with self._inswapper_b1_lock:
                inp = images if images.is_contiguous() else images.contiguous()
                emb = embedding if embedding.is_contiguous() else embedding.contiguous()
                B = inp.shape[0]
                if emb.shape[0] not in (1, B):
                    raise ValueError(
                        f"InSwapper batched: embedding batch {emb.shape[0]} vs images {B}"
                    )
                result = torch_model(inp, emb)  # [B, 3, 128, 128] float32
                output.copy_(result)
                if self.models_processor.uses_cuda_ep_for_thread():
                    torch.cuda.current_stream().synchronize()

    def calc_swapper_latent_ghost(self, source_embedding):
        latent = source_embedding.reshape((1, -1))

        return latent

    def calc_swapper_latent_alphaface(
        self, source_embedding: np.ndarray
    ) -> np.ndarray | None:
        """Projects the shared W600K ArcFace embedding into AlphaFace ID space."""
        if self._alphaface_emap is None:
            emap_path = models_dir / "alphaface" / "emp.npy"
            try:
                emap = np.load(emap_path).astype(np.float32)
            except (OSError, ValueError) as exc:
                print(f"[ERROR] Could not load AlphaFace identity projection: {exc}")
                return None
            if emap.shape != (512, 512):
                print(
                    "[ERROR] AlphaFace identity projection must have shape (512, 512)."
                )
                return None
            self._alphaface_emap = emap

        embedding = np.asarray(source_embedding, dtype=np.float32).reshape(1, -1)
        if embedding.shape[1] != 512:
            print("[ERROR] AlphaFace requires a 512-dimensional ArcFace embedding.")
            return None
        latent = np.dot(embedding, self._alphaface_emap)
        latent_norm = np.linalg.norm(latent)
        if not np.isfinite(latent_norm) or latent_norm <= 1e-12:
            print("[ERROR] AlphaFace produced an invalid identity projection.")
            return None
        return latent / latent_norm

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        if source_embedding is None:
            return None
        # FS-PERF-05: reuse shared _ensure_emap / _calc_emap_latent helpers
        if not self._ensure_emap():
            print("[ERROR] Emap could not be loaded for latent calculation.")
            n_e = source_embedding / l2norm(source_embedding)
            return n_e.reshape((1, -1))

        return self._calc_emap_latent(source_embedding)

    def run_iss_swapper(self, image, embedding, output, version="A"):
        model_name = f"InStyleSwapper256 Version {version}"
        model = self._load_swapper_model(model_name)
        if not model:
            print(f"[ERROR] {model_name} model not loaded.")
            return

        io_binding = model.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "target",
            image,
            session=model,
        )
        embedding = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "source",
            embedding,
            session=model,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            "output",
            output,
            session=model,
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def calc_swapper_latent_simswap512(self, source_embedding):
        latent = source_embedding.reshape(1, -1)
        # latent /= np.linalg.norm(latent)
        latent = latent / np.linalg.norm(latent, axis=1, keepdims=True)
        return latent

    def run_swapper_simswap512(self, image, embedding, output):
        model_name = "SimSwap512"
        model = self._load_swapper_model(model_name)
        if not model:
            print("[ERROR] SimSwap512 model not loaded.")
            return

        io_binding = model.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "input",
            image,
            session=model,
        )
        embedding = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "onnx::Gemm_1",
            embedding,
            session=model,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            "output",
            output,
            session=model,
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def run_swapper_ghostface(
        self, image, embedding, output, swapper_model="GhostFace-v2"
    ):
        model_name = None
        if swapper_model == "GhostFace-v1":
            model_name = "GhostFacev1"
        elif swapper_model == "GhostFace-v2":
            model_name = "GhostFacev2"
        elif swapper_model == "GhostFace-v3":
            model_name = "GhostFacev3"

        if not model_name:
            print(f"[ERROR] Unknown GhostFace model version: {swapper_model}")
            return

        ghostfaceswap_model = self._load_swapper_model(model_name)
        if not ghostfaceswap_model:
            print(f"[ERROR] {model_name} model not loaded.")
            return

        # FS-ROBUST-02: introspect output name dynamically instead of hardcoding node IDs
        output_name = self._session_io_names(model_name, ghostfaceswap_model)[
            "outputs"
        ][0]

        io_binding = ghostfaceswap_model.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "target",
            image,
            session=ghostfaceswap_model,
        )
        embedding = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "source",
            embedding,
            session=ghostfaceswap_model,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            output_name,
            output,
            session=ghostfaceswap_model,
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(
            model_name, ghostfaceswap_model, io_binding
        )

    @torch.no_grad()
    def run_swapper_alphaface(
        self, image: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        model_name = "AlphaFace"
        model = self._load_swapper_model(model_name)
        if not model:
            output.zero_()
            print(
                "[ERROR] AlphaFace model not loaded. Run 'python download_models.py' "
                "to install model_assets/alphaface/alphaface_swapper_fused_norm.onnx."
            )
            return

        io_binding = model.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "target",
            image.contiguous(),
            session=model,
        )
        embedding = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "source_embedding",
            embedding.contiguous(),
            session=model,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            "output",
            output.contiguous(),
            session=model,
        )
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    @torch.no_grad()
    def run_swapper_alphaface_batched(
        self, images: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Batched AlphaFace inference for sub-pixel 512px phase-shift resolution mode."""
        model_name = "AlphaFace"
        model = self._load_swapper_model(model_name)
        if not model:
            output.zero_()
            print("[ERROR] AlphaFace model not loaded for batched execution.")
            return

        images = images.contiguous()
        embedding = embedding.contiguous()
        output = output.contiguous()
        batch_size = int(images.shape[0])

        for idx in range(batch_size):
            self.run_swapper_alphaface(
                images[idx : idx + 1], embedding, output[idx : idx + 1]
            )

    def run_swapper_ghostface_batched(
        self,
        images: torch.Tensor,
        embedding: torch.Tensor,
        output: torch.Tensor,
        swapper_model: str = "GhostFace-v2",
    ) -> bool:
        """Try one ORT run with batch B>1. Returns False if binding/engine rejects batch."""
        model_name = None
        if swapper_model == "GhostFace-v1":
            model_name = "GhostFacev1"
        elif swapper_model == "GhostFace-v2":
            model_name = "GhostFacev2"
        elif swapper_model == "GhostFace-v3":
            model_name = "GhostFacev3"
        if not model_name:
            return False

        ghostfaceswap_model = self._load_swapper_model(model_name)
        if not ghostfaceswap_model:
            return False

        B = int(images.shape[0])
        if B < 1:
            return False
        emb = embedding if embedding.is_contiguous() else embedding.contiguous()
        if emb.shape[0] == 1 and B > 1:
            emb = emb.expand(B, -1).contiguous()
        elif emb.shape[0] != B:
            return False

        inp = images if images.is_contiguous() else images.contiguous()
        out = output if output.is_contiguous() else output.contiguous()

        output_name = self._session_io_names(model_name, ghostfaceswap_model)[
            "outputs"
        ][0]

        io_binding = ghostfaceswap_model.io_binding()
        try:
            inp = self.models_processor.bind_ort_io_input(
                io_binding,
                model_name,
                "target",
                inp,
                session=ghostfaceswap_model,
            )
            emb = self.models_processor.bind_ort_io_input(
                io_binding,
                model_name,
                "source",
                emb,
                session=ghostfaceswap_model,
            )
            self.models_processor.bind_ort_io_output(
                io_binding,
                model_name,
                output_name,
                out,
                session=ghostfaceswap_model,
            )
            self._run_model_with_lazy_build_check(
                model_name, ghostfaceswap_model, io_binding
            )
            return True
        except Exception as e:
            print(
                f"[WARN] GhostFace batched ORT bind/run failed (B={B}): {e!s:.200}",
                flush=True,
            )
            return False

    @staticmethod
    def _hyperswap_ui_to_model_name(swapper_model: str) -> Optional[str]:
        """Map UI ``HyperSwap-vN`` selection to ORT catalog name ``HyperSwapvN``."""
        if swapper_model == "HyperSwap-v1":
            return "HyperSwapv1"
        if swapper_model == "HyperSwap-v2":
            return "HyperSwapv2"
        if swapper_model == "HyperSwap-v3":
            return "HyperSwapv3"
        return None

    def _hyperswap_output_name(self, model_name: str, model) -> str:
        """Introspect first ONNX output name (cached), same pattern as GhostFace."""
        return self._session_io_names(model_name, model)["outputs"][0]

    def hyperswap_native_mask_ready(self) -> bool:
        """False once the native ``mask`` output proved unusable for this session."""
        return not self._hyperswap_native_mask_disabled

    def _disable_hyperswap_native_mask(self, reason: str) -> None:
        if self._hyperswap_native_mask_disabled:
            return
        self._hyperswap_native_mask_disabled = True
        print(
            f"[WARN] HyperSwap native mask unavailable; falling back to the standard "
            f"masks for this session. {reason}",
            flush=True,
        )

    def _hyperswap_mask_output_name(self, model_name: str, model) -> Optional[str]:
        """Name of the second ONNX output; ``None`` when the export only has ``output``."""
        if self._hyperswap_native_mask_disabled:
            return None
        outputs = self._session_io_names(model_name, model)["outputs"]
        if len(outputs) < 2:
            self._disable_hyperswap_native_mask(
                f"{model_name} exports a single output."
            )
            return None
        return outputs[1]

    def _run_hyperswap_pass(
        self,
        model_name: str,
        model,
        target: torch.Tensor,
        source: torch.Tensor,
        output: torch.Tensor,
        mask_output: Optional[torch.Tensor],
    ) -> bool:
        """One bind+run. With ``mask_output`` set, returns False instead of raising so the
        caller can retry without the extra output."""
        mask_name = None
        if mask_output is not None:
            mask_name = self._hyperswap_mask_output_name(model_name, model)
            if mask_name is None:
                return False

        try:
            io_binding = model.io_binding()
            target = self.models_processor.bind_ort_io_input(
                io_binding,
                model_name,
                "target",
                target,
                session=model,
            )
            source = self.models_processor.bind_ort_io_input(
                io_binding,
                model_name,
                "source",
                source,
                session=model,
            )
            self.models_processor.bind_ort_io_output(
                io_binding,
                model_name,
                self._hyperswap_output_name(model_name, model),
                output,
                session=model,
            )
            if mask_name is not None:
                self.models_processor.bind_ort_io_output(
                    io_binding,
                    model_name,
                    mask_name,
                    mask_output,
                    session=model,
                )
            self._run_model_with_lazy_build_check(model_name, model, io_binding)
            return True
        except Exception as e:
            if mask_name is None:
                raise
            self._disable_hyperswap_native_mask(f"First error: {e!s:.200}")
            return False

    def calc_hyperswap_latent(self, source_embedding):
        """FaceFusion HyperSwap: L2-normalized 512-D ArcFace row (1, 512)."""
        if source_embedding is None or len(source_embedding) == 0:
            return None
        v = np.asarray(source_embedding, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return None
        return (v / n).reshape(1, -1)

    def run_hyperswap(
        self, image, embedding, output, swapper_model="HyperSwap-v3", mask_output=None
    ) -> bool:
        """Run one HyperSwap inference. Returns True when ``mask_output`` was filled with
        the model's native mask."""
        model_name = self._hyperswap_ui_to_model_name(swapper_model)
        if not model_name:
            print(f"[ERROR] Unknown HyperSwap model: {swapper_model}")
            return False

        model = self._load_swapper_model(model_name)
        if not model:
            print(f"[ERROR] {model_name} model not loaded.")
            return False

        if mask_output is not None and self._run_hyperswap_pass(
            model_name, model, image, embedding, output, mask_output
        ):
            return True

        self._run_hyperswap_pass(model_name, model, image, embedding, output, None)
        return False

    def run_hyperswap_batched(
        self,
        images: torch.Tensor,
        embedding: torch.Tensor,
        output: torch.Tensor,
        swapper_model: str = "HyperSwap-v3",
        mask_output: Optional[torch.Tensor] = None,
    ) -> bool:
        """Try one ORT run with batch B>=1. Returns False if binding/engine rejects batch.

        When ``mask_output`` is given it is filled with the native mask; check
        ``hyperswap_native_mask_ready()`` afterwards to know whether it is usable.
        """
        if self._hyperswap_ort_batch_session_disabled:
            return False

        model_name = self._hyperswap_ui_to_model_name(swapper_model)
        if not model_name:
            return False

        model = self._load_swapper_model(model_name)
        if not model:
            return False

        B = int(images.shape[0])
        if B < 1:
            return False
        emb = embedding if embedding.is_contiguous() else embedding.contiguous()
        if emb.shape[0] == 1 and B > 1:
            emb = emb.expand(B, -1).contiguous()
        elif emb.shape[0] != B:
            return False

        inp = images if images.is_contiguous() else images.contiguous()
        out = output if output.is_contiguous() else output.contiguous()
        msk = None
        if mask_output is not None:
            msk = (
                mask_output if mask_output.is_contiguous() else mask_output.contiguous()
            )

        try:
            if msk is not None and self._run_hyperswap_pass(
                model_name, model, inp, emb, out, msk
            ):
                return True
            self._run_hyperswap_pass(model_name, model, inp, emb, out, None)
            return True
        except Exception as e:
            self._hyperswap_ort_batch_session_disabled = True
            if not self._hyperswap_ort_batch_fail_logged:
                self._hyperswap_ort_batch_fail_logged = True
                print(
                    f"[WARN] HyperSwap batched ORT bind/run failed (B={B}); "
                    f"using per-face for this session. First error: {e!s:.200}",
                    flush=True,
                )
            return False

    def run_blendswap(self, target_rgb_256, source_rgb_112, output):
        """FaceFusion blendswap_256: ``source`` = 112² RGB [0,1], ``target`` = 256² RGB [0,1]."""
        model_name = "BlendSwap256"
        model = self._load_swapper_model(model_name)
        if not model:
            print(f"[ERROR] {model_name} model not loaded.")
            return

        if not target_rgb_256.is_contiguous():
            target_rgb_256 = target_rgb_256.contiguous()
        if not source_rgb_112.is_contiguous():
            source_rgb_112 = source_rgb_112.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()

        io_binding = model.io_binding()
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()
        target_rgb_256 = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "target",
            target_rgb_256,
            session=model,
        )
        source_rgb_112 = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "source",
            source_rgb_112,
            session=model,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            "output",
            output,
            session=model,
        )
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def run_uniface(self, target_norm_256, source_rgb_256, output):
        """FaceFusion uniface_256: ``target`` = 256² RGB normalized (0.5/0.5), ``source`` = 256² RGB [0,1]."""
        model_name = "UniFace256"
        model = self._load_swapper_model(model_name)
        if not model:
            print(f"[ERROR] {model_name} model not loaded.")
            return

        if not target_norm_256.is_contiguous():
            target_norm_256 = target_norm_256.contiguous()
        if not source_rgb_256.is_contiguous():
            source_rgb_256 = source_rgb_256.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()

        io_binding = model.io_binding()
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()
        target_norm_256 = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "target",
            target_norm_256,
            session=model,
        )
        source_rgb_256 = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "source",
            source_rgb_256,
            session=model,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            "output",
            output,
            session=model,
        )
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def calc_rehiface_source_latent(self, source_embedding):
        """ReHiFace-S: ArcFace 512-D → crossface_hififace → L2-normalized (1, 512)."""
        if source_embedding is None or len(source_embedding) == 0:
            return None
        cross = self.models_processor.get_onnx_session("CrossFaceHiFaceS")
        if cross is None:
            cross = self.models_processor.load_model("CrossFaceHiFaceS")
        if cross is None:
            print("[ERROR] CrossFaceHiFaceS model not loaded.")
            return None

        dev = self.models_processor.get_effective_torch_device()
        td_in = self.models_processor.get_ort_io_torch_dtype(
            "CrossFaceHiFaceS", "input", is_output=False
        )
        emb = (
            torch.from_numpy(
                np.asarray(source_embedding, dtype=np.float32).reshape(1, -1)
            )
            .to(dtype=td_in, device=dev)
            .contiguous()
        )
        td_out = self.models_processor.get_ort_io_torch_dtype(
            "CrossFaceHiFaceS", "output", is_output=True
        )
        out_t = torch.empty((1, 512), dtype=td_out, device=dev).contiguous()
        io_binding = cross.io_binding()
        emb = self.models_processor.bind_ort_io_input(
            io_binding,
            "CrossFaceHiFaceS",
            "input",
            emb,
            session=cross,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            "CrossFaceHiFaceS",
            "output",
            out_t,
            session=cross,
        )
        self._run_model_with_lazy_build_check("CrossFaceHiFaceS", cross, io_binding)

        v = out_t.detach().float().cpu().numpy().reshape(-1)
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return v.reshape(1, -1)
        return (v / n).reshape(1, -1).astype(np.float32)

    def calc_crossface_simswap_latent(self, source_embedding):
        """SimSwap512-CrossFace: ArcFace w600k 512-D -> crossface_simswap -> L2 (1, 512)."""
        if source_embedding is None or len(source_embedding) == 0:
            return None
        cross = self.models_processor.get_onnx_session("CrossFaceSimSwap")
        if cross is None:
            cross = self.models_processor.load_model("CrossFaceSimSwap")
        if cross is None:
            print("[ERROR] CrossFaceSimSwap model not loaded.")
            return None

        dev = self.models_processor.get_effective_torch_device()
        td_in = self.models_processor.get_ort_io_torch_dtype(
            "CrossFaceSimSwap", "input", is_output=False
        )
        emb = (
            torch.from_numpy(
                np.asarray(source_embedding, dtype=np.float32).reshape(1, -1)
            )
            .to(dtype=td_in, device=dev)
            .contiguous()
        )
        td_out = self.models_processor.get_ort_io_torch_dtype(
            "CrossFaceSimSwap", "output", is_output=True
        )
        out_t = torch.empty((1, 512), dtype=td_out, device=dev).contiguous()
        io_binding = cross.io_binding()
        emb = self.models_processor.bind_ort_io_input(
            io_binding,
            "CrossFaceSimSwap",
            "input",
            emb,
            session=cross,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            "CrossFaceSimSwap",
            "output",
            out_t,
            session=cross,
        )
        self._run_model_with_lazy_build_check("CrossFaceSimSwap", cross, io_binding)

        v = out_t.detach().float().cpu().numpy().reshape(-1)
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return v.reshape(1, -1)
        return (v / n).reshape(1, -1).astype(np.float32)

    def run_rehiface(self, image, embedding, output):
        """HiFiFace unofficial 256 (FaceFusion): target NCHW [-1,1], source (1,512) L2-normalized."""
        model_name = "ReHiFaceS"
        model = self._load_swapper_model(model_name)
        if not model:
            print(f"[ERROR] {model_name} model not loaded.")
            return

        io_binding = model.io_binding()
        image = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "target",
            image,
            session=model,
        )
        embedding = self.models_processor.bind_ort_io_input(
            io_binding,
            model_name,
            "source",
            embedding,
            session=model,
        )
        self.models_processor.bind_ort_io_output(
            io_binding,
            model_name,
            "output",
            output,
            session=model,
        )

        self._run_model_with_lazy_build_check(model_name, model, io_binding)
