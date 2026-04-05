import torch
import threading
from torchvision.transforms import v2
from app.processors.utils import faceutil
import numpy as np
from numpy.linalg import norm as l2norm
from typing import TYPE_CHECKING, List, Optional
import kornia.geometry.transform as kgm

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FaceSwappers:
    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.current_swapper_model = None
        self.current_arcface_model = None
        self._session_io_name_cache: dict = {}  # FS-PERF-02: cache input/output names keyed by session id
        self._io_cache_lock = threading.Lock()
        self._inswapper_init_lock = threading.Lock()
        self._w600k_lock = threading.Lock()
        self._inswapper_b1_lock = threading.Lock()
        self._inswapper_torch = None  # InSwapperTorch instance
        self._inswapper_runner_b1: Optional[object] = None  # CUDA graph runner for B=1
        self._w600k_torch: Optional[object] = None  # IResNet50Torch
        self._w600k_runner: Optional[object] = None  # _CapturedGraph or eager model
        self.resize_112 = v2.Resize(
            (112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )
        self.swapper_models = [
            "Inswapper128",
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
            "SimSwapArcFace",
            "GhostArcFace",
            "CSCSArcFace",
            "CSCSIDArcFace",
        ]

    def unload_models(self):
        with self.models_processor.model_lock:
            for model_name in (
                *self.swapper_models,
                *self._crossface_aux_model_names,
            ):
                self.models_processor.unload_model(model_name)
            for model_name in self.arcface_models:
                self.models_processor.unload_model(model_name)

    def _manage_model(self, new_model_name):
        # FS-RACE-01: protect read-modify-write of current_swapper_model with lock
        with self.models_processor.model_lock:
            if (
                self.current_swapper_model
                and self.current_swapper_model != new_model_name
            ):
                self.models_processor.unload_model(self.current_swapper_model)
            # FS-BUG-07: current_swapper_model is committed only after load confirmation (see _load_swapper_model)

    def _load_swapper_model(self, model_name):
        """Handles loading and swapping of swapper models."""
        self._manage_model(model_name)
        model = self.models_processor.models.get(model_name)
        if not model:
            model = self.models_processor.load_model(model_name)
        # FS-BUG-07: only commit state after load is confirmed non-None
        if model is not None:
            with self.models_processor.model_lock:
                self.current_swapper_model = model_name
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
            if self.models_processor.device == "cuda":
                torch.cuda.current_stream().synchronize()
            elif self.models_processor.device != "cpu":
                # This handles synchronization for other execution providers (e.g., DirectML)
                self.models_processor.syncvec.cpu()

            self.models_processor.run_session_with_iobinding(ort_session, io_binding)

        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def run_recognize_direct(
        self, img, kps, similarity_type="Opal", arcface_model="Inswapper128ArcFace"
    ):
        # FS-RACE-01: protect read-modify-write of current_arcface_model with lock
        with self.models_processor.model_lock:
            if (
                self.current_arcface_model
                and self.current_arcface_model != arcface_model
            ):
                self.models_processor.unload_model(self.current_arcface_model)
            self.current_arcface_model = arcface_model

        ort_session = self.models_processor.models.get(arcface_model)
        if not ort_session:
            ort_session = self.models_processor.load_model(arcface_model)

        if not ort_session:
            print(
                f"[WARN] ArcFace model '{arcface_model}' failed to load. Skipping recognition."
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
        img: torch.Tensor,
        kps_list: List[np.ndarray],
        similarity_type: str,
        arcface_model: str,
    ) -> Optional[List[Optional[np.ndarray]]]:
        """
        One ORT inference for B>1 faces (Inswapper128ArcFace + Opal/Pearl, non-Custom).
        Returns None to fall back to per-face run_recognize_direct.
        """
        if len(kps_list) < 2:
            return None
        if arcface_model != "Inswapper128ArcFace":
            return None
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

        ort_session = self.models_processor.models.get(arcface_model)
        if not ort_session:
            ort_session = self.models_processor.load_model(arcface_model)
        if not ort_session:
            return None

        try:
            crops = [
                self._chw112_for_inswapper_arcface(img, kps, similarity_type)
                for kps in kps_list
            ]
            batch = torch.stack(crops, dim=0).float().clone()
            if batch.max() <= 1.0:
                batch = batch * 255.0
            batch.sub_(127.5).div_(127.5)

            session_id = id(ort_session)
            with self._io_cache_lock:
                if session_id not in self._session_io_name_cache:
                    self._session_io_name_cache[session_id] = {
                        "input": ort_session.get_inputs()[0].name,
                        "outputs": [o.name for o in ort_session.get_outputs()],
                    }
                input_name = self._session_io_name_cache[session_id]["input"]
                output_names = self._session_io_name_cache[session_id]["outputs"]

            io_binding = ort_session.io_binding()
            io_binding.bind_input(
                name=input_name,
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=tuple(batch.shape),
                buffer_ptr=batch.data_ptr(),
            )
            for name in output_names:
                io_binding.bind_output(name, self.models_processor.device)

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
            return [emb_arr[i].flatten().astype(np.float32, copy=False) for i in range(len(kps_list))]
        except Exception as e:
            print(f"[WARN] ArcFace batch inference failed, falling back per-face: {e}")
            return None

    def run_recognize(
        self, img, kps, similarity_type="Opal", face_swapper_model="Inswapper128"
    ):
        arcface_model = self.models_processor.get_arcface_model(face_swapper_model)
        return self.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def recognize(self, arcface_model, img, face_kps, similarity_type):
        """
        Generates the face embedding using the specified ArcFace model and alignment strategy.

        Args:
            arcface_model (str): Name of the model to use.
            img (torch.Tensor): Input image tensor (CHW).
            face_kps (np.ndarray): 5 facial landmarks.
            similarity_type (str): Alignment strategy ('Optimal', 'Pearl', 'Opal').

        Returns:
            tuple: (embedding numpy array, cropped_face tensor HWC)
        """
        ort_session = self.models_processor.models.get(arcface_model)
        if not ort_session:
            # This is a safety check; run_recognize_direct should prevent this.
            return None, None

        # --- ALIGNMENT STRATEGIES ---
        if similarity_type == "Optimal":
            img, _ = faceutil.warp_face_by_face_landmark_5(
                img,
                face_kps,
                mode="arcfacemap",
                interpolation=v2.InterpolationMode.BILINEAR,
            )

        elif similarity_type == "Pearl":
            dst = self.models_processor.arcface_dst.copy()
            dst[:, 0] += 8.0
            tform = faceutil.similarity_transform_from_correspondences(face_kps, dst)

            # OPTIMIZED: Direct GPU Warp to 128x128 using Kornia
            M_tensor = (
                torch.from_numpy(tform.params[0:2]).float().unsqueeze(0).to(img.device)
            )
            img_b = img.unsqueeze(0) if img.dim() == 3 else img
            img = kgm.warp_affine(
                img_b.float(),
                M_tensor,
                dsize=(128, 128),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

            # Fast resize to standard 112
            img = v2.functional.resize(img, [112, 112], antialias=True)

        else:
            # Mode 3: Opal (Standard / Default)
            tform = faceutil.similarity_transform_from_correspondences(
                face_kps, self.models_processor.arcface_dst
            )

            # OPTIMIZED: Direct GPU Warp to 112x112 using Kornia (bypasses torchvision crop/affine)
            M_tensor = (
                torch.from_numpy(tform.params[0:2]).float().unsqueeze(0).to(img.device)
            )
            img_b = img.unsqueeze(0) if img.dim() == 3 else img
            img = kgm.warp_affine(
                img_b.float(),
                M_tensor,
                dsize=(112, 112),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

        # --- NORMALIZATION & PRE-PROCESSING ---
        cropped_image = img.permute(1, 2, 0).clone()  # Store for display/debug (H,W,3)

        # Ensure float format
        if img.dtype == torch.uint8:
            img = img.float()

        # We MUST clone the image before doing in-place math if we are
        # not strictly sure that we own a brand new Kornia tensor.
        # "Optimal" mode might pass a reference, causing Race Conditions across threads.
        img = img.clone()

        # OPTIMIZED: In-Place math operations (.sub_ and .div_) to save VRAM fragmentation
        if arcface_model == "Inswapper128ArcFace":
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

        # FS-PERF-02: cache input/output names by session id to avoid repeated ONNX introspection
        # Lock prevents 'dictionary changed size during iteration' crashes when multiple
        # workers encounter a new model ID simultaneously.
        session_id = id(ort_session)
        with self._io_cache_lock:
            if session_id not in self._session_io_name_cache:
                self._session_io_name_cache[session_id] = {
                    "input": ort_session.get_inputs()[0].name,
                    "outputs": [o.name for o in ort_session.get_outputs()],
                }
            input_name = self._session_io_name_cache[session_id]["input"]
            output_names = self._session_io_name_cache[session_id]["outputs"]

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )

        for name in output_names:
            io_binding.bind_output(name, self.models_processor.device)

        # Run the model with lazy build handling (TensorRT safety)
        self._run_model_with_lazy_build_check(arcface_model, ort_session, io_binding)

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
        model = self.models_processor.models.get(model_name)
        if not model:
            print("[ERROR] CSCSArcFace model not loaded in recognize_cscs.")
            return None, None

        io_binding = model.io_binding()

        # SAFETY: Clear bindings to prevent thread caching errors
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(name="output", device_type=self.models_processor.device)

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
        model = self.models_processor.models.get(model_name)
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

        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(name="output", device_type=self.models_processor.device)

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
        io_binding.bind_input(
            name="input_1",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="input_2",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def _calc_emap_latent(self, source_embedding):
        """FS-PERF-05: shared emap-based latent computation extracted from
        calc_inswapper_latent and calc_swapper_latent_iss."""
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

        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 128, 128),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 128, 128),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def run_inswapper_batched(
        self, images: torch.Tensor, embedding: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Batched InSwapper inference for pixel-shift resolution mode."""

        torch_model = self._get_inswapper_torch()
        with torch.no_grad():
            # Same lock as B=1 CUDA-graph path: one shared InSwapperTorch instance.
            with self._inswapper_b1_lock:
                inp = images if images.is_contiguous() else images.contiguous()
                emb = (
                    embedding
                    if embedding.is_contiguous()
                    else embedding.contiguous()
                )
                B = inp.shape[0]
                if emb.shape[0] not in (1, B):
                    raise ValueError(
                        f"InSwapper batched: embedding batch {emb.shape[0]} vs images {B}"
                    )
                result = torch_model(inp, emb)  # [B, 3, 128, 128] float32
                output.copy_(result)
                if self.models_processor.device == "cuda":
                    torch.cuda.current_stream().synchronize()

    def calc_swapper_latent_ghost(self, source_embedding):
        latent = source_embedding.reshape((1, -1))

        return latent

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
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
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
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
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="onnx::Gemm_1",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
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
        session_id = id(ghostfaceswap_model)
        with self._io_cache_lock:
            if session_id not in self._session_io_name_cache:
                self._session_io_name_cache[session_id] = {
                    "input": ghostfaceswap_model.get_inputs()[0].name,
                    "outputs": [o.name for o in ghostfaceswap_model.get_outputs()],
                }
            output_name = self._session_io_name_cache[session_id]["outputs"][0]

        io_binding = ghostfaceswap_model.io_binding()
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling
        self._run_model_with_lazy_build_check(
            model_name, ghostfaceswap_model, io_binding
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

        session_id = id(ghostfaceswap_model)
        with self._io_cache_lock:
            if session_id not in self._session_io_name_cache:
                self._session_io_name_cache[session_id] = {
                    "input": ghostfaceswap_model.get_inputs()[0].name,
                    "outputs": [o.name for o in ghostfaceswap_model.get_outputs()],
                }
            output_name = self._session_io_name_cache[session_id]["outputs"][0]

        io_binding = ghostfaceswap_model.io_binding()
        try:
            io_binding.bind_input(
                name="target",
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=(B, 3, 256, 256),
                buffer_ptr=inp.data_ptr(),
            )
            io_binding.bind_input(
                name="source",
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=(B, 512),
                buffer_ptr=emb.data_ptr(),
            )
            io_binding.bind_output(
                name=output_name,
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=(B, 3, 256, 256),
                buffer_ptr=out.data_ptr(),
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

    def calc_hyperswap_latent(self, source_embedding):
        """FaceFusion HyperSwap: L2-normalized 512-D ArcFace row (1, 512)."""
        if source_embedding is None or len(source_embedding) == 0:
            return None
        v = np.asarray(source_embedding, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return v.reshape(1, -1)
        return (v / n).reshape(1, -1)

    def run_hyperswap(
        self, image, embedding, output, swapper_model="HyperSwap-v3"
    ):
        if swapper_model == "HyperSwap-v1":
            model_name = "HyperSwapv1"
        elif swapper_model == "HyperSwap-v2":
            model_name = "HyperSwapv2"
        elif swapper_model == "HyperSwap-v3":
            model_name = "HyperSwapv3"
        else:
            print(f"[ERROR] Unknown HyperSwap model: {swapper_model}")
            return

        model = self._load_swapper_model(model_name)
        if not model:
            print(f"[ERROR] {model_name} model not loaded.")
            return

        io_binding = model.io_binding()
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def run_hyperswap_batched(
        self,
        images: torch.Tensor,
        embedding: torch.Tensor,
        output: torch.Tensor,
        swapper_model: str = "HyperSwap-v3",
    ) -> bool:
        if swapper_model == "HyperSwap-v1":
            model_name = "HyperSwapv1"
        elif swapper_model == "HyperSwap-v2":
            model_name = "HyperSwapv2"
        elif swapper_model == "HyperSwap-v3":
            model_name = "HyperSwapv3"
        else:
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

        io_binding = model.io_binding()
        try:
            io_binding.bind_input(
                name="target",
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=(B, 3, 256, 256),
                buffer_ptr=inp.data_ptr(),
            )
            io_binding.bind_input(
                name="source",
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=(B, 512),
                buffer_ptr=emb.data_ptr(),
            )
            io_binding.bind_output(
                name="output",
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=(B, 3, 256, 256),
                buffer_ptr=out.data_ptr(),
            )
            self._run_model_with_lazy_build_check(model_name, model, io_binding)
            return True
        except Exception as e:
            print(
                f"[WARN] HyperSwap batched ORT bind/run failed (B={B}): {e!s:.200}",
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
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=target_rgb_256.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 112, 112),
            buffer_ptr=source_rgb_112.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
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
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=target_norm_256.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=source_rgb_256.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )
        self._run_model_with_lazy_build_check(model_name, model, io_binding)

    def calc_rehiface_source_latent(self, source_embedding):
        """ReHiFace-S: ArcFace 512-D → crossface_hififace → L2-normalized (1, 512)."""
        if source_embedding is None or len(source_embedding) == 0:
            return None
        cross = self.models_processor.models.get("CrossFaceHiFaceS")
        if cross is None:
            cross = self.models_processor.load_model("CrossFaceHiFaceS")
        if cross is None:
            print("[ERROR] CrossFaceHiFaceS model not loaded.")
            return None

        emb = (
            torch.from_numpy(np.asarray(source_embedding, dtype=np.float32).reshape(1, -1))
            .contiguous()
            .to(self.models_processor.device)
        )
        out_t = torch.empty(
            (1, 512),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()
        io_binding = cross.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=emb.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=out_t.data_ptr(),
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
        cross = self.models_processor.models.get("CrossFaceSimSwap")
        if cross is None:
            cross = self.models_processor.load_model("CrossFaceSimSwap")
        if cross is None:
            print("[ERROR] CrossFaceSimSwap model not loaded.")
            return None

        emb = (
            torch.from_numpy(np.asarray(source_embedding, dtype=np.float32).reshape(1, -1))
            .contiguous()
            .to(self.models_processor.device)
        )
        out_t = torch.empty(
            (1, 512),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()
        io_binding = cross.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=emb.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=out_t.data_ptr(),
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
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        self._run_model_with_lazy_build_check(model_name, model, io_binding)
