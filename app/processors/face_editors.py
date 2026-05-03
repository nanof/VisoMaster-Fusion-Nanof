import pickle
import hashlib
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict
import platform
import os
import threading

import torch
import numpy as np
import torch.nn.functional

from torchvision.transforms import v2

from app.processors.models_data import models_dir
from app.processors.utils import faceutil

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

SYSTEM_PLATFORM = platform.system()

# Thread-local: LivePortrait stats during apply_face_expression_restorer (expr_profile).
_lp_stitch_tls = threading.local()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def lp_stitch_perf_env_active() -> bool:
    """Env-based opt-in: LP_STITCH, PERF_STAGES, or PERF_BUNDLE."""
    if _env_truthy("VISIOMASTER_PERF_LP_STITCH"):
        return True
    if _env_truthy("VISIOMASTER_PERF_STAGES"):
        return True
    if _env_truthy("VISIOMASTER_PERF_BUNDLE"):
        return True
    return False


def liveportrait_wall_ms_enabled() -> bool:
    """If set, accumulate host-side wall time around stitch/warp ONNX binds (rough vs GPU)."""
    return _env_truthy("VISIOMASTER_PERF_LIVEPORTRAIT_MS")


@contextmanager
def expression_lp_stitch_profile_scope(models_processor: "ModelsProcessor"):
    """Count lp_stitch / lp_warp_decode during Face Expression Restorer; print on exit."""
    tls = _lp_stitch_tls
    overlay = False
    try:
        overlay = bool(
            models_processor.main_window.control.get(
                "PipelineProfileOverlayEnableToggle", False
            )
        ) or bool(
            models_processor.main_window.control.get(
                "PipelineProfileDockEnableToggle", False
            )
        )
    except Exception:
        overlay = False
    active = lp_stitch_perf_env_active() or overlay
    tls.count = 0
    tls.warp_count = 0
    tls.stitch_ms = 0.0
    tls.warp_ms = 0.0
    prev = getattr(tls, "expr_profile", False)
    tls.expr_profile = active
    try:
        yield
    finally:
        if active:
            n = int(getattr(tls, "count", 0))
            wn = int(getattr(tls, "warp_count", 0))
            line = (
                f"[PERF-LIVEPORTRAIT] apply_face_expression_restorer "
                f"lp_stitch_calls={n} lp_warp_decode_calls={wn}"
            )
            if liveportrait_wall_ms_enabled():
                sm = float(getattr(tls, "stitch_ms", 0.0))
                wm = float(getattr(tls, "warp_ms", 0.0))
                line += f" stitch_wall_ms={sm:.2f} warp_wall_ms={wm:.2f}"
            print(line, flush=True)
        tls.expr_profile = prev


def slice_lp_motion_batch(motion: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Slice one batch element from lp_motion_extractor output (after flag_refine_info)."""
    out: Dict[str, Any] = {}
    for key, val in motion.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] > index:
            out[key] = val[index : index + 1].contiguous()
        else:
            out[key] = val
    return out


class FaceEditors:
    """
    Manages face editing functionalities, primarily using the LivePortrait model pipeline.
    This class handles motion extraction, feature extraction, stitching, warping,
    and post-processing effects like makeup application. It is designed to work with
    ONNX Runtime using I/O binding for high performance.
    """

    def __init__(self, models_processor: "ModelsProcessor"):
        """
        Initializes the FaceEditors class.

        Args:
            models_processor (ModelsProcessor): A reference to the main ModelsProcessor instance.
        """
        self.models_processor = models_processor
        self.editor_lock = threading.Lock()
        self.current_face_editor_type: str | None = None
        self.editor_models: Dict[str, list[str]] = {
            "Human-Face": [
                "LivePortraitMotionExtractor",
                "LivePortraitAppearanceFeatureExtractor",
                "LivePortraitStitchingEye",
                "LivePortraitStitchingLip",
                "LivePortraitStitching",
                "LivePortraitWarpingSpade",
            ]
        }
        # Pre-create a faded mask for cropping operations to be used in the LivePortrait pipeline.
        self.lp_mask_crop = faceutil.create_faded_inner_mask(
            size=(512, 512),
            border_thickness=5,
            fade_thickness=15,
            blur_radius=5,
            device=self.models_processor.device,
        )
        self.lp_mask_crop = torch.unsqueeze(self.lp_mask_crop, 0)
        try:
            # Load a pre-calculated lip array used for lip retargeting in the LivePortrait model.
            self.lp_lip_array = np.array(self.load_lip_array())
        except Exception as e:
            # FE-02: broaden exception handling to catch any load failure, not just FileNotFoundError
            import logging

            logging.warning(f"lip_array load failed: {e}")
            self.lp_lip_array = None

    def load_lip_array(self):
        """
        Loads the lip array data from a pickle file required by the LivePortrait model.
        A SHA-256 digest of the file is verified before loading as a security measure.
        # FE-01: SHA-256 check fallback — to lock in a known-good hash, set
        #   KNOWN_LIP_ARRAY_SHA256 to the hex digest of the trusted file, e.g.:
        #   KNOWN_LIP_ARRAY_SHA256 = "abcdef1234..."
        #   and uncomment the verification block below.

        Returns:
            The loaded data from the pickle file.
        """
        import logging

        # FE-01: SHA-256 verification before loading the pickle file
        # Update KNOWN_LIP_ARRAY_SHA256 with the trusted file's digest to enforce integrity.
        KNOWN_LIP_ARRAY_SHA256 = (
            None  # Set to a hex-digest string to enable strict check
        )

        # Use os.path.join for better cross-platform compatibility.
        lip_array_path = os.path.join(models_dir, "liveportrait_onnx", "lip_array.pkl")

        with open(lip_array_path, "rb") as f:
            file_bytes = f.read()

        actual_sha256 = hashlib.sha256(file_bytes).hexdigest()
        if KNOWN_LIP_ARRAY_SHA256 is not None:
            if actual_sha256 != KNOWN_LIP_ARRAY_SHA256:
                raise RuntimeError(
                    f"lip_array.pkl failed SHA-256 verification. "
                    f"Expected: {KNOWN_LIP_ARRAY_SHA256}, Got: {actual_sha256}"
                )
        else:
            logging.debug(f"lip_array.pkl SHA-256: {actual_sha256}")

        return pickle.loads(file_bytes)

    def unload_models(self):
        if self.current_face_editor_type:
            models_to_unload = self.editor_models.get(self.current_face_editor_type, [])
            for model_name in models_to_unload:
                self.models_processor.unload_model(model_name)
            self.current_face_editor_type = None

    def _manage_editor_models(self, face_editor_type: str):
        """
        Manages loading and unloading of model groups for different face editor types.
        If the editor type changes, it unloads the models of the previous type.
        """
        with self.editor_lock:
            if (
                self.current_face_editor_type
                and self.current_face_editor_type != face_editor_type
            ):
                self.unload_models()
            self.current_face_editor_type = face_editor_type

    def _run_onnx_io_binding(
        self,
        model_name: str,
        inputs: Dict[str, torch.Tensor],
        output_spec: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Delegates to ``ModelsProcessor.run_onnx_io_binding`` (graph-aligned dtypes, multi-GPU bind ids)."""
        return self.models_processor.run_onnx_io_binding(
            model_name, inputs, output_spec
        )

    def lp_motion_extractor(self, img, face_editor_type="Human-Face", **kwargs) -> dict:
        """
        Extracts motion-related information from an input image using the LivePortrait motion extractor model.
        This includes head pose (pitch, yaw, roll), expression, and keypoints.
        It supports both TensorRT (async) and ONNX Runtime backends.

        Args:
            img (torch.Tensor): The input image tensor (C, H, W).
            face_editor_type (str): The type of face editor model to use.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the extracted motion parameters.
        """
        self._manage_editor_models(face_editor_type)
        kp_info = {}
        with torch.no_grad():
            model_name = "LivePortraitMotionExtractor"

            if face_editor_type == "Human-Face":
                if not self.models_processor.models.get(model_name):
                    self.models_processor.models[model_name] = (
                        self.models_processor.load_model(model_name)
                    )

            td_in = self.models_processor.get_ort_io_torch_dtype(
                model_name, "img", is_output=False
            )
            I_s = img.float().mul(1.0 / 255.0).clamp(0, 1).to(td_in)
            if I_s.dim() == 3:
                I_s = I_s.unsqueeze(0)
            elif I_s.dim() != 4:
                raise ValueError(
                    "lp_motion_extractor expects img [C,H,W] or [B,C,H,W], got "
                    f"shape {tuple(img.shape)}"
                )
            bs = int(I_s.shape[0])
            if bs < 1 or bs > 8:
                raise ValueError(f"lp_motion_extractor batch size must be 1..8, got {bs}")
            I_s = I_s.contiguous()

            dev = self.models_processor.device
            _od = lambda n: self.models_processor.get_ort_io_torch_dtype(
                model_name, n, is_output=True
            )
            inputs = {"img": I_s}
            output_spec = {
                "pitch": torch.empty((bs, 66), dtype=_od("pitch"), device=dev).contiguous(),
                "yaw": torch.empty((bs, 66), dtype=_od("yaw"), device=dev).contiguous(),
                "roll": torch.empty((bs, 66), dtype=_od("roll"), device=dev).contiguous(),
                "t": torch.empty((bs, 3), dtype=_od("t"), device=dev).contiguous(),
                "exp": torch.empty((bs, 63), dtype=_od("exp"), device=dev).contiguous(),
                "scale": torch.empty((bs, 1), dtype=_od("scale"), device=dev).contiguous(),
                "kp": torch.empty((bs, 63), dtype=_od("kp"), device=dev).contiguous(),
            }

            kp_info = self._run_onnx_io_binding(model_name, inputs, output_spec)

            # Post-process the raw model output to a more usable format.
            if kwargs.get("flag_refine_info", True):
                bs = kp_info["kp"].shape[0]
                # Convert headpose predictions to degrees.
                kp_info["pitch"] = faceutil.headpose_pred_to_degree(kp_info["pitch"])[
                    :, None
                ]  # Bx1
                kp_info["yaw"] = faceutil.headpose_pred_to_degree(kp_info["yaw"])[
                    :, None
                ]  # Bx1
                kp_info["roll"] = faceutil.headpose_pred_to_degree(kp_info["roll"])[
                    :, None
                ]  # Bx1
                # Reshape keypoints and expression tensors.
                kp_info["kp"] = kp_info["kp"].reshape(bs, -1, 3)  # BxNx3
                kp_info["exp"] = kp_info["exp"].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def lp_appearance_feature_extractor(self, img, face_editor_type="Human-Face"):
        """
        Extracts the appearance feature volume from an image using the LivePortrait appearance model.
        This feature volume is a 3D representation of the face's appearance.

        Args:
            img (torch.Tensor): The input image tensor (C, H, W).
            face_editor_type (str): The type of face editor model to use.

        Returns:
            torch.Tensor: The extracted appearance feature volume.
        """
        self._manage_editor_models(face_editor_type)
        with torch.no_grad():
            model_name = "LivePortraitAppearanceFeatureExtractor"

            if face_editor_type == "Human-Face":
                if not self.models_processor.models.get(model_name):
                    self.models_processor.models[model_name] = (
                        self.models_processor.load_model(model_name)
                    )

            td_in = self.models_processor.get_ort_io_torch_dtype(
                model_name, "img", is_output=False
            )
            td_out = self.models_processor.get_ort_io_torch_dtype(
                model_name, "output", is_output=True
            )
            I_s = img.float().mul(1.0 / 255.0).clamp(0, 1).to(td_in)
            I_s = torch.unsqueeze(I_s, 0).contiguous()

            inputs = {"img": I_s}
            output_spec = {
                "output": torch.empty(
                    (1, 32, 16, 64, 64),
                    dtype=td_out,
                    device=self.models_processor.device,
                ).contiguous()
            }
            results = self._run_onnx_io_binding(model_name, inputs, output_spec)
            output = results["output"]

        return output

    def lp_retarget_eye(
        self,
        kp_source: torch.Tensor,
        eye_close_ratio: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        """
        Calculates the motion delta to adjust eye keypoints based on a target eye-close ratio.

        Args:
            kp_source (torch.Tensor): BxNx3 source keypoints.
            eye_close_ratio (torch.Tensor): Bx3 target eye-close ratio.
            face_editor_type (str): The type of face editor model to use.

        Returns:
            torch.Tensor: BxNx3 delta to be added to the source keypoints to achieve the target expression.
        """
        self._manage_editor_models(face_editor_type)
        with torch.no_grad():
            model_name = "LivePortraitStitchingEye"

            if face_editor_type == "Human-Face":
                if not self.models_processor.models.get(model_name):
                    self.models_processor.models[model_name] = (
                        self.models_processor.load_model(model_name)
                    )

            td_in = self.models_processor.get_ort_io_torch_dtype(
                model_name, "input", is_output=False
            )
            td_out = self.models_processor.get_ort_io_torch_dtype(
                model_name, "output", is_output=True
            )
            feat_eye = faceutil.concat_feat(kp_source, eye_close_ratio).to(td_in).contiguous()

            inputs = {"input": feat_eye}
            output_spec = {
                "output": torch.empty(
                    (1, 63),
                    dtype=td_out,
                    device=self.models_processor.device,
                ).contiguous()
            }
            results = self._run_onnx_io_binding(model_name, inputs, output_spec)
            delta = results["output"]

        # Reshape the output delta to match the keypoint format.
        return delta.reshape(-1, kp_source.shape[1], 3)

    def lp_retarget_lip(
        self,
        kp_source: torch.Tensor,
        lip_close_ratio: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        """
        Calculates the motion delta to adjust lip keypoints based on a target lip-close ratio.

        Args:
            kp_source (torch.Tensor): BxNx3 source keypoints.
            lip_close_ratio (torch.Tensor): Bx2 target lip-close ratio.
            face_editor_type (str): The type of face editor model to use.

        Returns:
            torch.Tensor: BxNx3 delta to be added to the source keypoints.
        """
        self._manage_editor_models(face_editor_type)
        with torch.no_grad():
            model_name = "LivePortraitStitchingLip"

            if face_editor_type == "Human-Face":
                if not self.models_processor.models.get(model_name):
                    self.models_processor.models[model_name] = (
                        self.models_processor.load_model(model_name)
                    )

            td_in = self.models_processor.get_ort_io_torch_dtype(
                model_name, "input", is_output=False
            )
            td_out = self.models_processor.get_ort_io_torch_dtype(
                model_name, "output", is_output=True
            )
            feat_lip = faceutil.concat_feat(kp_source, lip_close_ratio).to(td_in).contiguous()

            inputs = {"input": feat_lip}
            output_spec = {
                "output": torch.empty(
                    (1, 63),
                    dtype=td_out,
                    device=self.models_processor.device,
                ).contiguous()
            }
            results = self._run_onnx_io_binding(model_name, inputs, output_spec)
            delta = results["output"]

        return delta.reshape(-1, kp_source.shape[1], 3)

    def lp_stitch(
        self,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        """
        Calculates the raw stitching delta between source and driving keypoints.

        Args:
            kp_source (torch.Tensor): BxNx3 source keypoints.
            kp_driving (torch.Tensor): BxNx3 driving keypoints.
            face_editor_type (str): The type of face editor model to use.

        Returns:
            torch.Tensor: A raw delta tensor representing the difference.
        """
        self._manage_editor_models(face_editor_type)
        tls = _lp_stitch_tls
        prof = getattr(tls, "expr_profile", False)
        t0 = time.perf_counter() if prof and liveportrait_wall_ms_enabled() else None
        with torch.no_grad():
            model_name = "LivePortraitStitching"

            if face_editor_type == "Human-Face":
                if not self.models_processor.models.get(model_name):
                    self.models_processor.models[model_name] = (
                        self.models_processor.load_model(model_name)
                    )

            td_in = self.models_processor.get_ort_io_torch_dtype(
                model_name, "input", is_output=False
            )
            td_out = self.models_processor.get_ort_io_torch_dtype(
                model_name, "output", is_output=True
            )
            feat_stitching = faceutil.concat_feat(kp_source, kp_driving).to(td_in).contiguous()

            inputs = {"input": feat_stitching}
            output_spec = {
                "output": torch.empty(
                    (1, 65),
                    dtype=td_out,
                    device=self.models_processor.device,
                ).contiguous()
            }
            results = self._run_onnx_io_binding(model_name, inputs, output_spec)
            delta = results["output"]

        if t0 is not None:
            tls.stitch_ms = float(getattr(tls, "stitch_ms", 0.0)) + (
                time.perf_counter() - t0
            )
        if prof:
            tls.count = int(getattr(tls, "count", 0)) + 1

        return delta

    def lp_stitching(
        self,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        """
        Performs the full stitching process by calculating a reference delta and applying
        the true motion delta to animate the driving keypoints.

        Args:
            kp_source (torch.Tensor): BxNx3 source keypoints.
            kp_driving (torch.Tensor): BxNx3 driving keypoints.
            face_editor_type (str): The type of face editor model to use.

        Returns:
            torch.Tensor: The new, animated driving keypoints.
        """
        bs, num_kp = kp_source.shape[:2]

        # Calculate a "default" delta by comparing the source keypoints to themselves.
        # This establishes a baseline for a neutral expression.
        kp_driving_default = kp_source.clone()
        # FE-04: call self.lp_stitch directly, not via models_processor
        default_delta = self.lp_stitch(
            kp_source, kp_driving_default, face_editor_type=face_editor_type
        )

        # Separate the default delta into expression and translation/rotation components.
        default_delta_exp = (
            default_delta[..., : 3 * num_kp].reshape(bs, num_kp, 3).clone()
        )
        default_delta_tx_ty = (
            default_delta[..., 3 * num_kp : 3 * num_kp + 2].reshape(bs, 1, 2).clone()
        )

        # Calculate the new delta based on the actual driving keypoints.
        kp_driving_new = kp_driving.clone()
        # FE-04: call self.lp_stitch directly, not via models_processor
        delta = self.lp_stitch(
            kp_source, kp_driving_new, face_editor_type=face_editor_type
        )

        # Separate the new delta into components.
        delta_exp = delta[..., : 3 * num_kp].reshape(bs, num_kp, 3).clone()
        delta_tx_ty = delta[..., 3 * num_kp : 3 * num_kp + 2].reshape(bs, 1, 2).clone()

        # The true motion delta is the difference between the new delta and the default delta.
        delta_exp_diff = delta_exp - default_delta_exp
        delta_tx_ty_diff = delta_tx_ty - default_delta_tx_ty

        # Apply the motion delta to the driving keypoints to create the final animation.
        kp_driving_new += delta_exp_diff
        kp_driving_new[..., :2] += delta_tx_ty_diff

        return kp_driving_new

    def lp_warp_decode(
        self,
        feature_3d: torch.Tensor,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        """
        Generates the final animated image by warping the 3D feature volume according to the
        source and driving keypoints, then decoding the result into an image.

        Args:
            feature_3d (torch.Tensor): Bx32x16x64x64 appearance feature volume.
            kp_source (torch.Tensor): BxNx3 source keypoints.
            kp_driving (torch.Tensor): BxNx3 animated driving keypoints.
            face_editor_type (str): The type of face editor model to use.

        Returns:
            torch.Tensor: The final warped and decoded image tensor.
        """
        self._manage_editor_models(face_editor_type)
        tls = _lp_stitch_tls
        prof = getattr(tls, "expr_profile", False)
        t0 = time.perf_counter() if prof and liveportrait_wall_ms_enabled() else None
        with torch.no_grad():
            model_name = "LivePortraitWarpingSpade"

            if face_editor_type == "Human-Face":
                if not self.models_processor.models.get(model_name):
                    self.models_processor.models[model_name] = (
                        self.models_processor.load_model(model_name)
                    )

            t_f = self.models_processor.get_ort_io_torch_dtype(
                model_name, "feature_3d", is_output=False
            )
            t_kd = self.models_processor.get_ort_io_torch_dtype(
                model_name, "kp_driving", is_output=False
            )
            t_ks = self.models_processor.get_ort_io_torch_dtype(
                model_name, "kp_source", is_output=False
            )
            t_out = self.models_processor.get_ort_io_torch_dtype(
                model_name, "out", is_output=True
            )
            feature_3d = feature_3d.to(t_f).contiguous()
            kp_source = kp_source.to(t_ks).contiguous()
            kp_driving = kp_driving.to(t_kd).contiguous()

            inputs = {
                "feature_3d": feature_3d,
                "kp_driving": kp_driving,
                "kp_source": kp_source,
            }
            output_spec = {
                "out": torch.empty(
                    (1, 3, 512, 512),
                    dtype=t_out,
                    device=self.models_processor.device,
                ).contiguous()
            }
            results = self._run_onnx_io_binding(model_name, inputs, output_spec)
            out = results["out"]

        if t0 is not None:
            tls.warp_ms = float(getattr(tls, "warp_ms", 0.0)) + (
                time.perf_counter() - t0
            )
        if prof:
            tls.warp_count = int(getattr(tls, "warp_count", 0)) + 1

        # Paste-back (Kornia) in FrameEdits is exercised in FP32 in practice.
        if out.dtype == torch.float16:
            out = out.float()
        return out

    def _get_faceparser_labels_via_facemasks(
        self, img_uint8_3x512x512: torch.Tensor
    ) -> torch.Tensor:
        """
        Gets semantic face parsing labels by calling the face parser model.
        The model runs on a 512x512 image and this function returns native
        512x512 labels for full-resolution compatibility with other modules.

        Args:
            img_uint8_3x512x512 (torch.Tensor): Input image tensor [3,512,512] uint8 (0..255).

        Returns:
            torch.Tensor: Label map tensor [512,512] of type torch.long.
        """
        fm = getattr(self.models_processor, "face_masks", None)
        if fm is None or not hasattr(fm, "_faceparser_labels"):
            raise RuntimeError(
                "models_processor.face_masks._faceparser_labels is not available."
            )
        return fm._faceparser_labels(img_uint8_3x512x512, None)

    def face_parser_makeup_direct_rgb_masked(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        color=None,
        blend_factor: float = 0.2,
    ) -> torch.Tensor:
        """
        Applies a specified RGB color to a masked region using Photoshop-like Overlay blending
        for realistic hair and makeup (preserving highlights and shadows).
        """
        device = img.device
        color = color or [230, 50, 20]
        blend_factor = float(max(0.0, min(1.0, blend_factor)))

        # Normalize target color to [0,1] range.
        r, g, b = [c / 255.0 for c in color]
        tar_color = torch.tensor([r, g, b], dtype=torch.float32, device=device).view(
            3, 1, 1
        )

        # Ensure mask is a float tensor in [0,1] range.
        if mask.dtype == torch.bool:
            m = mask.float()
        else:
            m = mask.clamp(0.0, 1.0).float()
        m = m.unsqueeze(0)  # [1,H,W]

        # Calculate the base weight.
        w = m * blend_factor

        t512_mask = v2.Resize(
            (512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )
        w = t512_mask(w)
        w = w.clamp(0, 1)

        gauss = v2.GaussianBlur(kernel_size=5, sigma=1.5)
        w = gauss(w)

        img_f = img.float() / 255.0

        cond = img_f < 0.5
        overlay = torch.where(
            cond, 2.0 * img_f * tar_color, 1.0 - 2.0 * (1.0 - img_f) * (1.0 - tar_color)
        )

        overlay = overlay.clamp(0.0, 1.0)

        out = img_f * (1.0 - w) + overlay * w

        # Convert back to uint8 [0,255] range.
        out = (out * 255.0).clamp(0, 255).to(torch.uint8)
        return out

    def face_parser_makeup_direct_rgb(
        self, img, parsing, part=(17,), color=None, blend_factor=0.2
    ):
        """
        Applies makeup to specific parts of a face based on a semantic parsing map.

        Args:
            img (torch.Tensor): Image tensor [3,H,W] uint8.
            parsing (torch.Tensor): Parsing map, can be [H,W] labels or [1,19,H,W] logits.
            part (tuple, optional): A tuple of class indices to apply makeup to. Defaults to (17,) (hair).
            color (list, optional): [R,G,B] color. Defaults to None.
            blend_factor (float, optional): Blending factor. Defaults to 0.2.
        """
        device = img.device
        color = color or [230, 50, 20]
        blend_factor = float(max(0.0, min(1.0, blend_factor)))

        # Convert parsing map (logits or labels) to a label tensor.
        if parsing.dim() == 2:
            labels = parsing.to(torch.long)
        elif parsing.dim() == 4 and parsing.shape[0] == 1 and parsing.shape[1] == 19:
            labels = parsing.argmax(dim=1).squeeze(0).to(torch.long)
        else:
            raise ValueError(
                f"Unsupported parsing tensor shape: {tuple(parsing.shape)}"
            )

        # Create a boolean mask for the target parts.
        if isinstance(part, tuple):
            m = torch.zeros_like(labels, dtype=torch.bool, device=device)
            for p in part:
                m |= labels == int(p)
        else:
            m = labels == int(part)

        # Apply the color to the generated mask.
        return self.face_parser_makeup_direct_rgb_masked(
            img=img, mask=m, color=color, blend_factor=blend_factor
        )

    def apply_face_makeup(self, img, parameters):
        """
        Orchestrates the application of various makeup effects to a face image based on parameters.

        Args:
            img (torch.Tensor): Input image [3,512,512] uint8.
            parameters (dict): A dictionary of makeup parameters from the UI.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - out_img (torch.Tensor): The output image with makeup applied [3,512,512] uint8.
                - combined_mask (torch.Tensor): A combined mask of all applied effects [1,512,512] float.
        """
        device = img.device

        # 1. Get semantic labels from the face parser model.
        labels = self._get_faceparser_labels_via_facemasks(
            img
        )  # Returns [256,256] long tensor

        # 2. Create a working copy of the image.
        out = img.clone()

        # 3. Apply color to each enabled facial area.
        if parameters.get("FaceMakeupEnableToggle", False):
            color = [
                parameters["FaceMakeupRedSlider"],
                parameters["FaceMakeupGreenSlider"],
                parameters["FaceMakeupBlueSlider"],
            ]
            out = self.face_parser_makeup_direct_rgb(
                out,
                labels,
                part=(1, 7, 8, 10),
                color=color,
                blend_factor=parameters["FaceMakeupBlendAmountDecimalSlider"],
            )

        if parameters.get("HairMakeupEnableToggle", False):
            color = [
                parameters["HairMakeupRedSlider"],
                parameters["HairMakeupGreenSlider"],
                parameters["HairMakeupBlueSlider"],
            ]
            out = self.face_parser_makeup_direct_rgb(
                out,
                labels,
                # FE-08: part must be a tuple, not a bare int
                part=(17,),
                color=color,
                blend_factor=parameters["HairMakeupBlendAmountDecimalSlider"],
            )

        if parameters.get("EyeBrowsMakeupEnableToggle", False):
            color = [
                parameters["EyeBrowsMakeupRedSlider"],
                parameters["EyeBrowsMakeupGreenSlider"],
                parameters["EyeBrowsMakeupBlueSlider"],
            ]
            out = self.face_parser_makeup_direct_rgb(
                out,
                labels,
                part=(2, 3),
                color=color,
                blend_factor=parameters["EyeBrowsMakeupBlendAmountDecimalSlider"],
            )

        if parameters.get("LipsMakeupEnableToggle", False):
            color = [
                parameters["LipsMakeupRedSlider"],
                parameters["LipsMakeupGreenSlider"],
                parameters["LipsMakeupBlueSlider"],
            ]
            out = self.face_parser_makeup_direct_rgb(
                out,
                labels,
                part=(12, 13),
                color=color,
                blend_factor=parameters["LipsMakeupBlendAmountDecimalSlider"],
            )

        # 4. Generate a combined mask of all applied effects for debugging or further processing.
        face_attributes = {
            1: parameters.get("FaceMakeupEnableToggle", False),
            2: parameters.get("EyeBrowsMakeupEnableToggle", False),
            3: parameters.get("EyeBrowsMakeupEnableToggle", False),
            4: parameters.get("EyesMakeupEnableToggle", False),
            5: parameters.get("EyesMakeupEnableToggle", False),
            7: parameters.get("FaceMakeupEnableToggle", False),
            8: parameters.get("FaceMakeupEnableToggle", False),
            10: parameters.get("FaceMakeupEnableToggle", False),
            12: parameters.get("LipsMakeupEnableToggle", False),
            13: parameters.get("LipsMakeupEnableToggle", False),
            17: parameters.get("HairMakeupEnableToggle", False),
        }

        combined_mask = torch.zeros_like(labels, dtype=torch.float32, device=device)

        for attr, enabled in face_attributes.items():
            if not enabled:
                continue

            combined_mask = torch.max(combined_mask, (labels == int(attr)).float())

        combined_mask = combined_mask.unsqueeze(0)

        t512_mask = v2.Resize(
            (512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )
        combined_mask = t512_mask(combined_mask)
        # FE-09: combined_mask is a float tensor in [0,1]; clamp to (0,1) not (0,255)
        combined_mask = combined_mask.clamp(0, 1)

        out_final = out.to(torch.uint8)
        return out_final, combined_mask
