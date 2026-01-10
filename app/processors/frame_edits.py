from typing import TYPE_CHECKING

import torch
import threading
import numpy as np
from torchvision.transforms import v2
from skimage import transform as trans

from app.processors.utils import faceutil

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FrameEdits:
    """
    Manages Face Editing operations (Expression restoration, LivePortrait editing, Makeup).
    Moves high-level processing logic out of FrameWorker.

    This class handles the 'LivePortrait' pipeline, including:
    - Motion Extraction (Pose, Expression)
    - Temporal Smoothing (SmartSmoother & OneEuroFilter)
    - Feature Retargeting (Eyes, Lips)
    - Image Warping and Pasting
    """

    def __init__(self, models_processor: "ModelsProcessor"):
        """
        Initializes the FrameEdits class.

        Args:
            models_processor: Reference to the central model manager (provides models & device).
        """
        self.models_processor = models_processor

        # Transforms will be updated per frame/settings via set_transforms
        self.t256_face = None
        self.interpolation_expression_faceeditor_back = None

        # SHARED STATE INITIALIZATION
        with self.models_processor.model_lock:
            if not hasattr(self.models_processor, "shared_smoothing_states"):
                self.models_processor.shared_smoothing_states = {
                    # Use of sequential wrapper
                    "target_smoother": faceutil.SequentialSmartSmoother(),
                    "driving_smoother": faceutil.SequentialSmartSmoother(),
                    "eyes_ratio_filter": faceutil.OneEuroFilter(
                        min_cutoff=1.0, beta=1.2
                    ),
                    "lip_ratio_filter": faceutil.OneEuroFilter(
                        min_cutoff=0.8, beta=1.0
                    ),
                    "eyes_lock": threading.Lock(),
                    "lips_lock": threading.Lock(),
                    "last_global_frame": -1,
                }

    def _get_shared_state(self):
        return getattr(self.models_processor, "shared_smoothing_states", None)

    def reset_temporal_state(self):
        state = self._get_shared_state()
        if state:
            state["driving_smoother"].reset()
            state["target_smoother"].reset()
            with state["eyes_lock"]:
                state["eyes_ratio_filter"].reset()
            with state["lips_lock"]:
                state["lip_ratio_filter"].reset()
            state["last_global_frame"] = -1

    def check_and_update_temporal_gap(self, frame_number: int):
        """
        Intelligent Reset Logic for Async Processing.
        Detects if we jumped too far (Scene Cut) or went backward (Seek).
        """
        state = self._get_shared_state()
        if not state:
            return

        last_frame = state["last_global_frame"]

        # Heuristic: If we jumped more than 30 frames forward or went backward, treat as new scene.
        # This handles scene cuts or user seeking in the timeline.
        GAP_THRESHOLD = 30

        if last_frame != -1:
            if (
                abs(frame_number - last_frame) > GAP_THRESHOLD
                or frame_number < last_frame - 5
            ):
                # Force resynchronization of the sequencer
                # We do not print to console in production to avoid log spamming
                state["driving_smoother"].set_next_frame_index(frame_number)
                state["target_smoother"].set_next_frame_index(frame_number)
                state["last_global_frame"] = frame_number
                return

        # Update high-water mark only if we moved forward naturally
        if frame_number > state["last_global_frame"]:
            state["last_global_frame"] = frame_number

    def set_transforms(self, t256_face, interpolation_expression_faceeditor_back):
        """
        Updates the scaling transforms and interpolation modes based on current control settings.
        Called from FrameWorker.set_scaling_transforms.
        """
        self.t256_face = t256_face
        self.interpolation_expression_faceeditor_back = (
            interpolation_expression_faceeditor_back
        )

    def apply_face_expression_restorer(
        self,
        driving: torch.Tensor,
        target: torch.Tensor,
        parameters: dict,
        frame_number: int = -1,
    ) -> torch.Tensor:
        """
        Restores the expression of the face using the LivePortrait model pipeline.

        Features:
        - Target Smoothing to reduce global jitter.
        - Ratio Smoothing (Eyes/Lips) to reduce high-frequency jitter on cilia/mouth.
        - Smart Dynamic Boost for Micro-Expressions: Amplifies subtle motions while
          protecting strong expressions from distortion.

        Args:
            driving: The original face tensor (source of expression).
            target: The swapped face tensor (destination of expression).
            parameters: Dictionary containing UI parameters.

        Returns:
            torch.Tensor: The expression-restored face image.
        """
        # 1. SETUP THE ASYNCHRONOUS CONTEXT
        if frame_number >= 0:
            self.check_and_update_temporal_gap(frame_number)

        current_stream = torch.cuda.current_stream()

        with torch.cuda.stream(current_stream):
            # --- CONFIGURATION ---
            use_mean_eyes = parameters.get("LandmarkMeanEyesToggle", False)
            use_smoothing = parameters.get(
                "FaceExpressionTemporalSmoothingToggle", True
            )
            state = self._get_shared_state()
            # Sanitized Mode Selection
            mode_raw = parameters.get("FaceExpressionModeSelection", "Advanced")
            mode = mode_raw.strip() if isinstance(mode_raw, str) else "Advanced"

            # PARAMETER: Micro-Expression Strength
            # 1.0 = Realistic Transfer. < 1.0 = Dampened. > 1.0 = Boosted.
            micro_expression_boost = parameters.get(
                "FaceExpressionMicroExpressionBoostDecimalSlider", 0.50
            )

            # --- 1. DRIVING FACE PROCESSING ---
            # Detect landmarks on the driving face
            _, driving_lmk_crop, _ = self.models_processor.run_detect_landmark(
                driving,
                bbox=np.array([0, 0, 512, 512]),
                det_kpss=[],
                detect_mode="203",
                score=0.5,
                from_points=False,
                use_mean_eyes=use_mean_eyes,
            )

            interp_mode = (
                self.interpolation_expression_faceeditor_back
                if self.interpolation_expression_faceeditor_back is not None
                else v2.InterpolationMode.BILINEAR
            )

            if self.t256_face is None:
                self.t256_face = v2.Resize(
                    (256, 256),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=False,
                )

            # Warp driving face
            driving_face_512, _, _ = faceutil.warp_face_by_face_landmark_x(
                driving,
                driving_lmk_crop,
                dsize=512,
                scale=parameters.get("FaceExpressionCropScaleBothDecimalSlider", 2.3),
                vy_ratio=parameters.get(
                    "FaceExpressionVYRatioBothDecimalSlider", -0.125
                ),
                interpolation=interp_mode,
            )

            driving_face_256 = self.t256_face(driving_face_512)

            # Calculate Raw Ratios (Eyes/Lips openness)
            c_d_eyes_lst = faceutil.calc_eye_close_ratio(driving_lmk_crop[None])
            c_d_lip_lst = faceutil.calc_lip_close_ratio(driving_lmk_crop[None])

            # Apply Smoothing to Ratios (Crucial for eyelashes and lip stability)
            if use_smoothing:
                # Manual Lock for simple filters
                with state["eyes_lock"]:
                    c_d_eyes_lst = state["eyes_ratio_filter"](
                        c_d_eyes_lst, t=frame_number
                    )
                with state["lips_lock"]:
                    c_d_lip_lst = state["lip_ratio_filter"](c_d_lip_lst, t=frame_number)

            # Extract Motion from Driving Face
            x_d_i_info = self.models_processor.lp_motion_extractor(
                driving_face_256, "Human-Face"
            )

            # --- SEQUENTIAL SMOOTHING BLOCK (DRIVING) ---
            if use_smoothing:

                def smooth_driving_logic(smoother, info, t_idx):
                    (
                        info["pitch"],
                        info["yaw"],
                        info["roll"],
                        info["t"],
                        info["scale"],
                    ) = smoother.smooth_pose(
                        info["pitch"],
                        info["yaw"],
                        info["roll"],
                        info["t"],
                        info["scale"],
                        frame_t=t_idx,
                    )
                    info["exp"] = smoother.smooth_expression(info["exp"], frame_t=t_idx)
                    return info

                # Thread waits his turn here
                x_d_i_info = state["driving_smoother"].process_ordered(
                    frame_number, smooth_driving_logic, x_d_i_info, frame_number
                )

                pose_damping = 0.8
                x_d_i_info["pitch"] *= pose_damping
                x_d_i_info["yaw"] *= pose_damping
                x_d_i_info["roll"] *= pose_damping

            # --- 2. TARGET FACE ---
            target = target.clamp(0, 255).type(torch.uint8)
            _, source_lmk, _ = self.models_processor.run_detect_landmark(
                target,
                bbox=np.array([0, 0, 512, 512]),
                det_kpss=[],
                detect_mode="203",
                score=0.5,
                from_points=False,
                use_mean_eyes=use_mean_eyes,
            )

            target_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                target,
                source_lmk,
                dsize=512,
                scale=parameters.get("FaceExpressionCropScaleBothDecimalSlider", 2.3),
                vy_ratio=parameters.get(
                    "FaceExpressionVYRatioBothDecimalSlider", -0.125
                ),
                interpolation=interp_mode,
            )
            target_face_256 = self.t256_face(target_face_512)
            x_s_info = self.models_processor.lp_motion_extractor(
                target_face_256, "Human-Face"
            )

            # --- SEQUENTIAL SMOOTHING BLOCK (TARGET) ---
            if use_smoothing:

                def smooth_target_logic(smoother, info, t_idx):
                    (
                        info["pitch"],
                        info["yaw"],
                        info["roll"],
                        info["t"],
                        info["scale"],
                    ) = smoother.smooth_pose(
                        info["pitch"],
                        info["yaw"],
                        info["roll"],
                        info["t"],
                        info["scale"],
                        frame_t=t_idx,
                    )
                    info["exp"] = smoother.smooth_expression(info["exp"], frame_t=t_idx)
                    return info

                x_s_info = state["target_smoother"].process_ordered(
                    frame_number, smooth_target_logic, x_s_info, frame_number
                )

            # Prepare Target Features
            x_c_s = x_s_info["kp"]
            R_s = faceutil.get_rotation_matrix(
                x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
            )
            f_s = self.models_processor.lp_appearance_feature_extractor(
                target_face_256, "Human-Face"
            )
            x_s = faceutil.transform_keypoint(x_s_info)

            face_editor_type = parameters.get("FaceEditorTypeSelection", "Human-Face")

            # --- 3. ZERO-TRANSLATION PRE-CALCULATION ---
            default_delta_raw = self.models_processor.lp_stitch(
                x_s, x_s, face_editor_type
            )
            default_delta_exp = default_delta_raw[..., :-2].reshape(x_s.shape[0], 21, 3)

            # --- INDICES DEFINITION ---
            # Integrated Logic: Brows -> Eyes, Jaw -> Lips
            # Excluding 'face' (cheeks/nose) to preserve identity shape.
            eye_indices = [1, 2, 11, 13, 15, 16, 18]
            lip_indices = [3, 6, 12, 14, 17, 19, 20]

            # Anchor
            R_anchor = R_s
            t_anchor = x_s_info["t"].clone()
            t_anchor[..., 2].fill_(0)
            scale_anchor = x_s_info["scale"]

            # Load Lip Array (Neutral reference for lips)
            lp_lip_array = torch.from_numpy(self.models_processor.lp_lip_array).to(
                dtype=torch.float32, device=self.models_processor.device
            )

            # --- SHARED HELPER FUNCTION ---
            def get_component_motion(
                indices,
                driving_exp,
                multiplier,
                extra_delta=0,
                is_relative=False,
                neutral_ref=None,
                use_boost=False,
            ):
                """
                Helper to calculate motion with 'Smart Dynamic Boost'.
                Args:
                    use_boost: If True, applies the Micro-Expression Logic.
                """
                delta_local = x_s_info["exp"].clone()

                # [CALIBRATION] Internal Attenuation
                # Raw LivePortrait transfer at 1.0 is often topologically exaggerated (Joker effect).
                # We apply a base attenuation of 0.6 so that UI Slider 1.0 feels "Natural".
                calibration_factor = 0.60

                if is_relative:
                    # Relative Motion Calculation
                    ref = neutral_ref if neutral_ref is not None else 0
                    if isinstance(ref, torch.Tensor) and ref.shape[-2] == 21:
                        ref_part = ref[..., indices, :]
                    else:
                        ref_part = ref

                    # Calculate the raw difference (motion intent)
                    raw_diff = driving_exp[:, indices, :] - ref_part

                    # Apply Calibration immediately to the raw difference
                    raw_diff = raw_diff * calibration_factor

                    # --- SMART DYNAMIC BOOST ---
                    # Logic: If boost > 1.0, only enhance small signals (micro-expressions).
                    # Large signals are kept closer to 1.0 to avoid distortion.
                    boost_val = micro_expression_boost if use_boost else 1.0

                    if use_boost and boost_val > 1.0:
                        magnitude = torch.abs(raw_diff)
                        decay = torch.exp(-10.0 * magnitude)
                        dynamic_scale = 1.0 + (boost_val - 1.0) * decay
                        diff = raw_diff * dynamic_scale
                    else:
                        diff = raw_diff * boost_val

                    delta_local[:, indices, :] = x_s_info["exp"][:, indices, :] + diff
                else:
                    # Absolute Motion (Rarely used, but dampened for safety)
                    # We blend the driving exp with the source exp to dampen it.
                    target_exp = driving_exp[:, indices, :]
                    current_exp = x_s_info["exp"][:, indices, :]

                    # Linear Interpolation towards target based on calibration
                    delta_local[:, indices, :] = (
                        current_exp * (1 - calibration_factor)
                        + target_exp * calibration_factor
                    )

                # Projection & Refinement
                x_proj = scale_anchor * (x_c_s @ R_anchor + delta_local) + t_anchor
                raw_delta = self.models_processor.lp_stitch(
                    x_s, x_proj, face_editor_type
                )
                refinement_exp = raw_delta[..., :-2].reshape(x_s.shape[0], 21, 3)

                x_target = x_proj + (refinement_exp - default_delta_exp) + extra_delta
                return (x_target - x_s) * multiplier

            accumulated_motion = torch.zeros_like(x_s)

            # --- MODE PROCESSING ---
            if mode == "Simple":
                # SIMPLE MODE: Explicitly enabled features, ignoring Advanced Toggles
                driving_multiplier = parameters.get(
                    "FaceExpressionFriendlyFactorDecimalSlider", 1.0
                )

                # Logic: "all" means Eyes+Lips. "Face" is ignored in UI logic.
                animation_region = parameters.get(
                    "FaceExpressionAnimationRegionSelection", "all"
                )
                if not animation_region:
                    animation_region = "all"

                has_eyes = "eyes" in animation_region or "all" in animation_region
                has_lips = "lips" in animation_region or "all" in animation_region

                # Lip Normalization Logic (Optional)
                flag_normalize_lip = parameters.get(
                    "FaceExpressionNormalizeLipsEnableToggle", True
                )
                lip_normalize_threshold = parameters.get(
                    "FaceExpressionNormalizeLipsThresholdDecimalSlider", 0.03
                )
                lips_retarget_delta = 0
                if flag_normalize_lip and source_lmk is not None:
                    # Use Smoothed Ratios
                    c_d_lip_before = [0.0]
                    combined_lip_ratio = faceutil.calc_combined_lip_ratio(
                        c_d_lip_before, source_lmk, device=self.models_processor.device
                    )
                    if combined_lip_ratio[0][0] >= lip_normalize_threshold:
                        lips_retarget_delta = self.models_processor.lp_retarget_lip(
                            x_s, combined_lip_ratio
                        )

                # Execute logic (Relative for Eyes, Relative for Lips in Simple Mode)
                if has_eyes:
                    accumulated_motion += get_component_motion(
                        eye_indices,
                        x_d_i_info["exp"],
                        driving_multiplier,
                        is_relative=True,
                        use_boost=False,
                    )

                if has_lips:
                    accumulated_motion += get_component_motion(
                        lip_indices,
                        x_d_i_info["exp"],
                        driving_multiplier,
                        extra_delta=lips_retarget_delta,
                        is_relative=True,
                        neutral_ref=lp_lip_array,
                        use_boost=False,
                    )

            else:
                # ADVANCED MODE
                driving_multiplier_eyes = parameters.get(
                    "FaceExpressionFriendlyFactorEyesDecimalSlider", 1.0
                )
                driving_multiplier_lips = parameters.get(
                    "FaceExpressionFriendlyFactorLipsDecimalSlider", 1.0
                )

                flag_activate_eyes = parameters.get("FaceExpressionEyesToggle", False)
                flag_activate_lips = parameters.get("FaceExpressionLipsToggle", False)

                flag_relative_eyes = parameters.get(
                    "FaceExpressionRelativeEyesToggle", False
                )
                flag_relative_lips = parameters.get(
                    "FaceExpressionRelativeLipsToggle", False
                )

                # --- Normalization Config ---
                flag_normalize_eyes = parameters.get(
                    "FaceExpressionNormalizeEyesBothEnableToggle", True
                )
                eyes_normalize_threshold = parameters.get(
                    "FaceExpressionNormalizeEyesThresholdBothDecimalSlider", 0.40
                )
                eyes_normalize_max = parameters.get(
                    "FaceExpressionNormalizeEyesMaxBothDecimalSlider", 0.50
                )
                combined_eyes_ratio_normalize = None

                # Calculate Normalized Eye Ratio using SMOOTHED list
                if flag_normalize_eyes and source_lmk is not None:
                    c_d_eyes_normalize = c_d_eyes_lst  # Already smoothed above
                    eyes_ratio = np.array([c_d_eyes_normalize[0][0]], dtype=np.float32)
                    eyes_ratio_normalize = max(eyes_ratio, 0.10)
                    eyes_ratio_l = min(c_d_eyes_normalize[0][0], eyes_normalize_max)
                    eyes_ratio_r = min(c_d_eyes_normalize[0][1], eyes_normalize_max)
                    eyes_ratio_max = np.array(
                        [[eyes_ratio_l, eyes_ratio_r]], dtype=np.float32
                    )

                    if eyes_ratio_normalize > eyes_normalize_threshold:
                        combined_eyes_ratio_normalize = (
                            faceutil.calc_combined_eye_ratio_norm(
                                eyes_ratio_max,
                                source_lmk,
                                device=self.models_processor.device,
                            )
                        )
                    else:
                        combined_eyes_ratio_normalize = (
                            faceutil.calc_combined_eye_ratio(
                                eyes_ratio_max,
                                source_lmk,
                                device=self.models_processor.device,
                            )
                        )

                if flag_activate_eyes:
                    eyes_retarget_delta = 0
                    if parameters.get(
                        "FaceExpressionRetargetingEyesBothEnableToggle", False
                    ):
                        eye_mult = parameters.get(
                            "FaceExpressionRetargetingEyesMultiplierBothDecimalSlider",
                            1.0,
                        )

                        if (
                            flag_normalize_eyes
                            and combined_eyes_ratio_normalize is not None
                        ):
                            target_eye_ratio = combined_eyes_ratio_normalize
                        else:
                            # Use Smoothed Ratios
                            target_eye_ratio = faceutil.calc_combined_eye_ratio(
                                c_d_eyes_lst,
                                source_lmk,
                                device=self.models_processor.device,
                            )

                        eyes_retarget_delta = self.models_processor.lp_retarget_eye(
                            x_s, target_eye_ratio * eye_mult, face_editor_type
                        )

                    accumulated_motion += get_component_motion(
                        eye_indices,
                        x_d_i_info["exp"],
                        driving_multiplier_eyes,
                        extra_delta=eyes_retarget_delta,
                        is_relative=flag_relative_eyes,
                        neutral_ref=0,
                        use_boost=True,
                    )

                if flag_activate_lips:
                    lips_retarget_delta = 0
                    if parameters.get(
                        "FaceExpressionRetargetingLipsBothEnableToggle", False
                    ):
                        lip_mult = parameters.get(
                            "FaceExpressionRetargetingLipsMultiplierBothDecimalSlider",
                            1.0,
                        )
                        # Use Smoothed Ratios
                        c_d_lip = faceutil.calc_combined_lip_ratio(
                            c_d_lip_lst, source_lmk, device=self.models_processor.device
                        )
                        lips_retarget_delta = self.models_processor.lp_retarget_lip(
                            x_s, c_d_lip * lip_mult, face_editor_type
                        )

                    accumulated_motion += get_component_motion(
                        lip_indices,
                        x_d_i_info["exp"],
                        driving_multiplier_lips,
                        extra_delta=lips_retarget_delta,
                        is_relative=flag_relative_lips,
                        neutral_ref=lp_lip_array,
                        use_boost=True,
                    )

            # --- 5. GENERATE FINAL IMAGE ---
            x_d_i_new = x_s + accumulated_motion

            out = self.models_processor.lp_warp_decode(
                f_s, x_s, x_d_i_new, face_editor_type
            )
            out = torch.squeeze(out).clamp_(0, 1)

            # --- 6. PASTE BACK ---
            t = trans.SimilarityTransform()
            t.params[0:2] = M_c2o
            dsize = (target.shape[1], target.shape[2])

            out = faceutil.pad_image_by_size(out, dsize)
            out = v2.functional.affine(
                out,
                t.rotation * 57.2958,  # Convert radians to degrees
                translate=(t.translation[0], t.translation[1]),
                scale=t.scale,
                shear=(0.0, 0.0),
                interpolation=v2.InterpolationMode.BILINEAR,
                center=(0, 0),
            )
            out = v2.functional.crop(out, 0, 0, dsize[0], dsize[1])

        out = out.mul_(255.0).clamp_(0, 255).type(torch.float32)
        return out

    def swap_edit_face_core(
        self,
        img: torch.Tensor,
        swap_restorecalc: torch.Tensor,
        parameters: dict,
        control: dict,
        **kwargs,
    ) -> torch.Tensor:
        """
        Applies Face Editor manipulations (Pose, Gaze, Expression) to the face via manual sliders.
        Optimized: Removed explicit CPU/GPU sync.

        Args:
            img: The original image/frame.
            swap_restorecalc: The reference face for detection (usually detection happens earlier).
            parameters: Global parameters dictionary.
            control: UI control dictionary.

        Returns:
            torch.Tensor: The manipulated face image.
        """

        use_mean_eyes = parameters.get("LandmarkMeanEyesToggle", False)

        if parameters["FaceEditorEnableToggle"]:
            # 1. SETUP THE ASYNCHRONOUS CONTEXT
            current_stream = torch.cuda.current_stream()

            with torch.cuda.stream(current_stream):
                init_source_eye_ratio = 0.0
                init_source_lip_ratio = 0.0

                # Detection
                _, lmk_crop, _ = self.models_processor.run_detect_landmark(
                    swap_restorecalc,
                    bbox=np.array([0, 0, 512, 512]),
                    det_kpss=[],
                    detect_mode="203",
                    score=0.5,
                    from_points=False,
                    use_mean_eyes=use_mean_eyes,
                )
                source_eye_ratio = faceutil.calc_eye_close_ratio(lmk_crop[None])
                source_lip_ratio = faceutil.calc_lip_close_ratio(lmk_crop[None])
                init_source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
                init_source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)

                interp_mode = (
                    self.interpolation_expression_faceeditor_back
                    if self.interpolation_expression_faceeditor_back is not None
                    else v2.InterpolationMode.BILINEAR
                )

                # Prepare Image
                original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                    img,
                    lmk_crop,
                    dsize=512,
                    scale=parameters["FaceEditorCropScaleDecimalSlider"],
                    vy_ratio=parameters["FaceEditorVYRatioDecimalSlider"],
                    interpolation=interp_mode,
                )

                if self.t256_face is None:
                    self.t256_face = v2.Resize(
                        (256, 256),
                        interpolation=v2.InterpolationMode.BILINEAR,
                        antialias=False,
                    )

                original_face_256 = self.t256_face(original_face_512)

                # Extract features
                x_s_info = self.models_processor.lp_motion_extractor(
                    original_face_256, parameters["FaceEditorTypeSelection"]
                )
                x_d_info_user_pitch = x_s_info["pitch"] + parameters["HeadPitchSlider"]
                x_d_info_user_yaw = x_s_info["yaw"] + parameters["HeadYawSlider"]
                x_d_info_user_roll = x_s_info["roll"] + parameters["HeadRollSlider"]

                R_s_user = faceutil.get_rotation_matrix(
                    x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
                )
                R_d_user = faceutil.get_rotation_matrix(
                    x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll
                )

                f_s_user = self.models_processor.lp_appearance_feature_extractor(
                    original_face_256, parameters["FaceEditorTypeSelection"]
                )
                x_s_user = faceutil.transform_keypoint(x_s_info)

                # --- Create Tensors from Manual Sliders ---
                device = self.models_processor.device

                # Position
                mov_x = torch.tensor(parameters["XAxisMovementDecimalSlider"]).to(
                    device
                )
                mov_y = torch.tensor(parameters["YAxisMovementDecimalSlider"]).to(
                    device
                )
                mov_z = torch.tensor(parameters["ZAxisMovementDecimalSlider"]).to(
                    device
                )

                # Eyes/Gaze
                eyeball_direction_x = torch.tensor(
                    parameters["EyeGazeHorizontalDecimalSlider"]
                ).to(device)
                eyeball_direction_y = torch.tensor(
                    parameters["EyeGazeVerticalDecimalSlider"]
                ).to(device)
                wink = torch.tensor(parameters["EyeWinkDecimalSlider"]).to(device)
                eyebrow = torch.tensor(parameters["EyeBrowsDirectionDecimalSlider"]).to(
                    device
                )

                # Mouth
                smile = torch.tensor(parameters["MouthSmileDecimalSlider"]).to(device)
                lip_variation_zero = torch.tensor(
                    parameters["MouthPoutingDecimalSlider"]
                ).to(device)
                lip_variation_one = torch.tensor(
                    parameters["MouthPursingDecimalSlider"]
                ).to(device)
                lip_variation_two = torch.tensor(
                    parameters["MouthGrinDecimalSlider"]
                ).to(device)
                lip_variation_three = torch.tensor(
                    parameters["LipsCloseOpenSlider"]
                ).to(device)

                x_c_s = x_s_info["kp"]
                delta_new = x_s_info["exp"]
                scale_new = x_s_info["scale"]
                t_new = x_s_info["t"]

                # Calculate New Rotation Matrix
                R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

                # --- Apply Modifications to Expression Delta ---
                if eyeball_direction_x != 0 or eyeball_direction_y != 0:
                    delta_new = faceutil.update_delta_new_eyeball_direction(
                        eyeball_direction_x, eyeball_direction_y, delta_new
                    )
                if smile != 0:
                    delta_new = faceutil.update_delta_new_smile(smile, delta_new)
                if wink != 0:
                    delta_new = faceutil.update_delta_new_wink(wink, delta_new)
                if eyebrow != 0:
                    delta_new = faceutil.update_delta_new_eyebrow(eyebrow, delta_new)
                if lip_variation_zero != 0:
                    delta_new = faceutil.update_delta_new_lip_variation_zero(
                        lip_variation_zero, delta_new
                    )
                if lip_variation_one != 0:
                    delta_new = faceutil.update_delta_new_lip_variation_one(
                        lip_variation_one, delta_new
                    )
                if lip_variation_two != 0:
                    delta_new = faceutil.update_delta_new_lip_variation_two(
                        lip_variation_two, delta_new
                    )
                if lip_variation_three != 0:
                    delta_new = faceutil.update_delta_new_lip_variation_three(
                        lip_variation_three, delta_new
                    )
                if mov_x != 0:
                    delta_new = faceutil.update_delta_new_mov_x(-mov_x, delta_new)
                if mov_y != 0:
                    delta_new = faceutil.update_delta_new_mov_y(mov_y, delta_new)

                # Calculate final driving keypoints
                x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
                eyes_delta, lip_delta = None, None

                # --- Retargeting Sliders (Opening/Closing) ---
                input_eye_ratio = max(
                    min(
                        init_source_eye_ratio
                        + parameters["EyesOpenRatioDecimalSlider"],
                        0.80,
                    ),
                    0.00,
                )
                if input_eye_ratio != init_source_eye_ratio:
                    combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio(
                        [[float(input_eye_ratio)]],
                        lmk_crop,
                        device=self.models_processor.device,
                    )
                    eyes_delta = self.models_processor.lp_retarget_eye(
                        x_s_user,
                        combined_eye_ratio_tensor,
                        parameters["FaceEditorTypeSelection"],
                    )

                input_lip_ratio = max(
                    min(
                        init_source_lip_ratio
                        + parameters["LipsOpenRatioDecimalSlider"],
                        0.80,
                    ),
                    0.00,
                )
                if input_lip_ratio != init_source_lip_ratio:
                    combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio(
                        [[float(input_lip_ratio)]],
                        lmk_crop,
                        device=self.models_processor.device,
                    )
                    lip_delta = self.models_processor.lp_retarget_lip(
                        x_s_user,
                        combined_lip_ratio_tensor,
                        parameters["FaceEditorTypeSelection"],
                    )

                # Add retargeting deltas to the main motion
                x_d_new = (
                    x_d_new
                    + (eyes_delta if eyes_delta is not None else 0)
                    + (lip_delta if lip_delta is not None else 0)
                )

                # Optional Stitching
                flag_stitching_retargeting_input: bool = kwargs.get(
                    "flag_stitching_retargeting_input", True
                )
                if flag_stitching_retargeting_input:
                    x_d_new = self.models_processor.lp_stitching(
                        x_s_user, x_d_new, parameters["FaceEditorTypeSelection"]
                    )

                # Generate Image
                out = self.models_processor.lp_warp_decode(
                    f_s_user, x_s_user, x_d_new, parameters["FaceEditorTypeSelection"]
                )
                out = torch.squeeze(out)
                out = out.clamp_(0, 1)

            # --- POST-PROCESSING (Paste Back) ---
            t = trans.SimilarityTransform()
            t.params[0:2] = M_c2o
            dsize = (img.shape[1], img.shape[2])
            out = faceutil.pad_image_by_size(out, dsize)
            out = v2.functional.affine(
                out,
                t.rotation * 57.2958,
                translate=(t.translation[0], t.translation[1]),
                scale=t.scale,
                shear=(0.0, 0.0),
                interpolation=interp_mode,
                center=(0, 0),
            )
            out = v2.functional.crop(out, 0, 0, dsize[0], dsize[1])  # cols, rows

            img = out
            img = img.mul_(255.0).clamp_(0, 255).type(torch.float32)

        return img

    def swap_edit_face_core_makeup(
        self,
        img: torch.Tensor,
        kps: np.ndarray,
        parameters: dict,
        control: dict,
        **kwargs,
    ) -> torch.Tensor:
        """
        Applies digital makeup to the face using face parser masks.

        Args:
            img: The original image tensor.
            kps: Keypoints of the face.
            parameters: Global parameters dictionary.
            control: Control settings dictionary.

        Returns:
            torch.Tensor: Image with makeup applied.
        """
        use_mean_eyes = parameters.get("LandmarkMeanEyesToggle", False)

        if (
            parameters["FaceMakeupEnableToggle"]
            or parameters["HairMakeupEnableToggle"]
            or parameters["EyeBrowsMakeupEnableToggle"]
            or parameters["LipsMakeupEnableToggle"]
        ):
            _, lmk_crop, _ = self.models_processor.run_detect_landmark(
                img,
                bbox=[],
                det_kpss=kps,
                detect_mode="203",
                score=0.5,
                from_points=False,
                use_mean_eyes=use_mean_eyes,
            )

            # Use the interpolation mode passed from FrameWorker, or default to BILINEAR
            interp_mode = (
                self.interpolation_expression_faceeditor_back
                if self.interpolation_expression_faceeditor_back is not None
                else v2.InterpolationMode.BILINEAR
            )

            # Prepare Image
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                img,
                lmk_crop,
                dsize=512,
                scale=parameters["FaceEditorCropScaleDecimalSlider"],
                vy_ratio=parameters["FaceEditorVYRatioDecimalSlider"],
                interpolation=interp_mode,
            )

            out, mask_out = self.models_processor.apply_face_makeup(
                original_face_512, parameters
            )

            # Apply strict 1 check (kept for non-regression / future toggle)
            if 1:
                # Gaussian blur for soft blending of the crop mask
                gauss = v2.GaussianBlur(kernel_size=5 * 2 + 1, sigma=(5 + 1) * 0.2)
                out = torch.clamp(torch.div(out, 255.0), 0, 1).type(torch.float32)
                mask_crop = gauss(self.models_processor.lp_mask_crop)
                img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)

        return img
