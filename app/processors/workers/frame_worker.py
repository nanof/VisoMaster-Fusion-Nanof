import traceback
from typing import TYPE_CHECKING, Dict, cast
import threading
import queue
import math
from math import floor, ceil
from app.ui.widgets import widget_components
import torch
from skimage import transform as trans
import kornia.enhance as ke
import kornia.color as kc


from torchvision.transforms import v2
import torchvision
from torchvision import transforms

import numpy as np
import torch.nn.functional as F

from app.processors.utils import faceutil
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.helpers.miscellaneous import (
    ParametersDict,
    get_scaling_transforms,
    draw_bounding_boxes_on_detected_faces,
    paint_landmarks_on_image,
    keypoints_adjustments,
    get_grid_for_pasting,
)
from app.helpers.vr_utils import EquirectangularConverter, PerspectiveConverter
from app.helpers.typing_helper import ParametersTypes
from app.processors.frame_enhancers import FrameEnhancers
from app.processors.frame_edits import FrameEdits

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

torchvision.disable_beta_transforms_warning()


class FrameWorker(threading.Thread):
    """
    Worker thread responsible for processing a single video frame or image.
    Can operate in two modes:
    1. Pool Worker: Persistently runs, fetching tasks from a queue.
    2. Single Frame Worker: Runs once for a specific frame (e.g., UI preview).

    Handles the entire pipeline: Detection -> Swapping -> Enhancement -> Post-processing.
    """

    def __init__(
        self,
        main_window: "MainWindow",
        # Pool worker args (frame_queue is a task queue)
        frame_queue: queue.Queue | None = None,
        worker_id: int = -1,
        # Single-frame worker args
        frame: np.ndarray | None = None,
        frame_number: int = -1,
        is_single_frame: bool = False,
    ):
        super().__init__()
        # This event will be used to signal the thread to stop
        self.stop_event = threading.Event()

        # Scaling transforms (initialized in set_scaling_transforms)
        self.t512 = None
        self.t384 = None
        self.t256 = None
        self.t128 = None
        self.interpolation_get_cropped_face_kps = None
        self.interpolation_original_face_128_384 = None
        self.interpolation_original_face_512 = None
        self.interpolation_Untransform = None
        self.interpolation_scaleback = None
        self.t256_face = None
        self.interpolation_expression_faceeditor_back = None
        self.interpolation_block_shift = None

        # --- Architecture References ---
        self.main_window = main_window
        self.models_processor = main_window.models_processor
        self.video_processor = main_window.video_processor

        # Initialize Helpers
        self.frame_enhancers = FrameEnhancers(self.models_processor)
        self.frame_edits = FrameEdits(self.models_processor)

        # Mode-specific args
        self.frame_queue = frame_queue  # This is now the TASK queue
        self.worker_id = worker_id

        # Single-frame data
        self.frame = frame  # Will be None in pool mode until a task is dequeued
        self.frame_number = (
            frame_number  # Will be -1 in pool mode until a task is dequeued
        )
        self.is_single_frame = is_single_frame

        # Determine mode & Thread Name
        self.is_pool_worker = (frame_queue is not None) and (worker_id != -1)

        if self.is_pool_worker:
            self.name = f"FrameWorker-Pool-{worker_id}"
        else:
            self.name = f"FrameWorker-Single-{frame_number}"

        self.parameters: Dict[
            str, ParametersTypes
        ] = {}  # Will be populated from main_window.parameters or task

        self.last_processed_frame_number = -1
        self.last_detected_faces = []

        # VR specific constants
        self.VR_PERSPECTIVE_RENDER_SIZE = 512  # Pixels, for rendering perspective crops
        self.VR_DYNAMIC_FOV_PADDING_FACTOR = (
            1.0  # Padding factor for dynamic FOV calculation
        )
        self.is_view_face_compare: bool = False
        self.is_view_face_mask: bool = False
        self.lock = threading.Lock()

    def set_scaling_transforms(self, control_params):
        """Initializes the torchvision transforms based on user interpolation settings."""
        (
            self.t512,
            self.t384,
            self.t256,
            self.t128,
            self.interpolation_get_cropped_face_kps,
            self.interpolation_original_face_128_384,
            self.interpolation_original_face_512,
            self.interpolation_Untransform,
            self.interpolation_scaleback,
            self.t256_face,
            self.interpolation_expression_faceeditor_back,
            self.interpolation_block_shift,
        ) = get_scaling_transforms(control_params)

        # Pass relevant transforms to FrameEdits helper
        self.frame_edits.set_transforms(
            self.t256_face, self.interpolation_expression_faceeditor_back
        )

    def run(self):
        """
        Main thread execution loop.
        - In Pool Mode: Loops, gets tasks from self.frame_queue, calls process_and_emit_task().
        - In Single-Frame Mode: Calls process_and_emit_task() just once.
        """
        if self.is_pool_worker:
            # --- Pool Worker Mode ---
            while not self.stop_event.is_set():
                task = None  # Ensure task is defined for 'finally'
                try:
                    # Block until a task is available or a poison pill is received
                    # Use a timeout to periodically check the stop_event
                    task = self.frame_queue.get(timeout=1.0)

                    if task is None:
                        # Poison pill received: Exit the loop
                        print(f"[INFO] {self.name} received poison pill. Exiting.")
                        break  # 'finally' will call task_done()

                    if self.stop_event.is_set():
                        # Stopped while waiting, discard task
                        break  # 'finally' will call task_done()

                    # Unpack the task which includes frame and specific parameters
                    (
                        self.frame_number,
                        self.frame,
                        local_params_from_feeder,
                        local_control_from_feeder,
                    ) = task

                    # Store them locally in the worker
                    self.parameters = local_params_from_feeder
                    self.local_control_state_from_feeder = local_control_from_feeder

                    # Process the frame
                    self.process_and_emit_task()

                except queue.Empty:
                    # Timeout occurred, just loop again to check stop_event
                    continue
                except Exception as e:
                    # An error happened *during* processing
                    print(
                        f"[ERROR] Error in {self.name} (frame {self.frame_number}): {e}"
                    )
                    traceback.print_exc()

                finally:
                    # This block executes *no matter what* (success, exception, or break)
                    if task is not None and self.frame_queue is not None:
                        try:
                            self.frame_queue.task_done()
                        except ValueError:
                            # Safe to ignore if queue was cleared externally
                            pass

        else:
            # --- Single-Frame Mode ---
            if self.stop_event.is_set():
                print(f"[WARN] {self.name} cancelled before start.")
                return
            try:
                # Single-Frame worker MUST use the *current* global state
                # to reflect immediate UI changes.
                with self.main_window.models_processor.model_lock:
                    local_parameters_copy = self.main_window.parameters.copy()
                    local_control_copy = self.main_window.control.copy()

                # Ensure parameter dicts exist (failsafe for new faces)
                active_target_face_ids = list(self.main_window.target_faces.keys())
                for face_id_key in active_target_face_ids:
                    if str(face_id_key) not in local_parameters_copy:
                        local_parameters_copy[str(face_id_key)] = (
                            self.main_window.default_parameters.copy()
                        )

                # Store locally
                self.parameters = local_parameters_copy
                self.local_control_state_from_feeder = local_control_copy

                # Run once
                self.process_and_emit_task()
            except Exception as e:
                print(f"[ERROR] Error in {self.name}: {e}")
                traceback.print_exc()

    def process_and_emit_task(self):
        """
        Processes self.frame using the configured parameters and emits the result signal.
        Does NOT interact with the task queue.
        """
        try:
            local_control_state = self.local_control_state_from_feeder

            # Get UI state (safe reads)
            self.is_view_face_compare = self.main_window.faceCompareCheckBox.isChecked()
            self.is_view_face_mask = self.main_window.faceMaskCheckBox.isChecked()

            # Determine if processing is needed
            needs_processing = (
                self.main_window.swapfacesButton.isChecked()
                or self.main_window.editFacesButton.isChecked()
                or local_control_state.get("FrameEnhancerEnableToggle", False)
                or local_control_state.get(
                    "ModeEnableToggle", False
                )  # Always processes in this mode
            )

            if needs_processing:
                # Ensure input frame is C-contiguous for PyTorch/OpenCV compatibility
                if not self.frame.flags["C_CONTIGUOUS"]:
                    self.frame = np.ascontiguousarray(self.frame)

                # Process Frame (returns BGR, uint8)
                processed_frame_bgr_np_uint8 = self.process_frame(
                    local_control_state, self.stop_event
                )

                # Ensure output is C-contiguous for Qt display
                self.frame = np.ascontiguousarray(processed_frame_bgr_np_uint8)
            else:
                # If no processing, just convert RGB to BGR for display
                self.frame = self.frame[..., ::-1]
                self.frame = np.ascontiguousarray(self.frame)

            # Check stop event again
            if self.stop_event.is_set():
                print(f"[WARN] {self.name} cancelled during process_frame.")
                return

            # Create Pixmap and Emit Signals
            pixmap = common_widget_actions.get_pixmap_from_frame(
                self.main_window, self.frame
            )

            if self.video_processor.file_type == "webcam" and not self.is_single_frame:
                self.video_processor.webcam_frame_processed_signal.emit(
                    pixmap, self.frame
                )
            elif not self.is_single_frame:
                self.video_processor.frame_processed_signal.emit(
                    self.frame_number, pixmap, self.frame
                )
            else:  # Single frame processing (image or paused video)
                self.video_processor.single_frame_processed_signal.emit(
                    self.frame_number, pixmap, self.frame
                )

        except Exception as e:
            print(f"[ERROR] Error in {self.name} (frame {self.frame_number}): {e}")
            traceback.print_exc()

    def _apply_denoiser_pass(
        self,
        image_tensor_cxhxw_uint8: torch.Tensor,
        control: dict,
        pass_suffix: str,
        kv_map: Dict | None,
    ) -> torch.Tensor:
        """Helper to run the diffusion-based denoiser (Ref-LDM)."""
        use_exclusive_path = control.get("UseReferenceExclusivePathToggle", False)
        denoiser_seed_from_slider_val = int(control.get("DenoiserBaseSeedSlider", 1))

        denoiser_mode_key = f"DenoiserModeSelection{pass_suffix}"
        denoiser_mode_val = control.get(denoiser_mode_key, "Single Step (Fast)")

        ddim_steps_key = f"DenoiserDDIMStepsSlider{pass_suffix}"
        ddim_steps_val = int(control.get(ddim_steps_key, 20))

        cfg_scale_key = f"DenoiserCFGScaleDecimalSlider{pass_suffix}"
        cfg_scale_val = float(control.get(cfg_scale_key, 1.0))

        single_step_t_key = f"DenoiserSingleStepTimestepSlider{pass_suffix}"
        single_step_t_val = int(control.get(single_step_t_key, 1))

        sharpen_key = f"DenoiserLatentSharpeningDecimalSlider{pass_suffix}"
        sharpen_val = float(control.get(sharpen_key, 0.0))

        if not kv_map:
            if use_exclusive_path:
                if control.get("CommandLineDebugEnableToggle", False):
                    print(
                        f"[ERROR] Denoiser {pass_suffix}: No source face for K/V, but 'Exclusive Reference Path' is ON. Skipping."
                    )
                return image_tensor_cxhxw_uint8

        denoised_image = self.models_processor.apply_denoiser_unet(
            image_tensor_cxhxw_uint8,
            reference_kv_map=kv_map,
            use_reference_exclusive_path=use_exclusive_path,
            denoiser_mode=denoiser_mode_val,
            base_seed=denoiser_seed_from_slider_val,
            denoiser_single_step_t=single_step_t_val,
            denoiser_ddim_steps=ddim_steps_val,
            denoiser_cfg_scale=cfg_scale_val,
            latent_sharpening_strength=sharpen_val,
        )
        return denoised_image

    def _find_best_target_match(self, detected_embedding_np, control_global):
        """Finds the best matching source face for a detected target face."""
        best_target_button = None
        best_params_pd = None
        highest_sim = -1.0

        for target_id, target_button_widget in list(
            self.main_window.target_faces.items()
        ):
            face_specific_params_dict = self.parameters.get(target_id, {})

            default_params_dict = (
                dict(self.main_window.default_parameters.data)
                if isinstance(self.main_window.default_parameters, ParametersDict)
                else dict(
                    self.main_window.default_parameters.data
                )  # Assume .data is always the target
            )

            current_params_pd = ParametersDict(
                dict(face_specific_params_dict), cast(dict, default_params_dict)
            )
            target_embedding_np = target_button_widget.get_embedding(
                control_global["RecognitionModelSelection"]
            )
            if target_embedding_np is None:
                continue
            sim = self.models_processor.findCosineDistance(
                detected_embedding_np, target_embedding_np
            )

            if (
                sim >= current_params_pd["SimilarityThresholdSlider"]
                and sim > highest_sim
            ):
                highest_sim = sim
                best_target_button = target_button_widget
                best_params_pd = current_params_pd
        return best_target_button, best_params_pd, highest_sim

    def _process_single_vr_perspective_crop_multi(
        self,
        perspective_crop_torch_rgb_uint8: torch.Tensor,
        target_face_button: "widget_components.TargetFaceCardButton",
        parameters_for_face: ParametersDict,
        control_global: dict,
        kps_5_on_crop_param: np.ndarray,
        kps_all_on_crop_param: np.ndarray | None,
        swap_button_is_checked_global: bool,
        edit_button_is_checked_global: bool,
        eye_side_for_debug: str = "",
        kv_map_for_swap: Dict | None = None,
    ) -> torch.Tensor:
        """Processes a single perspective crop extracted from a VR frame."""
        processed_crop_torch_rgb_uint8 = perspective_crop_torch_rgb_uint8.clone()
        if kps_5_on_crop_param is None or kps_5_on_crop_param.size == 0:
            return processed_crop_torch_rgb_uint8

        if not (swap_button_is_checked_global or edit_button_is_checked_global):
            return processed_crop_torch_rgb_uint8

        arcface_model_for_swap = self.models_processor.get_arcface_model(
            parameters_for_face["SwapModelSelection"]
        )
        s_e_for_swap_np = None
        if swap_button_is_checked_global:
            s_e_for_swap_np = target_face_button.assigned_input_embedding.get(
                arcface_model_for_swap
            )
            if (
                s_e_for_swap_np is None
                or not isinstance(s_e_for_swap_np, np.ndarray)
                or s_e_for_swap_np.size == 0
                or np.isnan(s_e_for_swap_np).any()
                or np.isinf(s_e_for_swap_np).any()
            ):
                s_e_for_swap_np = None

        t_e_for_swap_np = target_face_button.get_embedding(arcface_model_for_swap)
        dfm_model_instance_local = None
        if parameters_for_face["SwapModelSelection"] == "DeepFaceLive (DFM)":
            dfm_model_name = parameters_for_face["DFMModelSelection"]
            if dfm_model_name:
                dfm_model_instance_local = self.models_processor.load_dfm_model(
                    dfm_model_name
                )

        s_e_for_swap_core = s_e_for_swap_np if swap_button_is_checked_global else None

        if (
            swap_button_is_checked_global
            and (
                s_e_for_swap_core is not None
                or (
                    parameters_for_face["SwapModelSelection"] == "DeepFaceLive (DFM)"
                    and dfm_model_instance_local is not None
                )
            )
        ) or edit_button_is_checked_global:
            source_kps = None
            if target_face_button and target_face_button.assigned_input_faces:
                first_input_id = list(target_face_button.assigned_input_faces.keys())[0]
                store = target_face_button.assigned_input_faces[first_input_id]
                if "kps_5" in store:
                    source_kps = store["kps_5"]

            kps_5_on_crop_param = keypoints_adjustments(
                kps_5_on_crop_param, parameters_for_face, source_kps=source_kps
            )
            try:
                (
                    swapped_face_512_torch_rgb_uint8,
                    comprehensive_mask_1x512x512_from_swap_core,
                    _,
                ) = self.swap_core(
                    perspective_crop_torch_rgb_uint8,
                    kps_5_on_crop_param,
                    kps=kps_all_on_crop_param,
                    s_e=s_e_for_swap_core,
                    t_e=t_e_for_swap_np,
                    parameters=parameters_for_face.data,
                    control=control_global,
                    dfm_model_name=parameters_for_face["DFMModelSelection"],
                    is_perspective_crop=True,
                    kv_map=kv_map_for_swap,
                )
            except Exception as e_swap_core:
                print(
                    f"[ERROR] Error in swap_core for VR crop {eye_side_for_debug}: {e_swap_core}"
                )
                traceback.print_exc()
                swapped_face_512_torch_rgb_uint8 = cast(v2.Resize, self.t512)(
                    perspective_crop_torch_rgb_uint8
                )
                comprehensive_mask_1x512x512_from_swap_core = torch.zeros(
                    (1, 512, 512),
                    dtype=torch.float32,
                    device=perspective_crop_torch_rgb_uint8.device,
                )

            tform_persp_to_512template = self.get_face_similarity_tform(
                parameters_for_face["SwapModelSelection"], kps_5_on_crop_param
            )

            # Define the 512x512 resizer for masks
            t512_mask = v2.Resize(
                (512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
            )

            if (
                comprehensive_mask_1x512x512_from_swap_core is None
                or comprehensive_mask_1x512x512_from_swap_core.numel() == 0
            ):
                # Fallback mask logic
                persp_final_combined_mask_1x512x512_float_for_paste = (
                    t512_mask(self.get_border_mask(parameters_for_face.data)[0]).float()
                    if swap_button_is_checked_global
                    else torch.zeros(
                        (1, 512, 512),
                        dtype=torch.float32,
                        device=perspective_crop_torch_rgb_uint8.device,
                    )
                )
            else:
                # Primary path
                persp_final_combined_mask_1x512x512_float_for_paste = (
                    comprehensive_mask_1x512x512_from_swap_core.float()
                )

                if parameters_for_face.get("BordermaskEnableToggle", False):
                    border_mask_128, _ = self.get_border_mask(parameters_for_face.data)
                    border_mask_512 = t512_mask(border_mask_128)
                    persp_final_combined_mask_1x512x512_float_for_paste *= (
                        border_mask_512
                    )

            persp_final_combined_mask_3x512x512_float_for_paste = (
                persp_final_combined_mask_1x512x512_float_for_paste.repeat(3, 1, 1)
            )
            masked_swapped_face_to_paste_float = (
                swapped_face_512_torch_rgb_uint8.float()
                * persp_final_combined_mask_3x512x512_float_for_paste
            )

            crop_h, crop_w = (
                perspective_crop_torch_rgb_uint8.shape[1],
                perspective_crop_torch_rgb_uint8.shape[2],
            )
            _, source_grid_normalized_xy_persp = get_grid_for_pasting(
                tform_persp_to_512template,
                crop_h,
                crop_w,
                512,
                512,
                perspective_crop_torch_rgb_uint8.device,
            )
            pasted_face_on_persp_float = torch.nn.functional.grid_sample(
                masked_swapped_face_to_paste_float.unsqueeze(0),
                source_grid_normalized_xy_persp,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).squeeze(0)
            transformed_mask_on_persp_float = torch.nn.functional.grid_sample(
                persp_final_combined_mask_3x512x512_float_for_paste.unsqueeze(0),
                source_grid_normalized_xy_persp,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0)
            blended_persp_crop_float = (
                pasted_face_on_persp_float
                + perspective_crop_torch_rgb_uint8.float()
                * (1.0 - transformed_mask_on_persp_float)
            )
            processed_crop_torch_rgb_uint8 = torch.clamp(
                blended_persp_crop_float, 0, 255
            ).byte()

            if edit_button_is_checked_global:
                _, _, kps_all_for_editor_list = self.models_processor.run_detect(
                    processed_crop_torch_rgb_uint8,
                    control_global["DetectorModelSelection"],
                    max_num=1,
                    score=control_global["DetectorScoreSlider"] / 100.0,
                    input_size=(
                        processed_crop_torch_rgb_uint8.shape[1],
                        processed_crop_torch_rgb_uint8.shape[2],
                    ),
                    use_landmark_detection=True,
                    landmark_detect_mode="203",
                    landmark_score=control_global["LandmarkDetectScoreSlider"] / 100.0,
                    from_points=True,
                    rotation_angles=[0],
                    use_mean_eyes=control_global.get("LandmarkMeanEyesToggle", False),
                )
                kps_all_for_editor_on_crop = (
                    kps_all_for_editor_list[0]
                    if len(kps_all_for_editor_list) > 0
                    else None
                )
                if (
                    kps_all_for_editor_on_crop is not None
                    and kps_all_for_editor_on_crop.size > 0
                ):
                    processed_crop_torch_rgb_uint8 = (
                        self.frame_edits.swap_edit_face_core(
                            processed_crop_torch_rgb_uint8,
                            processed_crop_torch_rgb_uint8,
                            parameters_for_face.data,
                            control_global,
                        )
                    )
                    if any(
                        parameters_for_face.get(f, False)
                        for f in (
                            "FaceMakeupEnableToggle",
                            "HairMakeupEnableToggle",
                            "EyeBrowsMakeupEnableToggle",
                            "LipsMakeupEnableToggle",
                        )
                    ):
                        processed_crop_torch_rgb_uint8 = (
                            self.frame_edits.swap_edit_face_core_makeup(
                                processed_crop_torch_rgb_uint8,
                                kps_all_on_crop_param,
                                parameters_for_face.data,
                                control_global,
                            )
                        )

        return processed_crop_torch_rgb_uint8

    def process_frame(self, control: dict, stop_event: threading.Event):
        """
        Routing method:
        - Checks inputs.
        - Sets up transforms.
        - Dispatches to either VR180 or Standard processing logic.
        - Applies common global enhancers.
        """
        # Check 1: At the very beginning
        if stop_event.is_set():
            return self.frame[..., ::-1]  # Return original BGR frame

        # Keep last frame number for reference
        self.last_processed_frame_number = self.frame_number

        self.set_scaling_transforms(control)
        img_numpy_rgb_uint8 = self.frame

        # Prepare the base tensor
        processed_tensor_rgb_uint8 = (
            torch.from_numpy(img_numpy_rgb_uint8)
            .to(self.models_processor.device)
            .permute(2, 0, 1)
        )

        # --- ROUTING LOGIC ---
        if control.get("VR180ModeEnableToggle", False):
            processed_tensor_rgb_uint8 = self._process_frame_vr180(
                processed_tensor_rgb_uint8, img_numpy_rgb_uint8, control, stop_event
            )
        else:
            processed_tensor_rgb_uint8 = self._process_frame_standard(
                processed_tensor_rgb_uint8, control, stop_event
            )

        # --- Common Post-Processing (Enhancers, etc.) ---
        compare_mode_active = self.is_view_face_mask or self.is_view_face_compare

        if control["FrameEnhancerEnableToggle"] and not compare_mode_active:
            # Check 5: Before final heavy operation
            if stop_event.is_set():
                return img_numpy_rgb_uint8[..., ::-1]

            processed_tensor_rgb_uint8 = self.frame_enhancers.enhance_core(
                processed_tensor_rgb_uint8, control=control
            )

        final_img_np_rgb_uint8 = (
            processed_tensor_rgb_uint8.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        )
        if not final_img_np_rgb_uint8.flags["C_CONTIGUOUS"]:
            final_img_np_rgb_uint8 = np.ascontiguousarray(final_img_np_rgb_uint8)

        return final_img_np_rgb_uint8[..., ::-1]

    def _process_frame_vr180(
        self,
        original_equirect_tensor_for_vr: torch.Tensor,
        img_numpy_rgb_uint8: np.ndarray,
        control: dict,
        stop_event: threading.Event,
    ) -> torch.Tensor:
        """
        Handles the specific logic for VR180/360 frames:
        - Equirectangular detection
        - Perspective cropping
        - Processing per crop
        - Stitching back
        """
        swap_button_is_checked_global = self.main_window.swapfacesButton.isChecked()
        edit_button_is_checked_global = self.main_window.editFacesButton.isChecked()

        equirect_converter = EquirectangularConverter(
            img_numpy_rgb_uint8, device=self.models_processor.device
        )

        # Detection on full equirectangular image
        bboxes_eq_np, _, _ = self.models_processor.run_detect(
            original_equirect_tensor_for_vr,
            control["DetectorModelSelection"],
            max_num=control["MaxFacesToDetectSlider"],
            score=control["DetectorScoreSlider"] / 100.0,
            input_size=(512, 512),
            use_landmark_detection=False,  # VR usually detects faces first, then landmarks on crops
            landmark_detect_mode=control["LandmarkDetectModelSelection"],
            landmark_score=control["LandmarkDetectScoreSlider"] / 100.0,
            from_points=False,
            rotation_angles=[0]
            if not control["AutoRotationToggle"]
            else [0, 90, 180, 270],
            use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
        )

        if not isinstance(bboxes_eq_np, np.ndarray):
            bboxes_eq_np = np.array(bboxes_eq_np)

        # Filtering Block: De-duplicate nearby bounding boxes.
        if bboxes_eq_np.ndim == 2 and bboxes_eq_np.shape[0] > 1:
            initial_box_count = bboxes_eq_np.shape[0]
            areas = (bboxes_eq_np[:, 2] - bboxes_eq_np[:, 0]) * (
                bboxes_eq_np[:, 3] - bboxes_eq_np[:, 1]
            )
            sorted_indices = np.argsort(areas)[::-1]
            centers_x = (bboxes_eq_np[:, 0] + bboxes_eq_np[:, 2]) / 2.0
            centers_y = (bboxes_eq_np[:, 1] + bboxes_eq_np[:, 3]) / 2.0
            widths = bboxes_eq_np[:, 2] - bboxes_eq_np[:, 0]
            indices_to_keep = []
            suppressed_indices = np.zeros(initial_box_count, dtype=bool)

            for i in range(initial_box_count):
                idx1 = sorted_indices[i]
                if suppressed_indices[idx1]:
                    continue
                indices_to_keep.append(idx1)
                for j in range(initial_box_count):
                    if idx1 == j or suppressed_indices[j]:
                        continue
                    dist_x = centers_x[idx1] - centers_x[j]
                    dist_y = centers_y[idx1] - centers_y[j]
                    distance = np.sqrt(dist_x**2 + dist_y**2)
                    if distance < widths[idx1] * 0.5:
                        suppressed_indices[j] = True
            bboxes_eq_np = bboxes_eq_np[indices_to_keep]

        processed_perspective_crops_details = {}
        analyzed_faces_for_vr = []

        for bbox_eq_single in bboxes_eq_np:
            if stop_event.is_set():
                break

            theta, phi = equirect_converter.calculate_theta_phi_from_bbox(
                bbox_eq_single
            )
            original_eye_side = (
                "L"
                if (bbox_eq_single[0] + bbox_eq_single[2]) / 2
                < equirect_converter.width / 2
                else "R"
            )
            angular_width_deg = (
                (bbox_eq_single[2] - bbox_eq_single[0])
                / equirect_converter.width
                * 360.0
            )
            angular_height_deg = (
                (bbox_eq_single[3] - bbox_eq_single[1])
                / equirect_converter.height
                * 180.0
            )

            dynamic_fov_for_crop = np.clip(
                max(angular_width_deg, angular_height_deg)
                * self.VR_DYNAMIC_FOV_PADDING_FACTOR,
                15.0,
                100.0,
            )

            face_crop_tensor = equirect_converter.get_perspective_crop(
                dynamic_fov_for_crop,
                theta,
                phi,
                self.VR_PERSPECTIVE_RENDER_SIZE,
                self.VR_PERSPECTIVE_RENDER_SIZE,
            )
            if face_crop_tensor is None or face_crop_tensor.numel() == 0:
                continue

            # Landmark detection on the crop
            crop_size = self.VR_PERSPECTIVE_RENDER_SIZE
            padding = int(crop_size * 0.025)
            dummy_bbox_on_crop = np.array(
                [padding, padding, crop_size - padding, crop_size - padding]
            )

            kpss_5_crop_list, kpss_crop_list, _ = (
                self.models_processor.run_detect_landmark(
                    img=face_crop_tensor,
                    bbox=dummy_bbox_on_crop,
                    det_kpss=[],
                    detect_mode=control["LandmarkDetectModelSelection"],
                    score=control["LandmarkDetectScoreSlider"] / 100.0,
                    from_points=False,
                    use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
                )
            )

            kpss_5_crop = [kpss_5_crop_list] if len(kpss_5_crop_list) > 0 else []
            kpss_crop = [kpss_crop_list] if len(kpss_crop_list) > 0 else []

            if not (
                isinstance(kpss_5_crop, np.ndarray)
                and kpss_5_crop.shape[0] > 0
                or isinstance(kpss_5_crop, list)
                and len(kpss_5_crop) > 0
            ):
                del face_crop_tensor
                continue

            kps_on_crop = kpss_5_crop[0]
            kps_all_on_crop = (
                kpss_crop[0]
                if isinstance(kpss_crop, np.ndarray) and kpss_crop.shape[0] > 0
                else None
            )

            face_emb_crop, _ = self.models_processor.run_recognize_direct(
                face_crop_tensor,
                kps_on_crop,
                control["SimilarityTypeSelection"],
                control["RecognitionModelSelection"],
            )

            best_target_button_vr, best_params_for_target_vr, _ = (
                self._find_best_target_match(face_emb_crop, control)
            )

            if best_target_button_vr:
                denoiser_on = (
                    control.get("DenoiserUNetEnableBeforeRestorersToggle", False)
                    or control.get("DenoiserAfterFirstRestorerToggle", False)
                    or control.get("DenoiserAfterRestorersToggle", False)
                )
                if (
                    denoiser_on
                    and best_target_button_vr.assigned_kv_map is None
                    and best_target_button_vr.assigned_input_faces
                ):
                    best_target_button_vr.calculate_assigned_input_embedding()

                analyzed_faces_for_vr.append(
                    {
                        "theta": theta,
                        "phi": phi,
                        "original_eye_side": original_eye_side,
                        "face_crop_tensor": face_crop_tensor,
                        "kps_on_crop": kps_on_crop,
                        "kps_all_on_crop": kps_all_on_crop,
                        "target_button": best_target_button_vr,
                        "params": best_params_for_target_vr,
                        "fov_used_for_crop": dynamic_fov_for_crop,
                    }
                )
            else:
                del face_crop_tensor

        # Process collected faces
        for item_data in analyzed_faces_for_vr:
            processed_crop_for_stitching = (
                self._process_single_vr_perspective_crop_multi(
                    item_data["face_crop_tensor"],
                    item_data["target_button"],
                    item_data["params"],
                    control,
                    kps_5_on_crop_param=item_data["kps_on_crop"],
                    kps_all_on_crop_param=item_data["kps_all_on_crop"],
                    swap_button_is_checked_global=swap_button_is_checked_global,
                    edit_button_is_checked_global=edit_button_is_checked_global,
                    eye_side_for_debug=item_data["original_eye_side"],
                    kv_map_for_swap=item_data["target_button"].assigned_kv_map,
                )
                if swap_button_is_checked_global or edit_button_is_checked_global
                else item_data["face_crop_tensor"]
            )

            processed_perspective_crops_details[
                f"{item_data['original_eye_side']}_{item_data['theta']}_{item_data['phi']}"
            ] = {
                "tensor_rgb_uint8": processed_crop_for_stitching,
                "theta": item_data["theta"],
                "phi": item_data["phi"],
                "fov_used_for_crop": item_data["fov_used_for_crop"],
            }
            del item_data["face_crop_tensor"]

        # Stitch back
        final_equirect_torch_cxhxw_rgb_uint8 = original_equirect_tensor_for_vr.clone()
        p2e_converter = PerspectiveConverter(
            img_numpy_rgb_uint8, device=self.models_processor.device
        )

        for eye_side, data in processed_perspective_crops_details.items():
            p2e_converter.stitch_single_perspective(
                target_equirect_torch_cxhxw_rgb_uint8=final_equirect_torch_cxhxw_rgb_uint8,
                processed_crop_torch_cxhxw_rgb_uint8=data["tensor_rgb_uint8"],
                theta=data["theta"],
                phi=data["phi"],
                fov=data["fov_used_for_crop"],
                is_left_eye=("L" in eye_side.split("_")[0]),
            )

        processed_tensor_rgb_uint8 = final_equirect_torch_cxhxw_rgb_uint8

        # Cleanup
        del (
            equirect_converter,
            p2e_converter,
            processed_perspective_crops_details,
            analyzed_faces_for_vr,
        )
        torch.cuda.empty_cache()

        return processed_tensor_rgb_uint8

    def _process_frame_standard(
        self,
        processed_tensor_rgb_uint8: torch.Tensor,
        control: dict,
        stop_event: threading.Event,
    ) -> torch.Tensor:
        """
        Handles the standard (flat) processing logic:
        - Rotation
        - Detection
        - Swapping/Editing
        - Overlays (BBox, Landmarks, Comparison)
        """
        swap_button_is_checked_global = self.main_window.swapfacesButton.isChecked()
        edit_button_is_checked_global = self.main_window.editFacesButton.isChecked()
        det_faces_data_for_display = []

        img = processed_tensor_rgb_uint8
        img_x, img_y = img.size(2), img.size(1)
        scale_applied = False

        # Downscale for processing if too large (standard pipeline optimization)
        if img_x < 512 or img_y < 512:
            if img_x <= img_y:
                new_h, new_w = int(512 * img_y / img_x), 512
            else:
                new_h, new_w = 512, int(512 * img_x / img_y)
            img = v2.Resize((new_h, new_w), antialias=False)(img)
            scale_applied = True

        # Manual Rotation
        if control["ManualRotationEnableToggle"]:
            img = v2.functional.rotate(
                img,
                angle=control["ManualRotationAngleSlider"],
                interpolation=v2.InterpolationMode.BILINEAR,
                expand=True,
            )

        # Detection Setup
        use_landmark, landmark_mode, from_points = (
            control["LandmarkDetectToggle"],
            control["LandmarkDetectModelSelection"],
            control["DetectFromPointsToggle"],
        )
        if edit_button_is_checked_global:
            use_landmark, landmark_mode, from_points = True, "203", True

        # --- Tracking Logic ---
        detection_interval = int(control.get("FaceDetectionIntervalSlider", 1))
        previous_faces_arg = None

        if (
            len(self.last_detected_faces) > 0
            and self.frame_number % detection_interval != 0
            and self.frame_number == self.last_processed_frame_number + 1
        ):
            previous_faces_arg = self.last_detected_faces

        bboxes, kpss_5, kpss = self.models_processor.run_detect(
            img,
            control["DetectorModelSelection"],
            max_num=control["MaxFacesToDetectSlider"],
            score=control["DetectorScoreSlider"] / 100.0,
            input_size=(512, 512),
            use_landmark_detection=use_landmark,
            landmark_detect_mode=landmark_mode,
            landmark_score=control["LandmarkDetectScoreSlider"] / 100.0,
            from_points=from_points,
            rotation_angles=[0]
            if not control["AutoRotationToggle"]
            else [0, 90, 180, 270],
            use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
            previous_detections=previous_faces_arg,
        )

        # Update State for next frame
        self.last_detected_faces = []
        if isinstance(bboxes, np.ndarray) and bboxes.shape[0] > 0:
            for i in range(len(bboxes)):
                self.last_detected_faces.append(
                    {
                        "bbox": bboxes[i],
                        "score": 1.0,
                    }
                )

        if (
            isinstance(kpss_5, np.ndarray)
            and kpss_5.shape[0] > 0
            or isinstance(kpss_5, list)
            and len(kpss_5) > 0
        ):
            for i in range(len(kpss_5)):
                face_emb, _ = self.models_processor.run_recognize_direct(
                    img,
                    kpss_5[i],
                    control["SimilarityTypeSelection"],
                    control["RecognitionModelSelection"],
                )
                det_faces_data_for_display.append(
                    {
                        "kps_5": kpss_5[i],
                        "kps_all": kpss[i],
                        "embedding": face_emb,
                        "bbox": bboxes[i],
                        "original_face": None,
                        "swap_mask": None,
                    }
                )

        # Swapping / Editing Loop
        if det_faces_data_for_display:
            if control["SwapOnlyBestMatchEnableToggle"]:
                # --- Branch: Swap Only Best Match ---
                for _, target_face in self.main_window.target_faces.items():
                    if stop_event.is_set():
                        break

                    params = ParametersDict(
                        self.parameters[target_face.face_id],
                        self.main_window.default_parameters.data,
                    )
                    best_fface, best_score = None, -1.0

                    for fface in det_faces_data_for_display:
                        tgt, tgt_params, score = self._find_best_target_match(
                            fface["embedding"], control
                        )
                        if tgt and tgt.face_id == target_face.face_id:
                            if (
                                score >= tgt_params["SimilarityThresholdSlider"]
                                and score > best_score
                            ):
                                best_score = score
                                best_fface = fface

                    if best_fface is not None and (
                        swap_button_is_checked_global or edit_button_is_checked_global
                    ):
                        denoiser_on = (
                            control.get(
                                "DenoiserUNetEnableBeforeRestorersToggle", False
                            )
                            or control.get("DenoiserAfterFirstRestorerToggle", False)
                            or control.get("DenoiserAfterRestorersToggle", False)
                        )
                        if (
                            denoiser_on
                            and target_face.assigned_kv_map is None
                            and target_face.assigned_input_faces
                        ):
                            target_face.calculate_assigned_input_embedding()

                        # --- MORPHING: Swap Only Best Match ---
                        source_kps = None
                        if target_face and target_face.assigned_input_faces:
                            first_input_id = list(
                                target_face.assigned_input_faces.keys()
                            )[0]
                            store = target_face.assigned_input_faces[first_input_id]
                            if "kps_5" in store:
                                source_kps = store["kps_5"]

                        best_fface["kps_5"] = keypoints_adjustments(
                            best_fface["kps_5"], params, source_kps=source_kps
                        )

                        s_e = None
                        arcface_model = self.models_processor.get_arcface_model(
                            params["SwapModelSelection"]
                        )
                        if (
                            swap_button_is_checked_global
                            and params["SwapModelSelection"] != "DeepFaceLive (DFM)"
                        ):
                            s_e = target_face.assigned_input_embedding.get(
                                arcface_model
                            )
                            if s_e is not None and np.isnan(s_e).any():
                                s_e = None

                        img, best_fface["original_face"], best_fface["swap_mask"] = (
                            self.swap_core(
                                img,
                                best_fface["kps_5"],
                                best_fface["kps_all"],
                                s_e=s_e,
                                t_e=target_face.get_embedding(arcface_model),
                                parameters=params,
                                control=control,
                                dfm_model_name=params["DFMModelSelection"],
                                kv_map=target_face.assigned_kv_map,
                            )
                        )
                        if edit_button_is_checked_global and any(
                            params[f]
                            for f in (
                                "FaceMakeupEnableToggle",
                                "HairMakeupEnableToggle",
                                "EyeBrowsMakeupEnableToggle",
                                "LipsMakeupEnableToggle",
                            )
                        ):
                            img = self.frame_edits.swap_edit_face_core_makeup(
                                img, best_fface["kps_all"], params.data, control
                            )

            else:
                # --- Branch: Swap All Matches ---
                for fface in det_faces_data_for_display:
                    if stop_event.is_set():
                        break
                    best_target, params, _ = self._find_best_target_match(
                        fface["embedding"], control
                    )

                    if best_target and (
                        swap_button_is_checked_global or edit_button_is_checked_global
                    ):
                        denoiser_on = (
                            control.get(
                                "DenoiserUNetEnableBeforeRestorersToggle", False
                            )
                            or control.get("DenoiserAfterFirstRestorerToggle", False)
                            or control.get("DenoiserAfterRestorersToggle", False)
                        )
                        if (
                            denoiser_on
                            and best_target.assigned_kv_map is None
                            and best_target.assigned_input_faces
                        ):
                            best_target.calculate_assigned_input_embedding()

                        # --- MORPHING: Branch Swap All Matches ---
                        source_kps = None
                        if best_target and best_target.assigned_input_faces:
                            first_input_id = list(
                                best_target.assigned_input_faces.keys()
                            )[0]
                            store = best_target.assigned_input_faces[first_input_id]
                            if "kps_5" in store:
                                source_kps = store["kps_5"]

                        fface["kps_5"] = keypoints_adjustments(
                            fface["kps_5"], params, source_kps=source_kps
                        )

                        arcface_model = self.models_processor.get_arcface_model(
                            params["SwapModelSelection"]
                        )
                        s_e = None
                        if (
                            swap_button_is_checked_global
                            and params["SwapModelSelection"] != "DeepFaceLive (DFM)"
                        ):
                            s_e = best_target.assigned_input_embedding.get(
                                arcface_model
                            )
                            if s_e is not None and np.isnan(s_e).any():
                                s_e = None

                        img, fface["original_face"], fface["swap_mask"] = (
                            self.swap_core(
                                img,
                                fface["kps_5"],
                                fface["kps_all"],
                                s_e=s_e,
                                t_e=best_target.get_embedding(arcface_model),
                                parameters=params,
                                control=control,
                                dfm_model_name=params["DFMModelSelection"],
                                kv_map=best_target.assigned_kv_map,
                            )
                        )
                        if edit_button_is_checked_global and any(
                            params[f]
                            for f in (
                                "FaceMakeupEnableToggle",
                                "HairMakeupEnableToggle",
                                "EyeBrowsMakeupEnableToggle",
                                "LipsMakeupEnableToggle",
                            )
                        ):
                            img = self.frame_edits.swap_edit_face_core_makeup(
                                img, fface["kps_all"], params.data, control
                            )

        # Undo Rotation / Scaling
        if control["ManualRotationEnableToggle"]:
            img = v2.functional.rotate(
                img,
                angle=-control["ManualRotationAngleSlider"],
                interpolation=v2.InterpolationMode.BILINEAR,
                expand=True,
            )
        if scale_applied:
            img = v2.Resize(
                (img_y, img_x),
                interpolation=self.interpolation_scaleback,
                antialias=False,
            )(img)

        processed_tensor_rgb_uint8 = img

        # --- Overlays ---
        if control["ShowAllDetectedFacesBBoxToggle"] and det_faces_data_for_display:
            processed_tensor_rgb_uint8 = draw_bounding_boxes_on_detected_faces(
                processed_tensor_rgb_uint8, det_faces_data_for_display
            )

        if control["ShowLandmarksEnableToggle"] and det_faces_data_for_display:
            landmarks_data = self._resolve_landmarks_to_draw(
                det_faces_data_for_display, control
            )
            if landmarks_data:
                temp_permuted = processed_tensor_rgb_uint8.permute(1, 2, 0)
                temp_permuted = paint_landmarks_on_image(temp_permuted, landmarks_data)
                processed_tensor_rgb_uint8 = temp_permuted.permute(2, 0, 1)

        compare_mode_active = self.is_view_face_mask or self.is_view_face_compare
        if compare_mode_active and det_faces_data_for_display:
            processed_tensor_rgb_uint8 = self.get_compare_faces_image(
                processed_tensor_rgb_uint8, det_faces_data_for_display, control
            )

        return processed_tensor_rgb_uint8

    def _resolve_landmarks_to_draw(self, det_faces_data: list, control: dict) -> list:
        """
        Helper to determine which landmarks to draw and in what color based on matches.
        """
        landmarks_to_draw = []
        for fface_data in det_faces_data:
            _, matched_params, _ = self._find_best_target_match(
                fface_data["embedding"], control
            )
            if matched_params:
                use_adj = matched_params["LandmarksPositionAdjEnableToggle"]
                keypoints = (
                    fface_data.get("kps_5") if use_adj else fface_data.get("kps_all")
                )
                kcolor = (255, 0, 0) if use_adj else (0, 255, 255)

                if keypoints is not None:
                    landmarks_to_draw.append({"kps": keypoints, "color": kcolor})

        return landmarks_to_draw

    def get_compare_faces_image(
        self, img: torch.Tensor, det_faces_data: list, control: dict
    ) -> torch.Tensor:
        imgs_to_vstack = []
        for _, fface in enumerate(det_faces_data):
            best_target_for_compare, parameters_for_face, _ = (
                self._find_best_target_match(fface["embedding"], control)
            )
            if best_target_for_compare and parameters_for_face:
                modified_face = self.get_cropped_face_using_kps(
                    img, fface["kps_5"], parameters_for_face
                )
                if control["FrameEnhancerEnableToggle"]:
                    enhanced_version = self.frame_enhancers.enhance_core(
                        modified_face.clone(), control=control
                    )
                    if enhanced_version.shape[1:] != modified_face.shape[1:]:
                        enhanced_version = v2.Resize(
                            modified_face.shape[1:], antialias=True
                        )(enhanced_version)
                    modified_face = enhanced_version
                imgs_to_cat_horizontally = []
                original_face_from_swap_core = fface.get("original_face")
                if original_face_from_swap_core is not None:
                    imgs_to_cat_horizontally.append(
                        original_face_from_swap_core.permute(2, 0, 1)
                    )
                imgs_to_cat_horizontally.append(modified_face)
                swap_mask_from_swap_core = fface.get("swap_mask")
                if swap_mask_from_swap_core is not None:
                    mask_chw = swap_mask_from_swap_core.permute(2, 0, 1)
                    if mask_chw.shape[0] == 1:
                        mask_chw = mask_chw.repeat(3, 1, 1)
                    imgs_to_cat_horizontally.append(mask_chw)
                if imgs_to_cat_horizontally:
                    min_h = min(t.shape[1] for t in imgs_to_cat_horizontally)
                    resized_imgs_to_cat = []
                    for t_img in imgs_to_cat_horizontally:
                        if t_img.shape[1] != min_h:
                            aspect_ratio = t_img.shape[2] / t_img.shape[1]
                            new_w = (
                                int(min_h * aspect_ratio)
                                if aspect_ratio > 0
                                else t_img.shape[2]
                            )
                            resized_imgs_to_cat.append(
                                v2.Resize((min_h, new_w), antialias=True)(t_img)
                            )
                        else:
                            resized_imgs_to_cat.append(t_img)
                    imgs_to_vstack.append(torch.cat(resized_imgs_to_cat, dim=2))

        if imgs_to_vstack:
            max_width_for_vstack = max(
                img_strip.size(2) for img_strip in imgs_to_vstack
            )
            padded_strips_for_vstack = [
                torch.nn.functional.pad(
                    img_strip, (0, max_width_for_vstack - img_strip.size(2), 0, 0)
                )
                for img_strip in imgs_to_vstack
            ]
            return torch.cat(padded_strips_for_vstack, dim=1)
        return img

    def get_cropped_face_using_kps(
        self, img: torch.Tensor, kps_5: np.ndarray, parameters: dict
    ) -> torch.Tensor:
        tform = self.get_face_similarity_tform(parameters["SwapModelSelection"], kps_5)
        face_512_aligned = v2.functional.affine(
            img,
            angle=tform.rotation * 57.2958,
            translate=(tform.translation[0], tform.translation[1]),
            scale=tform.scale,
            shear=(0.0, 0.0),
            center=(0, 0),
            interpolation=self.interpolation_get_cropped_face_kps,
        )
        return v2.functional.crop(face_512_aligned, 0, 0, 512, 512)

    def get_face_similarity_tform(
        self, swapper_model: str, kps_5: np.ndarray
    ) -> trans.SimilarityTransform:
        if (
            swapper_model != "GhostFace-v1"
            and swapper_model != "GhostFace-v2"
            and swapper_model != "GhostFace-v3"
            and swapper_model != "CSCS"
        ):
            dst = faceutil.get_arcface_template(image_size=512, mode="arcface128")
            dst = np.squeeze(dst)
            # Use instance initialization + .estimate() for older skimage versions
            if hasattr(trans.SimilarityTransform, "from_estimate"):
                tform = trans.SimilarityTransform.from_estimate(kps_5, dst)
            else:
                tform = trans.SimilarityTransform()
                tform.estimate(kps_5, dst)
        elif swapper_model == "CSCS":
            # Use instance initialization + .estimate() for older skimage versions
            if hasattr(trans.SimilarityTransform, "from_estimate"):
                tform = trans.SimilarityTransform.from_estimate(
                    kps_5, self.models_processor.FFHQ_kps
                )
            else:
                tform = trans.SimilarityTransform()
                tform.estimate(kps_5, self.models_processor.FFHQ_kps)
        else:
            tform = trans.SimilarityTransform()
            dst = faceutil.get_arcface_template(image_size=512, mode="arcfacemap")
            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
            tform.params[0:2] = M
        return tform

    def get_transformed_and_scaled_faces(self, tform, img):
        original_face_512 = v2.functional.affine(
            img,
            tform.rotation * 57.2958,
            (tform.translation[0], tform.translation[1]),
            tform.scale,
            0,
            center=(0, 0),
            interpolation=self.interpolation_original_face_512,
        )
        original_face_512 = v2.functional.crop(original_face_512, 0, 0, 512, 512)
        original_face_384 = self.t384(original_face_512)
        original_face_256 = self.t256(original_face_512)
        original_face_128 = self.t128(original_face_256)
        return (
            original_face_512,
            original_face_384,
            original_face_256,
            original_face_128,
        )

    def get_affined_face_dim_and_swapping_latents(
        self,
        original_faces: tuple,
        swapper_model,
        dfm_model_name,
        s_e,
        t_e,
        parameters,
        cmddebug,
        tform,
    ):
        original_face_512, original_face_384, original_face_256, original_face_128 = (
            original_faces
        )

        dfm_model_instance = None
        input_face_affined = None
        dim = 1
        latent = None

        # Helper to apply Identity Boost (Face Likeness) using SLERP & LERP
        def apply_likeness_with_norm_preservation(
            source_latent: torch.Tensor, target_latent: torch.Tensor, params: dict
        ) -> torch.Tensor:
            if not params.get("FaceLikenessEnableToggle", False):
                return source_latent

            factor = float(params.get("FaceLikenessFactorDecimalSlider", 0.0))
            if factor == 0.0:
                return source_latent

            # 1. Capture original energy (Norms are generally constant in ArcFace)
            s_norm = torch.norm(source_latent)
            t_norm = torch.norm(target_latent)

            if s_norm < 1e-6 or t_norm < 1e-6:
                return source_latent

            # 2. Normalize to get directional vectors on the hypersphere
            s_dir = source_latent / s_norm
            t_dir = target_latent / t_norm

            if factor < 0.0:
                # --- INTERPOLATION (SLERP) ---
                # Move naturally towards the target face along the sphere
                t = 1.0 + factor

                cos_theta = torch.sum(s_dir * t_dir)
                cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)
                theta = torch.acos(cos_theta)
                sin_theta = torch.sin(theta)

                if sin_theta < 1e-3:
                    blended_dir = (1.0 - t) * t_dir + t * s_dir
                else:
                    weight_t = torch.sin((1.0 - t) * theta) / sin_theta
                    weight_s = torch.sin(t * theta) / sin_theta
                    blended_dir = weight_t * t_dir + weight_s * s_dir
            else:
                # --- EXTRAPOLATION (LERP) ---
                # Push the vector away from the target to exaggerate the source identity
                difference_vector = s_dir - t_dir
                blended_dir = s_dir + (factor * difference_vector)

            # 3. Always restore original Source Energy to prevent latent space corruption
            blended_dir = blended_dir / torch.norm(blended_dir)
            final_latent = blended_dir * s_norm

            return final_latent

        # --- Inswapper128 Logic ---
        if swapper_model == "Inswapper128":
            latent = (
                torch.from_numpy(self.models_processor.calc_inswapper_latent(s_e))
                .float()
                .to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(self.models_processor.calc_inswapper_latent(t_e))
                .float()
                .to(self.models_processor.device)
            )

            latent = apply_likeness_with_norm_preservation(
                latent, dst_latent, parameters
            )

            dim = 1
            if parameters["SwapperResAutoSelectEnableToggle"]:
                if tform.scale <= 1.00:
                    dim = 4
                    input_face_affined = original_face_512
                elif tform.scale <= 1.75:
                    dim = 3
                    input_face_affined = original_face_384
                elif tform.scale <= 2:
                    dim = 2
                    input_face_affined = original_face_256
                else:
                    dim = 1
                    input_face_affined = original_face_128
            else:
                if parameters["SwapperResSelection"] == "128":
                    dim = 1
                    input_face_affined = original_face_128
                elif parameters["SwapperResSelection"] == "256":
                    dim = 2
                    input_face_affined = original_face_256
                elif parameters["SwapperResSelection"] == "384":
                    dim = 3
                    input_face_affined = original_face_384
                elif parameters["SwapperResSelection"] == "512":
                    dim = 4
                    input_face_affined = original_face_512

        # --- InStyleSwapper Logic ---
        elif swapper_model in (
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
        ):
            version = swapper_model[-1]
            latent = (
                torch.from_numpy(
                    self.models_processor.calc_swapper_latent_iss(s_e, version)
                )
                .float()
                .to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.models_processor.calc_swapper_latent_iss(t_e, version)
                )
                .float()
                .to(self.models_processor.device)
            )

            latent = apply_likeness_with_norm_preservation(
                latent, dst_latent, parameters
            )

            if (
                (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version A"
                    and parameters["InStyleResAEnableToggle"]
                )
                or (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version B"
                    and parameters["InStyleResBEnableToggle"]
                )
                or (
                    parameters["SwapModelSelection"] == "InStyleSwapper256 Version C"
                    and parameters["InStyleResCEnableToggle"]
                )
            ):
                dim = 4
                input_face_affined = original_face_512
            else:
                dim = 2
                input_face_affined = original_face_256

        # --- SimSwap Logic ---
        elif swapper_model == "SimSwap512":
            latent = (
                torch.from_numpy(
                    self.models_processor.calc_swapper_latent_simswap512(s_e)
                )
                .float()
                .to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(
                    self.models_processor.calc_swapper_latent_simswap512(t_e)
                )
                .float()
                .to(self.models_processor.device)
            )

            latent = apply_likeness_with_norm_preservation(
                latent, dst_latent, parameters
            )

            dim = 4
            input_face_affined = original_face_512

        # --- GhostFace Logic ---
        elif (
            swapper_model == "GhostFace-v1"
            or swapper_model == "GhostFace-v2"
            or swapper_model == "GhostFace-v3"
        ):
            latent = (
                torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(s_e))
                .float()
                .to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(t_e))
                .float()
                .to(self.models_processor.device)
            )

            latent = apply_likeness_with_norm_preservation(
                latent, dst_latent, parameters
            )

            dim = 2
            input_face_affined = original_face_256

        # --- CSCS Logic ---
        elif swapper_model == "CSCS":
            latent = (
                torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(s_e))
                .float()
                .to(self.models_processor.device)
            )
            dst_latent = (
                torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(t_e))
                .float()
                .to(self.models_processor.device)
            )

            latent = apply_likeness_with_norm_preservation(
                latent, dst_latent, parameters
            )

            dim = 2
            input_face_affined = original_face_256

        # --- DFM Logic ---
        if swapper_model == "DeepFaceLive (DFM)" and dfm_model_name:
            dfm_model_instance = self.models_processor.load_dfm_model(dfm_model_name)
            latent = []
            input_face_affined = original_face_512
            dim = 4

        return input_face_affined, dfm_model_instance, dim, latent

    def get_swapped_and_prev_face(
        self,
        output,
        input_face_affined,
        original_face_512,
        latent,
        itex,
        dim,
        swapper_model,
        dfm_model,
        parameters,
    ):
        if parameters["PreSwapSharpnessDecimalSlider"] != 1.0:
            input_face_affined = input_face_affined.permute(2, 0, 1)
            input_face_affined = v2.functional.adjust_sharpness(
                input_face_affined, parameters["PreSwapSharpnessDecimalSlider"]
            )
            input_face_affined = input_face_affined.permute(1, 2, 0)

        prev_face = input_face_affined.clone()

        if swapper_model == "Inswapper128":
            for k in range(itex):
                # Lists to hold independent memory buffers for this iteration
                tile_inputs = []
                tile_outputs = []
                tile_coords = []

                # 1. PREPARATION PHASE (CPU)
                for j in range(dim):
                    for i in range(dim):
                        tile = input_face_affined[j::dim, i::dim]
                        t_in = tile.permute(2, 0, 1).contiguous().unsqueeze(0)
                        t_out = torch.empty_like(t_in)

                        tile_inputs.append(t_in)
                        tile_outputs.append(t_out)
                        tile_coords.append((j, i))

                # 2. EXECUTION PHASE (GPU - Async)
                with torch.no_grad():
                    for idx in range(len(tile_inputs)):
                        self.models_processor.run_inswapper(
                            tile_inputs[idx], latent, tile_outputs[idx]
                        )

                # 3. SYNCHRONIZATION PHASE
                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()

                # 4. RECONSTRUCTION PHASE (CPU)
                for idx, (j, i) in enumerate(tile_coords):
                    if tile_outputs[idx].sum() < 1.0:
                        res = tile_inputs[idx]
                    else:
                        res = tile_outputs[idx]

                    res_hwc = res.squeeze(0).permute(1, 2, 0)
                    output[j::dim, i::dim] = res_hwc

                # 5. STATE UPDATE
                prev_face = input_face_affined.clone()
                input_face_affined = output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)

        elif swapper_model in (
            "InStyleSwapper256 Version A",
            "InStyleSwapper256 Version B",
            "InStyleSwapper256 Version C",
        ):
            version = swapper_model[-1]
            dim_res = dim // 2

            for k in range(itex):
                tile_inputs = []
                tile_outputs = []
                tile_coords = []

                for j in range(dim_res):
                    for i in range(dim_res):
                        tile = input_face_affined[j::dim_res, i::dim_res]
                        t_in = tile.permute(2, 0, 1).contiguous().unsqueeze(0)
                        t_out = torch.empty_like(t_in)

                        tile_inputs.append(t_in)
                        tile_outputs.append(t_out)
                        tile_coords.append((j, i))

                with torch.no_grad():
                    for idx in range(len(tile_inputs)):
                        self.models_processor.run_iss_swapper(
                            tile_inputs[idx], latent, tile_outputs[idx], version
                        )

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()

                for idx, (j, i) in enumerate(tile_coords):
                    if tile_outputs[idx].sum() < 1.0:
                        res = tile_inputs[idx]
                    else:
                        res = tile_outputs[idx]

                    res_hwc = res.squeeze(0).permute(1, 2, 0)
                    output[j::dim_res, i::dim_res] = res_hwc

                prev_face = input_face_affined.clone()
                input_face_affined = output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)

        elif swapper_model == "SimSwap512":
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty(
                    (1, 3, 512, 512),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                ).contiguous()

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()

                self.models_processor.run_swapper_simswap512(
                    input_face_disc, latent, swapper_output
                )

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()

                if swapper_output.sum() < 1.0:
                    swapper_output = input_face_disc

                swapper_output = torch.squeeze(swapper_output)
                swapper_output = swapper_output.permute(1, 2, 0)

                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()

                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)

        elif (
            swapper_model == "GhostFace-v1"
            or swapper_model == "GhostFace-v2"
            or swapper_model == "GhostFace-v3"
        ):
            for k in range(itex):
                input_face_disc = torch.mul(input_face_affined, 255.0).permute(2, 0, 1)
                input_face_disc = torch.div(input_face_disc.float(), 127.5)
                input_face_disc = torch.sub(input_face_disc, 1)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty(
                    (1, 3, 256, 256),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                ).contiguous()

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()

                self.models_processor.run_swapper_ghostface(
                    input_face_disc, latent, swapper_output, swapper_model
                )

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()

                swapper_output = swapper_output[0]
                if swapper_output.sum() < 1.0:
                    pass
                swapper_output = swapper_output.permute(1, 2, 0)
                swapper_output = torch.mul(swapper_output, 127.5)
                swapper_output = torch.add(swapper_output, 127.5)

                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()
                input_face_affined = torch.div(input_face_affined, 255)

                output = swapper_output.clone()
                output = torch.clamp(output, 0, 255)

        elif swapper_model == "CSCS":
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = v2.functional.normalize(
                    input_face_disc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
                )
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty(
                    (1, 3, 256, 256),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                ).contiguous()

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()

                self.models_processor.run_swapper_cscs(
                    input_face_disc, latent, swapper_output
                )

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()

                swapper_output = torch.squeeze(swapper_output)
                swapper_output = torch.add(torch.mul(swapper_output, 0.5), 0.5)
                swapper_output = swapper_output.permute(1, 2, 0)

                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()

                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)

        elif swapper_model == "DeepFaceLive (DFM)" and dfm_model:
            out_celeb, _, _ = dfm_model.convert(
                original_face_512,
                parameters["DFMAmpMorphSlider"] / 100,
                rct=parameters["DFMRCTColorToggle"],
            )
            prev_face = input_face_affined.clone()
            input_face_affined = out_celeb.clone()
            output = out_celeb.clone()

        output = output.permute(2, 0, 1)
        swap = self.t512(output)
        return swap, prev_face

    def get_border_mask(self, parameters):
        """Creates the border fade mask based on sliders."""
        border_mask = torch.ones(
            (128, 128), dtype=torch.float32, device=self.models_processor.device
        )
        border_mask = torch.unsqueeze(border_mask, 0)

        if not parameters.get("BordermaskEnableToggle", False):
            return border_mask, border_mask.clone()

        top = parameters["BorderTopSlider"]
        left = parameters["BorderLeftSlider"]
        right = 128 - parameters["BorderRightSlider"]
        bottom = 128 - parameters["BorderBottomSlider"]

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        border_mask_calc = border_mask.clone()

        blur_amount = parameters["BorderBlurSlider"]
        blur_kernel_size = blur_amount * 2 + 1
        if blur_kernel_size > 1:
            sigma_val = max(blur_amount * 0.15 + 0.1, 1e-6)
            gauss = transforms.GaussianBlur(blur_kernel_size, sigma=sigma_val)
            border_mask = gauss(border_mask)
        return border_mask, border_mask_calc

    def get_dynamic_side_mask(
        self, yaw_deg, pitch_deg, height, width, device, parameters, kps_5, tform
    ):
        """
        Smart Profile Masking:
        Instead of a blind gradient, this uses the projected eye positions to ensure
        we NEVER mask the eyes.
        """
        mask = torch.ones((1, height, width), dtype=torch.float32, device=device)

        if not parameters.get("ProfileAngleMaskEnableToggle", False):
            return mask

        start_angle = parameters.get("ProfileAngleMaskThresholdSlider", 20)
        max_strength = parameters.get("ProfileAngleMaskStrengthSlider", 100) / 100.0

        if tform is not None:
            kps_proj = tform(kps_5)
            le_x = kps_proj[0][0]
            re_x = kps_proj[1][0]
        else:
            le_x = width * 0.35
            re_x = width * 0.65

        le_x_norm = np.clip(le_x / width, 0.0, 1.0)
        re_x_norm = np.clip(re_x / width, 0.0, 1.0)
        eye_safety_margin = 0.05

        abs_yaw = abs(yaw_deg)
        if abs_yaw > start_angle:
            angle_excess = max(0, abs_yaw - start_angle)
            strength_yaw = min(angle_excess / 45.0, 1.0) * max_strength
            linspace_x = torch.linspace(0, 1, width, device=device).view(1, 1, width)

            if yaw_deg > 0:
                # Looking Right -> Mask Left side
                fade_end = max(0.0, le_x_norm - eye_safety_margin)
                if fade_end > 0.05:
                    grad_yaw = torch.clamp(linspace_x / fade_end, 0, 1)
                    grad_yaw = 1.0 - (1.0 - grad_yaw) * strength_yaw
                    mask = mask * grad_yaw
            else:
                # Looking Left -> Mask Right side
                fade_start = min(1.0, re_x_norm + eye_safety_margin)
                if fade_start < 0.95:
                    grad_yaw = torch.clamp(
                        (linspace_x - fade_start) / (1.0 - fade_start), 0, 1
                    )
                    grad_yaw = 1.0 - grad_yaw
                    mask_r = torch.ones_like(linspace_x)
                    mask_r[linspace_x > fade_start] = 1.0 - (
                        (linspace_x[linspace_x > fade_start] - fade_start)
                        / (1.0 - fade_start)
                    )
                    grad_yaw = 1.0 - (1.0 - mask_r) * strength_yaw
                    mask = mask * grad_yaw

        return mask

    def swap_core(
        self,
        img: torch.Tensor,
        kps_5: np.ndarray,
        kps: np.ndarray | bool = False,
        s_e: np.ndarray | None = None,
        t_e: np.ndarray | None = None,
        parameters: dict | None = None,
        control: dict | None = None,
        dfm_model_name: str | None = None,
        is_perspective_crop: bool = False,
        kv_map: Dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Core function for face swapping. Handles:
        1. Alignment and Scaling.
        2. Swapping (Model inference).
        3. Blending and Masking (XSeg, Occluder, Texture Transfer).
        4. Color Correction.
        5. Restoration (GFPGAN/CodeFormer).
        6. Reverse alignment (Untransform).
        """
        valid_s_e = s_e if isinstance(s_e, np.ndarray) else None
        valid_t_e = t_e if isinstance(t_e, np.ndarray) else None
        parameters = parameters if parameters is not None else {}
        control = control if control is not None else {}
        swapper_model = parameters["SwapModelSelection"]
        self.set_scaling_transforms(control)

        debug = control.get("CommandLineDebugEnableToggle", False)
        debug_info: dict[str, str] = {}

        tform = self.get_face_similarity_tform(swapper_model, kps_5)

        t512_mask = v2.Resize(
            (512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=True
        )
        t128_mask = v2.Resize(
            (128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=True
        )

        original_face_512, original_face_384, original_face_256, original_face_128 = (
            self.get_transformed_and_scaled_faces(tform, img)
        )
        original_faces = (
            original_face_512,
            original_face_384,
            original_face_256,
            original_face_128,
        )
        swap = original_face_512
        prev_face = None

        # --- SWAPPING INFERENCE ---
        if valid_s_e is not None or (
            swapper_model == "DeepFaceLive (DFM)" and dfm_model_name
        ):
            input_face_affined, dfm_model_instance, dim, latent = (
                self.get_affined_face_dim_and_swapping_latents(
                    original_faces,
                    swapper_model,
                    dfm_model_name,
                    valid_s_e,
                    valid_t_e,
                    parameters,
                    debug,
                    tform,
                )
            )

            # Optional Face Scaling adjustment
            if parameters["FaceAdjEnableToggle"]:
                input_face_affined = v2.functional.affine(
                    input_face_affined,
                    0,
                    (0, 0),
                    1 + parameters["FaceScaleAmountSlider"] / 100,
                    0,
                    center=(dim * 128 / 2, dim * 128 / 2),
                    interpolation=v2.InterpolationMode.BILINEAR,
                )

            itex = 1
            if parameters["StrengthEnableToggle"]:
                itex = ceil(parameters["StrengthAmountSlider"] / 100.0)

            output_size = int(128 * dim)
            output = torch.zeros(
                (output_size, output_size, 3),
                dtype=torch.float32,
                device=self.models_processor.device,
            )
            input_face_affined = input_face_affined.permute(1, 2, 0).contiguous()
            input_face_affined = torch.div(input_face_affined, 255.0)

            swap, prev_face = self.get_swapped_and_prev_face(
                output,
                input_face_affined,
                original_face_512,
                latent,
                itex,
                dim,
                swapper_model,
                dfm_model_instance,
                parameters,
            )

        else:
            swap = original_face_512
            if parameters["StrengthEnableToggle"]:
                itex = ceil(parameters["StrengthAmountSlider"] / 100.0)
                prev_face = torch.div(swap, 255.0)
                prev_face = prev_face.permute(1, 2, 0)

        if parameters["StrengthEnableToggle"]:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters["StrengthAmountSlider"], 100) * 0.01
                if alpha == 0:
                    alpha = 1

                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)

                if prev_face.shape[-1] != swap.shape[-1]:
                    prev_face = v2.Resize(
                        (swap.shape[-2], swap.shape[-1]), antialias=True
                    )(prev_face)

                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1 - alpha)
                swap = torch.add(swap, prev_face)

        # --- DYNAMIC MASKS INITIALIZATION ---
        border_mask, border_mask_calc = self.get_border_mask(parameters)
        current_swap_h, current_swap_w = swap.shape[1], swap.shape[2]

        yaw_deg, pitch_deg = faceutil.calc_face_yaw_pitch(kps_5)
        side_mask = self.get_dynamic_side_mask(
            yaw_deg,
            pitch_deg,
            current_swap_h,
            current_swap_w,
            self.models_processor.device,
            parameters,
            kps_5,
            tform,
        )

        if (
            border_mask.shape[1] != current_swap_h
            or border_mask.shape[2] != current_swap_w
        ):
            resizer = v2.Resize((current_swap_h, current_swap_w), antialias=True)
            border_mask = resizer(border_mask)
            border_mask_calc = resizer(border_mask_calc)

        border_mask = border_mask * side_mask
        border_mask_calc = border_mask_calc * side_mask

        swap_mask = torch.ones(
            (current_swap_h, current_swap_w),
            dtype=torch.float32,
            device=self.models_processor.device,
        )
        swap_mask = torch.unsqueeze(swap_mask, 0)
        swap_mask_noFP = border_mask.clone()

        BgExclude = torch.ones(
            (512, 512), dtype=torch.float32, device=self.models_processor.device
        )
        BgExclude = torch.unsqueeze(BgExclude, 0)
        diff_mask = BgExclude.clone()
        texture_mask_view = BgExclude.clone()
        restore_mask = BgExclude.clone()
        texture_exclude_512 = BgExclude.clone()

        calc_mask = BgExclude.clone()
        calc_mask_dill = BgExclude.clone()
        mask_forcalc_512 = BgExclude.clone()

        M_ref = tform.params[0:2]
        ones_column_ref = np.ones((kps_5.shape[0], 1), dtype=np.float32)
        kps_ref = np.hstack([kps_5, ones_column_ref]) @ M_ref.T

        swap = torch.clamp(swap, 0.0, 255.0)

        # --- FACE EDITING (Beginning) ---
        # Expression Restorer beginning
        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced") == "Simple"
            )
            and parameters["FaceExpressionBeforeTypeSelection"] == "Beginning"
        ):
            swap = self.frame_edits.apply_face_expression_restorer(
                original_face_512, swap, cast(dict, parameters)
            )

        # Face editor beginning
        if (
            parameters["FaceEditorEnableToggle"]
            and self.main_window.editFacesButton.isChecked()
            and parameters["FaceEditorBeforeTypeSelection"] == "Beginning"
        ):
            editor_mask = swap_mask.clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.frame_edits.swap_edit_face_core(swap, swap, parameters, control)

        # First Denoiser pass - Before Restorers
        if control.get("DenoiserUNetEnableBeforeRestorersToggle", False):
            swap = self._apply_denoiser_pass(swap, control, "Before", kv_map)

        # --- MOUTH ENHANCEMENT & ALIGNMENT (PRE-RESTORER) ---
        paste_after_restorer = parameters.get("MouthParserStretchAfterToggle", False)

        if not paste_after_restorer:
            mouth_overlay_pkg = None
            if hasattr(self.models_processor, "face_masks"):
                mouth_overlay_pkg = self.models_processor.face_masks.get_mouth_overlay(
                    swap, original_face_512, parameters
                )

            if mouth_overlay_pkg is not None:
                overlay_rgb, overlay_mask = mouth_overlay_pkg
                if overlay_rgb is not None and overlay_mask is not None:
                    if overlay_rgb.shape[-1] != swap.shape[-1]:
                        overlay_rgb = v2.Resize(
                            (swap.shape[-2], swap.shape[-1]), antialias=True
                        )(overlay_rgb)
                        overlay_mask = v2.Resize(
                            (swap.shape[-2], swap.shape[-1]), antialias=True
                        )(overlay_mask.unsqueeze(0)).squeeze(0)

                    swap = swap * (1.0 - overlay_mask) + overlay_rgb * overlay_mask

        # --- RESTORATION 1 ---
        swap_original = swap.clone()

        if parameters["FaceRestorerEnableToggle"]:
            swap_restorecalc = self.models_processor.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetTypeSelection"],
                parameters["FaceRestorerTypeSelection"],
                parameters["FaceRestorerBlendSlider"],
                parameters["FaceFidelityWeightDecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=1,
            )
        else:
            swap_restorecalc = swap.clone()

        # Occluder
        if parameters["OccluderEnableToggle"]:
            mask = self.models_processor.face_masks.apply_occlusion(
                original_face_256,
                parameters["OccluderSizeSlider"],
                parameters=parameters,
                original_face_512=swap_restorecalc,
            )
            if mask.shape[-1] != swap_mask.shape[-1]:
                mask = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(mask)
            swap_mask = torch.mul(swap_mask, mask)

            gauss = transforms.GaussianBlur(
                parameters["OccluderXSegBlurSlider"] * 2 + 1,
                (parameters["OccluderXSegBlurSlider"] + 1) * 0.2,
            )
            swap_mask = gauss(swap_mask)

            if swap_mask_noFP.shape[-1] != swap_mask.shape[-1]:
                swap_mask_noFP = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(swap_mask_noFP)
            swap_mask_noFP *= swap_mask

        # --- MASKS (Parser / CLIPs / Restore) ---
        need_any_parser = (
            parameters.get("FaceParserEnableToggle", False)
            or (
                parameters.get("DFLXSegEnableToggle", False)
                and (
                    (
                        parameters.get("XSegMouthEnableToggle", False)
                        and parameters.get("DFLXSegSizeSlider", 0)
                        != parameters.get("DFLXSeg2SizeSlider", 0)
                    )
                    or parameters.get("XSegExcludeInnerMouthToggle", False)
                )
            )
            or (
                parameters.get("TransferTextureEnableToggle", False)
                or parameters.get("DifferencingEnableToggle", False)
            )
            and (parameters.get("ExcludeMaskEnableToggle", False))
        )

        FaceParser_mask = None
        mouth_512 = None
        inner_mouth_protection_512 = None

        if need_any_parser:
            out = self.models_processor.process_masks_and_masks(
                swap_restorecalc,
                original_face_512,
                parameters,
                control,
            )
            if not parameters.get("FaceParserEndToggle", False):
                FaceParser_mask = out.get("FaceParser_mask", None)

            texture_exclude_512 = out.get("texture_mask", texture_exclude_512)
            mouth_512 = out.get("mouth", None)
            inner_mouth_protection_512 = out.get("inner_mouth_protection", None)

        if FaceParser_mask is not None:
            if FaceParser_mask.shape[-1] != swap_mask.shape[-1]:
                FaceParser_mask = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(FaceParser_mask)
            swap_mask = swap_mask * FaceParser_mask

        # CLIPs
        if parameters.get("ClipEnableToggle", False):
            mask_clip = self.models_processor.run_CLIPs(
                original_face_512,
                parameters["ClipText"],
                parameters["ClipAmountSlider"],
            )
            if mask_clip.shape[-1] != swap_mask.shape[-1]:
                mask_clip = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(mask_clip)
            swap_mask *= mask_clip
            if swap_mask_noFP.shape[-1] != mask_clip.shape[-1]:
                swap_mask_noFP = v2.Resize(
                    (mask_clip.shape[-2], mask_clip.shape[-1]), antialias=True
                )(swap_mask_noFP)
            swap_mask_noFP *= mask_clip

        # Restore Eyes/Mouth
        if parameters.get("RestoreMouthEnableToggle", False) or parameters.get(
            "RestoreEyesEnableToggle", False
        ):
            M = tform.params[0:2]
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            dst_kps_5 = np.hstack([kps_5, ones_column]) @ M.T

            img_swap_mask = torch.ones(
                (1, 512, 512), dtype=torch.float32, device=self.models_processor.device
            )
            img_orig_mask = torch.zeros(
                (1, 512, 512), dtype=torch.float32, device=self.models_processor.device
            )

            if parameters.get("RestoreMouthEnableToggle", False):
                img_swap_mask = self.models_processor.restore_mouth(
                    img_orig_mask,
                    img_swap_mask,
                    dst_kps_5,
                    parameters["RestoreMouthBlendAmountSlider"] / 100.0,
                    parameters["RestoreMouthFeatherBlendSlider"],
                    parameters["RestoreMouthSizeFactorSlider"] / 100.0,
                    parameters["RestoreXMouthRadiusFactorDecimalSlider"],
                    parameters["RestoreYMouthRadiusFactorDecimalSlider"],
                    parameters["RestoreXMouthOffsetSlider"],
                    parameters["RestoreYMouthOffsetSlider"],
                ).clamp(0, 1)

            if parameters.get("RestoreEyesEnableToggle", False):
                img_swap_mask = self.models_processor.restore_eyes(
                    img_orig_mask,
                    img_swap_mask,
                    dst_kps_5,
                    parameters["RestoreEyesBlendAmountSlider"] / 100.0,
                    parameters["RestoreEyesFeatherBlendSlider"],
                    parameters["RestoreEyesSizeFactorDecimalSlider"],
                    parameters["RestoreXEyesRadiusFactorDecimalSlider"],
                    parameters["RestoreYEyesRadiusFactorDecimalSlider"],
                    parameters["RestoreXEyesOffsetSlider"],
                    parameters["RestoreYEyesOffsetSlider"],
                    parameters["RestoreEyesSpacingOffsetSlider"],
                ).clamp(0, 1)

            if parameters.get("RestoreEyesMouthBlurSlider", 0) > 0:
                b = parameters["RestoreEyesMouthBlurSlider"]
                gauss = transforms.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                img_swap_mask = gauss(img_swap_mask)

            if img_swap_mask.shape[-1] != swap_mask.shape[-1]:
                mask_resized = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(img_swap_mask)
            else:
                mask_resized = img_swap_mask
            swap_mask = swap_mask * mask_resized

        # --- DFL XSeg ---
        t256_near = v2.Resize(
            (256, 256), interpolation=v2.InterpolationMode.NEAREST, antialias=False
        )

        if parameters.get("DFLXSegEnableToggle", False):
            img_xseg_256 = t256_near(original_face_512)
            mouth_256 = None
            inner_mouth_protection_256 = None
            if (
                parameters.get("DFLXSegEnableToggle", False)
                and parameters.get("XSegMouthEnableToggle", False)
                and parameters.get("DFLXSegSizeSlider", 0)
                != parameters.get("DFLXSeg2SizeSlider", 0)
                and mouth_512 is not None
            ):
                mouth_256 = t256_near(mouth_512.unsqueeze(0))

            if (
                parameters.get("XSegExcludeInnerMouthToggle", False)
                and inner_mouth_protection_512 is not None
            ):
                inner_mouth_protection_256 = t256_near(
                    inner_mouth_protection_512.unsqueeze(0)
                ).squeeze(0)

            img_mask_256, mask_forcalc_256, mask_forcalc_dill_256, outpred_noFP_256 = (
                self.models_processor.apply_dfl_xseg(
                    img_xseg_256,
                    -parameters["DFLXSegSizeSlider"],
                    mouth_256 if mouth_256 is not None else 0,
                    parameters,
                    inner_mouth_mask=inner_mouth_protection_256,
                )
            )

            if img_mask_256.shape[-1] != swap_mask.shape[-1]:
                img_mask_res = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(img_mask_256)
                outpred_noFP_res = v2.Resize(
                    (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
                )(outpred_noFP_256)
            else:
                img_mask_res = img_mask_256
                outpred_noFP_res = outpred_noFP_256

            mask_forcalc_512 = t512_mask(mask_forcalc_256)
            mask_forcalc_dill_512 = t512_mask(mask_forcalc_dill_256)

            mask_forcalc_512 = 1 - mask_forcalc_512
            mask_forcalc_dill_512 = 1 - mask_forcalc_dill_512
            calc_mask = mask_forcalc_512
            calc_mask_dill = mask_forcalc_dill_512

            if swap_mask_noFP.shape[-1] != outpred_noFP_res.shape[-1]:
                swap_mask_noFP = v2.Resize(
                    (outpred_noFP_res.shape[-2], outpred_noFP_res.shape[-1]),
                    antialias=True,
                )(swap_mask_noFP)

            swap_mask_noFP = swap_mask_noFP * (1.0 - outpred_noFP_res)
            swap_mask = swap_mask * (1.0 - img_mask_res)
        else:
            calc_mask = t512_mask(swap_mask.clone()).clamp(0, 1)
            calc_mask_dill = calc_mask.clone()
            mask_forcalc_512 = calc_mask.clone()

        mask_autocolor = calc_mask.clone()
        mask_autocolor = mask_autocolor > 0.05

        # Auto Restore (First Pass)
        if (
            parameters["FaceRestorerEnableToggle"]
            and parameters["FaceRestorerAutoEnableToggle"]
        ):
            original_face_512_autorestore = original_face_512.clone()
            swap_original_autorestore = swap_original.clone()
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"]) / 100.0
            adjust_sharpness = float(parameters["FaceRestorerAutoSharpAdjustSlider"])
            scale_factor = round(tform.scale, 2)
            automasktoggle = parameters["FaceRestorerAutoMaskEnableToggle"]
            automaskadjust = parameters["FaceRestorerAutoSharpMaskAdjustDecimalSlider"]
            automaskblur = 2
            restore_mask = mask_forcalc_512.clone()

            alpha_auto, blur_value = self.face_restorer_auto(
                original_face_512_autorestore,
                swap_original_autorestore,
                swap_restorecalc,
                alpha_restorer,
                adjust_sharpness,
                scale_factor,
                debug,
                restore_mask,
                automasktoggle,
                automaskadjust,
                automaskblur,
            )

            if blur_value > 0:
                kernel_size = 2 * blur_value + 1
                sigma = blur_value * 0.1
                gaussian_blur = transforms.GaussianBlur(
                    kernel_size=kernel_size, sigma=sigma
                )
                swap = gaussian_blur(swap_original)
                debug_info["Restore1"] = f": {-blur_value:.2f}"
            elif isinstance(alpha_auto, torch.Tensor):
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
            elif alpha_auto != 0:
                swap = swap_restorecalc * alpha_auto + swap_original * (1 - alpha_auto)
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto * 100:.2f}"
            else:
                swap = swap_original
                if debug:
                    debug_info["Restore1"] = f": {alpha_auto * 100:.2f}"

        elif parameters["FaceRestorerEnableToggle"]:
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"]) / 100.0
            swap = torch.add(
                torch.mul(swap_restorecalc, alpha_restorer),
                torch.mul(swap_original, 1 - alpha_restorer),
            ).contiguous()

        # Expression Restorer (After First)
        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced") == "Simple"
            )
            and parameters["FaceExpressionBeforeTypeSelection"]
            == "After First Restorer"
        ):
            swap = self.frame_edits.apply_face_expression_restorer(
                original_face_512, swap, cast(dict, parameters)
            )

        # Face Editor (After First)
        if (
            parameters["FaceEditorEnableToggle"]
            and self.main_window.editFacesButton.isChecked()
            and parameters["FaceEditorBeforeTypeSelection"] == "After First Restorer"
        ):
            editor_mask = swap_mask.clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.frame_edits.swap_edit_face_core(
                swap, swap_restorecalc, parameters, control
            )
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = v2.Resize((swap.shape[-2], swap.shape[-1]), antialias=True)(
                    swap_mask_noFP
                )
            else:
                swap_mask = swap_mask_noFP

        # Second Denoiser pass - After First Restorer
        if control.get("DenoiserAfterFirstRestorerToggle", False):
            swap = self._apply_denoiser_pass(swap, control, "AfterFirst", kv_map)

        # --- RESTORATION 2 ---
        if (
            parameters["FaceRestorerEnable2Toggle"]
            and not parameters["FaceRestorerEnable2EndToggle"]
        ):
            swap_original2 = swap.clone()
            swap2 = self.models_processor.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetType2Selection"],
                parameters["FaceRestorerType2Selection"],
                parameters["FaceRestorerBlend2Slider"],
                parameters["FaceFidelityWeight2DecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=2,
            )

            if parameters["FaceRestorerAutoEnable2Toggle"]:
                original_face_512_autorestore2 = original_face_512.clone()
                swap_original_autorestore2 = swap_original2.clone()
                alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
                adjust_sharpness2 = float(
                    parameters["FaceRestorerAutoSharpAdjust2Slider"]
                )
                scale_factor2 = round(tform.scale, 2)
                automasktoggle2 = parameters["FaceRestorerAutoMask2EnableToggle"]
                automaskadjust2 = parameters[
                    "FaceRestorerAutoSharpMask2AdjustDecimalSlider"
                ]
                automaskblur2 = 2
                restore_mask = mask_forcalc_512.clone()

                alpha_auto2, blur_value2 = self.face_restorer_auto(
                    original_face_512_autorestore2,
                    swap_original_autorestore2,
                    swap2,
                    alpha_restorer2,
                    adjust_sharpness2,
                    scale_factor2,
                    debug,
                    restore_mask,
                    automasktoggle2,
                    automaskadjust2,
                    automaskblur2,
                )

                if blur_value2 > 0:
                    kernel_size = 2 * blur_value2 + 1
                    sigma = blur_value2 * 0.1
                    gaussian_blur = transforms.GaussianBlur(
                        kernel_size=kernel_size, sigma=sigma
                    )
                    swap = gaussian_blur(swap_original2)
                    debug_info["Restore2"] = f": {-blur_value2:.2f}"
                elif isinstance(alpha_auto2, torch.Tensor):
                    swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                elif alpha_auto2 != 0:
                    swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                    if debug:
                        debug_info["Restore2"] = f": {alpha_auto2 * 100:.2f}"
                else:
                    swap = swap_original2
                    if debug:
                        debug_info["Restore2"] = f": {alpha_auto2 * 100:.2f}"
            else:
                alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
                swap = torch.add(
                    torch.mul(swap2, alpha_restorer2),
                    torch.mul(swap_original2, 1 - alpha_restorer2),
                )

        # Expression (After Second)
        if (
            parameters["FaceExpressionEnableBothToggle"]
            and (
                parameters["FaceExpressionLipsToggle"]
                or parameters["FaceExpressionEyesToggle"]
                or parameters["FaceExpressionBrowsToggle"]
                or parameters["FaceExpressionGeneralToggle"]
                or parameters.get("FaceExpressionModeSelection", "Advanced") == "Simple"
            )
            and parameters["FaceExpressionBeforeTypeSelection"]
            == "After Second Restorer"
        ):
            swap = self.frame_edits.apply_face_expression_restorer(
                original_face_512, swap, cast(dict, parameters)
            )

        # Editor (After Second)
        if (
            parameters["FaceEditorEnableToggle"]
            and self.main_window.editFacesButton.isChecked()
            and parameters["FaceEditorBeforeTypeSelection"] == "After Second Restorer"
        ):
            editor_mask = t512_mask(swap_mask).clone()
            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.frame_edits.swap_edit_face_core(swap, swap, parameters, control)
            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = v2.Resize((swap.shape[-2], swap.shape[-1]), antialias=True)(
                    swap_mask_noFP
                )
            else:
                swap_mask = swap_mask_noFP

        # --- AUTO COLOR (Mask 512) ---
        if parameters.get("AutoColorEnableToggle", False):
            if parameters["AutoColorTransferTypeSelection"] == "Test":
                swap = faceutil.histogram_matching(
                    original_face_512, swap, parameters["AutoColorBlendAmountSlider"]
                )
            elif parameters["AutoColorTransferTypeSelection"] == "Test_Mask":
                swap = faceutil.histogram_matching_withmask(
                    original_face_512,
                    swap,
                    mask_autocolor,
                    parameters["AutoColorBlendAmountSlider"],
                )
            elif parameters["AutoColorTransferTypeSelection"] == "DFL_Test":
                swap = faceutil.histogram_matching_DFL_test(
                    original_face_512, swap, parameters["AutoColorBlendAmountSlider"]
                )
            elif parameters["AutoColorTransferTypeSelection"] == "DFL_Orig":
                swap = faceutil.histogram_matching_DFL_Orig(
                    original_face_512,
                    swap,
                    mask_autocolor,
                    parameters["AutoColorBlendAmountSlider"],
                )

        # --- TRANSFER TEXTURE ---
        if parameters.get("TransferTextureEnableToggle", False):
            # 1. Ensure resolutions match target 512x512
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)
                swap_mask = t512_mask(swap_mask)
                swap_mask_noFP = t512_mask(swap_mask_noFP)

            mask_input_vgg = t128_mask(calc_mask.clone())
            mask_vgg_512 = torch.ones(
                (1, 512, 512), dtype=torch.float32, device=self.models_processor.device
            )

            TextureFeatureLayerTypeSelection = "combo_relu3_3_relu3_1"
            upper_thresh = parameters["TextureUpperLimitSlider"] / 100.0

            # 2. VGG Mask Processing
            if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                # Fetch threshold values from UI
                thr = (
                    parameters["VGGMaskThresholdSlider"]
                    if parameters.get("ExcludeVGGMaskEnableToggle", False)
                    else 0
                )
                soft = 100
                mode = "smooth"

                # Retrieve BOTH the thresholded mask and the raw normalized difference (Size: 128x128)
                mask_vgg_raw, diff_norm_texture_raw = (
                    self.models_processor.apply_vgg_mask_simple(
                        swap,
                        original_face_512,
                        mask_input_vgg,
                        center_pct=thr,
                        softness_pct=soft,
                        feature_layer=TextureFeatureLayerTypeSelection,
                        mode=mode,
                    )
                )

                # Upscale to 512x512 IMMEDIATELY to prevent tensor mismatch
                mask_vgg_512 = t512_mask(mask_vgg_raw).clamp(0.0, 1.0)
                diff_norm_texture_512 = t512_mask(diff_norm_texture_raw).clamp(0.0, 1.0)

                # Fallback to the raw difference texture if manipulation is disabled (Restoring old behavior)
                if not parameters.get("ExcludeVGGMaskEnableToggle", False):
                    mask_vgg_512 = diff_norm_texture_512.clone()

                # Optional VGG specific blur
                if parameters.get("TextureBlendAmountSlider", 0) > 0:
                    b = parameters["TextureBlendAmountSlider"]
                    gauss = transforms.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                    mask_vgg_512 = gauss(mask_vgg_512.float())

            # 3. Features Exclusion Logic (Eyes, Mouth, etc.)
            if parameters.get("ExcludeMaskEnableToggle", False):
                # texture_exclude_512: 1 means KEEP texture (skin), 0 means REMOVE texture (eyes/mouth)
                feature_mask = texture_exclude_512.clone().float()

                # This creates a smooth gradient transition instead of a harsh binary cut-off.
                if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                    blur_val = parameters.get("FaceParserBlurTextureSlider", 0)
                    if blur_val > 0:
                        kernel_size = int(blur_val * 2 + 1)
                        sigma = max((blur_val + 1) * 0.2, 1e-6)
                        blur_op = transforms.GaussianBlur(kernel_size, sigma=sigma)
                        feature_mask = blur_op(feature_mask)

                # Combine VGG mask with the spatial FaceParser mask
                if parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                    # Clamp upper limits to protect extreme highlights/differences
                    mask_vgg_512 = torch.where(
                        mask_vgg_512 >= upper_thresh, upper_thresh, mask_vgg_512
                    )

                mask_final_512 = (
                    torch.max(mask_vgg_512 * (1.0 - feature_mask), 1.0 - calc_mask_dill)
                ).clamp(0.0, 1.0)

            elif parameters.get("ExcludeOriginalVGGMaskEnableToggle", False):
                # Clamp upper limits to protect extreme highlights/differences
                mask_vgg_512 = torch.where(
                    mask_vgg_512 >= upper_thresh, upper_thresh, mask_vgg_512
                )
                # Protect background if no spatial exclusion is active
                mask_final_512 = torch.max(mask_vgg_512, 1.0 - calc_mask_dill).clamp(
                    0.0, 1.0
                )

            else:
                # Fallback to raw mask if everything is disabled
                mask_final_512 = (1.0 - mask_forcalc_512).clamp(0.0, 1.0)

            # 4. AutoColor Backup Logic
            if parameters.get("AutoColorEnableToggle", False):
                swap_texture_backup = swap.clone()
            else:
                swap_texture_backup = faceutil.histogram_matching_DFL_Orig(
                    original_face_512, swap.clone(), mask_autocolor, 100
                )

            # 5. Gradient / Texture Generation Settings
            TransferTextureKernelSizeSlider = 12
            TransferTextureSigmaDecimalSlider = 4.00
            TransferTextureWeightSlider = 1
            TransferTexturePhiDecimalSlider = 9.7
            TransferTextureGammaDecimalSlider = 0.5

            if parameters.get("TransferTextureModeEnableToggle", False):
                TransferTextureLambdSlider = 8
                TransferTextureThetaSlider = 8
            else:
                TransferTextureLambdSlider = 2
                TransferTextureThetaSlider = 1

            clip_limit = (
                parameters["TransferTextureClipLimitDecimalSlider"]
                if parameters.get("TransferTextureClaheEnableToggle", False)
                else 0.0
            )
            alpha_clahe = parameters["TransferTextureAlphaClaheDecimalSlider"]
            grid_size = (4, 4)
            global_gamma = parameters["TransferTexturePreGammaDecimalSlider"]
            global_contrast = parameters["TransferTexturePreContrastDecimalSlider"]

            gradient_texture = self.gradient_magnitude(
                original_face_512,
                calc_mask_dill,
                TransferTextureKernelSizeSlider,
                TransferTextureWeightSlider,
                TransferTextureSigmaDecimalSlider,
                TransferTextureLambdSlider,
                TransferTextureGammaDecimalSlider,
                TransferTexturePhiDecimalSlider,
                TransferTextureThetaSlider,
                clip_limit,
                alpha_clahe,
                grid_size,
                global_gamma,
                global_contrast,
            )

            gradient_texture = faceutil.histogram_matching_DFL_Orig(
                original_face_512, gradient_texture, mask_autocolor, 100
            )

            if parameters["FaceParserBlurTextureSlider"] > 0:
                orig = mask_final_512.clone()
                gauss = transforms.GaussianBlur(
                    parameters["FaceParserBlurTextureSlider"] * 2 + 1,
                    (parameters["FaceParserBlurTextureSlider"] + 1) * 0.2,
                )
                mask_final_512 = gauss(mask_final_512.type(torch.float32))
                mask_final_512 = torch.max(mask_final_512, orig).clamp(0.0, 1.0)
            # 6. Final Blending
            # alpha_t modulates the overall strength, w determines the per-pixel application map
            alpha_t = parameters["TransferTextureBlendAmountSlider"] / 100.0
            w = alpha_t * (1.0 - mask_final_512)
            w = w.clamp(0.0, 1.0)

            swap = (swap_texture_backup * (1.0 - w) + gradient_texture * w).clamp(
                0, 255
            )
            texture_mask_view = (1.0 - mask_final_512).clone()

        # --- DIFFERENCING ---
        if parameters.get("DifferencingEnableToggle", False):
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)
                swap_mask = t512_mask(swap_mask)
                swap_mask_noFP = t512_mask(swap_mask_noFP)

            diff_mask_128 = t128_mask(calc_mask.clone())
            swapped_face_resized = swap.clone()
            original_face_resized = original_face_512.clone()
            FeatureLayerTypeSelection = "combo_relu3_3_relu3_1"

            lower_thresh = parameters["DifferencingLowerLimitThreshSlider"] / 100.0
            upper_thresh = parameters["DifferencingUpperLimitThreshSlider"] / 100.0
            middle_value = parameters["DifferencingMiddleLimitValueSlider"] / 100.0
            upper_value = parameters["DifferencingUpperLimitValueSlider"] / 100.0

            mask_diff_128, diff_norm_texture = (
                self.models_processor.apply_perceptual_diff_onnx(
                    swapped_face_resized,
                    original_face_resized,
                    diff_mask_128,
                    lower_thresh,
                    0,
                    upper_thresh,
                    upper_value,
                    middle_value,
                    FeatureLayerTypeSelection,
                    False,
                )
            )

            eps = 1e-6
            inv_lower = 1.0 / max(lower_thresh, eps)
            inv_mid = 1.0 / max((upper_thresh - lower_thresh), eps)
            inv_high = 1.0 / max((1.0 - upper_thresh), eps)

            res_low = diff_norm_texture * inv_lower * middle_value
            res_mid = middle_value + (diff_norm_texture - lower_thresh) * inv_mid * (
                upper_value - middle_value
            )
            res_high = upper_value + (diff_norm_texture - upper_thresh) * inv_high * (
                1.0 - upper_value
            )

            piece = torch.where(
                diff_norm_texture < lower_thresh,
                res_low,
                torch.where(diff_norm_texture > upper_thresh, res_high, res_mid),
            )

            mask512 = t512_mask(piece)
            if parameters.get("DifferencingBlendAmountSlider", 0) > 0:
                b = parameters["DifferencingBlendAmountSlider"]
                gauss = transforms.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                mask512 = gauss(mask512.float())

            mask512 = torch.max((mask512), 1 - calc_mask_dill)
            mask512 = (mask512).clamp(0, 1)

            swap = (swap * mask512 + original_face_512 * (1.0 - mask512)).clamp(0, 255)
            diff_mask = 1 - mask512.clone()

        # Face Editor (After Texture Transfer)
        if (
            parameters["FaceEditorEnableToggle"]
            and self.main_window.editFacesButton.isChecked()
            and parameters["FaceEditorBeforeTypeSelection"] == "After Texture Transfer"
        ):
            editor_mask = t512_mask(swap_mask).clone()
            if swap.shape[-1] != 512:
                swap = t512_mask(swap)

            swap = swap * editor_mask + original_face_512 * (1 - editor_mask)
            swap = self.frame_edits.swap_edit_face_core(swap, swap, parameters, control)

            if swap_mask_noFP.shape[-1] != swap.shape[-1]:
                swap_mask = v2.Resize((swap.shape[-2], swap.shape[-1]), antialias=True)(
                    swap_mask_noFP
                )
            else:
                swap_mask = swap_mask_noFP

        # --- COLOR CORRECTIONS ---
        if parameters["ColorEnableToggle"]:
            swap = torch.unsqueeze(swap, 0).contiguous()
            swap = v2.functional.adjust_gamma(
                swap, parameters["ColorGammaDecimalSlider"], 1.0
            )
            swap = torch.squeeze(swap)
            swap = swap.permute(1, 2, 0).type(torch.float32)

            del_color = torch.tensor(
                [
                    parameters["ColorRedSlider"],
                    parameters["ColorGreenSlider"],
                    parameters["ColorBlueSlider"],
                ],
                device=self.models_processor.device,
            )
            swap += del_color
            swap = torch.clamp(swap, min=0.0, max=255.0)
            swap = swap.permute(2, 0, 1) / 255.0

            swap = v2.functional.adjust_brightness(
                swap, parameters["ColorBrightnessDecimalSlider"]
            )
            swap = v2.functional.adjust_contrast(
                swap, parameters["ColorContrastDecimalSlider"]
            )
            swap = v2.functional.adjust_saturation(
                swap, parameters["ColorSaturationDecimalSlider"]
            )
            swap = v2.functional.adjust_sharpness(
                swap, parameters["ColorSharpnessDecimalSlider"]
            )
            swap = v2.functional.adjust_hue(swap, parameters["ColorHueDecimalSlider"])

            swap = swap * 255.0

        # --- RESTORATION 2 (END) ---
        if (
            parameters["FaceRestorerEnable2Toggle"]
            and parameters["FaceRestorerEnable2EndToggle"]
        ):
            swap_original2 = swap.clone()
            swap2 = self.models_processor.apply_facerestorer(
                swap,
                parameters["FaceRestorerDetType2Selection"],
                parameters["FaceRestorerType2Selection"],
                parameters["FaceRestorerBlend2Slider"],
                parameters["FaceFidelityWeight2DecimalSlider"],
                control["DetectorScoreSlider"],
                kps_ref,
                slot_id=2,
            )

            if parameters["FaceRestorerAutoEnable2Toggle"]:
                original_face_512_autorestore2 = original_face_512.clone()
                swap_original_autorestore2 = swap_original2.clone()
                alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
                adjust_sharpness2 = float(
                    parameters["FaceRestorerAutoSharpAdjust2Slider"]
                )
                scale_factor2 = round(tform.scale, 2)
                automasktoggle2 = parameters["FaceRestorerAutoMask2EnableToggle"]
                automaskadjust2 = parameters[
                    "FaceRestorerAutoSharpMask2AdjustDecimalSlider"
                ]
                automaskblur2 = 2
                restore_mask = mask_forcalc_512.clone()

                alpha_auto2, blur_value2 = self.face_restorer_auto(
                    original_face_512_autorestore2,
                    swap_original_autorestore2,
                    swap2,
                    alpha_restorer2,
                    adjust_sharpness2,
                    scale_factor2,
                    debug,
                    restore_mask,
                    automasktoggle2,
                    automaskadjust2,
                    automaskblur2,
                )

                if blur_value2 > 0:
                    kernel_size = 2 * blur_value2 + 1
                    sigma = blur_value2 * 0.1
                    gaussian_blur = transforms.GaussianBlur(
                        kernel_size=kernel_size, sigma=sigma
                    )
                    swap = gaussian_blur(swap_original2)
                    debug_info["Restore2"] = f": {-blur_value2:.2f}"
                elif isinstance(alpha_auto2, torch.Tensor):
                    swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                elif alpha_auto2 != 0:
                    swap = swap2 * alpha_auto2 + swap_original2 * (1 - alpha_auto2)
                    if debug:
                        debug_info["Restore2"] = f": {alpha_auto2 * 100:.2f}"
                else:
                    swap = swap_original2
                    if debug:
                        debug_info["Restore2"] = f": {alpha_auto2 * 100:.2f}"
            else:
                alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
                swap = torch.add(
                    torch.mul(swap2, alpha_restorer2),
                    torch.mul(swap_original2, 1 - alpha_restorer2),
                )

        # Third denoiser pass - After restorers
        if control.get("DenoiserAfterRestorersToggle", False):
            swap = self._apply_denoiser_pass(swap, control, "After", kv_map)

        # --- MOUTH ENHANCEMENT & ALIGNMENT (POST-RESTORER) ---
        if parameters.get("MouthParserStretchAfterToggle", False):
            mouth_overlay_pkg = None
            if hasattr(self.models_processor, "face_masks"):
                # 'swap' now contains the fully restored face
                mouth_overlay_pkg = self.models_processor.face_masks.get_mouth_overlay(
                    swap, original_face_512, parameters
                )

            if mouth_overlay_pkg is not None:
                overlay_rgb, overlay_mask = mouth_overlay_pkg
                if overlay_rgb is not None and overlay_mask is not None:
                    if overlay_rgb.shape[-1] != swap.shape[-1]:
                        overlay_rgb = v2.Resize(
                            (swap.shape[-2], swap.shape[-1]), antialias=True
                        )(overlay_rgb)
                        overlay_mask = v2.Resize(
                            (swap.shape[-2], swap.shape[-1]), antialias=True
                        )(overlay_mask.unsqueeze(0)).squeeze(0)

                    swap = swap * (1.0 - overlay_mask) + overlay_rgb * overlay_mask

        # --- FACE PARSER (END) ---
        if parameters.get("FaceParserEnableToggle") and parameters.get(
            "FaceParserEndToggle"
        ):
            out = self.models_processor.process_masks_and_masks(
                swap,
                original_face_512,
                parameters,
                control,
            )

            FaceParser_mask = out.get("FaceParser_mask", None)

            if FaceParser_mask is not None:
                if FaceParser_mask.shape[-1] != swap_mask.shape[-1]:
                    FaceParser_mask = v2.Resize(
                        (swap.shape[-2], swap.shape[-1]), antialias=True
                    )(FaceParser_mask)

                swap_mask = swap_mask * FaceParser_mask

        # Recalculate AutoColor Mask
        calc_mask = t512_mask(swap_mask.clone()).clamp(0, 1)
        mask_autocolor = calc_mask.clone()
        mask_autocolor = mask_autocolor > 0.05

        # AutoColor End (EndingColorTransfer)
        if parameters.get("EndingColorTransferEnableToggle", False):
            if parameters["EndingColorTransferTypeSelection"] == "Test":
                swap = faceutil.histogram_matching(
                    original_face_512, swap, parameters["EndingColorBlendAmountSlider"]
                )
            elif parameters["EndingColorTransferTypeSelection"] == "Test_Mask":
                swap = faceutil.histogram_matching_withmask(
                    original_face_512,
                    swap,
                    mask_autocolor,
                    parameters["EndingColorBlendAmountSlider"],
                )
            elif parameters["EndingColorTransferTypeSelection"] == "DFL_Test":
                swap = faceutil.histogram_matching_DFL_test(
                    original_face_512, swap, parameters["EndingColorBlendAmountSlider"]
                )
            elif parameters["EndingColorTransferTypeSelection"] == "DFL_Orig":
                swap = faceutil.histogram_matching_DFL_Orig(
                    original_face_512,
                    swap,
                    mask_autocolor,
                    parameters["EndingColorBlendAmountSlider"],
                )

        # Final blending
        if (
            parameters["FinalBlendAdjEnableToggle"]
            and parameters["FinalBlendAmountSlider"] > 0
        ):
            final_blur_strength = parameters["FinalBlendAmountSlider"]
            kernel_size = 2 * final_blur_strength + 1
            sigma = final_blur_strength * 0.1
            gaussian_blur = transforms.GaussianBlur(
                kernel_size=kernel_size, sigma=sigma
            )
            swap = gaussian_blur(swap)

        # Artefacts: Jpeg
        if parameters["JPEGCompressionEnableToggle"]:
            jpeg_q = int(parameters["JPEGCompressionAmountSlider"])
            if jpeg_q != 100:
                s = float(tform.scale)
                gamma = 0.60
                strength = 0.80
                q_min = 14
                q_max = 100

                jpeg_q_eff = faceutil._map_jpeg_quality(
                    base_q=jpeg_q,
                    face_scale=s,
                    gamma=gamma,
                    strength=strength,
                    q_min=q_min,
                    q_max=q_max,
                )
                if debug:
                    debug_info["JPEG Quality"] = f"{jpeg_q_eff}"

                swap2 = faceutil.jpegBlur(swap, jpeg_q_eff)
                blend = parameters["JPEGCompressionBlendSlider"] / 100.0
                swap = torch.add(swap2 * blend, swap * (1.0 - blend))

        # Artefacts: BlockShift
        if parameters["BlockShiftEnableToggle"]:
            base_quality = parameters["BlockShiftAmountSlider"]
            max_px = parameters["BlockShiftMaxAmountSlider"]

            swap2 = self.apply_block_shift_gpu_jitter(
                swap,
                block_size=base_quality,
                max_amount_pixels=float(max_px),
                seed=1337,
            )

            block_shift_blend = parameters["BlockShiftBlendAmountSlider"] / 100.0
            swap = swap2 * block_shift_blend + swap * (1.0 - block_shift_blend)
            swap = torch.add(
                torch.mul(swap2, block_shift_blend),
                torch.mul(swap, 1 - block_shift_blend),
            )

        if parameters["ColorNoiseDecimalSlider"] > 0:
            swap = swap.to(torch.float32)
            noise = (
                (torch.rand_like(swap, dtype=torch.float32) - 0.5)
                * 2
                * parameters["ColorNoiseDecimalSlider"]
            )
            swap = torch.clamp(swap + noise, 0.0, 255.0)

        if control.get("AnalyseImageEnableToggle", False):
            image_analyse_swap = self.analyze_image(swap)
            if debug:
                debug_info["JS: "] = image_analyse_swap

        if debug and debug_info:
            one_liner = ", ".join(f"{key}={value}" for key, value in debug_info.items())
            print(f"[DEBUG] {one_liner}")

        if is_perspective_crop:
            return t512_mask(swap), t512_mask(swap_mask), None

        # Mask Post-Processing (Final Blend)
        gauss = transforms.GaussianBlur(
            parameters["OverallMaskBlendAmountSlider"] * 2 + 1,
            (parameters["OverallMaskBlendAmountSlider"] + 1) * 0.2,
        )
        swap_mask = gauss(swap_mask)

        if border_mask.shape[-1] != swap_mask.shape[-1]:
            border_mask = v2.Resize(
                (swap_mask.shape[-2], swap_mask.shape[-1]), antialias=True
            )(border_mask)

        swap_mask = torch.mul(swap_mask, border_mask)

        if swap.shape[-1] != 512:
            swap = t512_mask(swap)
            swap_mask = t512_mask(swap_mask)

        swap = torch.mul(swap, swap_mask)

        # --- VIEW MODES ---
        original_face_512_clone = None
        if self.is_view_face_compare:
            original_face_512_clone = original_face_512.clone()
            original_face_512_clone = original_face_512_clone.type(torch.uint8)
            original_face_512_clone = original_face_512_clone.permute(1, 2, 0)

        swap_mask_clone = None
        if self.is_view_face_mask:
            mask_show_type = parameters["MaskShowSelection"]
            if mask_show_type == "swap_mask":
                if (
                    parameters["FaceEditorEnableToggle"]
                    and self.main_window.editFacesButton.isChecked()
                ):
                    swap_mask_clone = torch.ones_like(swap_mask).clone()
                else:
                    swap_mask_clone = swap_mask.clone()
            elif mask_show_type == "diff":
                swap_mask_clone = diff_mask.clone()
            elif mask_show_type == "texture":
                swap_mask_clone = texture_mask_view.clone()

            if swap_mask_clone is not None:
                if swap_mask_clone.shape[-1] != 512:
                    swap_mask_clone = t512_mask(swap_mask_clone)
                swap_mask_clone = torch.sub(1, swap_mask_clone)
                swap_mask_clone = torch.cat(
                    (swap_mask_clone, swap_mask_clone, swap_mask_clone), 0
                )
                swap_mask_clone = swap_mask_clone.permute(1, 2, 0)
                swap_mask_clone = torch.mul(swap_mask_clone, 255.0).type(torch.uint8)

        # --- UNTRANSFORM (PASTE BACK) ---
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0, 0], [0, 511], [511, 0], [511, 511]])

        x = IM512[0][0] * corners[:, 0] + IM512[0][1] * corners[:, 1] + IM512[0][2]
        y = IM512[1][0] * corners[:, 0] + IM512[1][1] * corners[:, 1] + IM512[1][2]

        left = floor(np.min(x))
        if left < 0:
            left = 0
        top = floor(np.min(y))
        if top < 0:
            top = 0
        right = ceil(np.max(x))
        if right > img.shape[2]:
            right = img.shape[2]
        bottom = ceil(np.max(y))
        if bottom > img.shape[1]:
            bottom = img.shape[1]

        swap = v2.functional.pad(swap, (0, 0, img.shape[2] - 512, img.shape[1] - 512))
        swap = v2.functional.affine(
            swap,
            tform.inverse.rotation * 57.2958,
            (tform.inverse.translation[0], tform.inverse.translation[1]),
            tform.inverse.scale,
            0,
            interpolation=self.interpolation_Untransform,
            center=(0, 0),
        )
        swap = swap[0:3, top:bottom, left:right]

        swap_mask = v2.functional.pad(
            swap_mask, (0, 0, img.shape[2] - 512, img.shape[1] - 512)
        )
        swap_mask = v2.functional.affine(
            swap_mask,
            tform.inverse.rotation * 57.2958,
            (tform.inverse.translation[0], tform.inverse.translation[1]),
            tform.inverse.scale,
            0,
            interpolation=v2.InterpolationMode.BILINEAR,
            center=(0, 0),
        )
        swap_mask = swap_mask[0:1, top:bottom, left:right]
        swap_mask_minus = swap_mask.clone()
        swap_mask_minus = torch.sub(1, swap_mask)

        img_crop = img[0:3, top:bottom, left:right]
        img_crop = torch.mul(swap_mask_minus, img_crop)

        swap = torch.add(swap, img_crop)
        swap = swap.type(torch.uint8)
        swap = swap.clamp(0, 255)

        img[0:3, top:bottom, left:right] = swap

        return img, original_face_512_clone, swap_mask_clone

    @torch.no_grad()
    def gradient_magnitude(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        kernel_size: int,
        weighting_strength: float,
        sigma: float,
        lambd: float,
        gamma: float,
        psi: float,
        theta_count: int,
        clip_limit: float,
        alpha_clahe: float,
        grid_size: tuple[int, int],
        global_gamma: float,
        global_contrast: float,
    ) -> torch.Tensor:
        """
        Calculates the weighted Gabor magnitude for texture transfer.

        Args:
            image: Tensor [C, H, W] in [0..255]
            mask:  Tensor [C, H, W] (0/1)
        Returns:
            Tensor [C, H, W] – weighted Gabor magnitude
        """

        C, H, W = image.shape
        image = image.float() / 255.0
        mask = mask.bool()

        # 1) Global Gamma & Contrast
        if global_gamma != 1.0:
            image = image.pow(global_gamma)
        if global_contrast != 1.0:
            m_gc = image.mean((1, 2), keepdim=True)
            image = (image - m_gc) * global_contrast + m_gc

        # 2) CLAHE in L-channel (with alpha_clahe blending)
        if clip_limit > 0.0:
            image = image.unsqueeze(0).clamp(0, 1)  # [1,3,H,W]
            mask_b3 = mask.unsqueeze(0)  # [1,3,H,W]

            lab = kc.rgb_to_lab(image)  # [1,3,H,W]
            L = lab[:, 0:1, :, :] / 100.0  # [1,1,H,W]

            mb = mask_b3[:, 0:1, :, :]  # [1,1,H,W]
            area_l = mb.sum((2, 3), keepdim=True).clamp(min=1)
            mean_l = (L * mb).sum((2, 3), keepdim=True) / area_l
            Lf = torch.where(mb, L, mean_l)
            Leq = ke.equalize_clahe(
                Lf,
                clip_limit=clip_limit,
                grid_size=grid_size,
                slow_and_differentiable=False,
            ).clamp(0, 1)
            L_blend = alpha_clahe * Leq + (1 - alpha_clahe) * L
            Lnew = torch.where(mb, L_blend, L)

            lab_eq = torch.cat([Lnew * 100.0, lab[:, 1:, :, :]], dim=1)  # [1,3,H,W]
            x_eq = kc.lab_to_rgb(lab_eq)
            image = x_eq.squeeze(0)

        # 3) Gabor Filter setup
        kernel_size = max(1, 2 * kernel_size - 1)
        if theta_count == 10:
            theta_values = torch.tensor([math.pi / 4], device=image.device)
        else:
            theta_values = torch.linspace(
                0, math.pi, theta_count + 1, device=image.device
            )[:-1]

        # 4) Single Gabor Filter call
        magnitude = self.apply_gabor_filter_torch(
            image, kernel_size, sigma, lambd, gamma, psi, theta_values
        )  # [C, H, W]

        # 5) Invert
        max_mv = magnitude.amax((1, 2), keepdim=True)
        inverted = max_mv - magnitude  # [C, H, W]

        # 6) Weighting
        if weighting_strength > 0:
            img_m = image * mask
            weighted = inverted * (
                (1 - weighting_strength) + weighting_strength * img_m
            )
        else:
            weighted = inverted

        return weighted * 255  # [C, H, W]

    def apply_gabor_filter_torch(
        self, image, kernel_size, sigma, lambd, gamma, psi, theta_values
    ):
        """
        Applies Gabor filter bank to image.

        Args:
            image: Tensor [C, H, W]
            theta_values: Tensor [N]
        Returns:
            Tensor [C, H, W]
        """
        C, H, W = image.shape
        image = image.unsqueeze(0)  # → [1, C, H, W]

        N = theta_values.shape[0]

        kernels = self.get_gabor_kernels(
            kernel_size, sigma, lambd, gamma, psi, theta_values, image.device
        )  # [N, 1, k, k]

        # expand to all channels:
        weight = kernels.repeat_interleave(C, dim=0)  # → [N*C, 1, k, k]
        out = F.conv2d(
            image,  # [1, C, H, W]
            weight,
            padding=kernel_size // 2,
            groups=C,  # each channel group gets N filters
        )  # out: [1, N*C, H, W]

        # reshape to [N, C, H, W]:
        out = out.squeeze(0).view(N, C, H, W)
        magnitudes = out.amax(dim=0)
        return magnitudes

    def get_gabor_kernels(
        self, kernel_size, sigma, lambd, gamma, psi, theta_values, device
    ):
        """
        Returns: Tensor [N, 1, k, k]
        """
        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-half, half, kernel_size, device=device),
            torch.linspace(-half, half, kernel_size, device=device),
            indexing="ij",
        )

        kernels = []
        for theta in theta_values:
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

            gb = torch.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / sigma**2)
            gb *= torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernels.append(gb)

        return torch.stack(kernels).unsqueeze(1)  # → [N, 1, k, k]

    def face_restorer_auto(
        self,
        original_face_512,  # [3,H,W], float in [0..255]
        swap_original,  # [3,H,W]
        swap,  # [3,H,W]
        alpha,  # initial scalar alpha (ignored; we binary search below)
        adjust_sharpness,
        scale_factor,
        debug,
        swap_mask,
        alpha_map_enable: bool = False,
        alpha_map_strength: float = 0.5,
        alpha_map_blur: int = 7,
    ):
        """Auto-Restorer: Blends between restored and original image based on sharpness."""
        # Baseline sharpness of original
        scores_original = self.sharpness_score(original_face_512)
        score_new_original = (
            scores_original["combined"].item() * 100 + adjust_sharpness / 10.0
        )

        # Binary search for scalar alpha
        alpha = 1.0
        max_iterations = 7
        alpha_min, alpha_max = 0.0, 1.0
        tolerance = 0.5
        min_alpha_change = 0.05
        iteration = 0
        prev_alpha = alpha
        iteration_blur = 0

        while iteration < max_iterations:
            swap2 = swap * alpha + swap_original * (1 - alpha)
            swap2_masked = swap2.clone()

            scores_swap = self.sharpness_score(swap2_masked)
            score_new_swap = scores_swap["combined"].item() * 100
            sharpness_diff = score_new_swap - score_new_original

            if abs(sharpness_diff) < tolerance:
                break

            if sharpness_diff < 0:
                if alpha > 0.99:
                    prev_alpha = alpha
                    break
                alpha_min = alpha
                alpha = (alpha + alpha_max) / 2.0
            else:
                alpha_max = alpha
                alpha = (alpha + alpha_min) / 2.0

            # Very small alpha -> blur fallback on base
            if sharpness_diff >= 0 and alpha < 0.07:
                prev_alpha = 0.0
                base = swap_original
                max_blur_strength = 10
                for bs in range(0, max_blur_strength + 1):
                    if bs == 0:
                        kernel_size = 1
                        sigma = 1e-6
                    else:
                        kernel_size = 2 * bs + 1
                        sigma = max(bs, 1e-6)
                    gaussian_blur = transforms.GaussianBlur(kernel_size, sigma)
                    swap2_blurred = gaussian_blur(base)
                    scores_swap_b = self.sharpness_score(swap2_blurred)
                    score_new_swap_b = scores_swap_b["combined"].item() * 100.0
                    sharpness_diff_b = score_new_swap_b - score_new_original

                    if sharpness_diff_b < 0:
                        iteration_blur = 0 if bs == 0 else (bs - 1)
                        break
                    if abs(sharpness_diff_b) <= tolerance:
                        iteration_blur = bs
                        break
                    iteration_blur = bs
                break

            if abs(prev_alpha - alpha) < min_alpha_change:
                prev_alpha = (prev_alpha + alpha) / 2.0
                if abs(prev_alpha) <= 0.05:
                    prev_alpha = 0.0
                break

            prev_alpha = alpha
            iteration += 1

        # Per-pixel alpha map, derived from sharpness distribution
        if alpha_map_enable and (prev_alpha > 0.0):
            # Build the *final* composite (for a stable map), then sharpness map of it
            swap_final = swap * prev_alpha + swap_original * (1 - prev_alpha)

            s_map = self.sharpness_map(
                swap_final,
                mask=swap_mask,
                tenengrad_thresh=0.05,
                comb_weight=0.5,
                smooth_kernel=alpha_map_blur
                if (alpha_map_blur and alpha_map_blur % 2 == 1)
                else 0,
            )

            # Mean sharpness inside mask (or global)
            if swap_mask is not None:
                m = (
                    (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0))
                    .float()
                    .to(s_map.device)
                )
                denom = m.sum().clamp_min(1.0)
                mu = (s_map * m).sum() / denom
            else:
                mu = s_map.mean()

            # Deviation map around mean, scale around prev_alpha
            dev = (s_map - mu).clamp(-1.0, 1.0)
            alpha_map = prev_alpha * (1.0 + alpha_map_strength * dev)
            alpha_map = alpha_map.clamp(0.0, 1.0)

            # Keep outside-face area at scalar prev_alpha (if a mask is provided)
            if swap_mask is not None:
                m = (
                    (swap_mask if swap_mask.dim() == 2 else swap_mask.squeeze(0))
                    .float()
                    .to(alpha_map.device)
                )
                alpha_map = alpha_map * m + prev_alpha * (1.0 - m)

            return alpha_map.unsqueeze(0), iteration_blur

        # Fallback: scalar like before
        return prev_alpha, iteration_blur

    def sharpness_score(
        self,
        image: torch.Tensor,
        mask: torch.Tensor = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5,
    ) -> dict:
        """
        Calculates three sharpness metrics on an RGB image:
          1) var_lap: Variance of Laplacian
          2) tten: Thresholded Tenengrad (Proportion of strong edges)
          3) combined: comb_weight*var_lap + (1-comb_weight)*tten

        Args:
            image: Tensor [3, H, W], float in [0..1]
            mask:  optional Tensor [H, W] or [1, H, W] with 1=valid, 0=ignore
            tenengrad_thresh: Threshold for Tenengrad (0..1)
            comb_weight: Weight for var_lap in combination (0..1)

        Returns:
            {
              "var_lap": float Tensor,
              "ttengrad": float Tensor,
              "combined": float Tensor
            }
        """
        image = image / 255.0

        # 1) Grayscale [1,1,H,W]
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)

        # 2) Optional Mask on [H,W]
        if mask is not None:
            m = mask.float()
            if m.dim() == 3:  # [1,H,W]
                m = m.squeeze(0)
        else:
            m = None

        def valid_count(t):
            return m.sum().clamp(min=1.0) if m is not None else t.numel()

        # --- Variance of Laplacian ---
        lap = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=image.device, dtype=torch.float32
        ).view(1, 1, 3, 3)
        L = F.conv2d(gray, lap, padding=1).squeeze()  # [H,W]
        L2 = L.pow(2)
        if m is not None:
            L = L * m
            L2 = L2 * m
        cnt = valid_count(L2)
        mean_L2 = L2.sum() / cnt
        mean_L = L.sum() / cnt
        var_lap = (mean_L2 - mean_L.pow(2)).clamp(min=0.0)

        # --- Thresholded Tenengrad ---
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            device=image.device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        Gx = F.conv2d(gray, sobel_x, padding=1).squeeze()  # [H,W]
        Gy = F.conv2d(gray, sobel_y, padding=1).squeeze()
        G = (Gx.pow(2) + Gy.pow(2)).sqrt()
        if m is not None:
            G = G * m
        total = cnt
        strong = (G > tenengrad_thresh).float().sum()
        ttengrad = strong / total

        # --- Combined Score ---
        combined = comb_weight * var_lap + (1 - comb_weight) * ttengrad

        return {"var_lap": var_lap, "ttengrad": ttengrad, "combined": combined}

    def sharpness_map(
        self,
        image: torch.Tensor,  # [3,H,W], float in [0..255]
        mask: torch.Tensor | None = None,
        tenengrad_thresh: float = 0.05,
        comb_weight: float = 0.5,
        smooth_kernel: int = 5,  # odd; 0/1 = no blur
    ) -> torch.Tensor:
        """
        Returns a normalized per-pixel sharpness map in [0..1] with shape [H,W].
        Combines Laplacian energy + gradient magnitude (Tenengrad-like).
        """
        eps = 1e-8
        device = image.device

        # [3,H,W] -> [1,1,H,W] gray, range [0..1]
        gray = (image / 255.0).mean(dim=0, keepdim=True).unsqueeze(0)

        # Convs
        lap_k = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        lap = F.conv2d(gray, lap_k, padding=1).squeeze(0).squeeze(0)  # [H,W]
        gx = F.conv2d(gray, sobel_x, padding=1).squeeze(0).squeeze(0)  # [H,W]
        gy = F.conv2d(gray, sobel_y, padding=1).squeeze(0).squeeze(0)
        grad = (gx.pow(2) + gy.pow(2)).sqrt()  # [H,W]

        # Robust normalization via percentiles inside mask (if given)
        def robust_norm(x, msk):
            if msk is not None:
                sel = x[msk > 0]
                if sel.numel() < 16:  # fallback if mask tiny
                    sel = x.reshape(-1)
            else:
                sel = x.reshape(-1)
            p5 = (
                torch.quantile(sel, 0.05)
                if sel.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            p95 = (
                torch.quantile(sel, 0.95)
                if sel.numel() > 0
                else torch.tensor(1.0, device=device)
            )
            y = (x - p5) / (p95 - p5 + eps)
            return y.clamp_(0, 1)

        m = None
        if mask is not None:
            m = (mask if mask.dim() == 2 else mask.squeeze(0)).float().to(device)

        lap_n = robust_norm(lap.abs(), m)
        grad_n = robust_norm(grad, m)

        smap = comb_weight * lap_n + (1.0 - comb_weight) * grad_n  # [H,W]

        # Optional smoothing to avoid noisy alpha
        if smooth_kernel and smooth_kernel >= 3 and smooth_kernel % 2 == 1:
            k = smooth_kernel
            smap3 = smap.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            gb = transforms.GaussianBlur(kernel_size=k, sigma=max(1, k // 2))
            smap = gb(smap3).squeeze(0).squeeze(0)

        return smap.clamp(0, 1)

    @torch.no_grad()
    def apply_block_shift_gpu_jitter(
        self,
        img: torch.Tensor,
        block_size: int,
        max_amount_pixels: float,
        *,
        seed: int = 1337,
        pad_mode: str = "replicate",
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        MPEG-like Block Jitter: shifts every BxB block field by a
        deterministic (bx, by)-dependent offset in pixels.

        Args:
            img: Tensor [C, H, W] (BGR/RGB agnostic). CPU or CUDA.
            block_size: Block size B (e.g. 8).
            max_amount_pixels: max |Offset| in pixels (applied to both axes).
            seed: global seed for deterministic offsets (frame-stable).
            pad_mode: Padding mode for border (replicate|reflect|zeros).
            align_corners: as in grid_sample.

        Returns:
            Tensor [C, H, W] – same Device/Dtype as input.
        """
        seed = seed + self.frame_number * 17
        assert img.ndim == 3, "expected [C,H,W]"
        C, H, W = img.shape
        device = img.device
        dtype = img.dtype

        # calculate on float32 for grid_sample if necessary
        work = (
            img
            if img.dtype in (torch.float32, torch.float16, torch.bfloat16)
            else img.float()
        )

        # Pad to multiples of B (bottom/right), crop back later
        B = int(2**block_size)
        H_pad = (B - (H % B)) % B
        W_pad = (B - (W % B)) % B
        if H_pad or W_pad:
            pad = (0, W_pad, 0, H_pad)  # (left, right, top, bottom)
            mode = {
                "replicate": "replicate",
                "reflect": "reflect",
                "zeros": "constant",
            }[pad_mode]
            work = F.pad(work[None], pad=pad, mode=mode).squeeze(0)
        Hp, Wp = work.shape[-2:]

        # Number of blocks
        nby = Hp // B
        nbx = Wp // B

        # --- deterministic offsets per block in range [-max, +max] ---
        # Build block coordinate fields
        by_grid, bx_grid = torch.meshgrid(
            torch.arange(nby, device=device, dtype=torch.float32),
            torch.arange(nbx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # simple Hash -> [0,1)
        h = torch.sin((bx_grid * 12.9898 + by_grid * 78.233 + float(seed)) * 43758.5453)
        frac = torch.frac(h * 0.5 + 0.5)

        # derive two independent offsets from hash
        max_amount_pixels = max_amount_pixels / 4
        dx_base = ((frac) * 2.0 - 1.0) * float(max_amount_pixels)

        # second "source": just another linear combo
        h2 = torch.sin(
            (bx_grid * 96.233 + by_grid * 15.987 + (float(seed) + 101)) * 12345.6789
        )
        frac2 = torch.frac(h2 * 0.5 + 0.5)
        dy_base = ((frac2) * 2.0 - 1.0) * float(max_amount_pixels)

        # upsample to pixel grid by tiling each block offset BxB
        dx = torch.repeat_interleave(
            torch.repeat_interleave(dx_base, B, dim=0), B, dim=1
        )  # [Hp,Wp]
        dy = torch.repeat_interleave(
            torch.repeat_interleave(dy_base, B, dim=0), B, dim=1
        )  # [Hp,Wp]

        # --- Build Flow-Field for grid_sample ---
        xs = torch.linspace(-1.0, 1.0, Wp, device=device)
        ys = torch.linspace(-1.0, 1.0, Hp, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [Hp,Wp]
        dx_norm = (2.0 * dx) / max(Wp - 1, 1)
        dy_norm = (2.0 * dy) / max(Hp - 1, 1)

        flow_x = grid_x + dx_norm
        flow_y = grid_y + dy_norm
        flow = torch.stack([flow_x, flow_y], dim=-1)  # [Hp,Wp,2]

        warped = F.grid_sample(
            work[None],
            flow[None],
            mode="bilinear",
            padding_mode="border",
            align_corners=align_corners,
        ).squeeze(0)

        # crop back to original size if padded
        if H_pad or W_pad:
            warped = warped[..., :H, :W]

        if warped.dtype != dtype:
            warped = warped.to(dtype)

        return warped

    def analyze_image(self, image):
        image = image.float() / 255.0
        C, H, W = image.shape
        grayscale = torch.mean(image, dim=0, keepdim=True)
        analysis = {}
        fft = torch.fft.fft2(grayscale)
        high_freq_energy = torch.mean(torch.abs(fft))
        analysis["jpeg_artifacts"] = min(high_freq_energy.item() / 50, 1.0)
        median_filtered = F.avg_pool2d(grayscale, 3, stride=1, padding=1)
        noise_map = torch.abs(grayscale - median_filtered)
        sp_noise = torch.mean((noise_map > 0.1).float())
        analysis["salt_pepper_noise"] = min(sp_noise.item() * 10, 1.0)
        local_var = F.avg_pool2d(grayscale**2, 5, stride=1, padding=2) - (
            F.avg_pool2d(grayscale, 5, stride=1, padding=2) ** 2
        )
        speckle_noise = torch.mean(local_var)
        analysis["speckle_noise"] = min(speckle_noise.item() * 50, 1.0)
        laplace_kernel = (
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                dtype=torch.float32,
                device=image.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        laplace_edges = F.conv2d(grayscale.unsqueeze(0), laplace_kernel, padding=1)
        edge_strength = torch.mean(torch.abs(laplace_edges))
        analysis["blur"] = 1.0 - min(edge_strength.item() * 5, 1.0)
        contrast = grayscale.std()
        analysis["low_contrast"] = 1.0 - min(contrast.item() * 10, 1.0)
        return analysis
