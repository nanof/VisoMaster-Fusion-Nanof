from typing import TYPE_CHECKING, Dict

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F

from app.processors.external.clipseg import CLIPDensePredT
from app.processors.models_data import models_dir

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_VGG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class FaceMasks:
    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self._morph_kernels: Dict[tuple, torch.Tensor] = {}
        self._kernel_cache: Dict[str, torch.Tensor] = {}
        self._meshgrid_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._blur_cache: Dict[tuple, transforms.GaussianBlur] = {}
        self.clip_model_loaded = False
        self.active_models: set[str] = set()

    def unload_models(self):
        """Unloads all models managed by this class."""
        with self.models_processor.model_lock:
            for model_name in list(self.active_models):
                self.models_processor.unload_model(model_name)
            self.active_models.clear()

    def _faceparser_labels(self, img_uint8_3x512x512: torch.Tensor) -> torch.Tensor:
        """
        Runs FaceParser on 512x512 input and returns NATIVE 512x512 labels.
        """
        model_name = "FaceParser"
        ort_session = self.models_processor.models.get(model_name)
        if not ort_session:
            ort_session = self.models_processor.load_model(model_name)

        if not ort_session:
            return torch.zeros(
                (512, 512), dtype=torch.long, device=img_uint8_3x512x512.device
            )

        # Preprocessing
        x = img_uint8_3x512x512.float().div(255.0)
        x = v2.functional.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        x = x.unsqueeze(0).contiguous()

        out = torch.empty((1, 19, 512, 512), device=self.models_processor.device)
        io = ort_session.io_binding()
        io.bind_input(
            "input",
            self.models_processor.device,
            0,
            np.float32,
            (1, 3, 512, 512),
            x.data_ptr(),
        )
        io.bind_output(
            "output",
            self.models_processor.device,
            0,
            np.float32,
            (1, 19, 512, 512),
            out.data_ptr(),
        )

        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            ort_session.run_with_iobinding(io)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

        labels_512 = out.argmax(dim=1).squeeze(0).to(torch.long)
        return labels_512

    def _create_aligned_mouth_overlay(
        self, img_orig, labels_orig, labels_swap, parameters
    ):
        """
        STABILIZED ALIGNMENT WITH BLENDING:
        1. Rigid alignment (Scale/Translate) based on width.
        2. Soft edges on the mask to prevent contour artifacts.
        """
        # 1. Define Regions
        mouth_classes = (labels_orig == 11) | (labels_orig == 12) | (labels_orig == 13)
        if mouth_classes.sum() == 0:
            return None, None

        mouth_classes_swap = (
            (labels_swap == 11) | (labels_swap == 12) | (labels_swap == 13)
        )
        if mouth_classes_swap.sum() == 0:
            return None, None

        hole_mask = labels_swap == 11
        if hole_mask.sum() == 0:
            return None, None

        # 2. Get Coordinates
        y_o, x_o = torch.where(mouth_classes)
        y_s, x_s = torch.where(mouth_classes_swap)

        # 3. Calculate Centroids
        cy_o, cx_o = y_o.float().mean(), x_o.float().mean()
        cy_s, cx_s = y_s.float().mean(), x_s.float().mean()

        # 4. Calculate Widths
        w_o = (x_o.max() - x_o.min()).float()
        w_s = (x_s.max() - x_s.min()).float()

        # 5. Calculate Transformations
        # Scale based on Width Ratio + 15% Safety Margin
        mouthzoom = parameters.get("MouthParserStretchDecimalSlider", 1.05)
        scale_factor = (w_s / (w_o + 1e-6)) * mouthzoom

        translate_x = cx_s - cx_o
        translate_y = cy_s - cy_o

        # 6. Apply Affine Transform
        overlay = v2.functional.affine(
            img_orig,
            angle=0,
            translate=[translate_x.item(), translate_y.item()],
            scale=scale_factor.item(),
            shear=[0.0, 0.0],
            interpolation=v2.InterpolationMode.BILINEAR,
            center=[cx_o.item(), cy_o.item()],
        )

        # 7. Create Mask with Soft Edges
        # Convert boolean mask to float
        overlay_mask = hole_mask.float()

        # Apply slight blur to edges to fix "cutting contours" artifacts
        # We unsqueeze to [1, H, W] for the transform, then squeeze back
        overlay_mask = v2.functional.gaussian_blur(
            overlay_mask.unsqueeze(0), kernel_size=5, sigma=1.5
        ).squeeze(0)

        return overlay, overlay_mask

    def get_mouth_overlay(self, swap_img, original_img, parameters):
        """
        Public helper to retrieve just the mouth overlay.
        Used by FrameWorker to inject the mouth BEFORE the Face Restorer runs.
        """
        # Check requirements
        if not parameters.get("MouthParserStretchToggle", False):
            return None

        # Run inference
        labels_swap = self._faceparser_labels(swap_img)
        labels_orig = self._faceparser_labels(original_img)

        # Generate Overlay using the Rigid Alignment logic
        return self._create_aligned_mouth_overlay(
            original_img, labels_orig, labels_swap, parameters
        )

    def process_masks_and_masks(
        self,
        swap_restorecalc: torch.Tensor,
        original_face_512: torch.Tensor,
        parameters: dict,
        control: dict,
    ) -> dict:
        device = self.models_processor.device
        mode = control.get("DilatationTypeSelection", "conv")
        result = {"swap_formask": swap_restorecalc}

        target_h, target_w = swap_restorecalc.shape[1], swap_restorecalc.shape[2]

        resize_to_target = v2.Resize(
            (target_h, target_w),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )

        # --- Check Requirements ---
        # Mouth stretch is now an independent trigger
        need_mouth_stretch = parameters.get("MouthParserStretchToggle", False)

        need_parser = parameters.get("FaceParserEnableToggle", False) or (
            (
                parameters.get("TransferTextureEnableToggle", False)
                or parameters.get("DifferencingEnableToggle", False)
            )
            and parameters.get("ExcludeMaskEnableToggle", False)
        )
        need_parser_mouth = (
            parameters.get("DFLXSegEnableToggle", False)
            and parameters.get("XSegMouthEnableToggle", False)
            and parameters.get("DFLXSegSizeSlider", 0)
            != parameters.get("DFLXSeg2SizeSlider", 0)
        )

        labels_swap = None
        labels_orig = None

        # We need labels if Parser is ON, OR MouthStretch is ON, OR XSegMouth is ON
        if need_parser or need_parser_mouth or need_mouth_stretch:
            labels_swap = self._faceparser_labels(swap_restorecalc)

        # We need Original labels if (Parser ON OR MouthStretch ON)
        should_get_orig_labels = need_mouth_stretch or (
            need_parser
            and (
                parameters.get("FaceParserEnableToggle", False)
                or parameters.get("ExcludeMaskEnableToggle", False)
            )
        )

        if should_get_orig_labels:
            labels_orig = self._faceparser_labels(original_face_512)

        # ---------- MOUTH FIT & ALIGN LOGIC ----------
        if need_mouth_stretch and labels_orig is not None and labels_swap is not None:
            # Use the new Rigid/Stable Alignment function
            overlay, overlay_mask = self._create_aligned_mouth_overlay(
                original_face_512, labels_orig, labels_swap, parameters
            )

            if overlay is not None:
                if overlay.shape[1] != target_h:
                    overlay = resize_to_target(overlay)
                    overlay_mask = v2.Resize(
                        (target_h, target_w), interpolation=v2.InterpolationMode.NEAREST
                    )(overlay_mask.unsqueeze(0)).squeeze(0)

                result["mouth_overlay_info"] = (overlay, overlay_mask)
                if control.get("CommandLineDebugEnableToggle", False):
                    print("[INFO] Mouth Align: Applied Stable Width Transform.")

        # ---------- MOUTH (Grouped Optimization) ----------
        if need_parser_mouth:
            mouth = torch.zeros((512, 512), device=device, dtype=torch.float32)
            mouth_groups = {}
            mouth_specs = {
                11: "XsegMouthParserSlider",
                12: "XsegUpperLipParserSlider",
                13: "XsegLowerLipParserSlider",
            }

            for cls, pname in mouth_specs.items():
                val = int(parameters.get(pname, 0))
                if val not in mouth_groups:
                    mouth_groups[val] = []
                mouth_groups[val].append(cls)

            for val, classes in mouth_groups.items():
                if val:
                    m = self._mask_from_labels_lut(labels_swap, classes)
                    m = self._dilate_binary(m, val, mode)
                    mouth = torch.maximum(mouth, m)

            result["mouth"] = resize_to_target(mouth.unsqueeze(0)).clamp(0, 1).squeeze()

        # ---------- FACEPARSER MASK (Grouped Optimization) ----------
        if parameters.get("FaceParserEnableToggle", False):
            fp = torch.zeros((512, 512), device=device, dtype=torch.float32)
            fp_groups = {}
            face_classes = {
                1: "FaceParserSlider",
                2: "LeftEyebrowParserSlider",
                3: "RightEyebrowParserSlider",
                4: "LeftEyeParserSlider",
                5: "RightEyeParserSlider",
                6: "EyeGlassesParserSlider",
                10: "NoseParserSlider",
                11: "MouthParserSlider",
                12: "UpperLipParserSlider",
                13: "LowerLipParserSlider",
                14: "NeckParserSlider",
                17: "HairParserSlider",
            }
            mouth_inside = parameters.get("MouthParserInsideToggle", False)

            for cls, pname in face_classes.items():
                val = int(parameters.get(pname, 0))
                if val == 0:
                    continue
                is_min = mouth_inside and cls == 11
                key = (val, is_min)
                if key not in fp_groups:
                    fp_groups[key] = []
                fp_groups[key].append(cls)

            for (val, is_min), classes in fp_groups.items():
                m1 = self._mask_from_labels_lut(labels_swap, classes)
                m1 = self._dilate_binary(m1, val, mode)

                if labels_orig is not None:
                    m2 = self._mask_from_labels_lut(labels_orig, classes)
                    if is_min:
                        comb = torch.minimum(m1, m2)
                    else:
                        comb = torch.maximum(m1, m2)
                else:
                    comb = m1
                fp = torch.maximum(fp, comb)

            if parameters.get("FaceBlurParserSlider", 0) > 0:
                b = parameters["FaceBlurParserSlider"]
                gauss = transforms.GaussianBlur(b * 2 + 1, (b + 1) * 0.2)
                fp = gauss(fp.unsqueeze(0).unsqueeze(0)).squeeze()

            mask_high_res = (1.0 - fp).unsqueeze(0)
            mask_final = resize_to_target(mask_high_res)

            if parameters.get("FaceParserBlendSlider", 0) > 0:
                mask_final = (
                    mask_final + parameters["FaceParserBlendSlider"] / 100.0
                ).clamp(0, 1)
            result["FaceParser_mask"] = mask_final

        # ---------- TEXTURE / DIFFERENCING EXCLUDE ----------
        if (
            parameters.get("TransferTextureEnableToggle", False)
            or parameters.get("DifferencingEnableToggle", False)
        ) and parameters.get("ExcludeMaskEnableToggle", False):
            tex = torch.zeros((512, 512), device=device, dtype=torch.float32)
            tex_o = torch.zeros((512, 512), device=device, dtype=torch.float32)

            tex_specs = {
                1: "FaceParserTextureSlider",
                2: "EyebrowParserTextureSlider",
                3: "EyebrowParserTextureSlider",
                4: "EyeParserTextureSlider",
                5: "EyeParserTextureSlider",
                10: "NoseParserTextureSlider",
                11: "MouthParserTextureSlider",
                12: "MouthParserTextureSlider",
                13: "MouthParserTextureSlider",
                14: "NeckParserTextureSlider",
            }

            face_val = int(parameters.get(tex_specs[1], 0))
            if face_val > 0:
                blend = parameters.get("FaceParserTextureSlider", 0) / 10.0
                m_s = self._mask_from_labels_lut(labels_swap, [1]) * blend
                tex = torch.maximum(tex, m_s)
                if labels_orig is not None:
                    m_o = self._mask_from_labels_lut(labels_orig, [1]) * blend
                    tex_o = torch.maximum(tex_o, m_o)

            tex_groups = {}
            for cls, pname in tex_specs.items():
                if cls == 1:
                    continue
                val = int(parameters.get(pname, 0))
                if val == 0:
                    continue
                if val not in tex_groups:
                    tex_groups[val] = []
                tex_groups[val].append(cls)

            for d, classes in tex_groups.items():
                m_s = self._mask_from_labels_lut(labels_swap, classes)
                m_o = (
                    self._mask_from_labels_lut(labels_orig, classes)
                    if labels_orig is not None
                    else torch.zeros_like(m_s)
                )

                if d > 0:
                    m_s = self._dilate_binary(m_s, d, mode)
                    m_o = self._dilate_binary(m_o, d, mode)
                    if parameters.get("FaceParserBlendTextureSlider", 0):
                        bl = parameters["FaceParserBlendTextureSlider"] / 100.0
                        m_s = (m_s + bl).clamp(0, 1)
                        m_o = (m_o + bl).clamp(0, 1)
                    tex = torch.maximum(tex, m_s)
                    tex_o = torch.maximum(tex_o, m_o)

                elif d < 0:
                    d_abs = abs(d)
                    m_s = self._dilate_binary(m_s, -d_abs, mode)
                    m_o = self._dilate_binary(m_o, -d_abs, mode)
                    if parameters.get("FaceParserBlendTextureSlider", 0):
                        bl = parameters["FaceParserBlendTextureSlider"] / 100.0
                        m_s = (m_s + bl).clamp(0, 1)
                        m_o = (m_o + bl).clamp(0, 1)
                    sub = torch.maximum(m_s, m_o)
                    tex = (tex - sub).clamp_min(0)
                    tex_o = (tex_o - sub).clamp_min(0)

            comb = torch.minimum(1.0 - tex.clamp(0, 1), 1.0 - tex_o.clamp(0, 1))
            result["texture_mask"] = comb.unsqueeze(0).clamp(0, 1)

        return result

    def _get_circle_kernel(self, r: int, device: str) -> torch.Tensor:
        key = (int(r), str(device))
        k = self._morph_kernels.get(key)
        if k is not None:
            return k
        rr = int(r)

        ys, xs = torch.meshgrid(
            torch.arange(-rr, rr + 1, device=device),
            torch.arange(-rr, rr + 1, device=device),
            indexing="ij",
        )
        kernel = ((xs * xs + ys * ys) <= rr * rr).float().unsqueeze(0).unsqueeze(0)
        self._morph_kernels[key] = kernel
        return kernel

    def _dilate_binary(
        self, m: torch.Tensor, r: int, mode: str = "conv"
    ) -> torch.Tensor:
        if r == 0:
            return m
        squeeze_back = False
        if m.dim() == 2:
            m_in = m.unsqueeze(0).unsqueeze(0)
            squeeze_back = True
        elif m.dim() == 4:
            m_in = m
        else:
            raise ValueError(f"_dilate_binary: unsupported shape {m.shape}")

        rr = abs(int(r))

        if mode == "pool":
            out = F.max_pool2d(m_in, kernel_size=2 * rr + 1, stride=1, padding=rr)
            out = (out > 0).float()
        elif mode == "iter_pool":
            out = m_in
            for _ in range(rr):
                out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
            out = (out > 0).float()
        else:
            kernel = self._get_circle_kernel(rr, m_in.device)
            hits = F.conv2d(m_in, kernel, padding=rr)
            out = (hits > 0).float()

        return out.squeeze(0).squeeze(0) if squeeze_back else out

    def _mask_from_labels_lut(
        self, labels: torch.Tensor, classes: list[int]
    ) -> torch.Tensor:
        lut = torch.zeros(19, device=labels.device, dtype=torch.float32)
        if classes:
            lut[torch.tensor(classes, device=labels.device, dtype=torch.long)] = 1.0
        return lut[labels]

    def apply_occlusion(self, img, amount):
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones(
            (256, 256), dtype=torch.float32, device=self.models_processor.device
        ).contiguous()

        self.models_processor.run_occluder(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = outpred > 0
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount > 0:
            if "3x3" not in self._kernel_cache:
                self._kernel_cache["3x3"] = torch.ones(
                    (1, 1, 3, 3),
                    dtype=torch.float32,
                    device=self.models_processor.device,
                )
            kernel = self._kernel_cache["3x3"]

            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount < 0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones(
                (1, 1, 3, 3), dtype=torch.float32, device=self.models_processor.device
            )

            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred

    def run_occluder(self, image, output):
        model_name = "Occluder"
        ort_session = self.models_processor.models.get(model_name)

        if not ort_session:
            ort_session = self.models_processor.load_model(model_name)
            if ort_session:
                self.active_models.add(model_name)

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="img",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 1, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            ort_session.run_with_iobinding(io_binding)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def apply_dfl_xseg(self, img, amount, mouth, parameters):
        amount2 = -parameters["DFLXSeg2SizeSlider"]
        amount_calc = -parameters["BackgroundParserTextureSlider"]

        img = img.type(torch.float32)
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones(
            (256, 256), dtype=torch.float32, device=self.models_processor.device
        ).contiguous()

        self.run_dfl_xseg(img, outpred)

        outpred = torch.clamp(outpred, min=0.0, max=1.0)
        outpred[outpred < 0.1] = 0
        outpred_calc = outpred.clone()

        outpred = 1.0 - outpred
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        outpred_calc = torch.where(outpred_calc < 0.1, 0, 1).float()
        outpred_calc = 1.0 - outpred_calc
        outpred_calc = torch.unsqueeze(outpred_calc, 0).type(torch.float32)

        outpred_calc_dill = outpred_calc.clone()

        if amount2 != amount:
            outpred2 = outpred.clone()

        if amount > 0:
            r = int(amount)
            k = 2 * r + 1
            outpred = F.max_pool2d(outpred, kernel_size=k, stride=1, padding=r)
            outpred = outpred.clamp(0, 1)

        elif amount < 0:
            r = int(-amount)
            k = 2 * r + 1
            outpred = 1 - outpred
            outpred = F.max_pool2d(outpred, kernel_size=k, stride=1, padding=r)
            outpred = 1 - outpred
            outpred = outpred.clamp(0, 1)

        blur_amount = parameters["OccluderXSegBlurSlider"]
        if blur_amount > 0:
            blur_key = (blur_amount, (blur_amount + 1) * 0.2)
            if blur_key not in self._blur_cache:
                kernel_size = blur_amount * 2 + 1
                sigma = (blur_amount + 1) * 0.2
                self._blur_cache[blur_key] = transforms.GaussianBlur(kernel_size, sigma)
            gauss = self._blur_cache[blur_key]
            outpred = gauss(outpred)

        outpred_noFP = outpred.clone()
        if amount2 != amount:
            if amount2 > 0:
                r2 = int(amount2)
                k2 = 2 * r2 + 1
                outpred2 = F.max_pool2d(outpred2, kernel_size=k2, stride=1, padding=r2)
                outpred2 = outpred2.clamp(0, 1)

            elif amount2 < 0:
                r2 = int(-amount2)
                k2 = 2 * r2 + 1
                outpred2 = 1 - outpred2
                outpred2 = F.max_pool2d(outpred2, kernel_size=k2, stride=1, padding=r2)
                outpred2 = 1 - outpred2
                outpred2 = outpred2.clamp(0, 1)

            blur_amount2 = parameters["XSeg2BlurSlider"]
            if blur_amount2 > 0:
                blur_key2 = (blur_amount2, (blur_amount2 + 1) * 0.2)
                if blur_key2 not in self._blur_cache:
                    kernel_size2 = blur_amount2 * 2 + 1
                    sigma2 = (blur_amount2 + 1) * 0.2
                    self._blur_cache[blur_key2] = transforms.GaussianBlur(
                        kernel_size2, sigma2
                    )
                gauss2 = self._blur_cache[blur_key2]
                outpred2 = gauss2(outpred2)

            outpred[mouth > 0.01] = outpred2[mouth > 0.01]

        outpred = torch.reshape(outpred, (1, 256, 256))

        if parameters["BgExcludeEnableToggle"] and amount_calc != 0:
            if amount_calc > 0:
                r2 = int(amount_calc)
                k2 = 2 * r2 + 1
                outpred_calc_dill = F.max_pool2d(
                    outpred_calc_dill, kernel_size=k2, stride=1, padding=r2
                )
                outpred_calc_dill = outpred_calc_dill.clamp(0, 1)
                if parameters["BGExcludeBlurAmountSlider"] > 0:
                    gauss = transforms.GaussianBlur(
                        parameters["BGExcludeBlurAmountSlider"] * 2 + 1,
                        (parameters["BGExcludeBlurAmountSlider"] + 1) * 0.2,
                    )
                    outpred_calc_dill = gauss(outpred_calc_dill.type(torch.float32))
                outpred_calc_dill = outpred_calc_dill.clamp(0, 1)
            elif amount_calc < 0:
                r2 = int(-amount_calc)
                k2 = 2 * r2 + 1
                outpred_calc_dill = 1 - outpred_calc_dill
                outpred_calc_dill = F.max_pool2d(
                    outpred_calc_dill, kernel_size=k2, stride=1, padding=r2
                )
                outpred_calc_dill = 1 - outpred_calc_dill
                if parameters["BGExcludeBlurAmountSlider"] > 0:
                    orig = outpred_calc_dill.clone()
                    gauss = transforms.GaussianBlur(
                        parameters["BGExcludeBlurAmountSlider"] * 2 + 1,
                        (parameters["BGExcludeBlurAmountSlider"] + 1) * 0.2,
                    )
                    outpred_calc_dill = gauss(outpred_calc_dill.type(torch.float32))
                    outpred_calc_dill = torch.max(outpred_calc_dill, orig)
                outpred_calc_dill = outpred_calc_dill.clamp(0, 1)
        return outpred, outpred_calc, outpred_calc_dill, outpred_noFP

    def run_dfl_xseg(self, image, output):
        model_name = "XSeg"
        ort_session = self.models_processor.models.get(model_name)
        if not ort_session:
            ort_session = self.models_processor.load_model(model_name)

        if not ort_session:
            return

        io_binding = ort_session.io_binding()
        io_binding.bind_input(
            name="in_face:0",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=image.size(),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_output(
            name="out_mask:0",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 1, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            ort_session.run_with_iobinding(io_binding)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def run_onnx(self, image_tensor, output_tensor, model_key):
        sess = self.models_processor.models.get(model_key)
        if sess is None:
            sess = self.models_processor.load_model(model_key)

        image_tensor = image_tensor.contiguous()
        io_binding = sess.io_binding()

        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=image_tensor.shape,
            buffer_ptr=image_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name="features",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output_tensor.shape,
            buffer_ptr=output_tensor.data_ptr(),
        )

        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_key)
        if is_lazy_build:
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_key}\n\nThis may take several minutes.",
            )

        try:
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            else:
                self.models_processor.syncvec.cpu()

            sess.run_with_iobinding(io_binding)
        finally:
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

        return output_tensor

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        device = img.device
        if not self.models_processor.clip_session:
            self.models_processor.clip_session = CLIPDensePredT(
                version="ViT-B/16", reduce_dim=64, complex_trans_conv=True
            )
            self.models_processor.clip_session.eval()
            self.models_processor.clip_session.load_state_dict(
                torch.load(f"{models_dir}/rd64-uni-refined.pth", weights_only=True),
                strict=False,
            )
            self.models_processor.clip_session.to(device)

        clip_mask = torch.ones((352, 352), device=device)
        img = img.float() / 255.0
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize((352, 352)),
            ]
        )
        CLIPimg = transform(img).unsqueeze(0).contiguous().to(device)

        if CLIPText != "":
            prompts = CLIPText.split(",")
            with torch.no_grad():
                preds = self.models_processor.clip_session(
                    CLIPimg.repeat(len(prompts), 1, 1, 1), prompts
                )[0]

            clip_mask = 1 - torch.sigmoid(preds[0][0])
            for i in range(len(prompts) - 1):
                clip_mask *= 1 - torch.sigmoid(preds[i + 1][0])

            thresh = CLIPAmount / 100.0
            clip_mask = (clip_mask > thresh).float()

        return clip_mask.unsqueeze(0)

    def soft_oval_mask(
        self, height, width, center, radius_x, radius_y, feather_radius=None
    ):
        if feather_radius is None:
            feather_radius = max(radius_x, radius_y) // 2

        cache_key = (height, width)
        if cache_key in self._meshgrid_cache:
            y, x = self._meshgrid_cache[cache_key]
        else:
            y, x = torch.meshgrid(
                torch.arange(height), torch.arange(width), indexing="ij"
            )
            self._meshgrid_cache[cache_key] = (y, x)

        normalized_distance = torch.sqrt(
            ((x - center[0]) / radius_x) ** 2 + ((y - center[1]) / radius_y) ** 2
        )
        mask = torch.clamp(
            (1 - normalized_distance) * (radius_x / feather_radius), 0, 1
        )
        return mask

    def restore_mouth(
        self,
        img_orig,
        img_swap,
        kpss_orig,
        blend_alpha=0.5,
        feather_radius=10,
        size_factor=0.5,
        radius_factor_x=1.0,
        radius_factor_y=1.0,
        x_offset=0,
        y_offset=0,
    ):
        left_mouth = np.array([int(val) for val in kpss_orig[3]])
        right_mouth = np.array([int(val) for val in kpss_orig[4]])

        mouth_center = (left_mouth + right_mouth) // 2
        mouth_base_radius = int(np.linalg.norm(left_mouth - right_mouth) * size_factor)

        radius_x = int(mouth_base_radius * radius_factor_x)
        radius_y = int(mouth_base_radius * radius_factor_y)

        mouth_center[0] += x_offset
        mouth_center[1] += y_offset

        ymin = max(0, mouth_center[1] - radius_y)
        ymax = min(img_orig.size(1), mouth_center[1] + radius_y)
        xmin = max(0, mouth_center[0] - radius_x)
        xmax = min(img_orig.size(2), mouth_center[0] + radius_x)

        mouth_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
        mouth_mask = self.soft_oval_mask(
            ymax - ymin,
            xmax - xmin,
            (radius_x, radius_y),
            radius_x,
            radius_y,
            feather_radius,
        ).to(img_orig.device)

        target_ymin = ymin
        target_ymax = ymin + mouth_region_orig.size(1)
        target_xmin = xmin
        target_xmax = xmin + mouth_region_orig.size(2)

        img_swap_mouth = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
        blended_mouth = (
            blend_alpha * img_swap_mouth + (1 - blend_alpha) * mouth_region_orig
        )

        img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = (
            mouth_mask * blended_mouth + (1 - mouth_mask) * img_swap_mouth
        )
        return img_swap

    def restore_eyes(
        self,
        img_orig,
        img_swap,
        kpss_orig,
        blend_alpha=0.5,
        feather_radius=10,
        size_factor=3.5,
        radius_factor_x=1.0,
        radius_factor_y=1.0,
        x_offset=0,
        y_offset=0,
        eye_spacing_offset=0,
    ):
        left_eye = np.array([int(val) for val in kpss_orig[0]])
        right_eye = np.array([int(val) for val in kpss_orig[1]])

        left_eye[0] += x_offset
        right_eye[0] += x_offset
        left_eye[1] += y_offset
        right_eye[1] += y_offset

        eye_distance = np.linalg.norm(left_eye - right_eye)
        base_eye_radius = int(eye_distance / size_factor)

        radius_x = int(base_eye_radius * radius_factor_x)
        radius_y = int(base_eye_radius * radius_factor_y)

        left_eye[0] += eye_spacing_offset
        right_eye[0] -= eye_spacing_offset

        def extract_and_blend_eye(
            eye_center,
            radius_x,
            radius_y,
            img_orig,
            img_swap,
            blend_alpha,
            feather_radius,
        ):
            ymin = max(0, eye_center[1] - radius_y)
            ymax = min(img_orig.size(1), eye_center[1] + radius_y)
            xmin = max(0, eye_center[0] - radius_x)
            xmax = min(img_orig.size(2), eye_center[0] + radius_x)

            eye_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
            eye_mask = self.soft_oval_mask(
                ymax - ymin,
                xmax - xmin,
                (radius_x, radius_y),
                radius_x,
                radius_y,
                feather_radius,
            ).to(img_orig.device)

            target_ymin = ymin
            target_ymax = ymin + eye_region_orig.size(1)
            target_xmin = xmin
            target_xmax = xmin + eye_region_orig.size(2)

            img_swap_eye = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
            blended_eye = (
                blend_alpha * img_swap_eye + (1 - blend_alpha) * eye_region_orig
            )

            img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = (
                eye_mask * blended_eye + (1 - eye_mask) * img_swap_eye
            )

        extract_and_blend_eye(
            left_eye,
            radius_x,
            radius_y,
            img_orig,
            img_swap,
            blend_alpha,
            feather_radius,
        )
        extract_and_blend_eye(
            right_eye,
            radius_x,
            radius_y,
            img_orig,
            img_swap,
            blend_alpha,
            feather_radius,
        )

        return img_swap

    def apply_fake_diff(
        self,
        swapped_face,
        original_face,
        lower_thresh,
        lower_value,
        upper_thresh,
        upper_value,
        middle_value,
        parameters,
    ):
        diff = torch.abs(swapped_face - original_face)

        sample = diff.reshape(-1)
        sample = sample[torch.randint(0, sample.numel(), (50_000,), device=diff.device)]
        diff_max = torch.quantile(sample, 0.99)
        diff = torch.clamp(diff, max=diff_max)

        diff_min = diff.min()
        diff_max = diff.max()
        diff_norm = (diff - diff_min) / (diff_max - diff_min)

        diff_mean = diff_norm.mean(dim=0)
        scale = diff_mean / lower_thresh
        result = torch.where(
            diff_mean < lower_thresh,
            lower_value + scale * (middle_value - lower_value),
            torch.empty_like(diff_mean),
        )

        middle_scale = (diff_mean - lower_thresh) / (upper_thresh - lower_thresh)
        result = torch.where(
            (diff_mean >= lower_thresh) & (diff_mean <= upper_thresh),
            middle_value + middle_scale * (upper_value - middle_value),
            result,
        )

        above_scale = (diff_mean - upper_thresh) / (1 - upper_thresh)
        result = torch.where(
            diff_mean > upper_thresh,
            upper_value + above_scale * (1 - upper_value),
            result,
        )

        return result.unsqueeze(0)

    def apply_perceptual_diff_onnx(
        self,
        swapped_face,
        original_face,
        swap_mask,
        lower_thresh,
        lower_value,
        upper_thresh,
        upper_value,
        middle_value,
        feature_layer,
        ExcludeVGGMaskEnableToggle,
    ):
        feature_shapes = {
            "combo_relu3_3_relu3_1": (1, 512, 128, 128),
        }

        model_key = feature_layer
        if model_key not in self.models_processor.models:
            self.models_processor.models[model_key] = self.models_processor.load_model(
                model_key
            )

        def preprocess(img):
            img = img.clone().float() / 255.0
            mean = _VGG_MEAN.to(img.device)
            std = _VGG_STD.to(img.device)
            return ((img - mean) / std).unsqueeze(0).contiguous()

        swapped = preprocess(swapped_face)
        original = preprocess(original_face)

        shape = feature_shapes[feature_layer]
        outpred = torch.empty(shape, dtype=torch.float32, device=swapped.device)
        outpred2 = torch.empty_like(outpred)

        swapped_feat = self.run_onnx(swapped, outpred, model_key)
        original_feat = self.run_onnx(original, outpred2, model_key)

        diff_map = torch.abs(swapped_feat - original_feat).mean(dim=1)[0]
        diff_map = diff_map * swap_mask.squeeze(0)

        sample = diff_map.reshape(-1)
        sample = sample[
            torch.randint(0, diff_map.numel(), (50_000,), device=diff_map.device)
        ]
        diff_max = torch.quantile(sample, 0.99)
        diff_map = torch.clamp(diff_map, max=diff_max)

        diff_min, diff_max = diff_map.amin(), diff_map.amax()
        diff_norm = (diff_map - diff_min) / (diff_max - diff_min + 1e-6)

        diff_norm_texture = diff_norm.clone()

        if ExcludeVGGMaskEnableToggle:
            eps = 1e-6
            inv_lower = 1.0 / max(lower_thresh, eps)
            inv_mid = 1.0 / max((upper_thresh - lower_thresh), eps)
            inv_high = 1.0 / max((1.0 - upper_thresh), eps)

            res_low = lower_value + diff_norm * inv_lower * (middle_value - lower_value)
            res_mid = middle_value + (diff_norm - lower_thresh) * inv_mid * (
                upper_value - middle_value
            )
            res_high = upper_value + (diff_norm - upper_thresh) * inv_high * (
                1.0 - upper_value
            )

            result = torch.where(
                diff_norm < lower_thresh,
                res_low,
                torch.where(diff_norm > upper_thresh, res_high, res_mid),
            )
        else:
            result = diff_norm

        return result.unsqueeze(0), diff_norm_texture.unsqueeze(0)
