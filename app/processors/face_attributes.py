"""Lightweight face attribute models (InsightFace GenderAge)."""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from app.processors.utils import faceutil

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

GENDERAGE_MODEL_NAME = "GenderAge"
GENDERAGE_INPUT_SIZE = 96
# InsightFace buffalo genderage.onnx graph uses mean=0 / std=1 (pixel scale 0–255).
GENDERAGE_INPUT_MEAN = 0.0
GENDERAGE_INPUT_STD = 1.0

GENDER_FILTER_ALL = "All"
GENDER_FILTER_FEMALE = "Female appearance only"
GENDER_FILTER_MALE = "Male appearance only"


def parse_gender_age_output(
    pred: np.ndarray,
) -> tuple[str, float, int]:
    """Parse InsightFace genderage ``fc1`` vector of length 3.

    Index 0 = female logit, 1 = male logit, 2 = age/100.
    Gender id matches InsightFace ``Face.sex``: 0→female, 1→male.
    """
    flat = np.asarray(pred, dtype=np.float64).reshape(-1)
    if flat.size < 3:
        raise ValueError(f"genderage output expected >=3 values, got {flat.size}")
    logits = flat[:2]
    # Softmax confidence of the chosen class (stable with large logits).
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    probs = exp / np.maximum(exp.sum(), 1e-12)
    gender_idx = int(np.argmax(logits))
    gender = "male" if gender_idx == 1 else "female"
    confidence = float(probs[gender_idx])
    age = int(np.round(float(flat[2]) * 100.0))
    return gender, confidence, age


def gender_filter_mode(control: dict[str, Any] | None) -> str:
    if not control:
        return GENDER_FILTER_ALL
    raw = control.get("GenderAppearanceFilterSelection", GENDER_FILTER_ALL)
    if raw in (GENDER_FILTER_FEMALE, GENDER_FILTER_MALE):
        return str(raw)
    return GENDER_FILTER_ALL


def gender_filter_min_confidence(control: dict[str, Any] | None) -> float:
    if not control:
        return 0.60
    try:
        return max(0.0, min(1.0, float(control.get("GenderAppearanceMinConfidenceSlider", 60)) / 100.0))
    except (TypeError, ValueError):
        return 0.60


def skip_swap_for_gender_appearance_filter(
    control: dict[str, Any] | None,
    gender: Optional[str],
    confidence: float,
) -> bool:
    """Return True when the detected face must not be swapped.

    Uncertain results (missing gender or confidence below threshold) do **not**
    skip — that was a common failure mode of over-aggressive previous filters.
    """
    mode = gender_filter_mode(control)
    if mode == GENDER_FILTER_ALL:
        return False
    if gender not in ("female", "male"):
        return False
    if float(confidence) < gender_filter_min_confidence(control):
        return False
    if mode == GENDER_FILTER_FEMALE and gender != "female":
        return True
    if mode == GENDER_FILTER_MALE and gender != "male":
        return True
    return False


def bbox_from_kps5(kps_5: np.ndarray) -> np.ndarray:
    """Approximate a detector-style bbox from 5 landmarks (inner face → pad out)."""
    pts = np.asarray(kps_5, dtype=np.float32).reshape(-1, 2)
    x1 = float(pts[:, 0].min())
    y1 = float(pts[:, 1].min())
    x2 = float(pts[:, 0].max())
    y2 = float(pts[:, 1].max())
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    # Landmarks sit inside the face; inflate toward a typical SCRFD box.
    pad_x = w * 0.55
    pad_y = h * 0.70
    return np.array(
        [x1 - pad_x, y1 - pad_y * 1.1, x2 + pad_x, y2 + pad_y * 0.55],
        dtype=np.float32,
    )


class FaceAttributes:
    """InsightFace GenderAge classifier (buffalo ``genderage.onnx``)."""

    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.active_models: set[str] = set()
        # Guards only the lazy session creation; inference runs unlocked like the
        # other ORT paths so multi-GPU workers are not serialized on this model.
        self._load_lock = threading.Lock()
        # track_id → (gender, confidence, age); cleared on media change by caller.
        self._track_cache: dict[int, tuple[str, float, int]] = {}
        self._track_cache_lock = threading.Lock()

    def unload_models(self) -> None:
        with self.models_processor.model_lock:
            for model_name in list(self.active_models):
                self.models_processor.unload_model(model_name)
            self.active_models.clear()
        self.clear_track_cache()

    def clear_track_cache(self) -> None:
        with self._track_cache_lock:
            self._track_cache.clear()

    def classify_face_gender(
        self,
        img_chw: torch.Tensor,
        bbox: Optional[np.ndarray] = None,
        kps_5: Optional[np.ndarray] = None,
        track_id: int = -1,
    ) -> tuple[Optional[str], float]:
        """Classify apparent gender for a single face (see ``classify_faces_gender``)."""
        results = self.classify_faces_gender(
            img_chw,
            [{"bbox": bbox, "kps_5": kps_5, "track_id": track_id}],
        )
        return results[0]

    def classify_faces_gender(
        self,
        img_chw: torch.Tensor,
        faces: list[dict[str, Any]],
    ) -> list[tuple[Optional[str], float]]:
        """Classify apparent gender for every face in one batched inference.

        Uses InsightFace Attribute preprocessing (bbox-centered similarity crop to
        96×96, mean=0 / std=1 — not ArcFace landmark align). Crops are taken from a
        small sub-window of the frame instead of warping the full-resolution image,
        which is what made the per-face cost scale with frame size.
        """
        results: list[tuple[Optional[str], float]] = [(None, 0.0)] * len(faces)
        pending_rows: list[torch.Tensor] = []
        pending_idx: list[int] = []
        pending_tid: list[int] = []

        for i, face in enumerate(faces):
            tid = int(face.get("track_id", -1) or -1)
            if tid >= 0:
                with self._track_cache_lock:
                    cached = self._track_cache.get(tid)
                if cached is not None:
                    results[i] = (cached[0], cached[1])
                    continue

            box = self._resolve_bbox(img_chw, face.get("bbox"), face.get("kps_5"))
            if box is None:
                continue
            try:
                row = self._genderage_input_row(img_chw, box)
            except Exception as exc:
                print(f"[WARN] GenderAge crop failed: {exc}")
                continue
            if row is None:
                continue
            pending_rows.append(row)
            pending_idx.append(i)
            pending_tid.append(tid)

        if not pending_rows:
            return results

        try:
            preds = self._infer_genderage_batch(torch.stack(pending_rows, dim=0))
        except Exception as exc:
            print(f"[WARN] GenderAge inference failed: {exc}")
            return results

        for row_i, out_i in enumerate(pending_idx):
            try:
                gender, confidence, age = parse_gender_age_output(preds[row_i])
            except Exception as exc:
                print(f"[WARN] GenderAge parse failed: {exc}")
                continue
            results[out_i] = (gender, confidence)
            tid = pending_tid[row_i]
            if tid >= 0 and confidence >= 0.55:
                with self._track_cache_lock:
                    self._track_cache[tid] = (gender, confidence, age)
                    if len(self._track_cache) > 256:
                        for drop_key in list(self._track_cache.keys())[:64]:
                            self._track_cache.pop(drop_key, None)

        return results

    def _resolve_bbox(
        self,
        img_chw: torch.Tensor,
        bbox: Optional[np.ndarray],
        kps_5: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if bbox is not None:
            arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
            if arr.size >= 4:
                return arr[:4]
        if kps_5 is not None:
            pts = np.asarray(kps_5, dtype=np.float32)
            if pts.size >= 10:
                return bbox_from_kps5(pts)
        # Last resort: whole tensor as face region (VR tight crops).
        if img_chw is not None and img_chw.dim() == 3:
            _, h, w = img_chw.shape
            pad = int(min(h, w) * 0.05)
            return np.array(
                [pad, pad, max(pad + 1, w - pad), max(pad + 1, h - pad)],
                dtype=np.float32,
            )
        return None

    def _genderage_input_row(
        self, img_chw: torch.Tensor, bbox: np.ndarray
    ) -> Optional[torch.Tensor]:
        """Build one normalized CHW 96×96 model input from a frame + bbox.

        Equivalent to InsightFace's ``face_align.transform`` (rotation=0), but the
        affine warp runs on a small sub-window instead of the whole frame — warping
        full-resolution pixels per face was costing several ms per face at 4K. The
        returned row always lives on the host, matching the CPU-EP session.
        """
        if img_chw is None or img_chw.dim() != 3:
            return None
        _, frame_h, frame_w = img_chw.shape
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        side = max(w, h) * 1.5
        scale = float(GENDERAGE_INPUT_SIZE) / side

        # Sub-window covering the sampled square, plus a margin for bilinear taps.
        margin = 4
        left = int(math.floor(center_x - side * 0.5)) - margin
        top = int(math.floor(center_y - side * 0.5)) - margin
        right = int(math.ceil(center_x + side * 0.5)) + margin
        bottom = int(math.ceil(center_y + side * 0.5)) + margin

        valid_left = max(left, 0)
        valid_top = max(top, 0)
        valid_right = min(right, frame_w)
        valid_bottom = min(bottom, frame_h)
        if valid_right <= valid_left or valid_bottom <= valid_top:
            return None

        sub = img_chw[:, valid_top:valid_bottom, valid_left:valid_right]
        # Inference runs on the CPU EP, so bring the (tiny) sub-window to the host
        # before warping instead of after: warping on the GPU costs an extra blocking
        # host->device copy of the affine matrix inside faceutil.transform. Measured
        # per face at 4K: 2.12 -> 1.22 ms with an idle GPU, 13.0 -> 11.9 ms while the
        # device is saturated (the remainder is unavoidably waiting for the GPU queue).
        sub = sub.to("cpu")
        pad_l = valid_left - left
        pad_t = valid_top - top
        pad_r = right - valid_right
        pad_b = bottom - valid_bottom
        if pad_l or pad_t or pad_r or pad_b:
            # Zero fill matches the out-of-bounds behaviour of the full-frame warp.
            sub = F.pad(
                sub.unsqueeze(0), (pad_l, pad_r, pad_t, pad_b), value=0
            ).squeeze(0)

        crop, _ = faceutil.transform(
            sub,
            (center_x - left, center_y - top),
            GENDERAGE_INPUT_SIZE,
            scale,
            0.0,
        )
        if crop.dim() != 3 or crop.shape[0] != 3:
            raise RuntimeError(f"unexpected genderage crop shape {tuple(crop.shape)}")

        # (pixel - mean) / std with mean=0, std=1 → float in ~[0, 255]
        inp = crop.to(dtype=torch.float32)
        if GENDERAGE_INPUT_MEAN != 0.0:
            inp = inp - GENDERAGE_INPUT_MEAN
        if GENDERAGE_INPUT_STD != 1.0:
            inp = inp / GENDERAGE_INPUT_STD
        return inp

    def _infer_genderage_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Run GenderAge once for an ``(N, 3, 96, 96)`` batch → ``(N, 3)`` numpy.

        Runs on the **CPU EP** (see ``ONNX_MODELS_FORCE_CPU_EP``): the CUDA EP puts
        every depthwise Conv of this graph into cuDNN fallback mode, which measured
        ~233 ms per call versus ~0.6 ms on CPU. ``_genderage_input_row`` already
        returns host tensors, so the ``to("cpu")`` below is a no-op safety net.
        """
        model_name = GENDERAGE_MODEL_NAME
        ort_session = self.models_processor.get_onnx_session(model_name)
        if not ort_session:
            with self._load_lock:
                ort_session = self.models_processor.get_onnx_session(model_name)
                if not ort_session:
                    ort_session = self.models_processor.load_model(model_name)
                    if ort_session:
                        self.active_models.add(model_name)
        if not ort_session:
            raise RuntimeError("GenderAge model failed to load")

        in_np_dtype = self.models_processor.get_ort_io_numpy_dtype(
            model_name, "data", is_output=False, session=ort_session
        )
        inp = np.ascontiguousarray(
            batch.detach().to(device="cpu", copy=False).numpy(), dtype=in_np_dtype
        )
        preds = ort_session.run(["fc1"], {"data": inp})[0]
        return np.asarray(preds, dtype=np.float32).reshape(int(batch.shape[0]), -1)
