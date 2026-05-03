"""
TensorRT EP dynamic shape profiles (batch axes) for ORT.

Separated from ``models_processor`` so unit tests can import without Qt/torch/einops.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import onnx


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _trt_clamp_i(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _trt_shape_tuple_str(dims: list[int]) -> str:
    return "x".join(str(x) for x in dims)


def _trt_profiles_from_onnx_inputs(
    onnx_path: str, batch_opt: int, batch_max: int
) -> dict[str, str] | None:
    try:
        model = onnx.load(onnx_path)
    except Exception:
        return None

    init_names = {x.name for x in model.graph.initializer}
    parts_min: list[str] = []
    parts_opt: list[str] = []
    parts_max: list[str] = []

    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        tt = inp.type.tensor_type
        if not tt.shape.dim:
            return None
        dims: list[int | None] = []
        for d in tt.shape.dim:
            if d.dim_value and d.dim_value > 0:
                dims.append(int(d.dim_value))
            elif d.dim_value == 0:
                dims.append(None)
            elif d.dim_param:
                dims.append(None)
            else:
                dims.append(None)

        dyn_idx = [i for i, v in enumerate(dims) if v is None]
        if not dyn_idx:
            s = _trt_shape_tuple_str([int(x) for x in dims])  # type: ignore[arg-type]
            parts_min.append(f"{inp.name}:{s}")
            parts_opt.append(f"{inp.name}:{s}")
            parts_max.append(f"{inp.name}:{s}")
        elif len(dyn_idx) == 1 and dyn_idx[0] == 0:
            tail = dims[1:]
            if not tail or any(x is None for x in tail):
                return None
            tail_i = [int(x) for x in tail]  # type: ignore[arg-type]
            tail_s = _trt_shape_tuple_str(tail_i)
            parts_min.append(f"{inp.name}:1x{tail_s}")
            parts_opt.append(f"{inp.name}:{batch_opt}x{tail_s}")
            parts_max.append(f"{inp.name}:{batch_max}x{tail_s}")
        else:
            return None

    if not parts_min:
        return None
    return {
        "trt_profile_min_shapes": ",".join(parts_min),
        "trt_profile_opt_shapes": ",".join(parts_opt),
        "trt_profile_max_shapes": ",".join(parts_max),
    }


def _control_int(control: Mapping[str, Any] | None, key: str, default: int) -> int:
    if not control or key not in control:
        return default
    raw = control.get(key)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def tensorrt_dynamic_shape_profile_opts(
    model_name: str,
    onnx_path: str | None = None,
    control: Mapping[str, Any] | None = None,
) -> dict[str, str] | None:
    if _env_truthy("VISIOMASTER_TRT_NO_DYNAMIC_PROFILES"):
        return None

    use_ui = bool(control and control.get("TrtDynamicBatchTuningToggle", False))
    if use_ui:
        max_swap = _trt_clamp_i(_control_int(control, "TrtMaxBatchSwapSlider", 16), 1, 32)
        opt_swap = _trt_clamp_i(_control_int(control, "TrtOptBatchSwapSlider", 4), 1, max_swap)
        max_mot = _trt_clamp_i(_control_int(control, "TrtMaxBatchLpMotionSlider", 8), 1, 8)
        opt_mot = _trt_clamp_i(_control_int(control, "TrtOptBatchLpMotionSlider", 2), 1, max_mot)
        max_stitch = _trt_clamp_i(
            _control_int(control, "TrtMaxBatchLpStitchSlider", 12), 1, 16
        )
        opt_stitch = _trt_clamp_i(
            _control_int(control, "TrtOptBatchLpStitchSlider", 4), 1, max_stitch
        )
        max_arc = _trt_clamp_i(_control_int(control, "TrtMaxBatchArcfaceSlider", 16), 1, 32)
        opt_arc = _trt_clamp_i(_control_int(control, "TrtOptBatchArcfaceSlider", 8), 1, max_arc)
    else:
        max_swap = _trt_clamp_i(_env_int("VISIOMASTER_TRT_MAX_BATCH_SWAP", 16), 1, 32)
        opt_swap = _trt_clamp_i(
            _env_int("VISIOMASTER_TRT_OPT_BATCH_SWAP", min(4, max_swap)), 1, max_swap
        )
        max_mot = _trt_clamp_i(_env_int("VISIOMASTER_TRT_MAX_BATCH_LP_MOTION", 8), 1, 8)
        opt_mot = _trt_clamp_i(
            _env_int("VISIOMASTER_TRT_OPT_BATCH_LP_MOTION", min(2, max_mot)), 1, max_mot
        )
        max_stitch = _trt_clamp_i(_env_int("VISIOMASTER_TRT_MAX_BATCH_LP_STITCH", 12), 1, 16)
        opt_stitch = _trt_clamp_i(
            _env_int("VISIOMASTER_TRT_OPT_BATCH_LP_STITCH", min(4, max_stitch)), 1, max_stitch
        )
        max_arc = _trt_clamp_i(_env_int("VISIOMASTER_TRT_MAX_BATCH_ARCFACE", 16), 1, 32)
        opt_arc = _trt_clamp_i(
            _env_int("VISIOMASTER_TRT_OPT_BATCH_ARCFACE", min(8, max_arc)), 1, max_arc
        )

    if model_name == "LivePortraitMotionExtractor" and _env_truthy(
        "VISIOMASTER_LP_MOTION_TRT_STATIC_BATCH"
    ):
        return None

    if model_name == "Inswapper128ArcFace":
        if _env_truthy("VISIOMASTER_TRT_SKIP_ARCFACE_ONNX_PROFILE"):
            return None
        if onnx_path and os.path.isfile(onnx_path):
            prof = _trt_profiles_from_onnx_inputs(onnx_path, opt_arc, max_arc)
            if prof:
                return prof
        return None

    specs: dict[str, tuple[str, str, str]] = {
        "LivePortraitMotionExtractor": (
            "img:1x3x256x256",
            f"img:{opt_mot}x3x256x256",
            f"img:{max_mot}x3x256x256",
        ),
        "LivePortraitStitching": (
            "input:1x126",
            f"input:{opt_stitch}x126",
            f"input:{max_stitch}x126",
        ),
        "LivePortraitStitchingEye": (
            "input:1x66",
            f"input:{opt_stitch}x66",
            f"input:{max_stitch}x66",
        ),
        "LivePortraitStitchingLip": (
            "input:1x65",
            f"input:{opt_stitch}x65",
            f"input:{max_stitch}x65",
        ),
        "Inswapper128": (
            "target:1x3x128x128,source:1x512",
            f"target:{opt_swap}x3x128x128,source:{opt_swap}x512",
            f"target:{max_swap}x3x128x128,source:{max_swap}x512",
        ),
    }
    swap_256 = (
        "GhostFacev1",
        "GhostFacev2",
        "GhostFacev3",
        "HyperSwapv1",
        "HyperSwapv2",
        "HyperSwapv3",
    )
    tpl_256 = (
        "target:1x3x256x256,source:1x512",
        f"target:{opt_swap}x3x256x256,source:{opt_swap}x512",
        f"target:{max_swap}x3x256x256,source:{max_swap}x512",
    )
    for m in swap_256:
        specs[m] = tpl_256

    tpl = specs.get(model_name)
    if not tpl:
        return None
    return {
        "trt_profile_min_shapes": tpl[0],
        "trt_profile_opt_shapes": tpl[1],
        "trt_profile_max_shapes": tpl[2],
    }


def merge_tensorrt_dynamic_shape_profiles(
    model_name: str,
    model_trt_options: dict,
    onnx_path: str | None = None,
    control: Mapping[str, Any] | None = None,
) -> dict:
    """Attach TRT min/opt/max shape profiles when this model has a registry entry."""
    prof = tensorrt_dynamic_shape_profile_opts(
        model_name, onnx_path=onnx_path, control=control
    )
    if not prof:
        return model_trt_options
    out = dict(model_trt_options)
    out.update(prof)
    if _env_truthy("VISIOMASTER_LOG_TRT_PROFILE"):
        print(
            f"[TRT-PROFILE] {model_name} min={prof['trt_profile_min_shapes']} "
            f"opt={prof['trt_profile_opt_shapes']} max={prof['trt_profile_max_shapes']}",
            flush=True,
        )
    return out
