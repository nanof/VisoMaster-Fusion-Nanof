"""
Host-side NVIDIA Image Scaling (NVScaler) constant buffer for OpenGL UBO.

Logic ported from NVIDIAGameWorks/NVIDIAImageScaling NIS_Config.h (MIT).
"""

from __future__ import annotations

import math
import os
import struct

# NVScaler en un solo paso: k_scale = input_w/output_w debe estar en (min_k, 1].
# min_k=0.5 ⇒ como mucho 2× por paso (valores menores suelen dar salida negra en el shader).
# Opcional: VISIOMASTER_NIS_MIN_K_SCALE (solo un paso; para >2× se usa cadena en el GL item).
_DEFAULT_MIN_K_SCALE = 0.5
# Límite de píxeles destino (rw×rh) para ejecutar NVScaler compute; por encima solo blit.
# Evita picos de VRAM y cierres del driver / TDR en Windows con preview muy grande.
_DEFAULT_MAX_OUTPUT_PIXELS = 12_000_000


def nis_max_compute_output_pixels() -> int:
    raw = os.environ.get("VISIOMASTER_NIS_MAX_OUTPUT_PIXELS", "").strip()
    if not raw:
        return _DEFAULT_MAX_OUTPUT_PIXELS
    try:
        v = int(raw)
    except ValueError:
        return _DEFAULT_MAX_OUTPUT_PIXELS
    return max(2_000_000, min(80_000_000, v))


def _nis_min_k_scale() -> float:
    raw = os.environ.get("VISIOMASTER_NIS_MIN_K_SCALE", "").strip()
    if not raw:
        return _DEFAULT_MIN_K_SCALE
    try:
        v = float(raw)
    except ValueError:
        return _DEFAULT_MIN_K_SCALE
    return max(0.05, min(0.499, v))


def nis_upscale_chain(w: int, h: int, rw: int, rh: int) -> list[tuple[int, int]]:
    """
    Lista de (out_w, out_h) por paso; en cada tramo el upscale es ≤ 1/min_k (por defecto 2×).
    Vacía si no hace falta upscale.
    """
    w = max(1, int(w))
    h = max(1, int(h))
    rw = max(1, int(rw))
    rh = max(1, int(rh))
    if w >= rw and h >= rh:
        return []
    lo = _nis_min_k_scale()
    max_ratio = 1.0 / lo
    steps: list[tuple[int, int]] = []
    cw, ch = w, h
    while cw < rw or ch < rh:
        sx = float(rw) / float(cw) if cw else 1.0
        sy = float(rh) / float(ch) if ch else 1.0
        scale = min(max_ratio, sx, sy)
        if scale <= 1.00001:
            break
        nw = min(rw, max(cw + 1, int(math.ceil(cw * scale))))
        nh = min(rh, max(ch + 1, int(math.ceil(ch * scale))))
        if nw <= cw and nh <= ch:
            break
        steps.append((nw, nh))
        cw, ch = nw, nh
        if len(steps) > 24:
            break
    if cw != rw or ch != rh:
        steps.append((rw, rh))
    return steps


def pack_nis_scaler_ubo(
    *,
    input_tex_w: int,
    input_tex_h: int,
    input_vp_w: int,
    input_vp_h: int,
    out_tex_w: int,
    out_tex_h: int,
    out_vp_w: int,
    out_vp_h: int,
    sharpness_0_1: float,
) -> bytes | None:
    """
    Build std140 bytes for layout(std140, binding=0) NISConstBlock in nis_scaler.comp.
    Returns None when no hay upscale (k_scale>1 por eje) o dimensiones inválidas.
    Para ratios >2× usar nis_upscale_chain + varios dispatch en el preview.
    """
    itw = max(1, int(input_tex_w))
    ith = max(1, int(input_tex_h))
    otw = max(1, int(out_tex_w))
    oth = max(1, int(out_tex_h))
    ivw = int(input_vp_w) if int(input_vp_w) > 0 else itw
    ivh = int(input_vp_h) if int(input_vp_h) > 0 else ith
    ovw = int(out_vp_w) if int(out_vp_w) > 0 else otw
    ovh = int(out_vp_h) if int(out_vp_h) > 0 else oth
    if ivw <= 0 or ivh <= 0 or ovw <= 0 or ovh <= 0:
        return None

    sharp = max(0.0, min(1.0, float(sharpness_0_1)))
    sharpen_slider = sharp - 0.5
    max_scale = 1.25 if sharpen_slider >= 0.0 else 1.75
    min_scale = 1.25 if sharpen_slider >= 0.0 else 1.0
    limit_scale = 1.25 if sharpen_slider >= 0.0 else 1.0

    k_detect_ratio = 2.0 * 1127.0 / 1024.0
    k_detect_thres = 64.0 / 1024.0
    k_min_contrast_ratio = 2.0
    k_max_contrast_ratio = 10.0
    k_sharp_start_y = 0.45
    k_sharp_end_y = 0.9
    k_sharp_strength_min = max(0.0, 0.4 + sharpen_slider * min_scale * 1.2)
    k_sharp_strength_max = 1.6 + sharpen_slider * max_scale * 1.8
    k_sharp_limit_min = max(0.1, 0.14 + sharpen_slider * limit_scale * 0.32)
    k_sharp_limit_max = 0.5 + sharpen_slider * limit_scale * 0.6

    k_ratio_norm = 1.0 / (k_max_contrast_ratio - k_min_contrast_ratio)
    k_sharp_scale_y = 1.0 / (k_sharp_end_y - k_sharp_start_y)
    k_sharp_strength_scale = k_sharp_strength_max - k_sharp_strength_min
    k_sharp_limit_scale = k_sharp_limit_max - k_sharp_limit_min

    k_src_norm_x = 1.0 / float(itw)
    k_src_norm_y = 1.0 / float(ith)
    k_dst_norm_x = 1.0 / float(otw)
    k_dst_norm_y = 1.0 / float(oth)
    k_scale_x = float(ivw) / float(ovw)
    k_scale_y = float(ivh) / float(ovh)
    lo = _nis_min_k_scale()
    if k_scale_x < lo or k_scale_x > 1.0 or k_scale_y < lo or k_scale_y > 1.0:
        return None

    # 18 floats + 8 uint + 2 floats → 112 bytes (std140, múltiplo de 16)
    return struct.pack(
        "18f8I2f",
        k_detect_ratio,
        k_detect_thres,
        k_min_contrast_ratio,
        k_ratio_norm,
        1.0,  # kContrastBoost
        1.0 / 255.0,  # kEps
        k_sharp_start_y,
        k_sharp_scale_y,
        k_sharp_strength_min,
        k_sharp_strength_scale,
        k_sharp_limit_min,
        k_sharp_limit_scale,
        k_scale_x,
        k_scale_y,
        k_dst_norm_x,
        k_dst_norm_y,
        k_src_norm_x,
        k_src_norm_y,
        0,
        0,
        int(ivw),
        int(ivh),
        0,
        0,
        int(ovw),
        int(ovh),
        0.0,
        0.0,
    )
