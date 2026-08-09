"""Simple lower-face blending for MuseTalk crops (no BiSeNet dependency)."""

from __future__ import annotations

import cv2
import numpy as np


def soft_lower_face_mask(
    h: int,
    w: int,
    *,
    centre_y: float = 0.64,
    radius_y: float = 0.30,
    radius_x: float = 0.42,
    strength: float = 1.0,
) -> np.ndarray:
    """Float32 mask HxW in [0,1]: full strength over the mouth, feathered edges.

    The mouth gets weight 1.0 over a plateau rather than the tip of a quadratic
    falloff. With a falloff the lips sat around 0.6, so a third of the original
    mouth was blended back in exactly where lip closure reads, which is what made
    the lip motion so hard to see. The ellipse still reaches 0 before the crop
    border so the paste has no visible seam.

    Geometry and ``strength`` are exposed because the right trade-off is not
    universal: MuseTalk drives lip shape from audio, so it normalises away much of
    a face's own lip character, and lowering ``strength`` buys that character back
    at the cost of lip movement.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = h * float(centre_y)
    cx = w * 0.5
    ry = h * float(radius_y)
    rx = w * float(radius_x)
    dist = ((yy - cy) / max(ry, 1.0)) ** 2 + ((xx - cx) / max(rx, 1.0)) ** 2
    # Flat at 1.0 while dist <= 0.55, then feathered down to 0 at the ellipse.
    mask = np.clip((1.0 - dist) / 0.45, 0.0, 1.0)
    # Fade top of crop so forehead/eyes stay from the swapped frame.
    top_fade = np.clip((yy - h * 0.36) / max(h * 0.10, 1.0), 0.0, 1.0)
    mask = mask * top_fade
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(w * 0.02, 1.0))
    if strength < 1.0:
        mask = mask * max(float(strength), 0.0)
    return mask.astype(np.float32)


_MOUTH_SHARPEN_SIGMA = 1.2
# Cap so a very soft neighbourhood cannot demand a halo-inducing boost.
_MOUTH_SHARPEN_MAX = 1.4


def _local_contrast(gray: np.ndarray, sigma: float = 2.0) -> float:
    """High-frequency energy: mean distance from a blurred copy of itself."""
    return float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)).mean())


def restore_mouth_detail(
    face_bgr: np.ndarray,
    reference_bgr: np.ndarray | None = None,
    *,
    sigma: float = _MOUTH_SHARPEN_SIGMA,
    max_amount: float = _MOUTH_SHARPEN_MAX,
) -> np.ndarray:
    """Sharpen the generated mouth only as much as its surroundings warrant.

    The VAE keeps mean colour to ~1/255 but strips fine detail, and a mouth
    blurrier than the face around it reads as the wrong colour even with matching
    channel means. How much is missing depends entirely on the footage: a sharp
    source lost 34% of its detail, while soft phone footage came out 33% *sharper*
    than its own blurry original. A fixed boost therefore over-sharpens exactly
    the material that needs it least, so the amount is derived per frame from
    ``reference_bgr`` — the untouched face next to the mouth.

    Sharpening is self-sourced: pulling high frequencies from the original mouth
    would ghost the old lip shape back in wherever the audio changed it.
    """
    if reference_bgr is None or reference_bgr.size < 64:
        return face_bgr
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    have = _local_contrast(gray)
    if have <= 1e-3:
        return face_bgr
    want = _local_contrast(
        cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    )
    # Unsharp with amount a scales high frequencies by roughly (1 + a).
    amount = min(max(want / have - 1.0, 0.0), max_amount)
    if amount <= 0.05:
        return face_bgr
    f32 = face_bgr.astype(np.float32)
    blurred = cv2.GaussianBlur(f32, (0, 0), sigmaX=sigma)
    return np.clip(f32 + amount * (f32 - blurred), 0, 255).astype(np.uint8)


# Chroma magnitude (distance from neutral in Cr/Cb) that separates lips from the
# rest: skin measured ~15-22 and lips ~24-30 on real footage, while teeth and the
# oral cavity sit well below. Gating the shift by this means an opened mouth whose
# teeth fall inside the original lip footprint is never tinted.
_LIP_CHROMA_LO = 12.0
_LIP_CHROMA_HI = 28.0


def transfer_lip_chroma(
    generated_bgr: np.ndarray,
    original_bgr: np.ndarray,
    lip_mask: np.ndarray,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Pull the generated lips' colour toward the original's, luminance untouched.

    Lip-sync lives in luminance — openness, teeth, the dark interior — while the
    identity of the lips lives in chrominance, and MuseTalk normalises that toward
    a magenta prior regardless of the face beneath. Shifting only the Cr/Cb *mean*
    over the lip pixels toward the original therefore recovers the real lip colour
    without moving the lips a pixel.

    A mean shift rather than a per-pixel copy because the two mouths need not share
    a pose: copying would drag the closed-mouth original onto an open generated
    one. The shift is additionally gated by how coloured each generated pixel is,
    so teeth revealed by an opening mouth keep their own near-neutral tone.
    """
    if lip_mask is None or strength <= 0.0:
        return generated_bgr
    m = np.ascontiguousarray(lip_mask, dtype=np.float32)
    if m.shape[:2] != generated_bgr.shape[:2] or float(m.sum()) < 16.0:
        return generated_bgr
    gen = cv2.cvtColor(generated_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    org = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    chroma = np.hypot(gen[:, :, 1] - 128.0, gen[:, :, 2] - 128.0)
    gate = np.clip(
        (chroma - _LIP_CHROMA_LO) / max(_LIP_CHROMA_HI - _LIP_CHROMA_LO, 1e-3),
        0.0,
        1.0,
    )
    weight = m * gate
    total = float(weight.sum())
    if total < 16.0:
        return generated_bgr
    for ch in (1, 2):
        gen_mean = float((gen[:, :, ch] * weight).sum() / total)
        org_mean = float((org[:, :, ch] * weight).sum() / total)
        shift = float(strength) * (org_mean - gen_mean)
        gen[:, :, ch] = np.clip(gen[:, :, ch] + weight * shift, 0.0, 255.0)
    return cv2.cvtColor(gen.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def blend_face_region(
    frame_bgr: np.ndarray,
    face_bgr_256: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    mask: np.ndarray | None = None,
    lip_mask: np.ndarray | None = None,
    lip_color_strength: float = 0.0,
    mask_options: dict | None = None,
) -> np.ndarray:
    """Paste ``face_bgr_256`` into ``frame_bgr`` at ``bbox``.

    ``mask`` is the segmented lower face when face parsing is available, which is
    what upstream MuseTalk uses. ``mask_options`` only drives the geometric
    fallback for when the parser model is missing.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1 = int(max(0, min(x1, w - 1)))
    x2 = int(max(0, min(x2, w)))
    y1 = int(max(0, min(y1, h - 1)))
    y2 = int(max(0, min(y2, h)))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return frame_bgr
    region_w, region_h = x2 - x1, y2 - y1
    resized = cv2.resize(
        face_bgr_256, (region_w, region_h), interpolation=cv2.INTER_LANCZOS4
    )
    region = frame_bgr[y1:y2, x1:x2]
    # Sharpen after the resize so the correction covers the resize blur too, and
    # judge it against the crop's own top band: the mask leaves that untouched, so
    # it shows how sharp this face actually is at this size on this footage.
    resized = restore_mouth_detail(resized, region[: max(int(region_h * 0.36), 8)])
    # Recover the real lip colour the model normalised toward magenta. Only the
    # chrominance moves and only over the parsed lips, so the lip-sync (luminance)
    # and the teeth are untouched. Earlier attempts corrected global colour, which
    # is dominated by skin (matching the median made the bias worse, -2.0 -> -3.2)
    # or corrected luminance, which undid the lip-sync; this touches neither.
    if lip_mask is not None and lip_color_strength > 0.0:
        if lip_mask.shape[:2] == (region_h, region_w):
            resized = transfer_lip_chroma(
                resized, region, lip_mask, strength=float(lip_color_strength)
            )
    if mask is None or mask.shape[:2] != (region_h, region_w):
        mask = soft_lower_face_mask(region_h, region_w, **(mask_options or {}))
    weights = mask.astype(np.float32)[..., None]
    roi = region.astype(np.float32)
    mixed = roi * (1.0 - weights) + resized.astype(np.float32) * weights
    out = frame_bgr.copy()
    out[y1:y2, x1:x2] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out


def expand_bbox(
    bbox: np.ndarray | list | tuple,
    frame_shape: tuple[int, ...],
    *,
    extra_margin: int = 10,
    pad_ratio: float = 0.0,
    vertical_shift: float = -0.06,
) -> tuple[int, int, int, int]:
    """Clamp/expand detector bbox; MuseTalk v1.5 adds margin on y2.

    ``pad_ratio`` and ``vertical_shift`` were tuned against ground truth: driving
    a clip with its own audio means the correct mouth is the one already on
    screen, so the framing that minimises the error against it is the one
    matching MuseTalk's training convention. Padding the detector box hurt badly
    (0.05 scored 18.9/255 and 0.12 scored 25.1/255 against 15.8 for no padding),
    and lifting the window by 6% of the box height — which seats the mouth lower
    in the crop, where the model expects it — brought that down to 11.0/255.
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    shift = bh * float(vertical_shift)
    y1 += shift
    y2 += shift
    x1 -= bw * pad_ratio
    x2 += bw * pad_ratio
    y1 -= bh * pad_ratio
    y2 += bh * pad_ratio + float(extra_margin)
    x1 = int(max(0, min(x1, w - 1)))
    x2 = int(max(0, min(x2, w)))
    y1 = int(max(0, min(y1, h - 1)))
    y2 = int(max(0, min(y2, h)))
    return x1, y1, x2, y2
