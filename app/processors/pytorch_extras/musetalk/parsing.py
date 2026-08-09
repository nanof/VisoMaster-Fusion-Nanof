"""MuseTalk's own blending mask, ported from the reference implementation.

Upstream (``musetalk/utils/blending.py`` plus ``face_parsing/__init__.py``) never
uses a geometric shape: it segments the face with BiSeNet and composites the
parsed lower face. Our first port approximated that with an ellipse, which put a
visible edge through the mouth — a quadratic plateau clipped to 1.0 has a
discontinuous derivative, and across teeth-against-shadow contrast that reads as
a hard line no matter how the radii are tuned.

The morphology constants below are upstream's and are calibrated for the
512x512 parse, so the label map is expected at that resolution.

Class ids are CelebAMask-HQ, which is what both upstream's weights and our
``FaceParsingBiSeNet18`` were trained on.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

SKIN = 1
NOSE = 10
MOUTH = 11
UPPER_LIP = 12
LOWER_LIP = 13

# Oral cavity plus both lips: everything lip-sync actually changes.
MOUTH_CLASSES = (MOUTH, UPPER_LIP, LOWER_LIP)

# Upstream defaults for v1.5 (``--left_cheek_width`` / ``--right_cheek_width``).
CHEEK_WIDTH = 90
EXPAND = 1.5
UPPER_BOUNDARY_RATIO = 0.5


def _jaw_kernel() -> np.ndarray:
    """Upstream's cone-with-a-tail dilation kernel.

    Mass sits below the anchor, so dilating the skin class grows it downwards
    past the jawline. That is what lets the chin move with the audio instead of
    the repaint stopping at the segmented jaw edge.
    """
    cone_height, tail_height = 21, 12
    total = cone_height + tail_height
    kernel = np.zeros((total, total), dtype=np.uint8)
    centre = total // 2
    for row in range(cone_height // 2, cone_height):
        width = 2 * (row - cone_height // 2) + 1
        kernel[row, centre - width // 2 : centre + width // 2 + 1] = 1
    base_width = int(kernel[cone_height - 1].sum())
    for row in range(cone_height, total):
        start = max(0, centre - base_width // 2)
        end = min(total, centre + base_width // 2 + 1)
        kernel[row, start:end] = 1
    return kernel


_JAW_KERNEL = _jaw_kernel()
# Flat and wide: pulls the mask in horizontally without shortening the chin.
_CHEEK_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 3))


def _cheek_mask(shape: tuple[int, int], left: int, right: int) -> np.ndarray:
    """Everything outside a central column, where the eroded mask is used.

    The left edge includes its boundary column because upstream draws this with
    ``cv2.rectangle``, whose second corner is inclusive.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    centre = shape[1] // 2
    mask[:, : max(centre - int(left) + 1, 0)] = 255
    mask[:, min(centre + int(right), shape[1]) :] = 255
    return mask


def jaw_region_mask(
    labels: np.ndarray,
    *,
    left_cheek_width: int = CHEEK_WIDTH,
    right_cheek_width: int = CHEEK_WIDTH,
) -> np.ndarray:
    """Upstream ``mode="jaw"``: a 0/255 mask of the repaintable face region.

    The cheeks get the eroded mask so the repaint cannot creep onto ears, hair
    or background, while the centre keeps the full dilated reach. The nose is
    excluded and the mouth and lips are forced in.
    """
    skin = (labels == SKIN).astype(np.uint8) * 255
    dilated = cv2.dilate(skin, _JAW_KERNEL, iterations=1)
    eroded = cv2.erode(dilated, _CHEEK_KERNEL, iterations=2)
    cheeks = _cheek_mask(labels.shape[:2], left_cheek_width, right_cheek_width)
    region = cv2.bitwise_and(eroded, cheeks)
    region = cv2.bitwise_or(region, cv2.bitwise_and(dilated, cv2.bitwise_not(cheeks)))
    out = np.zeros(labels.shape[:2], dtype=np.uint8)
    out[(region == 255) & (labels != NOSE)] = 255
    out[np.isin(labels, (MOUTH, UPPER_LIP, LOWER_LIP))] = 255
    return out


def mouth_region_mask(labels: np.ndarray) -> np.ndarray:
    """0/255 mask of the oral cavity and both lips.

    This is the whole of what audio changes, unlike :func:`lip_region_mask`,
    which excludes the cavity because it guides a colour shift.
    """
    out = np.zeros(labels.shape[:2], dtype=np.uint8)
    out[np.isin(labels, MOUTH_CLASSES)] = 255
    return out


def lip_region_mask(labels: np.ndarray) -> np.ndarray:
    """0/255 mask of the lips only, for the chroma correction.

    The oral cavity (``MOUTH``) is deliberately excluded: only the upper and lower
    lip carry the identity colour MuseTalk normalises toward its magenta prior, and
    leaving the cavity out means the correction can never tint teeth.
    """
    out = np.zeros(labels.shape[:2], dtype=np.uint8)
    out[np.isin(labels, (UPPER_LIP, LOWER_LIP))] = 255
    return out


def crop_box_for(
    bbox: tuple[int, int, int, int], expand: float = EXPAND
) -> tuple[int, int, int, int]:
    """Upstream ``get_crop_box``: a square around the face box, ``expand`` times it.

    The mask is parsed on this larger square rather than on the face box, so the
    segmenter sees a whole head in context and the feather has room to fall off.
    """
    x1, y1, x2, y2 = bbox
    xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
    s = int(max(x2 - x1, y2 - y1) // 2 * expand)
    return xc - s, yc - s, xc + s, yc + s


def _padded_crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop, zero-filling out-of-frame area the way PIL's crop does upstream."""
    x_s, y_s, x_e, y_e = box
    h, w = frame.shape[:2]
    out = np.zeros((y_e - y_s, x_e - x_s, frame.shape[2]), dtype=frame.dtype)
    sx1, sy1 = max(x_s, 0), max(y_s, 0)
    sx2, sy2 = min(x_e, w), min(y_e, h)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y_s : sy2 - y_s, sx1 - x_s : sx2 - x_s] = frame[sy1:sy2, sx1:sx2]
    return out


def _parse_and_place(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    parse_labels,
    expand: float,
):
    """Parse the expanded square once and return the label map plus its geometry.

    Shared so both the repaint mask and the lip mask come from a single BiSeNet
    call: segmenting the head twice per frame would double the parser's GPU cost
    for no benefit, the two masks being different reductions of the same labels.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    box = crop_box_for((x1, y1, x2, y2), expand)
    face_large = _padded_crop(frame_bgr, box)
    ch, cw = face_large.shape[:2]
    if ch < 8 or cw < 8:
        return None
    labels = parse_labels(face_large[:, :, ::-1])
    if labels is None:
        return None
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.size == 0:
        return None
    ox, oy = x1 - box[0], y1 - box[1]
    return labels, (ox, oy, x2 - x1, y2 - y1, ch, cw)


def _place_and_feather(
    region_512: np.ndarray,
    geom: tuple[int, int, int, int, int, int],
    *,
    upper_boundary_ratio: float | None,
    feather_ratio: float,
) -> np.ndarray | None:
    """Resize a 512-space mask onto ``bbox``, drop its top, and feather the edge.

    ``upper_boundary_ratio`` of None keeps the whole height, which the lip mask
    wants: unlike the repaint region it never reaches the eyes, so cutting the top
    would only clip the upper lip.
    """
    ox, oy, bw, bh, ch, cw = geom
    region = cv2.resize(region_512, (cw, ch), interpolation=cv2.INTER_LINEAR)
    inside = np.zeros((ch, cw), dtype=np.uint8)
    inside[oy : oy + bh, ox : ox + bw] = region[oy : oy + bh, ox : ox + bw]
    if upper_boundary_ratio is not None:
        top = int(ch * float(upper_boundary_ratio))
        if top > 0:
            inside[:top] = 0
    k = int(feather_ratio * cw // 2 * 2) + 1
    blurred = cv2.GaussianBlur(inside, (k, k), 0)
    mask = blurred[oy : oy + bh, ox : ox + bw].astype(np.float32) / 255.0
    if mask.shape != (bh, bw):
        return None
    return mask


class ParsedMasks(NamedTuple):
    """The three reductions of one parse that the blend can use.

    ``jaw`` is upstream's repaint region, ``lip`` guides the chroma correction and
    ``mouth`` is the tight cavity-plus-lips region used to repaint locally. All are
    float32 in [0,1] over ``bbox``, or None when parsing is unavailable.
    """

    jaw: np.ndarray | None
    lip: np.ndarray | None
    mouth: np.ndarray | None


def parsed_masks(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    parse_labels,
    *,
    expand: float = EXPAND,
    upper_boundary_ratio: float = UPPER_BOUNDARY_RATIO,
    left_cheek_width: int = CHEEK_WIDTH,
    right_cheek_width: int = CHEEK_WIDTH,
    strength: float = 1.0,
) -> ParsedMasks:
    """Every mask the blend needs, from a single parse."""
    placed = _parse_and_place(frame_bgr, bbox, parse_labels, expand)
    if placed is None:
        return ParsedMasks(None, None, None)
    labels, geom = placed
    jaw = _place_and_feather(
        jaw_region_mask(
            labels,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
        ),
        geom,
        upper_boundary_ratio=upper_boundary_ratio,
        feather_ratio=0.05,
    )
    if jaw is not None and strength < 1.0:
        jaw = jaw * max(float(strength), 0.0)
    # The lip mask feathers less: it guides a colour shift, not a paste, so a wide
    # falloff would only bleed the correction onto the skin around the lips.
    lip = _place_and_feather(
        lip_region_mask(labels),
        geom,
        upper_boundary_ratio=None,
        feather_ratio=0.02,
    )
    # Crisp: it is one half of a union that gets dilated and feathered as a whole.
    mouth = _place_and_feather(
        mouth_region_mask(labels),
        geom,
        upper_boundary_ratio=None,
        feather_ratio=0.0,
    )
    return ParsedMasks(jaw, lip, mouth)


def parsed_face_masks(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    parse_labels,
    *,
    expand: float = EXPAND,
    upper_boundary_ratio: float = UPPER_BOUNDARY_RATIO,
    left_cheek_width: int = CHEEK_WIDTH,
    right_cheek_width: int = CHEEK_WIDTH,
    strength: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(jaw, lip)`` only, for callers that do not repaint locally."""
    masks = parsed_masks(
        frame_bgr,
        bbox,
        parse_labels,
        expand=expand,
        upper_boundary_ratio=upper_boundary_ratio,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width,
        strength=strength,
    )
    return masks.jaw, masks.lip


def crop_mouth_mask(
    crop_bgr: np.ndarray,
    parse_labels,
    out_shape: tuple[int, int],
) -> np.ndarray | None:
    """Parse a face crop and return its mouth region resized to ``out_shape``.

    Used on MuseTalk's own output: the union of where the mouth *was* and where it
    *is* is the only region that can be repainted without leaving two mouths
    visible, and the second half of that union can only come from the generated
    crop. The crop shares its framing with the face box, so a plain resize lands
    the mask in the right place.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    labels = parse_labels(np.ascontiguousarray(crop_bgr[:, :, ::-1]))
    if labels is None:
        return None
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.size == 0:
        return None
    h, w = int(out_shape[0]), int(out_shape[1])
    if h < 2 or w < 2:
        return None
    region = cv2.resize(
        mouth_region_mask(labels), (w, h), interpolation=cv2.INTER_NEAREST
    )
    return region.astype(np.float32) / 255.0


def mouth_only_blend_mask(
    original_mouth: np.ndarray | None,
    generated_mouth: np.ndarray | None,
    *,
    padding_px: int = 6,
    feather_px: int = 9,
) -> np.ndarray | None:
    """Union of both mouth poses, grown by ``padding_px`` and feathered.

    Composited at full alpha this is the only way to avoid the doubled mouth and
    chin: alpha below 1.0 leaves the original mouth visible through the generated
    one, and any mask that does not cover *both* poses leaves part of the original
    mouth beside the new one. Growing the union and feathering hides the seam in
    the skin around the mouth, which lip-sync barely changes.
    """
    parts = [
        np.asarray(m, dtype=np.float32)
        for m in (original_mouth, generated_mouth)
        if m is not None and getattr(m, "size", 0)
    ]
    if not parts:
        return None
    shape = parts[0].shape[:2]
    if any(p.shape[:2] != shape for p in parts):
        return None
    union = np.zeros(shape, dtype=np.uint8)
    for part in parts:
        union[part > 0.5] = 255
    if not union.any():
        return None
    pad = max(int(padding_px), 0)
    grown = union
    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1,) * 2)
        grown = cv2.dilate(union, kernel, iterations=1)
    k = max(int(feather_px), 0)
    if k > 0:
        grown = cv2.GaussianBlur(grown, (k * 2 + 1,) * 2, 0)
        # The falloff belongs in the padding ring only. Left to blur freely it eats
        # into the lip border, and alpha under 1.0 exactly there is the doubled
        # mouth again, at its most visible edge.
        grown = np.maximum(grown, union)
    return grown.astype(np.float32) / 255.0


def parsed_lower_face_mask(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    parse_labels,
    *,
    expand: float = EXPAND,
    upper_boundary_ratio: float = UPPER_BOUNDARY_RATIO,
    left_cheek_width: int = CHEEK_WIDTH,
    right_cheek_width: int = CHEEK_WIDTH,
    strength: float = 1.0,
) -> np.ndarray | None:
    """Float32 mask in [0,1] covering ``bbox``, or None if parsing is unavailable.

    ``parse_labels`` takes an RGB crop and returns a 512x512 CelebAMask label
    map. Only the part covering ``bbox`` is returned because that is the only
    place the generated face is pasted; upstream composites the whole expanded
    square, but outside the face box that copies the original onto itself.

    ``upper_boundary_ratio`` is a fraction of the *expanded* square, so the
    default 0.5 cuts at the face box's own vertical centre — the mid-nose line,
    given how the crop is framed.
    """
    return parsed_face_masks(
        frame_bgr,
        bbox,
        parse_labels,
        expand=expand,
        upper_boundary_ratio=upper_boundary_ratio,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width,
        strength=strength,
    )[0]
