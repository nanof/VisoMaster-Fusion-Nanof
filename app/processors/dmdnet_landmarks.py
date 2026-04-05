"""Landmark helpers for DMDNet (68-point layout, InsightFace 106 → 68 mapping)."""

from __future__ import annotations

import numpy as np
import torch

# Same index map as InsightFace / common 106→68 reductions (see e.g. DMDNet test code).
_MAP_106_TO_68: list[int] = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    72,
    73,
    74,
    86,
    78,
    79,
    80,
    85,
    84,
    35,
    41,
    42,
    39,
    37,
    36,
    89,
    95,
    96,
    93,
    91,
    90,
    52,
    64,
    63,
    71,
    67,
    68,
    61,
    58,
    59,
    53,
    56,
    55,
    65,
    66,
    62,
    70,
    69,
    57,
    60,
    54,
]


def landmarks106_to_68_xy(points106: np.ndarray) -> np.ndarray:
    """Project 106-point face landmarks to a 68-point layout (xy, float32)."""
    pts = np.asarray(points106, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"Expected (N, 2+) landmark array, got {pts.shape}")
    if pts.shape[0] < 106:
        raise ValueError(f"Need at least 106 points for 106→68 map, got {pts.shape[0]}")
    out = np.zeros((68, 2), dtype=np.float64)
    for i, j in enumerate(_MAP_106_TO_68):
        out[i] = pts[j, :2]
    return out.astype(np.float32)


def _finalize_dmdnet_roi_box(
    xyxy: np.ndarray, *, min_side: float = 32.0, limit: float = 512.0
) -> np.ndarray:
    """DMDNet uses integer ROI slices on feature maps; boxes must have strictly positive area."""
    x1, y1, x2, y2 = (float(xyxy[i]) for i in range(4))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 < min_side:
        cx = 0.5 * (x1 + x2)
        x1 = cx - 0.5 * min_side
        x2 = cx + 0.5 * min_side
    if y2 - y1 < min_side:
        cy = 0.5 * (y1 + y2)
        y1 = cy - 0.5 * min_side
        y2 = cy + 0.5 * min_side
    x1 = max(0.0, min(x1, limit - min_side))
    y1 = max(0.0, min(y1, limit - min_side))
    x2 = max(x1 + min_side, min(x2, limit))
    y2 = max(y1 + min_side, min(y2, limit))
    ix1, iy1, ix2, iy2 = (
        int(np.floor(x1)),
        int(np.floor(y1)),
        int(np.ceil(x2)),
        int(np.ceil(y2)),
    )
    if ix2 <= ix1:
        ix2 = min(int(limit), ix1 + int(min_side))
    if iy2 <= iy1:
        iy2 = min(int(limit), iy1 + int(min_side))
    return np.array([ix1, iy1, ix2, iy2], dtype=np.float32)


def get_component_location_tensor(
    landmarks68_xy: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Four component bounding boxes [LE, RE, NO, MO] as in DMDNet ``main_test.py``."""
    lm = np.asarray(landmarks68_xy, dtype=np.float64).reshape(68, 2)
    lm = lm.copy()
    lm[lm > 504] = 504
    lm[lm < 8] = 8

    map_le_b = list(np.hstack((range(17, 22), range(36, 42))))
    map_re_b = list(np.hstack((range(22, 27), range(42, 48))))
    map_le = list(range(36, 42))
    map_re = list(range(42, 48))
    map_no = list(range(29, 36))
    map_mo = list(range(48, 68))

    mean_le = np.mean(lm[map_le], axis=0)
    l_le1 = max(float(mean_le[1] - np.min(lm[map_le_b, 1])), 4.0) * 1.3
    l_le2 = l_le1 / 1.9
    l_le_xy = l_le1 + l_le2
    l_le_lt = [l_le_xy / 2, l_le1]
    l_le_rb = [l_le_xy / 2, l_le2]
    loc_le = np.hstack((mean_le - l_le_lt + 1, mean_le + l_le_rb)).astype(int)

    mean_re = np.mean(lm[map_re], axis=0)
    l_re1 = max(float(mean_re[1] - np.min(lm[map_re_b, 1])), 4.0) * 1.3
    l_re2 = l_re1 / 1.9
    l_re_xy = l_re1 + l_re2
    l_re_lt = [l_re_xy / 2, l_re1]
    l_re_rb = [l_re_xy / 2, l_re2]
    loc_re = np.hstack((mean_re - l_re_lt + 1, mean_re + l_re_rb)).astype(int)

    mean_no = np.mean(lm[map_no], axis=0)
    l_no1 = max(
        float(np.max([mean_no[0] - lm[31][0], lm[35][0] - mean_no[0]]) * 1.25),
        12.0,
    )
    l_no2 = max(float((lm[33][1] - mean_no[1]) * 1.1), 12.0)
    l_no_xy = max(l_no1 * 2, l_no2 + 8.0)
    l_no_lt = [l_no_xy / 2, l_no_xy - l_no2]
    l_no_rb = [l_no_xy / 2, l_no2]
    loc_no = np.hstack((mean_no - l_no_lt + 1, mean_no + l_no_rb)).astype(int)

    mean_mo = np.mean(lm[map_mo], axis=0)
    l_mo = (
        max(
            float(np.max(np.max(lm[map_mo], axis=0) - np.min(lm[map_mo], axis=0)) / 2),
            16.0,
        )
        * 1.1
    )
    mo_o = mean_mo - l_mo + 1
    mo_t = mean_mo + l_mo
    mo_t[mo_t > 510] = 510
    loc_mo = np.hstack((mo_o, mo_t)).astype(int)

    fixed = [
        _finalize_dmdnet_roi_box(loc_le),
        _finalize_dmdnet_roi_box(loc_re),
        _finalize_dmdnet_roi_box(loc_no),
        _finalize_dmdnet_roi_box(loc_mo),
    ]
    stacked = np.stack(fixed, axis=0)
    return torch.from_numpy(stacked.astype(np.float32)).to(device)
