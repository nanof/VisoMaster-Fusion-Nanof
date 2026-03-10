"""Auto-mouth: detect mouth openness from pipeline landmarks; maintain on/stay/off state."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Minimum pixel span of the mouth (corners) to trust the ratio calculation.
# Prevents divide-by-near-zero when a face is very small or partially out of frame.
MIN_MOUTH_SPAN_PX: float = 8.0


def compute_lip_open_ratio_203(kps: np.ndarray | None) -> float | None:
    """Compute lip-open ratio using 203-point landmarks.

    Uses the same pairs as faceutil.calc_lip_close_ratio:
      vertical distance: kps[90] ↔ kps[102]
      horizontal span:   kps[48] ↔ kps[66]

    Returns None when landmarks are unavailable or the face is too small.
    """
    if kps is None or len(kps) < 203:
        return None
    span = float(np.linalg.norm(kps[48] - kps[66]))
    if span < MIN_MOUTH_SPAN_PX:
        return None
    vert = float(np.linalg.norm(kps[90] - kps[102]))
    return vert / (span + 1e-6)


def compute_lip_open_ratio_68(kps: np.ndarray | None) -> float | None:
    """Compute lip-open ratio using 68-point landmarks.

    vertical distance: kps[62] ↔ kps[66]
    horizontal span:   kps[48] ↔ kps[54]

    Returns None when landmarks are unavailable or the face is too small.
    """
    if kps is None or len(kps) < 68:
        return None
    span = float(np.linalg.norm(kps[48] - kps[54]))
    if span < MIN_MOUTH_SPAN_PX:
        return None
    vert = float(np.linalg.norm(kps[62] - kps[66]))
    return vert / (span + 1e-6)


@dataclass
class MouthOpennessState:
    """Per-face EMA state for the Auto Mouth Expression feature.

    Three update rules:
      1. ratio >= threshold  → activate (set active=True)
      2. ratio is None       → stay     (keep current active value unchanged)
      3. ratio <  threshold  → deactivate (set active=False)

    Rule 2 handles occlusion (object covering the mouth, e.g. spoon/fork/food)
    by keeping the feature enabled rather than falsely turning it off.
    """

    active: bool = False
    ema: float = 0.0

    def update(self, ratio: float | None, alpha: float, threshold: float) -> bool:
        """Update EMA and state. Returns the new active flag."""
        if ratio is None:
            return self.active  # rule 2 — stay
        self.ema = alpha * ratio + (1.0 - alpha) * self.ema
        self.active = self.ema >= threshold
        return self.active

    def reset(self) -> None:
        """Reset state to inactive (call when switching target faces or disabling)."""
        self.active = False
        self.ema = 0.0
