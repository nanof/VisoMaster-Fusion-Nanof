"""Unit tests for app.processors.mouth_openness."""
from __future__ import annotations

import numpy as np
import pytest

from app.processors.mouth_openness import (
    MIN_MOUTH_SPAN_PX,
    MouthOpennessState,
    compute_lip_open_ratio_203,
    compute_lip_open_ratio_68,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kps_203(n: int = 203, mouth_vert: float = 10.0, mouth_horiz: float = 50.0) -> np.ndarray:
    """Build a dummy kps array for the 203-point case.

    203-pt relevant indices:
      48, 66 — horizontal mouth corners
      90, 102 — vertical lip points

    Note: kps[66] is the *horizontal* right-mouth-corner in the 203-pt model.
    """
    assert n >= 203
    kps = np.zeros((n, 2), dtype=np.float32)
    kps[48] = [0.0, 0.0]
    kps[66] = [mouth_horiz, 0.0]        # right mouth corner (horiz)
    kps[90] = [mouth_horiz / 2, 0.0]
    kps[102] = [mouth_horiz / 2, mouth_vert]
    return kps


def _make_kps_68(n: int = 68, mouth_vert: float = 10.0, mouth_horiz: float = 50.0) -> np.ndarray:
    """Build a dummy kps array for the 68-point case.

    68-pt relevant indices:
      48, 54 — horizontal mouth corners
      62, 66 — vertical lip points
    """
    assert n >= 68
    kps = np.zeros((n, 2), dtype=np.float32)
    kps[48] = [0.0, 0.0]
    kps[54] = [mouth_horiz, 0.0]
    kps[62] = [mouth_horiz / 2, 0.0]
    kps[66] = [mouth_horiz / 2, mouth_vert]
    return kps


# ---------------------------------------------------------------------------
# compute_lip_open_ratio_203
# ---------------------------------------------------------------------------


class TestComputeLipOpenRatio203:
    def test_returns_none_when_kps_is_none(self):
        assert compute_lip_open_ratio_203(None) is None

    def test_returns_none_when_too_few_landmarks(self):
        kps = np.zeros((100, 2), dtype=np.float32)
        assert compute_lip_open_ratio_203(kps) is None

    def test_returns_none_when_exactly_202_landmarks(self):
        kps = np.zeros((202, 2), dtype=np.float32)
        assert compute_lip_open_ratio_203(kps) is None

    def test_returns_float_for_203_landmarks(self):
        kps = _make_kps_203(203, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert isinstance(result, float)

    def test_ratio_value_correct(self):
        # vert=10, horiz=50 → ratio ≈ 10/50 = 0.2
        kps = _make_kps_203(203, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert result == pytest.approx(10.0 / 50.0, rel=1e-4)

    def test_returns_none_when_mouth_span_too_small(self):
        kps = _make_kps_203(203, mouth_vert=2.0, mouth_horiz=MIN_MOUTH_SPAN_PX - 1.0)
        assert compute_lip_open_ratio_203(kps) is None

    def test_returns_value_when_span_exactly_at_min(self):
        # Guard is strict <, so span == MIN_MOUTH_SPAN_PX passes through and returns a value
        kps = _make_kps_203(203, mouth_horiz=MIN_MOUTH_SPAN_PX, mouth_vert=2.0)
        assert compute_lip_open_ratio_203(kps) is not None

    def test_returns_value_when_span_above_min(self):
        kps = _make_kps_203(203, mouth_horiz=MIN_MOUTH_SPAN_PX + 1.0, mouth_vert=5.0)
        result = compute_lip_open_ratio_203(kps)
        assert result is not None
        assert result > 0.0

    def test_closed_mouth_returns_low_ratio(self):
        # vert very small relative to horiz
        kps = _make_kps_203(203, mouth_vert=0.5, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert result < 0.05

    def test_wide_open_mouth_returns_high_ratio(self):
        kps = _make_kps_203(203, mouth_vert=30.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert result > 0.5

    def test_accepts_more_than_203_landmarks(self):
        kps = _make_kps_203(300, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_203(kps)
        assert result is not None


# ---------------------------------------------------------------------------
# compute_lip_open_ratio_68
# ---------------------------------------------------------------------------


class TestComputeLipOpenRatio68:
    def test_returns_none_when_kps_is_none(self):
        assert compute_lip_open_ratio_68(None) is None

    def test_returns_none_when_too_few_landmarks(self):
        kps = np.zeros((30, 2), dtype=np.float32)
        assert compute_lip_open_ratio_68(kps) is None

    def test_returns_none_when_exactly_67_landmarks(self):
        kps = np.zeros((67, 2), dtype=np.float32)
        assert compute_lip_open_ratio_68(kps) is None

    def test_returns_float_for_68_landmarks(self):
        kps = _make_kps_68(68, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_68(kps)
        assert isinstance(result, float)

    def test_ratio_value_correct(self):
        kps = _make_kps_68(68, mouth_vert=10.0, mouth_horiz=50.0)
        result = compute_lip_open_ratio_68(kps)
        assert result == pytest.approx(10.0 / 50.0, rel=1e-4)

    def test_returns_none_when_mouth_span_too_small(self):
        kps = _make_kps_68(68, mouth_horiz=MIN_MOUTH_SPAN_PX - 1.0)
        assert compute_lip_open_ratio_68(kps) is None

    def test_returns_value_when_span_above_min(self):
        kps = _make_kps_68(68, mouth_horiz=MIN_MOUTH_SPAN_PX + 1.0, mouth_vert=4.0)
        assert compute_lip_open_ratio_68(kps) is not None


# ---------------------------------------------------------------------------
# MouthOpennessState
# ---------------------------------------------------------------------------


class TestMouthOpennessState:
    # -- initial state --

    def test_initial_state_is_inactive(self):
        state = MouthOpennessState()
        assert state.active is False

    def test_initial_ema_is_zero(self):
        state = MouthOpennessState()
        assert state.ema == 0.0

    # -- update: rule 1 — ratio >= threshold → activate --

    def test_update_activates_when_ratio_above_threshold(self):
        state = MouthOpennessState()
        result = state.update(ratio=0.5, alpha=1.0, threshold=0.2)
        assert result is True
        assert state.active is True

    def test_update_activates_when_ratio_equals_threshold(self):
        state = MouthOpennessState()
        # alpha=1.0 → ema = ratio immediately
        result = state.update(ratio=0.2, alpha=1.0, threshold=0.2)
        assert result is True

    # -- update: rule 3 — ratio < threshold → deactivate --

    def test_update_deactivates_when_ratio_below_threshold(self):
        state = MouthOpennessState(active=True, ema=0.5)
        result = state.update(ratio=0.01, alpha=1.0, threshold=0.2)
        assert result is False
        assert state.active is False

    # -- update: rule 2 — ratio=None → stay --

    def test_update_stays_active_when_ratio_is_none(self):
        state = MouthOpennessState(active=True, ema=0.5)
        result = state.update(ratio=None, alpha=0.4, threshold=0.2)
        assert result is True
        assert state.active is True

    def test_update_stays_inactive_when_ratio_is_none(self):
        state = MouthOpennessState(active=False, ema=0.0)
        result = state.update(ratio=None, alpha=0.4, threshold=0.2)
        assert result is False
        assert state.active is False

    def test_update_none_does_not_change_ema(self):
        state = MouthOpennessState(active=True, ema=0.5)
        state.update(ratio=None, alpha=0.4, threshold=0.2)
        assert state.ema == 0.5

    # -- EMA smoothing --

    def test_ema_smoothing_with_alpha_one(self):
        state = MouthOpennessState()
        state.update(ratio=0.6, alpha=1.0, threshold=0.2)
        assert state.ema == pytest.approx(0.6)

    def test_ema_smoothing_with_alpha_half(self):
        state = MouthOpennessState()
        state.update(ratio=0.8, alpha=0.5, threshold=0.2)
        # ema = 0.5*0.8 + 0.5*0.0 = 0.4
        assert state.ema == pytest.approx(0.4)

    def test_ema_accumulates_across_updates(self):
        state = MouthOpennessState()
        state.update(ratio=0.8, alpha=0.5, threshold=0.9)  # ema=0.4, below threshold
        assert state.active is False
        state.update(ratio=0.8, alpha=0.5, threshold=0.5)  # ema=0.6, above threshold
        assert state.active is True

    def test_ema_decays_toward_zero(self):
        state = MouthOpennessState(active=True, ema=0.9)
        state.update(ratio=0.0, alpha=0.5, threshold=0.2)
        assert state.ema == pytest.approx(0.45)

    # -- reset --

    def test_reset_clears_active_and_ema(self):
        state = MouthOpennessState(active=True, ema=0.8)
        state.reset()
        assert state.active is False
        assert state.ema == 0.0

    # -- edge cases --

    def test_alpha_zero_ema_never_changes(self):
        state = MouthOpennessState(ema=0.0)
        state.update(ratio=1.0, alpha=0.0, threshold=0.2)
        assert state.ema == pytest.approx(0.0)
        assert state.active is False

    def test_high_threshold_not_triggered_by_normal_ratio(self):
        state = MouthOpennessState()
        state.update(ratio=0.15, alpha=1.0, threshold=0.50)
        assert state.active is False

    def test_returns_bool_not_generic_truthy(self):
        state = MouthOpennessState()
        result = state.update(ratio=0.5, alpha=1.0, threshold=0.2)
        assert isinstance(result, bool)
