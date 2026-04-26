"""Auto re-weight calibration (proportional to 1/ms_per_frame)."""
from __future__ import annotations

import pytest

from app.processors.gpu_scheduler import calibrate_weights_from_timings


def test_fast_gpu_is_normalized_to_max_weight():
    w = calibrate_weights_from_timings({0: 10.0, 1: 30.0}, max_weight=8)
    assert w[0] == 8  # fastest normalized to max
    assert 2 <= w[1] <= 3  # ~10/30 * 8 = 2.67


def test_symmetric_gpus_get_equal_weights():
    w = calibrate_weights_from_timings({0: 20.0, 1: 20.0}, max_weight=8)
    assert w[0] == w[1]


def test_uncommon_ratios_still_round_to_stable_ints():
    w = calibrate_weights_from_timings({0: 5.0, 1: 8.0, 2: 50.0}, max_weight=10)
    assert w[0] == 10
    # 5/8 * 10 = 6.25 -> 6
    assert w[1] == 6
    # 5/50 * 10 = 1.0 -> 1
    assert w[2] == 1


def test_partial_data_skips_missing_gpu():
    # Missing ms (0.0 / negative) should default to weight=1 without affecting
    # the faster GPU's normalization.
    w = calibrate_weights_from_timings({0: 10.0, 1: 0.0, 2: -5.0}, max_weight=8)
    assert w[0] == 8
    assert w[1] == 1
    assert w[2] == 1


def test_no_measurements_returns_all_ones():
    w = calibrate_weights_from_timings({0: 0.0, 1: 0.0}, max_weight=8)
    assert w == {0: 1, 1: 1}


@pytest.mark.parametrize("max_weight", [4, 6, 8, 12])
def test_weights_respect_max_weight_bound(max_weight: int) -> None:
    w = calibrate_weights_from_timings(
        {0: 10.0, 1: 30.0, 2: 60.0}, max_weight=max_weight
    )
    assert max(w.values()) == max_weight
    assert min(w.values()) >= 1
