"""The probe is the instrument we judge lip-sync by, so it must not lie.

Its verdict used to come from the mean change across the whole crop. That number
falls as the mask gets tighter and as the frame gets larger, so switching to the
anatomical mask on 1080p footage made a working repaint report "NO CHANGE".
"""

from __future__ import annotations

import numpy as np

from app.processors.pytorch_extras.musetalk.engine import _EffectProbe

BOX = (0, 0, 200, 260)


def _probe() -> _EffectProbe:
    return _EffectProbe(dump_dir=None, dump_frames=0)


def _pair(mouth_value: int, area: tuple[slice, slice] | None = None):
    before = np.full((260, 200, 3), 120, dtype=np.uint8)
    after = before.copy()
    if area is None:
        area = (slice(150, 200), slice(70, 130))
    after[area] = mouth_value
    return before, after


def _verdict(probe: _EffectProbe) -> str:
    cov = sum(probe._coverage) / len(probe._coverage)
    return "touching" if cov >= probe._COVERAGE_FLOOR else "not touching"


def test_a_small_but_strong_repaint_counts_as_touching_the_frame():
    """A tight mask on a big crop: 1.4% of the area, which the old rule missed."""
    probe = _probe()
    before, after = _pair(200)
    probe.record(before, after, BOX, 0)
    assert _verdict(probe) == "touching"


def test_the_reported_change_is_measured_inside_the_repaint():
    """80 units over the repainted pixels, not 80 diluted by the untouched crop."""
    probe = _probe()
    before, after = _pair(200)  # 120 -> 200
    probe.record(before, after, BOX, 0)
    assert probe._deltas[0] == 80.0


def test_the_reported_change_does_not_depend_on_mask_size():
    """The same repaint strength must read the same through a tighter mask."""
    wide = _probe()
    wide.record(*_pair(200, (slice(120, 240), slice(20, 180))), BOX, 0)
    tight = _probe()
    tight.record(*_pair(200, (slice(180, 200), slice(90, 110))), BOX, 0)
    assert wide._deltas[0] == tight._deltas[0]
    assert wide._coverage[0] > tight._coverage[0]


def test_an_untouched_frame_is_reported_as_untouched():
    probe = _probe()
    before = np.full((260, 200, 3), 120, dtype=np.uint8)
    probe.record(before, before.copy(), BOX, 0)
    assert _verdict(probe) == "not touching"
    assert probe._deltas[0] == 0.0


def test_imperceptible_noise_is_not_counted_as_a_repaint():
    """A rounding difference of one or two levels is not lip-sync working."""
    probe = _probe()
    before = np.full((260, 200, 3), 120, dtype=np.uint8)
    after = before.copy()
    after[150:200, 70:130] = 122
    probe.record(before, after, BOX, 0)
    assert _verdict(probe) == "not touching"


def test_coverage_is_a_fraction_of_the_crop():
    probe = _probe()
    probe.record(*_pair(200), BOX, 0)
    expected = (50 * 60) / (260 * 200)
    assert probe._coverage[0] == expected


def test_the_report_states_both_the_strength_and_the_area(capsys):
    probe = _probe()
    for i in range(probe.REPORT_EVERY):
        probe.record(*_pair(200), BOX, i)
    out = capsys.readouterr().out
    assert "inside the repaint" in out
    assert "% of the crop" in out
    assert "NO CHANGE" not in out
