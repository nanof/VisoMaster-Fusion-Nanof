"""Unit tests for the deterministic Weighted Round-Robin scheduler."""
from __future__ import annotations

from app.processors.gpu_scheduler import (
    WeightedScheduler,
    calibrate_weights_from_timings,
    distribute_threads_by_weights,
    normalize_mode,
    mode_enables_autotune,
    mode_enables_stealing,
)


def _run(sched: WeightedScheduler, n: int) -> list[int]:
    return [sched.next_gpu() for _ in range(n)]


def test_equal_weights_behave_like_round_robin():
    sched = WeightedScheduler(targets=[0, 1], weights={0: 1, 1: 1})
    seq = _run(sched, 8)
    # DRR picks highest deficit, ties broken by smallest id -> 0,1,0,1,...
    assert seq == [0, 1, 0, 1, 0, 1, 0, 1]


def test_weights_3_1_interleave_pattern():
    sched = WeightedScheduler(targets=[0, 1], weights={0: 3, 1: 1})
    seq = _run(sched, 16)
    # DRR with weights 3:1 produces the cycle ``0,0,1,0`` (tie on id=0) that
    # repeats every 4 picks. This is interleaved (no long burst on either GPU)
    # and maintains the 3:1 ratio over any multiple of 4 picks.
    assert seq == [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    # The slower GPU still receives exactly 1/4 of the frames.
    assert seq.count(1) == 4
    assert seq.count(0) == 12


def test_proportion_holds_over_large_window():
    sched = WeightedScheduler(targets=[0, 1, 2], weights={0: 5, 1: 2, 2: 1})
    seq = _run(sched, 800)
    counts = {0: seq.count(0), 1: seq.count(1), 2: seq.count(2)}
    # Exact shares: 5/8, 2/8, 1/8 of 800 = 500, 200, 100.
    assert counts[0] == 500
    assert counts[1] == 200
    assert counts[2] == 100


def test_update_weights_is_live():
    sched = WeightedScheduler(targets=[0, 1], weights={0: 1, 1: 1})
    assert _run(sched, 4) == [0, 1, 0, 1]
    sched.update_weights({0: 3, 1: 1})
    # After reweight deficits reset so the new schedule starts clean, and the
    # canonical DRR cycle for 3:1 (``0,0,1,0``) takes over.
    assert _run(sched, 8) == [0, 0, 1, 0, 0, 0, 1, 0]


def test_set_targets_adds_missing_with_default_weight():
    sched = WeightedScheduler(targets=[0], weights={})
    sched.set_targets([0, 1, 2], {})
    seq = _run(sched, 6)
    # With default weight=1 each, DRR emits 0,1,2,0,1,2,...
    assert seq == [0, 1, 2, 0, 1, 2]


def test_peek_sequence_does_not_mutate_live_state():
    sched = WeightedScheduler(targets=[0, 1], weights={0: 2, 1: 1})
    peeked = sched.peek_sequence(6)
    live = _run(sched, 6)
    assert peeked == live


def test_empty_targets_fallback_is_safe():
    sched = WeightedScheduler(targets=[], weights={})
    assert sched.next_gpu() == 0


def test_calibrate_weights_proportional_to_inverse_ms():
    # 10 ms vs 30 ms -> fast GPU should receive ~3x more work.
    weights = calibrate_weights_from_timings({0: 10.0, 1: 30.0}, max_weight=8)
    assert weights[0] == 8
    assert 2 <= weights[1] <= 3


def test_calibrate_weights_with_unmeasured_stays_at_one():
    weights = calibrate_weights_from_timings({0: 10.0, 1: 0.0}, max_weight=8)
    assert weights[0] == 8
    assert weights[1] == 1


def test_distribute_threads_proportional_to_weights():
    out = distribute_threads_by_weights(total_threads=4, weights={0: 3, 1: 1}, targets=[0, 1])
    assert sum(out.values()) == 4
    assert out[0] == 3 and out[1] == 1


def test_distribute_threads_min_one_per_gpu():
    # Even with zero requested per slow GPU, we keep at least 1.
    out = distribute_threads_by_weights(total_threads=2, weights={0: 100, 1: 1}, targets=[0, 1])
    assert out[1] >= 1
    assert sum(out.values()) == max(2, len(out))


def test_mode_normalization_and_flags():
    assert normalize_mode("Round-Robin") == "round_robin"
    assert normalize_mode("Weighted Manual") == "weighted_manual"
    assert normalize_mode("Weighted Auto") == "weighted_auto"
    assert normalize_mode("Hybrid") == "hybrid"
    assert normalize_mode(None) == "round_robin"
    assert normalize_mode("unknown-mode") == "round_robin"
    assert mode_enables_stealing("hybrid") is True
    assert mode_enables_stealing("weighted_manual") is False
    assert mode_enables_autotune("weighted_auto") is True
    assert mode_enables_autotune("hybrid") is True
    assert mode_enables_autotune("round_robin") is False
