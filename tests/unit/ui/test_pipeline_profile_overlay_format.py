"""Pipeline profile overlay text: multithread vs global mean column."""

from app.ui.widgets.actions.pipeline_profile_actions import (
    PIPELINE_PROFILE_FRAME_TOTAL_KEY,
    format_profile_overlay_multithread,
)


def test_format_profile_overlay_multithread_shows_per_thread_columns() -> None:
    per_thread = {
        "FrameWorker-Pool-0": {"read_frame_ms": 2.0, "std_swap_edit": 10.0},
        "FrameWorker-Pool-1": {"read_frame_ms": 4.0, "std_swap_edit": 20.0},
    }
    text = format_profile_overlay_multithread(per_thread, global_mean_column=False)
    assert "W0" in text
    assert "W1" in text
    assert "Avg" in text
    assert "mean across workers" not in text


def test_format_profile_overlay_global_mean_single_column() -> None:
    per_thread = {
        "FrameWorker-Pool-0": {"read_frame_ms": 2.0, "std_swap_edit": 10.0},
        "FrameWorker-Pool-1": {"read_frame_ms": 4.0, "std_swap_edit": 20.0},
    }
    text = format_profile_overlay_multithread(per_thread, global_mean_column=True)
    assert "mean across workers" in text
    assert "Mean" in text
    assert "W0" not in text
    assert "W1" not in text
    assert "Avg" not in text
    # read_frame mean (2+4)/2 = 3.0
    assert "  3.0" in text


def test_format_profile_overlay_frame_total_row_and_breakdown_total() -> None:
    per_thread = {
        "FrameWorker-Pool-0": {
            "read_frame_ms": 2.0,
            "std_swap_edit": 10.0,
            PIPELINE_PROFILE_FRAME_TOTAL_KEY: 25.0,
        },
    }
    text = format_profile_overlay_multithread(per_thread, global_mean_column=False)
    assert "Frame total (feeder ∑ + worker wall)" in text
    assert "  25.0" in text
    assert "Total (breakdown ∑)" in text
    # Breakdown sum excludes frame total key: 2 + 10 = 12
    assert "  12.0" in text
