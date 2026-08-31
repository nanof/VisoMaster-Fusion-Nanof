"""Feeder cache for Global Input Resize must react to size selection, not only the toggle."""

from app.processors.video_processor import VideoProcessor


def test_preview_target_height_none_when_toggle_off():
    assert (
        VideoProcessor._preview_target_height_from_scan_control(
            {
                "GlobalInputResizeToggle": False,
                "GlobalInputResizeSizeSelection": "540p",
            }
        )
        is None
    )


def test_preview_target_height_parses_selection():
    assert (
        VideoProcessor._preview_target_height_from_scan_control(
            {
                "GlobalInputResizeToggle": True,
                "GlobalInputResizeSizeSelection": "540p",
            }
        )
        == 540
    )


def test_feeder_cache_ignores_size_when_unchanged():
    control = {
        "GlobalInputResizeToggle": True,
        "GlobalInputResizeSizeSelection": "720p",
    }
    toggle, size, height, changed = VideoProcessor._refresh_feeder_input_resize_cache(
        control,
        cached_toggle=True,
        cached_size="720p",
        cached_height=720,
    )
    assert (toggle, size, height, changed) == (True, "720p", 720, False)


def test_feeder_cache_invalidates_when_size_changes_with_toggle_on():
    """Regression: playback used to keep 1080p after switching the dropdown to 540p."""
    control = {
        "GlobalInputResizeToggle": True,
        "GlobalInputResizeSizeSelection": "540p",
    }
    toggle, size, height, changed = VideoProcessor._refresh_feeder_input_resize_cache(
        control,
        cached_toggle=True,
        cached_size="1080p",
        cached_height=1080,
    )
    assert toggle is True
    assert size == "540p"
    assert height == 540
    assert changed is True


def test_feeder_cache_invalidates_when_toggle_turns_off():
    control = {
        "GlobalInputResizeToggle": False,
        "GlobalInputResizeSizeSelection": "540p",
    }
    toggle, size, height, changed = VideoProcessor._refresh_feeder_input_resize_cache(
        control,
        cached_toggle=True,
        cached_size="540p",
        cached_height=540,
    )
    assert toggle is False
    assert height is None
    assert changed is True


def test_input_downscale_needed_when_target_below_source():
    assert VideoProcessor._input_downscale_needed_for_source(1920, 1080, 540) is True
    assert VideoProcessor._input_downscale_needed_for_source(1920, 1080, 1080) is False
    assert VideoProcessor._input_downscale_needed_for_source(1280, 720, 1080) is False
    assert VideoProcessor._input_downscale_needed_for_source(1920, 1080, None) is False
