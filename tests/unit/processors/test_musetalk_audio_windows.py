"""Per-frame Whisper windows must be cut lazily, and cut identically.

The eager version copied a window per frame, which is about ten times the audio
itself, so a feature-length video asked CUDA for 11 GiB and lip-sync switched
itself off. The reference implementation of that loop is reproduced here so the
lazy view is pinned to the exact windows it used to produce.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from app.processors.pytorch_extras.musetalk.audio import (  # noqa: E402
    FrameFeatureWindows,
)

WINDOW = 10  # 2 * (2 left + 2 right + 1)
STEPS = 5  # Whisper hidden states
DEPTH = 8  # feature width, shortened for the test


def _timeline(length: int):
    """Distinct values everywhere, so a misaligned window cannot pass unnoticed."""
    n = length * STEPS * DEPTH
    return torch.arange(n, dtype=torch.float32).reshape(1, length, STEPS, DEPTH)


def _eager(timeline, num_frames: int, multiplier: float, window: int):
    """The loop this replaced, kept verbatim as the reference."""
    prompts = []
    for frame_index in range(num_frames):
        start = math.floor(frame_index * multiplier)
        clip = timeline[:, start : start + window]
        if clip.shape[1] != window:
            break
        prompts.append(clip)
    if not prompts:
        return torch.empty(0)
    stacked = torch.cat(prompts, dim=0)
    return stacked.reshape(stacked.shape[0], window * STEPS, DEPTH)


@pytest.mark.parametrize("fps", [25, 24, 30, 60])
def test_every_window_matches_the_eager_version(fps):
    multiplier = 50 / fps
    timeline = _timeline(400)
    frames = 120
    want = _eager(timeline, frames, multiplier, WINDOW)
    view = FrameFeatureWindows(timeline, frames, multiplier, WINDOW)
    assert len(view) == want.shape[0]
    for i in range(len(view)):
        assert torch.equal(view[i : i + 1], want[i : i + 1]), i


def test_the_shape_matches_the_eager_version():
    timeline = _timeline(400)
    want = _eager(timeline, 120, 2.0, WINDOW)
    view = FrameFeatureWindows(timeline, 120, 2.0, WINDOW)
    assert view.shape == tuple(want.shape)


def test_frames_whose_window_runs_off_the_end_are_dropped():
    """The eager loop broke out; the count has to stop in the same place."""
    timeline = _timeline(30)
    multiplier = 2.0
    want = _eager(timeline, 100, multiplier, WINDOW)
    view = FrameFeatureWindows(timeline, 100, multiplier, WINDOW)
    assert len(view) == want.shape[0] == 11


def test_no_per_frame_copies_are_made():
    """The whole point: memory follows the audio, not the frame count."""
    timeline = _timeline(2810)
    view = FrameFeatureWindows(timeline, 1400, 2.0, WINDOW)
    assert view._timeline.data_ptr() == timeline.data_ptr()
    eager_elements = len(view) * WINDOW * STEPS * DEPTH
    # At 25 fps a frame advances 2 steps of the timeline but copies a 10-step
    # window, so the eager stack was five times the timeline it was cut from.
    assert eager_elements == pytest.approx(5 * timeline.numel(), rel=0.01)


def test_a_batch_slice_returns_one_row_per_frame():
    view = FrameFeatureWindows(_timeline(400), 120, 2.0, WINDOW)
    batch = view[7:10]
    assert batch.shape == (3, WINDOW * STEPS, DEPTH)
    assert torch.equal(batch[1:2], view[8:9])


def test_an_empty_slice_keeps_the_row_width():
    view = FrameFeatureWindows(_timeline(400), 120, 2.0, WINDOW)
    empty = view[5:5]
    assert empty.shape == (0, WINDOW * STEPS, DEPTH)
    assert empty.dtype == view.dtype


def test_indexing_one_frame_drops_the_batch_dimension():
    view = FrameFeatureWindows(_timeline(400), 120, 2.0, WINDOW)
    assert view[3].shape == (WINDOW * STEPS, DEPTH)
    assert torch.equal(view[3], view[3:4][0])
    assert torch.equal(view[-1], view[len(view) - 1])


def test_an_out_of_range_frame_is_an_error():
    view = FrameFeatureWindows(_timeline(400), 120, 2.0, WINDOW)
    with pytest.raises(IndexError):
        view[len(view)]


def test_a_track_too_short_for_one_window_holds_no_frames():
    view = FrameFeatureWindows(_timeline(4), 100, 2.0, WINDOW)
    assert len(view) == 0
    assert view.shape[0] == 0


def test_moving_to_a_device_is_forwarded_to_the_timeline():
    """``prepare_audio`` parks the track in system RAM through this path."""
    view = FrameFeatureWindows(_timeline(400), 120, 2.0, WINDOW)
    assert view.to("cpu") is view
    assert view.is_cuda is False
    assert view.dtype == torch.float32
    assert view.to(dtype=torch.float16).dtype == torch.float16
