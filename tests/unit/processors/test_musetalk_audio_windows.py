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
    MuseTalkAudioProcessor,
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


@pytest.mark.parametrize("fps", [25, 24, 23.976, 29.97, 30, 60])
def test_every_window_matches_the_eager_version(fps):
    multiplier = 50 / fps
    timeline = _timeline(400)
    frames = 120
    want = _eager(timeline, frames, multiplier, WINDOW)
    view = FrameFeatureWindows(timeline, frames, multiplier, WINDOW)
    assert len(view) == want.shape[0]
    for i in range(len(view)):
        assert torch.equal(view[i : i + 1], want[i : i + 1]), i


def test_whisper_chunking_keeps_fractional_container_fps():
    """A seek deep into 29.97 fps media must not use a rounded 30 fps window."""

    class _Encoder:
        def __call__(self, feature, output_hidden_states=True):
            return type("_Out", (), {"hidden_states": [feature] * STEPS})()

    whisper = type("_Whisper", (), {"encoder": _Encoder()})()
    processor = MuseTalkAudioProcessor.__new__(MuseTalkAudioProcessor)
    duration_s = 10
    fps = 29.97
    chunks = processor.get_whisper_chunk(
        [torch.zeros((1, duration_s * 50 + 20, DEPTH))],
        torch.device("cpu"),
        torch.float32,
        whisper,
        librosa_length=duration_s * 16000,
        fps=fps,
    )

    assert chunks._multiplier == pytest.approx(50 / fps)
    assert len(chunks) == math.floor(duration_s * fps)


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


def test_pcm_segments_stream_without_holding_the_whole_track(tmp_path):
    """Long recordings must not ``librosa.load`` the full WAV into one buffer."""
    import wave

    import numpy as np

    from app.processors.pytorch_extras.musetalk.audio import _iter_pcm16k_segments

    sr = 16000
    # Just over one 30 s boundary so the iterator yields two segments.
    n = sr * 31
    samples = (np.linspace(-0.2, 0.2, n, dtype=np.float32) * 32767.0).astype(np.int16)
    wav = tmp_path / "longish.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(samples.tobytes())

    lengths = []
    totals = []
    for segment, total in _iter_pcm16k_segments(wav, segment_seconds=30):
        lengths.append(int(segment.shape[0]))
        totals.append(int(total))
        assert segment.nbytes <= 30 * sr * 4 + 64

    assert lengths == [30 * sr, sr]
    assert totals == [30 * sr, 31 * sr]


def test_whisper_encode_keeps_host_timeline_in_float16():
    """fp16 halves RAM for feature-length tracks; windows stay numerically close."""

    class _Encoder:
        def __call__(self, feature, output_hidden_states=True):
            return type("_Out", (), {"hidden_states": [feature] * STEPS})()

    whisper = type("_Whisper", (), {"encoder": _Encoder()})()
    processor = MuseTalkAudioProcessor.__new__(MuseTalkAudioProcessor)
    duration_s = 2
    chunks = processor.get_whisper_chunk(
        [torch.zeros((1, duration_s * 50 + 20, DEPTH))],
        torch.device("cpu"),
        torch.float32,
        whisper,
        librosa_length=duration_s * 16000,
        fps=25.0,
    )
    assert chunks.dtype == torch.float16
    assert len(chunks) == duration_s * 25
