"""Whisper feature extraction for MuseTalk (adapted from TMElyralab/MuseTalk, MIT)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class FrameFeatureWindows:
    """Per-frame Whisper windows, cut from the timeline only when asked for.

    Materialising one window per frame costs O(frames x window): the timeline runs
    at 50 steps per second, so at 25 fps a frame advances two steps yet copies a
    ten-step window, and the stack comes out five times the audio it was cut from
    — ten times at 50 fps. A feature-length video asked CUDA for 11 GiB that way
    and lip-sync silently switched itself off. Holding the timeline and slicing on
    demand is O(duration), and the engine only ever wants one frame at a time.

    Deliberately mimics the tensor it replaced — ``shape``, ``len`` and slicing —
    so the engine and its batcher did not have to learn about this class.
    """

    def __init__(
        self, timeline: Any, num_frames: int, multiplier: float, window: int
    ) -> None:
        self._timeline = timeline
        self._multiplier = float(multiplier)
        self._window = int(window)
        self._num_frames = self._count_full_windows(int(num_frames))
        steps, depth = timeline.shape[2], timeline.shape[3]
        self.shape = (self._num_frames, self._window * steps, depth)

    def _count_full_windows(self, num_frames: int) -> int:
        """Frames whose window fits, matching what the eager version kept.

        Integer arithmetic only, so counting even a two-hour track is free.
        """
        available = int(self._timeline.shape[1])
        for frame_index in range(max(num_frames, 0)):
            if math.floor(frame_index * self._multiplier) + self._window > available:
                return frame_index
        return max(num_frames, 0)

    def __len__(self) -> int:
        return self._num_frames

    @property
    def dtype(self) -> Any:
        return self._timeline.dtype

    @property
    def is_cuda(self) -> bool:
        return bool(getattr(self._timeline, "is_cuda", False))

    def to(self, *args: Any, **kwargs: Any) -> FrameFeatureWindows:
        self._timeline = self._timeline.to(*args, **kwargs)
        return self

    def _window_at(self, frame_index: int) -> Any:
        start = math.floor(int(frame_index) * self._multiplier)
        clip = self._timeline[:, start : start + self._window]
        # Same as the old ``rearrange(x, "b c h w -> b (c h) w")``.
        return clip.reshape(1, self.shape[1], self.shape[2])

    def __getitem__(self, key: Any) -> Any:
        import torch

        if isinstance(key, slice):
            wanted = range(*key.indices(self._num_frames))
            if not wanted:
                return self._timeline.new_zeros((0, self.shape[1], self.shape[2]))
            return torch.cat([self._window_at(i) for i in wanted], dim=0)
        index = int(key)
        if index < 0:
            index += self._num_frames
        if not 0 <= index < self._num_frames:
            raise IndexError(index)
        return self._window_at(index)[0]


class MuseTalkAudioProcessor:
    def __init__(self, feature_extractor_path: str | Path) -> None:
        from app.processors.pytorch_extras.musetalk.paths import (
            prepare_transformers_env,
        )

        prepare_transformers_env()
        from transformers import AutoFeatureExtractor

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(feature_extractor_path)
        )

    def get_audio_feature(
        self, wav_path: str | Path, weight_dtype: Any = None
    ) -> tuple[list[Any], int] | tuple[None, int]:
        import librosa

        wav_path = Path(wav_path)
        if not wav_path.is_file():
            return None, 0
        librosa_output, sampling_rate = librosa.load(str(wav_path), sr=16000)
        assert sampling_rate == 16000
        segment_length = 30 * sampling_rate
        segments = [
            librosa_output[i : i + segment_length]
            for i in range(0, len(librosa_output), segment_length)
        ]
        features: list[Any] = []
        for segment in segments:
            audio_feature = self.feature_extractor(
                segment, return_tensors="pt", sampling_rate=sampling_rate
            ).input_features
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)
        return features, int(len(librosa_output))

    def get_whisper_chunk(
        self,
        whisper_input_features: list[Any],
        device: Any,
        weight_dtype: Any,
        whisper: Any,
        librosa_length: int,
        fps: float = 25.0,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ) -> Any:
        import torch

        audio_feature_length_per_frame = 2 * (
            audio_padding_length_left + audio_padding_length_right + 1
        )
        whisper_feature_parts: list[Any] = []
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(device).to(weight_dtype)
            audio_feats = whisper.encoder(
                input_feature, output_hidden_states=True
            ).hidden_states
            # Straight to the host: only one 30-second segment needs to be resident
            # on the device, whereas joining the whole track there is what ran the
            # GPU out of memory on long videos.
            whisper_feature_parts.append(torch.stack(audio_feats, dim=2).to("cpu"))

        whisper_feature = torch.cat(whisper_feature_parts, dim=1)
        sr = 16000
        audio_fps = 50
        fps_i = max(1, int(round(float(fps))))
        whisper_idx_multiplier = audio_fps / fps_i
        num_frames = math.floor((librosa_length / sr) * fps_i)
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        whisper_feature = whisper_feature[:, :actual_length, ...]

        padding_nums = math.ceil(whisper_idx_multiplier)
        whisper_feature = torch.cat(
            [
                torch.zeros_like(
                    whisper_feature[:, : padding_nums * audio_padding_length_left]
                ),
                whisper_feature,
                torch.zeros_like(
                    whisper_feature[:, : padding_nums * 3 * audio_padding_length_right]
                ),
            ],
            1,
        )

        return FrameFeatureWindows(
            whisper_feature,
            num_frames,
            whisper_idx_multiplier,
            audio_feature_length_per_frame,
        )


def extract_wav_from_media(
    media_path: str | Path, out_wav: str | Path, *, sample_rate: int = 16000
) -> Path:
    """Extract mono PCM WAV via ffmpeg (must be on PATH)."""
    import subprocess

    media_path = Path(media_path)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        if "does not contain any stream" in err or "Output file #0 does not" in err:
            raise RuntimeError(f"'{media_path.name}' has no audio track")
        raise RuntimeError(
            f"ffmpeg could not extract audio from '{media_path.name}': {err}"
        )
    return out_wav
