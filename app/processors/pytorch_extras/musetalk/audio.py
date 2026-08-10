"""Whisper feature extraction for MuseTalk (adapted from TMElyralab/MuseTalk, MIT)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator


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


def _iter_pcm16k_segments(
    wav_path: Path, *, segment_seconds: int = 30, sample_rate: int = 16000
) -> Iterator[tuple[Any, int]]:
    """Yield (mono float32 PCM @ 16 kHz, sample_count_so_far) without holding the track.

    Long recordings used to ``librosa.load`` the whole file, then keep every 30 s
    mel feature in a list before Whisper ran. Peak RAM tracked the full duration
    twice. Streaming one segment at a time keeps the peak near a single chunk.
    """
    import numpy as np

    # Prefer soundfile when present: block-reads native 16 kHz mono without a
    # resample of the whole file. Falls through for other rates/channels.
    try:
        import soundfile as sf

        info = sf.info(str(wav_path))
        if int(info.samplerate) == sample_rate and info.channels == 1:
            total = 0
            with sf.SoundFile(str(wav_path)) as handle:
                frames = segment_seconds * sample_rate
                while True:
                    block = handle.read(frames, dtype="float32", always_2d=False)
                    if block is None or len(block) == 0:
                        break
                    segment = np.asarray(block, dtype=np.float32).reshape(-1)
                    total += int(segment.shape[0])
                    yield segment, total
            return
        duration_s = float(info.duration)
    except Exception:
        duration_s = None

    # Stdlib path for 16-bit mono PCM at the target rate (unit tests / no deps).
    if duration_s is None:
        try:
            import wave

            with wave.open(str(wav_path), "rb") as handle:
                if (
                    handle.getnchannels() == 1
                    and handle.getsampwidth() == 2
                    and handle.getframerate() == sample_rate
                ):
                    frames_per_seg = segment_seconds * sample_rate
                    total = 0
                    while True:
                        raw = handle.readframes(frames_per_seg)
                        if not raw:
                            break
                        segment = (
                            np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                        )
                        total += int(segment.shape[0])
                        yield segment, total
                    return
                duration_s = handle.getnframes() / float(handle.getframerate())
        except Exception:
            duration_s = None

    import librosa

    if duration_s is None:
        duration_s = float(librosa.get_duration(path=str(wav_path)))

    # Other rates/channels: librosa resamples one segment at a time via offset.
    offset = 0.0
    total = 0
    while offset < duration_s - 1e-6:
        segment, sr = librosa.load(
            str(wav_path),
            sr=sample_rate,
            mono=True,
            offset=offset,
            duration=float(segment_seconds),
        )
        assert sr == sample_rate
        if segment is None or len(segment) == 0:
            break
        segment = np.asarray(segment, dtype=np.float32).reshape(-1)
        total += int(segment.shape[0])
        yield segment, total
        offset += float(segment_seconds)


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
        """Extract mel features in 30 s segments (compat path for callers/tests)."""
        wav_path = Path(wav_path)
        if not wav_path.is_file():
            return None, 0
        features: list[Any] = []
        librosa_length = 0
        for segment, total in _iter_pcm16k_segments(wav_path):
            librosa_length = total
            audio_feature = self.feature_extractor(
                segment, return_tensors="pt", sampling_rate=16000
            ).input_features
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)
        if librosa_length <= 0:
            return None, 0
        return features, int(librosa_length)

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
        return self._windows_from_encoder_parts(
            self._encode_whisper_parts(
                whisper_input_features, device, weight_dtype, whisper
            ),
            librosa_length=librosa_length,
            fps=fps,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
        )

    def build_whisper_windows(
        self,
        wav_path: str | Path,
        device: Any,
        weight_dtype: Any,
        whisper: Any,
        fps: float = 25.0,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ) -> Any | None:
        """Stream the WAV, encode Whisper one segment at a time, return lazy windows.

        Preferred over ``get_audio_feature`` + ``get_whisper_chunk`` for long
        recordings: mel features are freed after each encode, and the timeline is
        kept as float16 in system RAM.
        """
        wav_path = Path(wav_path)
        if not wav_path.is_file():
            return None
        parts: list[Any] = []
        librosa_length = 0
        for segment, total in _iter_pcm16k_segments(wav_path):
            librosa_length = total
            mel = self.feature_extractor(
                segment, return_tensors="pt", sampling_rate=16000
            ).input_features
            if weight_dtype is not None:
                mel = mel.to(dtype=weight_dtype)
            parts.append(
                self._encode_one_whisper_part(mel, device, weight_dtype, whisper)
            )
            del mel
        if librosa_length <= 0 or not parts:
            return None
        return self._windows_from_encoder_parts(
            parts,
            librosa_length=librosa_length,
            fps=fps,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
        )

    def _encode_whisper_parts(
        self,
        whisper_input_features: list[Any],
        device: Any,
        weight_dtype: Any,
        whisper: Any,
    ) -> list[Any]:
        parts: list[Any] = []
        for input_feature in whisper_input_features:
            parts.append(
                self._encode_one_whisper_part(
                    input_feature, device, weight_dtype, whisper
                )
            )
        return parts

    @staticmethod
    def _encode_one_whisper_part(
        input_feature: Any, device: Any, weight_dtype: Any, whisper: Any
    ) -> Any:
        import torch

        feature = input_feature.to(device).to(weight_dtype)
        audio_feats = whisper.encoder(feature, output_hidden_states=True).hidden_states
        # Straight to the host as float16: only one 30-second segment needs to be
        # resident on the device, and a long timeline at fp32 is ~2× the RAM of the
        # half-precision copy apply_frame_bgr casts from anyway.
        stacked = torch.stack(audio_feats, dim=2).to(device="cpu", dtype=torch.float16)
        del feature, audio_feats
        return stacked

    def _windows_from_encoder_parts(
        self,
        whisper_feature_parts: list[Any],
        *,
        librosa_length: int,
        fps: float,
        audio_padding_length_left: int,
        audio_padding_length_right: int,
    ) -> Any:
        import torch

        audio_feature_length_per_frame = 2 * (
            audio_padding_length_left + audio_padding_length_right + 1
        )
        whisper_feature = torch.cat(whisper_feature_parts, dim=1)
        del whisper_feature_parts
        sr = 16000
        audio_fps = 50
        # Keep the container's exact rate. Rounding 23.976/29.97 to an integer
        # looks harmless near frame zero but drifts several seconds over a long
        # clip, which becomes especially visible after seeking deep into it.
        fps_f = max(1.0, float(fps))
        whisper_idx_multiplier = audio_fps / fps_f
        num_frames = math.floor((librosa_length / sr) * fps_f)
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
