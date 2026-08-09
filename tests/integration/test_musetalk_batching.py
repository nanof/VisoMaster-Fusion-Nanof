"""Batched MuseTalk inference must keep each frame paired with its own audio.

The engine coalesces concurrent ``apply_frame_bgr`` calls from the frame workers
into a single VAE/UNet pass. If the batch were assembled or scattered in the
wrong order, every frame would still change (so a "did it change?" check passes)
while being driven by another frame's audio.

Needs a CUDA device and the downloaded MuseTalk weights; skipped otherwise.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from app.processors.pytorch_extras.musetalk.paths import musetalk_assets_ready

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.integration]

INDICES = [0, 7, 14, 21, 28, 35]


def _require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    _require_cuda()
    if not musetalk_assets_ready():
        pytest.skip("MuseTalk weights not downloaded")

    from app.processors.pytorch_extras.musetalk import MuseTalkEngine

    wav = tmp_path_factory.mktemp("musetalk") / "tone.wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=220:duration=5",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg unavailable")

    eng = MuseTalkEngine()
    if not eng.load():
        pytest.skip("MuseTalk failed to load")
    if not eng.prepare_audio(Path(wav), fps=25.0):
        eng.unload()
        pytest.skip("could not build audio features")
    yield eng
    eng.unload()


def test_batched_frames_keep_their_own_audio(engine):
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3)).astype(np.uint8)
    bbox = [220, 100, 420, 340]

    # One at a time: nothing to coalesce with, so these are per-frame references.
    reference = {
        i: engine.apply_frame_bgr(frame, i, [bbox]).astype(np.int32) for i in INDICES
    }

    # Submitted together, which is what forces real batches.
    batched: dict[int, np.ndarray] = {}
    lock = threading.Lock()

    def submit(index: int) -> None:
        out = engine.apply_frame_bgr(frame, index, [bbox]).astype(np.int32)
        with lock:
            batched[index] = out

    threads = [threading.Thread(target=submit, args=(i,)) for i in INDICES]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert set(batched) == set(INDICES)
    for index in INDICES:
        distances = {
            other: int(np.abs(batched[index] - reference[other]).sum())
            for other in INDICES
        }
        closest = min(distances, key=lambda k: distances[k])
        assert closest == index, (
            f"frame {index} matched frame {closest}'s reference: {distances}"
        )
