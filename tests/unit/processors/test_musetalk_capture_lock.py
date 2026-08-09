"""MuseTalk must stand still while another model captures a CUDA graph.

Lip-sync infers on its own batch thread. When the face restorer built its CUDA
graph, that thread kept submitting work to the capturing stream and the driver
refused it ("operation not permitted when stream is capturing"), which then
poisoned the context for the restore itself. The app already serialises captures
through ``ModelsProcessor.cuda_graph_capture_lock``; the engine has to hold the
same lock around its own GPU work for that to protect it too.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.processors.pytorch_extras.musetalk.engine import (  # noqa: E402
    MuseTalkEngine,
    _CropRequest,
)


class _RecordingLock:
    """A lock that remembers whether it was held when the GPU was touched."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.held = False
        self.acquisitions = 0

    def __enter__(self):
        self._lock.acquire()
        self.held = True
        self.acquisitions += 1
        return self

    def __exit__(self, *_exc) -> bool:
        self.held = False
        self._lock.release()
        return False


class _FakeVae:
    def __init__(self, lock: _RecordingLock | None) -> None:
        self._lock = lock
        self.held_during_encode: bool | None = None
        self.vae = type("_Inner", (), {"dtype": torch.float32})()

    def get_latents_for_unet_batch(self, crops):
        self.held_during_encode = self._lock.held if self._lock else None
        return torch.zeros((len(crops), 8, 32, 32))

    def decode_latents(self, pred):
        return [np.zeros((256, 256, 3), np.uint8) for _ in range(pred.shape[0])]


class _FakeUnet:
    def __init__(self) -> None:
        self.model = self

    dtype = torch.float32

    def __call__(self, latents, _timesteps, encoder_hidden_states=None):
        return type("_Out", (), {"sample": latents})()


def _engine(lock: _RecordingLock | None) -> tuple[MuseTalkEngine, _FakeVae]:
    eng = MuseTalkEngine.__new__(MuseTalkEngine)
    vae = _FakeVae(lock)
    eng._vae = vae
    eng._unet = _FakeUnet()
    eng._pe = lambda audio: audio
    eng._whisper_chunks = torch.zeros((4, 50, 384))
    eng._device = torch.device("cpu")
    eng._timesteps = torch.zeros(1)
    eng._bbox_shift = 0
    eng._channels_last = False
    eng.gpu_capture_lock = lock
    return eng, vae


def _batch() -> list[_CropRequest]:
    return [_CropRequest(crop=np.zeros((256, 256, 3), np.uint8), audio_index=0)]


def test_inference_holds_the_capture_lock() -> None:
    lock = _RecordingLock()
    eng, vae = _engine(lock)
    eng._infer_batch(_batch())
    assert vae.held_during_encode is True
    assert lock.acquisitions == 1


def test_a_capture_in_progress_blocks_inference() -> None:
    """The real symptom: work submitted mid-capture. The lock must make us wait."""
    lock = _RecordingLock()
    eng, vae = _engine(lock)
    started = threading.Event()

    with lock:  # stand in for another model capturing its graph
        worker = threading.Thread(target=eng._infer_batch, args=(_batch(),))
        worker.start()
        started.wait(0.2)
        assert vae.held_during_encode is None  # never reached the GPU
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert vae.held_during_encode is True  # ran once the capture released


def test_inference_still_works_without_a_lock_injected() -> None:
    """An engine built outside ModelsProcessor has no lock and must not crash."""
    eng, vae = _engine(None)
    batch = _batch()
    eng._infer_batch(batch)
    assert batch[0].recon is not None
    assert vae.held_during_encode is None
