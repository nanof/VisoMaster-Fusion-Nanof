"""Wiring for MuseTalk FPS opts: compile flag, Whisper on CPU, channels_last."""

from __future__ import annotations

import numpy as np
import pytest

from app.processors.pytorch_extras.musetalk.engine import (
    MuseTalkEngine,
    musetalk_compile_enabled,
    musetalk_perf_enabled,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, False),
        ("", False),
        ("1", True),
        ("true", True),
        ("0", False),
        ("false", False),
        ("off", False),
    ],
)
def test_compile_flag_defaults_off(raw, expected, monkeypatch):
    if raw is None:
        monkeypatch.delenv("VISOFUSION_MUSETALK_COMPILE", raising=False)
    else:
        monkeypatch.setenv("VISOFUSION_MUSETALK_COMPILE", raw)
    assert musetalk_compile_enabled() is expected


def test_compile_flag_reads_ui_toggle_when_env_unset(monkeypatch):
    monkeypatch.delenv("VISOFUSION_MUSETALK_COMPILE", raising=False)
    assert musetalk_compile_enabled({"MuseTalkCompileToggle": True}) is True
    assert musetalk_compile_enabled({"MuseTalkCompileToggle": False}) is False


def test_compile_env_overrides_ui_toggle(monkeypatch):
    monkeypatch.setenv("VISOFUSION_MUSETALK_COMPILE", "0")
    assert musetalk_compile_enabled({"MuseTalkCompileToggle": True}) is False
    monkeypatch.setenv("VISOFUSION_MUSETALK_COMPILE", "1")
    assert musetalk_compile_enabled({"MuseTalkCompileToggle": False}) is True


def test_compile_empty_env_falls_through_to_ui(monkeypatch):
    monkeypatch.setenv("VISOFUSION_MUSETALK_COMPILE", "")
    assert musetalk_compile_enabled({"MuseTalkCompileToggle": True}) is True


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, False),
        ("", False),
        ("1", True),
        ("true", True),
        ("0", False),
    ],
)
def test_perf_flag_defaults_off(raw, expected, monkeypatch):
    if raw is None:
        monkeypatch.delenv("VISOFUSION_MUSETALK_PERF", raising=False)
    else:
        monkeypatch.setenv("VISOFUSION_MUSETALK_PERF", raw)
    assert musetalk_perf_enabled() is expected


def test_load_keeps_whisper_on_cpu(monkeypatch):
    """Realtime path must not park Whisper on the inference GPU."""
    torch = pytest.importorskip("torch")
    import sys
    import types
    import torch.nn as nn

    monkeypatch.setenv("VISOFUSION_MUSETALK_COMPILE", "0")
    monkeypatch.setattr(
        "app.processors.pytorch_extras.musetalk.engine.musetalk_assets_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        "app.processors.pytorch_extras.musetalk.engine.prepare_transformers_env",
        lambda: None,
    )

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def eval(self):
            return self

        def requires_grad_(self, *_a, **_k):
            return self

        @property
        def dtype(self):
            return self.w.dtype

    class _Vae:
        def __init__(self, *a, **k):
            self.vae = _Tiny()
            self.device = torch.device("cpu")

        def set_channels_last(self, enabled=True):
            pass

    class _Unet:
        def __init__(self, *a, **k):
            self.model = _Tiny()

    class _PE(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, x):
            return x

    class _Audio:
        def __init__(self, *a, **k):
            pass

    class _Whisper:
        @staticmethod
        def from_pretrained(_path):
            return _Tiny()

    import app.processors.pytorch_extras.musetalk.models as models_mod

    monkeypatch.setattr(models_mod, "MuseTalkVAE", _Vae)
    monkeypatch.setattr(models_mod, "MuseTalkUNet", _Unet)
    monkeypatch.setattr(models_mod, "PositionalEncoding", _PE)
    monkeypatch.setattr(
        "app.processors.pytorch_extras.musetalk.engine.MuseTalkAudioProcessor",
        _Audio,
    )

    fake_tf = types.ModuleType("transformers")
    fake_tf.WhisperModel = _Whisper
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    eng = MuseTalkEngine()
    assert eng.load(device="cpu", use_float16=False) is True
    try:
        assert next(eng._whisper.parameters()).device.type == "cpu"
        assert eng._compiled is False
    finally:
        eng.unload()


def test_infer_batch_uses_inference_mode_path():
    """Fake GPU-less batch must still complete under inference_mode + no channels_last."""
    torch = pytest.importorskip("torch")
    from app.processors.pytorch_extras.musetalk.engine import _CropRequest

    class _Vae:
        vae = type("_Inner", (), {"dtype": torch.float32})()

        def get_latents_for_unet_batch(self, crops):
            return torch.zeros((len(crops), 8, 32, 32))

        def decode_latents(self, pred):
            return [np.zeros((256, 256, 3), np.uint8) for _ in range(pred.shape[0])]

    class _Unet:
        def __init__(self):
            self.model = self

        dtype = torch.float32

        def __call__(self, latents, _timesteps, encoder_hidden_states=None):
            return type("_Out", (), {"sample": latents})()

    eng = MuseTalkEngine.__new__(MuseTalkEngine)
    eng._vae = _Vae()
    eng._unet = _Unet()
    eng._pe = lambda audio: audio
    eng._whisper_chunks = torch.zeros((4, 50, 384))
    eng._device = torch.device("cpu")
    eng._timesteps = torch.zeros(1)
    eng._channels_last = False
    eng.gpu_capture_lock = None
    batch = [_CropRequest(crop=np.zeros((256, 256, 3), np.uint8), audio_index=0)]
    eng._infer_batch(batch)
    assert batch[0].recon is not None


def test_compile_specs_cover_powers_of_two():
    from app.processors.pytorch_extras.musetalk.engine import _compile_batch_specs

    assert _compile_batch_specs(8) == [1, 2, 4, 8]
    assert _compile_batch_specs(6) == [1, 2, 4, 6]
    assert _compile_batch_specs(1) == [1]


def test_infer_batch_pads_unet_to_a_compiled_shape():
    """An odd batch must reach the compiled UNet padded, or Inductor recompiles."""
    torch = pytest.importorskip("torch")
    from app.processors.pytorch_extras.musetalk.engine import _CropRequest

    seen: list[int] = []

    class _Vae:
        vae = type("_Inner", (), {"dtype": torch.float32})()

        def get_latents_for_unet_batch(self, crops):
            return torch.zeros((len(crops), 8, 32, 32))

        def decode_latents(self, pred):
            return [np.zeros((256, 256, 3), np.uint8) for _ in range(pred.shape[0])]

    class _Unet:
        def __init__(self):
            self.model = self

        dtype = torch.float32

        def __call__(self, latents, _timesteps, encoder_hidden_states=None):
            seen.append(latents.shape[0])
            assert encoder_hidden_states.shape[0] == latents.shape[0]
            return type("_Out", (), {"sample": latents})()

    eng = MuseTalkEngine.__new__(MuseTalkEngine)
    eng._vae = _Vae()
    eng._unet = _Unet()
    eng._pe = lambda audio: audio
    eng._whisper_chunks = torch.zeros((8, 50, 384))
    eng._device = torch.device("cpu")
    eng._timesteps = torch.zeros(1)
    eng._channels_last = False
    eng._batch_specs = [1, 2, 4, 8]
    eng.gpu_capture_lock = None
    batch = [
        _CropRequest(crop=np.zeros((256, 256, 3), np.uint8), audio_index=i)
        for i in range(3)
    ]
    eng._infer_batch(batch)
    assert seen == [4]
    assert all(r.recon is not None for r in batch)


def _padding_engine(encoded: list[int], decoded: list[int], specs: list[int]):
    """Engine stub that records the batch size the VAE encode/decode receive."""
    import torch

    class _Vae:
        vae = type("_Inner", (), {"dtype": torch.float32})()

        def get_latents_for_unet_batch(self, crops):
            encoded.append(len(crops))
            return torch.zeros((len(crops), 8, 32, 32))

        def decode_latents(self, pred):
            decoded.append(pred.shape[0])
            return [np.zeros((256, 256, 3), np.uint8) for _ in range(pred.shape[0])]

    class _Unet:
        def __init__(self):
            self.model = self

        dtype = torch.float32

        def __call__(self, latents, _timesteps, encoder_hidden_states=None):
            return type("_Out", (), {"sample": latents})()

    eng = MuseTalkEngine.__new__(MuseTalkEngine)
    eng._vae = _Vae()
    eng._unet = _Unet()
    eng._pe = lambda audio: audio
    eng._whisper_chunks = torch.zeros((8, 50, 384))
    eng._device = torch.device("cpu")
    eng._timesteps = torch.zeros(1)
    eng._channels_last = False
    eng._batch_specs = specs
    eng.gpu_capture_lock = None
    return eng


def test_infer_batch_pads_the_vae_too():
    """The VAE is compiled per shape as well, so it must not see a raw batch."""
    pytest.importorskip("torch")
    from app.processors.pytorch_extras.musetalk.engine import _CropRequest

    encoded: list[int] = []
    decoded: list[int] = []
    eng = _padding_engine(encoded, decoded, [1, 2, 4, 8])
    batch = [
        _CropRequest(crop=np.zeros((256, 256, 3), np.uint8), audio_index=i)
        for i in range(3)
    ]
    eng._infer_batch(batch)
    assert encoded == [4]
    assert decoded == [4]
    # The padding rows are never handed back to a waiting worker.
    assert sum(r.recon is not None for r in batch) == 3


def test_infer_batch_does_not_pad_when_compile_is_off():
    """Padding is wasted VAE work when nothing is specialised on a shape."""
    pytest.importorskip("torch")
    from app.processors.pytorch_extras.musetalk.engine import _CropRequest

    encoded: list[int] = []
    decoded: list[int] = []
    eng = _padding_engine(encoded, decoded, [])
    batch = [
        _CropRequest(crop=np.zeros((256, 256, 3), np.uint8), audio_index=i)
        for i in range(3)
    ]
    eng._infer_batch(batch)
    assert encoded == [3]
    assert decoded == [3]


def test_infer_batch_stamps_perf_fields_when_enabled(monkeypatch):
    torch = pytest.importorskip("torch")
    from app.processors.pytorch_extras.musetalk.engine import _CropRequest

    monkeypatch.setenv("VISOFUSION_MUSETALK_PERF", "1")

    class _Vae:
        vae = type("_Inner", (), {"dtype": torch.float32})()

        def get_latents_for_unet_batch(self, crops):
            return torch.zeros((len(crops), 8, 32, 32))

        def decode_latents(self, pred):
            return [np.zeros((256, 256, 3), np.uint8) for _ in range(pred.shape[0])]

    class _Unet:
        def __init__(self):
            self.model = self

        dtype = torch.float32

        def __call__(self, latents, _timesteps, encoder_hidden_states=None):
            return type("_Out", (), {"sample": latents})()

    eng = MuseTalkEngine.__new__(MuseTalkEngine)
    eng._vae = _Vae()
    eng._unet = _Unet()
    eng._pe = lambda audio: audio
    eng._whisper_chunks = torch.zeros((4, 50, 384))
    eng._device = torch.device("cpu")
    eng._timesteps = torch.zeros(1)
    eng._channels_last = False
    eng._batch_specs = []
    eng.gpu_capture_lock = None
    batch = [
        _CropRequest(crop=np.zeros((256, 256, 3), np.uint8), audio_index=0),
        _CropRequest(crop=np.zeros((256, 256, 3), np.uint8), audio_index=1),
    ]
    eng._infer_batch(batch)
    assert batch[0].batch_size == 2
    assert batch[0].infer_ms >= 0.0
    assert batch[1].encode_ms == batch[0].encode_ms

