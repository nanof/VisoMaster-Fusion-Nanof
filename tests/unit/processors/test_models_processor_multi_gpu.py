from __future__ import annotations

import os
import sys
import threading
from unittest.mock import MagicMock

if "einops" not in sys.modules:
    sys.modules["einops"] = MagicMock()
if "omegaconf" not in sys.modules:
    sys.modules["omegaconf"] = MagicMock(OmegaConf=MagicMock())

from app.processors.models_processor import ModelsProcessor


def _make_models_processor_stub() -> ModelsProcessor:
    mp = ModelsProcessor.__new__(ModelsProcessor)
    mp.device = "cuda"
    mp.gpu_index = 0
    mp.provider_name = "CUDA"
    mp.emulate_multi_gpu = True
    mp.emulated_gpu_count = 2
    mp._thread_gpu_context = threading.local()
    mp.trt_ep_options = {
        "trt_engine_cache_path": "tensorrt-engines",
        "trt_timing_cache_path": "tensorrt-engines",
        "trt_ep_context_file_path": "tensorrt-engines",
    }
    return mp


def test_set_gpu_index_updates_trt_options_and_provider_device_id(monkeypatch):
    mp = _make_models_processor_stub()
    monkeypatch.setattr(
        "app.processors.models_processor.torch.cuda.is_available", lambda: False
    )

    resolved = mp.set_gpu_index(9, reconfigure_providers=True)

    assert resolved == 1
    assert mp.gpu_index == 1
    assert mp.trt_ep_options["device_id"] == 1
    # ORT TRT EP: engine cache path is relative to trt_ep_context_file_path when dumping ctx.
    assert mp.trt_ep_options["trt_engine_cache_path"] == "engines"
    assert mp.trt_ep_options["trt_timing_cache_path"] == "trt_timing.cache"
    assert mp.trt_ep_options["trt_ep_context_file_path"].endswith(
        f"{os.sep}tensorrt-engines{os.sep}gpu1"
    )
    assert mp.providers[0] == ("CUDAExecutionProvider", {"device_id": 1})


def test_thread_gpu_context_overrides_global_gpu_index():
    mp = _make_models_processor_stub()

    assert mp.get_active_ort_device_id() == 0
    mp.set_thread_gpu_index(1)
    assert mp.get_active_ort_device_id() == 1
    mp.clear_thread_gpu_index()
    assert mp.get_active_ort_device_id() == 0


def test_clamp_gpu_index_respects_physical_gpus_when_routing_targets_span_only_zero(
    monkeypatch,
):
    """Primary GPU must still switch to physical :1 when routing list implies count 1."""
    mp = _make_models_processor_stub()
    mp.emulate_multi_gpu = False
    mp.ui_multi_gpu_routing_enabled = True
    mp.ui_routing_targets = [0]
    mp.gpu_index = 0
    monkeypatch.setattr(
        "app.processors.models_processor.torch.cuda.is_available", lambda: True
    )
    monkeypatch.setattr(
        "app.processors.models_processor.torch.cuda.device_count", lambda: 2
    )

    assert mp.clamp_gpu_index(1) == 1


def test_get_compute_cuda_device_id_clamps_to_available_physical_gpus(monkeypatch):
    """Emulation can assign logical GPU 1 while only one physical CUDA device exists."""
    mp = _make_models_processor_stub()
    monkeypatch.setattr(
        "app.processors.models_processor.torch.cuda.is_available", lambda: True
    )
    monkeypatch.setattr(
        "app.processors.models_processor.torch.cuda.device_count", lambda: 1
    )

    assert mp.get_active_ort_device_id() == 0
    assert mp.get_compute_cuda_device_id() == 0

    mp.set_thread_gpu_index(1)
    assert mp.get_active_ort_device_id() == 1
    assert mp.get_compute_cuda_device_id() == 0


def test_ort_session_storage_key_splits_per_physical_gpu(monkeypatch):
    """Multi-GPU routing + 2 physical devices → distinct ORT session dict keys per CUDA ordinal."""
    mp = ModelsProcessor.__new__(ModelsProcessor)
    mp.device = "cuda"
    mp.ui_multi_gpu_routing_enabled = True
    mp.emulate_multi_gpu = False
    mp.gpu_index = 0
    mp._thread_gpu_context = threading.local()

    monkeypatch.setattr(mp, "_physical_cuda_device_count", lambda: 2)
    monkeypatch.setattr(mp, "_is_thread_cpu_routing", lambda: False)

    monkeypatch.setattr(mp, "get_compute_cuda_device_id", lambda: 0)
    assert ModelsProcessor._ort_session_storage_key(mp, "Inswapper128ArcFace") == (
        "Inswapper128ArcFace__cuda0"
    )

    monkeypatch.setattr(mp, "get_compute_cuda_device_id", lambda: 1)
    assert ModelsProcessor._ort_session_storage_key(mp, "Inswapper128ArcFace") == (
        "Inswapper128ArcFace__cuda1"
    )
