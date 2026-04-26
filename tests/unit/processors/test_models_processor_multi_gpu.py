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
