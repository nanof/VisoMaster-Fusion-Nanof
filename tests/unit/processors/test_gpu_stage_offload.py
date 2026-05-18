from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock

if "einops" not in sys.modules:
    sys.modules["einops"] = MagicMock()
if "omegaconf" not in sys.modules:
    sys.modules["omegaconf"] = MagicMock(OmegaConf=MagicMock())

from app.processors.models_processor import ModelsProcessor


def _stub_mp() -> ModelsProcessor:
    mp = ModelsProcessor.__new__(ModelsProcessor)
    mp.device = "cuda"
    mp.gpu_index = 0
    mp.emulate_multi_gpu = False
    mp.ui_multi_gpu_mode = "stage"
    mp.ui_multi_gpu_routing_enabled = False
    mp.ui_stage_offload_secondary_phys = 1
    mp.ui_offload_face_restorer = True
    mp.ui_offload_frame_enhancer = True
    mp._thread_gpu_context = threading.local()
    return mp


def test_should_offload_stage_requires_stage_mode(monkeypatch):
    mp = _stub_mp()
    monkeypatch.setattr(mp, "_physical_cuda_device_count", lambda: 2)
    monkeypatch.setattr(mp, "_primary_cuda_device_ordinal", lambda: 0)
    assert mp.should_offload_stage("facerestorer", parameters={"FaceRestorerEnableToggle": True})
    mp.ui_multi_gpu_mode = "off"
    assert not mp.should_offload_stage("facerestorer", parameters={"FaceRestorerEnableToggle": True})


def test_gpu_stage_context_restores_thread_gpu(monkeypatch):
    mp = _stub_mp()
    monkeypatch.setattr(mp, "_physical_cuda_device_count", lambda: 2)
    monkeypatch.setattr(mp, "_primary_cuda_device_ordinal", lambda: 0)
    monkeypatch.setattr(mp, "get_stage_offload_secondary_physical", lambda: 1)
    monkeypatch.setattr(mp, "_sync_torch_cuda_device", lambda: None)
    mp.set_thread_gpu_index(0)
    with mp.gpu_stage_context("facerestorer"):
        assert mp.get_compute_cuda_device_id() == 1
        assert mp._is_stage_offload_thread()
    assert mp.get_compute_cuda_device_id() == 0
    assert not mp._is_stage_offload_thread()


def test_ort_session_storage_key_stage_context(monkeypatch):
    mp = _stub_mp()
    monkeypatch.setattr(mp, "_physical_cuda_device_count", lambda: 2)
    monkeypatch.setattr(mp, "_is_thread_cpu_routing", lambda: False)
    monkeypatch.setattr(mp, "get_compute_cuda_device_id", lambda: 1)
    mp._thread_gpu_context.stage_offload_active = "facerestorer"
    assert ModelsProcessor._ort_session_storage_key(mp, "GFPGAN") == "GFPGAN__cuda1"
    del mp._thread_gpu_context.stage_offload_active
    assert ModelsProcessor._ort_session_storage_key(mp, "GFPGAN") == "GFPGAN"
