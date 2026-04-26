from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.processors.video_processor import VideoProcessor


def _mp_ui_multi():
    return SimpleNamespace(
        gpu_index=0,
        device="cuda",
        emulate_multi_gpu=False,
        ui_multi_gpu_routing_enabled=True,
        _physical_cuda_device_count=lambda: 2,
        get_ui_routing_targets_sorted=lambda: [0, 1],
        get_configured_gpu_count=lambda: 2,
        clamp_gpu_index=lambda idx: idx,
    )


def test_resolve_assigned_gpu_index_uses_frame_modulo_when_ui_multi_enabled():
    models_processor = _mp_ui_multi()
    dummy = SimpleNamespace(main_window=SimpleNamespace(models_processor=models_processor))

    assert VideoProcessor._resolve_assigned_gpu_index(dummy, frame_number=5) == 1
    assert VideoProcessor._resolve_assigned_gpu_index(dummy, frame_number=6) == 0


def test_resolve_assigned_gpu_index_always_primary_when_ui_multi_disabled():
    models_processor = SimpleNamespace(
        gpu_index=1,
        device="cuda",
        emulate_multi_gpu=False,
        ui_multi_gpu_routing_enabled=False,
        _physical_cuda_device_count=lambda: 2,
        get_ui_routing_targets_sorted=lambda: [0, 1],
        get_configured_gpu_count=lambda: 2,
        clamp_gpu_index=lambda idx: idx,
    )
    dummy = SimpleNamespace(main_window=SimpleNamespace(models_processor=models_processor))

    assert VideoProcessor._resolve_assigned_gpu_index(dummy, frame_number=5) == 1
    assert VideoProcessor._resolve_assigned_gpu_index(dummy, frame_number=6) == 1


def test_resolve_env_emulate_round_robins_when_ui_multi_disabled():
    models_processor = SimpleNamespace(
        gpu_index=0,
        device="cuda",
        emulate_multi_gpu=True,
        emulated_gpu_count=2,
        ui_multi_gpu_routing_enabled=False,
        _physical_cuda_device_count=lambda: 1,
        get_ui_routing_targets_sorted=lambda: [0],
        get_configured_gpu_count=lambda: 2,
        clamp_gpu_index=lambda idx: max(0, min(int(idx), 1)),
    )
    dummy = SimpleNamespace(main_window=SimpleNamespace(models_processor=models_processor))

    assert VideoProcessor._resolve_assigned_gpu_index(dummy, frame_number=5) == 1
    assert VideoProcessor._resolve_assigned_gpu_index(dummy, frame_number=6) == 0


def test_launch_async_single_frame_worker_passes_assigned_gpu(monkeypatch):
    created = {}

    class _FakeWorker:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.preview_generation = 0

        def start(self):
            return None

    monkeypatch.setattr("app.processors.video_processor.FrameWorker", _FakeWorker)

    dummy = SimpleNamespace(
        main_window=SimpleNamespace(),
        _resolve_assigned_gpu_index=lambda frame_number: 1,
        _log_multi_gpu_route=lambda *_args, **_kwargs: None,
        _current_single_frame_worker=None,
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    worker = VideoProcessor._launch_async_single_frame_worker(dummy, 7, frame, 4)

    assert created["assigned_gpu_index"] == 1
    assert created["frame_number"] == 7
    assert created["worker_id"] == -1
    assert worker.preview_generation == 4
