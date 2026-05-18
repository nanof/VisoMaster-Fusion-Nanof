from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.ui.widgets.actions import control_actions


def test_apply_saved_gpu_settings_clamps_primary_physical(monkeypatch):
    monkeypatch.setattr(
        "app.ui.widgets.actions.gpu_settings_actions._physical_cuda_count",
        lambda: 2,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.gpu_settings_actions.fill_primary_gpu_combo",
        lambda *a, **k: None,
    )

    models_processor = MagicMock()
    models_processor.set_gpu_index = MagicMock(return_value=1)
    models_processor.gpu_index = 0

    combo = MagicMock()
    combo.blockSignals = MagicMock(return_value=None)

    main_window = SimpleNamespace(
        control={"GpuPrimaryPhysicalIndex": "8"},
        models_processor=models_processor,
        parameter_widgets={"GpuPrimaryDeviceSelection": combo},
    )

    control_actions.apply_saved_gpu_index(main_window)

    models_processor.set_gpu_index.assert_called_once_with(1)
    assert main_window.control["GpuPrimaryPhysicalIndex"] == 1


def test_change_gpu_index_reconfigures_provider(monkeypatch):
    monkeypatch.setattr(
        "app.ui.widgets.actions.control_actions.detector_internal_size_ui.sync_detector_internal_size_combo",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.ui.widgets.actions.control_actions.common_widget_actions.update_gpu_memory_progressbar",
        lambda *_args, **_kwargs: None,
    )

    models_processor = SimpleNamespace(
        gpu_index=0,
        provider_name="CUDA",
        clamp_gpu_index=MagicMock(return_value=1),
        set_gpu_index=MagicMock(),
        switch_providers_priority=MagicMock(),
        clear_gpu_memory=MagicMock(),
        face_detectors=SimpleNamespace(clear_declared_input_side_cache=MagicMock()),
    )
    main_window = SimpleNamespace(
        models_processor=models_processor, video_processor=SimpleNamespace(stop_processing=MagicMock())
    )

    control_actions.change_gpu_index(main_window, "1")

    main_window.video_processor.stop_processing.assert_called_once()
    models_processor.set_gpu_index.assert_called_once_with(1)
    models_processor.switch_providers_priority.assert_called_once_with("CUDA")
    models_processor.clear_gpu_memory.assert_called_once()
