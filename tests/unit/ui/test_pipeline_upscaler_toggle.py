"""Pipeline upscaler toggle must act on the new value, not the stored one."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("PySide6")

from app.ui.widgets.actions import layout_actions  # noqa: F401,E402
from app.ui.widgets.actions import control_actions  # noqa: E402


class _ModelsProcessorStub:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []
        self.pipeline_upscaler_engine = None

    def ensure_pipeline_upscaler_loaded(self, model_type: str | None = None) -> bool:
        self.calls.append(("load", model_type))
        return False

    def unload_pipeline_upscaler(self) -> None:
        self.calls.append(("unload", None))


def _main_window(*, enabled: bool, model: str = "HYPIR SD2"):
    return types.SimpleNamespace(
        control={
            "PipelineUpscalerEnableToggle": enabled,
            "PipelineUpscalerTypeSelection": model,
            "nThreadsSlider": 1,
        },
        models_processor=_ModelsProcessorStub(),
        video_processor=None,
    )


def test_enabling_loads_even_though_control_still_holds_false():
    main_window = _main_window(enabled=False)
    control_actions.handle_pipeline_upscaler_toggle_change(
        main_window, True, "PipelineUpscalerEnableToggle"
    )
    assert main_window.models_processor.calls == [("load", "HYPIR SD2")]


def test_disabling_unloads_even_though_control_still_holds_true():
    main_window = _main_window(enabled=True)
    control_actions.handle_pipeline_upscaler_toggle_change(
        main_window, False, "PipelineUpscalerEnableToggle"
    )
    assert main_window.models_processor.calls == [("unload", None)]


def test_type_change_reloads_when_enabled():
    main_window = _main_window(enabled=True, model="HYPIR SD2")
    control_actions.handle_pipeline_upscaler_type_change(
        main_window, "SeedVR2 3B", "PipelineUpscalerTypeSelection"
    )
    assert main_window.models_processor.calls == [
        ("unload", None),
        ("load", "SeedVR2 3B"),
    ]


def test_type_change_ignored_when_disabled():
    main_window = _main_window(enabled=False)
    control_actions.handle_pipeline_upscaler_type_change(
        main_window, "SeedVR2 3B", "PipelineUpscalerTypeSelection"
    )
    assert main_window.models_processor.calls == []
