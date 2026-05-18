"""Regression: GpuWeightsEditor spin min/max must not collapse to 0..0."""

from __future__ import annotations

from app.ui.widgets.widget_components import (
    GpuThreadsPerGpuEditor,
    GpuWeightsEditor,
    _per_gpu_spin_bounds,
)


def test_gpu_weights_editor_bounds():
    assert _per_gpu_spin_bounds(GpuWeightsEditor) == (1, 1, 16)


def test_gpu_threads_per_gpu_editor_bounds():
    assert _per_gpu_spin_bounds(GpuThreadsPerGpuEditor) == (0, 0, 32)
