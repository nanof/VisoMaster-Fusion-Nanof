"""VRAM bar slot order: primary CUDA device first."""

from __future__ import annotations

from app.ui.widgets.actions.common_actions import _vram_physical_display_order


def test_primary_first_two_gpus():
    assert _vram_physical_display_order(2, 0) == [0, 1]
    assert _vram_physical_display_order(2, 1) == [1, 0]


def test_primary_first_clamps():
    assert _vram_physical_display_order(2, 9) == [1, 0]
