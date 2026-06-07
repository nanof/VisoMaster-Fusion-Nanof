"""Unit tests for Advanced lip motion merge in frame_edits."""

from __future__ import annotations

import pytest
import torch

from app.processors.frame_edits import _merge_lip_motion_candidates


class TestMergeLipMotionCandidates:
    def test_lerp_y_axis_and_absolute_z(self):
        lip_indices = [3, 6, 12]
        relative = torch.zeros(1, 21, 3)
        absolute = torch.zeros(1, 21, 3)

        relative[0, 3, 0] = 0.4
        relative[0, 3, 1] = 0.0
        relative[0, 6, 1] = 0.2

        absolute[0, 3, 0] = 0.1
        absolute[0, 3, 1] = 1.0
        absolute[0, 3, 2] = 0.8
        absolute[0, 6, 1] = 0.6
        absolute[0, 6, 2] = 0.3

        merged = _merge_lip_motion_candidates(relative, absolute, lip_indices)

        assert merged[0, 3, 0] == relative[0, 3, 0]
        assert merged[0, 3, 1].item() == pytest.approx(0.5)
        assert merged[0, 3, 2].item() == pytest.approx(0.8)
        assert merged[0, 6, 1].item() == pytest.approx(0.4)
        assert merged[0, 6, 2].item() == pytest.approx(0.3)
        assert merged[0, 12, 1].item() == pytest.approx(0.0)
        assert merged[0, 12, 2].item() == pytest.approx(0.0)

    def test_unlisted_indices_unchanged(self):
        lip_indices = [3]
        relative = torch.full((1, 21, 3), 0.25)
        absolute = torch.full((1, 21, 3), 0.75)

        merged = _merge_lip_motion_candidates(relative, absolute, lip_indices)

        assert merged[0, 3, 1].item() == pytest.approx(0.5)
        assert merged[0, 3, 2].item() == pytest.approx(0.75)
        assert merged[0, 10, 1].item() == pytest.approx(0.25)
