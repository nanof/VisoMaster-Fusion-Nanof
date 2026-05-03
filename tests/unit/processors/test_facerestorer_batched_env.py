"""PERF-008: env gating for batched primary face restorer (no ORT session required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture()
def mock_mp(monkeypatch):
    mp = MagicMock()
    mp.provider_name = "CUDAExecutionProvider"
    mp.device = "cuda"
    mp.get_effective_torch_device.return_value = "cpu"
    return mp


def test_restorer_ort_batch_disabled_by_env(mock_mp, monkeypatch):
    monkeypatch.setenv("VISIOMASTER_RESTORER_ORT_BATCH", "0")
    from app.processors.face_restorers import FaceRestorers

    fr = FaceRestorers(mock_mp)
    assert fr.restorer_ort_batched_attempt_enabled() is False


def test_try_apply_returns_none_for_batch_one(mock_mp, monkeypatch):
    monkeypatch.delenv("VISIOMASTER_RESTORER_ORT_BATCH", raising=False)
    from app.processors.face_restorers import FaceRestorers

    fr = FaceRestorers(mock_mp)
    x = torch.zeros(1, 3, 64, 64)
    assert fr.try_apply_facerestorer_batched_original_stack(x, "GFPGAN-v1.4", 0.9) is None


def test_try_apply_returns_none_for_unsupported_type(mock_mp, monkeypatch):
    monkeypatch.delenv("VISIOMASTER_RESTORER_ORT_BATCH", raising=False)
    from app.processors.face_restorers import FaceRestorers

    fr = FaceRestorers(mock_mp)
    x = torch.zeros(2, 3, 64, 64)
    assert fr.try_apply_facerestorer_batched_original_stack(x, "VQFR-v2", 0.5) is None
