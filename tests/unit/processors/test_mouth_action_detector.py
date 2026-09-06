"""Tests for MouthActionDetector ONNX Runtime path."""

from __future__ import annotations

import builtins
import os

import numpy as np
import pytest

from app.processors.mouth_action_detector import MouthActionDetector, _MODEL_PATH


def teardown_function() -> None:
    MouthActionDetector.unload()


def test_get_caches_failed_singleton_without_retry(monkeypatch):
    """A broken load must not be re-attempted on every frame."""
    MouthActionDetector._instance = None
    attempts = {"n": 0}

    def boom(self):
        attempts["n"] += 1
        raise AttributeError("onnxruntime boom")

    monkeypatch.setattr(MouthActionDetector, "_lazy_load", boom)

    a = MouthActionDetector.get()
    b = MouthActionDetector.get()

    assert a is b
    assert attempts["n"] == 1
    assert a.available is False
    assert a.load_error is not None
    assert "onnxruntime boom" in a.load_error

    MouthActionDetector._instance = None


def test_lazy_load_records_onnxruntime_import_failure(monkeypatch):
    import sys

    MouthActionDetector._instance = None
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)

    det = MouthActionDetector.get()
    assert det.available is False
    assert det.load_error is not None
    assert "onnxruntime is not installed" in det.load_error

    MouthActionDetector._instance = None


def test_providers_default_to_cpu(monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.delenv("VISOMASTER_MOUTH_ACTION_PROVIDER", raising=False)
    assert MouthActionDetector._providers() == ["CPUExecutionProvider"]


@pytest.mark.skipif(not os.path.isfile(_MODEL_PATH), reason="model.onnx not present")
def test_mouth_action_detector_loads_onnx_model(monkeypatch) -> None:
    monkeypatch.setenv("VISOMASTER_MOUTH_ACTION_PROVIDER", "cpu")
    detector = MouthActionDetector.get()

    assert detector.available, detector.load_error
    assert detector._input_name == "image_tensor:0"
    assert detector._boxes_name == "detected_boxes:0"
    assert detector._scores_name == "detected_scores:0"
    assert detector._classes_name == "detected_classes:0"


@pytest.mark.skipif(not os.path.isfile(_MODEL_PATH), reason="model.onnx not present")
def test_mouth_action_detector_score_returns_probability(monkeypatch) -> None:
    monkeypatch.setenv("VISOMASTER_MOUTH_ACTION_PROVIDER", "cpu")
    detector = MouthActionDetector.get()
    frame: np.ndarray = np.zeros((3, 320, 320), dtype=np.uint8)

    score = detector.score(frame)

    assert 0.0 <= score <= 1.0


@pytest.mark.skipif(not os.path.isfile(_MODEL_PATH), reason="model.onnx not present")
def test_mouth_action_detector_unload_clears_singleton(monkeypatch) -> None:
    monkeypatch.setenv("VISOMASTER_MOUTH_ACTION_PROVIDER", "cpu")
    first = MouthActionDetector.get()
    assert first.available

    MouthActionDetector.unload()

    second = MouthActionDetector.get()
    assert second is not first
    assert second.available
