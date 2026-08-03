"""Tests for MouthActionDetector TensorFlow import hardening."""

from __future__ import annotations

import builtins
import types


def test_tensorflow_import_guard_restores_feature_imported(monkeypatch):
    from app.processors import mouth_action_detector as mad
    import sys

    calls = {"n": 0}

    def _orig(*_a, **_k):
        calls["n"] += 1
        return "orig"

    fake_feat = types.ModuleType("shibokensupport.feature")
    fake_feat.feature_imported = _orig
    fake_loader = types.ModuleType("shibokensupport.signature.loader")
    fake_loader.feature_imported = _orig
    monkeypatch.setitem(
        sys.modules, "shibokensupport", types.ModuleType("shibokensupport")
    )
    monkeypatch.setitem(
        sys.modules,
        "shibokensupport.signature",
        types.ModuleType("shibokensupport.signature"),
    )
    monkeypatch.setitem(sys.modules, "shibokensupport.feature", fake_feat)
    monkeypatch.setitem(sys.modules, "shibokensupport.signature.loader", fake_loader)

    with mad._tensorflow_import_guard():
        assert fake_feat.feature_imported is not _orig
        assert fake_loader.feature_imported is not _orig
        assert fake_feat.feature_imported() is None

    assert fake_feat.feature_imported is _orig
    assert fake_loader.feature_imported is _orig
    assert calls["n"] == 0
    assert fake_feat.feature_imported() == "orig"


def test_get_caches_failed_singleton_without_retry(monkeypatch):
    """A broken TF import must not be re-attempted on every frame."""
    from app.processors.mouth_action_detector import MouthActionDetector

    MouthActionDetector._instance = None
    attempts = {"n": 0}

    def boom(self):
        attempts["n"] += 1
        raise AttributeError("_SixMetaPathImporter object has no attribute '_path'")

    monkeypatch.setattr(MouthActionDetector, "_lazy_load", boom)

    a = MouthActionDetector.get()
    b = MouthActionDetector.get()

    assert a is b
    assert attempts["n"] == 1
    assert a.available is False
    assert a.load_error is not None
    assert "_SixMetaPathImporter" in a.load_error

    MouthActionDetector._instance = None


def test_lazy_load_records_generic_import_failure(monkeypatch):
    from app.processors.mouth_action_detector import MouthActionDetector

    MouthActionDetector._instance = None
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise AttributeError("_SixMetaPathImporter object has no attribute '_path'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    det = MouthActionDetector.get()
    assert det.available is False
    assert det.load_error is not None
    assert "tensorflow import failed" in det.load_error

    MouthActionDetector._instance = None
