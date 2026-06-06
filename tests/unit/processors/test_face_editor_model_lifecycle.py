"""LivePortrait load/unload bookkeeping (VRAM leak regressions)."""

from __future__ import annotations

from app.processors.face_editors import FaceEditors


class _FakeModelsProcessor:
    def __init__(self):
        self.device = "cpu"
        self.models: dict[str, object] = {}
        self.models_trt: dict[str, object] = {}
        self.unloaded: list[str] = []

    def _ort_session_storage_key(self, model_name: str) -> str:
        return model_name

    def is_model_loaded(self, model_name: str) -> bool:
        key = self._ort_session_storage_key(model_name)
        return bool(self.models.get(key)) or self.models_trt.get(key) is not None

    def unload_model(self, model_name: str) -> None:
        self.unloaded.append(model_name)
        key = self._ort_session_storage_key(model_name)
        self.models.pop(key, None)
        self.models_trt.pop(key, None)


def test_unload_models_clears_sessions_even_without_editor_type_flag():
  mp = _FakeModelsProcessor()
  mp.models["LivePortraitMotionExtractor"] = object()
  mp.models_trt["LivePortraitStitching"] = object()
  editors = FaceEditors(mp)  # type: ignore[arg-type]
  editors.current_face_editor_type = None

  editors.unload_models()

  assert editors.current_face_editor_type is None
  assert not editors.are_models_loaded()
  assert "LivePortraitMotionExtractor" in mp.unloaded
  assert "LivePortraitStitching" in mp.unloaded


def test_are_models_loaded_detects_trt_native_sessions():
  mp = _FakeModelsProcessor()
  mp.models_trt["LivePortraitWarpingSpade"] = object()
  editors = FaceEditors(mp)  # type: ignore[arg-type]

  assert editors.are_models_loaded()

