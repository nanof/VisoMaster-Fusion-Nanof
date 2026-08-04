"""ORT I/O dtype helpers (imports lightweight ort_io_dtype_utils, not models_processor)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.processors.ort_io_dtype_utils import (
    _numpy_scalar_type_to_torch_dtype,
    _ort_warmup_numpy_dtype_for_input,
    resolve_session_io_dtype_maps,
)


class _FakeSession:
    def __init__(self, in_type: str = "tensor(float32)"):
        self._in_type = in_type

    def get_inputs(self):
        return [SimpleNamespace(name="data", type=self._in_type)]

    def get_outputs(self):
        return [SimpleNamespace(name="683", type="tensor(float32)")]


def test_ort_type_string_float16():
    arg = SimpleNamespace(type="tensor(float16)")
    assert _ort_warmup_numpy_dtype_for_input(arg) == np.float16


def test_numpy_scalar_to_torch_float16():
    assert _numpy_scalar_type_to_torch_dtype(np.float16) == torch.float16
    assert _numpy_scalar_type_to_torch_dtype(np.float32) == torch.float32


def test_ort_type_string_bool():
    arg = SimpleNamespace(type="tensor(bool)")
    assert _ort_warmup_numpy_dtype_for_input(arg) == np.bool_
    assert _numpy_scalar_type_to_torch_dtype(np.bool_) == torch.bool


def test_dtype_maps_cached_per_session():
    cache: dict = {}
    session = _FakeSession()
    first = resolve_session_io_dtype_maps(cache, "ArcFace", session)
    assert first[0] == {"data": np.float32}
    assert first[1] == {"683": np.float32}
    assert resolve_session_io_dtype_maps(cache, "ArcFace", session)[0] is first[0]


def test_dtype_maps_survive_unload_while_caller_holds_session():
    """Callers pass the session they already hold; registry unload must not matter."""
    cache: dict = {}
    session = _FakeSession()
    resolve_session_io_dtype_maps(cache, "ArcFace", session)

    # unload_model set the registry entry to None; the worker still holds `session`
    # and passes it explicitly (as face_swappers.recognize now does).
    ins, outs = resolve_session_io_dtype_maps(cache, "ArcFace", session)
    assert ins == {"data": np.float32}
    assert outs == {"683": np.float32}


def test_dtype_maps_reintrospect_after_reload_with_other_export():
    """A reload can swap FP32 for FP16, so cached dtypes must not outlive the session."""
    cache: dict = {}
    resolve_session_io_dtype_maps(cache, "ArcFace", _FakeSession("tensor(float32)"))
    reloaded = _FakeSession("tensor(float16)")
    ins, _ = resolve_session_io_dtype_maps(cache, "ArcFace", reloaded)
    assert ins == {"data": np.float16}


def test_dtype_maps_raise_when_session_is_none():
    with pytest.raises(RuntimeError, match="No ONNX session loaded"):
        resolve_session_io_dtype_maps({}, "ArcFace", None)


def test_dtype_maps_fall_back_to_alive_weakref_when_registry_cleared():
    """Helpers that allocate buffers before re-fetching the session still resolve."""
    cache: dict = {}
    session = _FakeSession()
    resolve_session_io_dtype_maps(cache, "Occluder", session)
    ins, outs = resolve_session_io_dtype_maps(cache, "Occluder", None)
    assert ins == {"data": np.float32}
    assert outs == {"683": np.float32}
