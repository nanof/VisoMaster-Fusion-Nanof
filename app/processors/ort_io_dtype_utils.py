"""ORT declared I/O dtype helpers (lightweight — no models_processor import chain)."""

from __future__ import annotations

import weakref
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

SessionIoDtypeCache = Dict[
    str,
    Tuple[Dict[str, type], Dict[str, type], Optional["weakref.ReferenceType"]],
]


def _ort_warmup_numpy_dtype_for_input(inp) -> type:
    """Match ORT declared input element type for session.run warmup (avoids float32 vs float16 mismatch)."""
    ty = (getattr(inp, "type", "") or "").lower()
    if "float16" in ty:
        return np.float16
    if "uint8" in ty:
        return np.uint8
    if "int64" in ty:
        return np.int64
    if "int32" in ty:
        return np.int32
    if "bool" in ty:
        return np.bool_
    return np.float32


def session_declared_numpy_dtype_for_name(
    session, tensor_name: str, *, is_output: bool
) -> type:
    """Declared element type for an ONNX I/O name (for sessions not registered in ModelsProcessor)."""
    items = session.get_outputs() if is_output else session.get_inputs()
    for arg in items:
        if arg.name == tensor_name:
            return _ort_warmup_numpy_dtype_for_input(arg)
    return np.float32


def resolve_session_io_dtype_maps(
    cache: SessionIoDtypeCache, model_name: str, session: Any
) -> Tuple[Dict[str, type], Dict[str, type]]:
    """``(input name -> np type, output name -> np type)`` for *session*, cached by identity.

    Callers that already hold an InferenceSession (typical after ``get_onnx_session`` /
    ``load_model``) should pass that object. Looking the session up again by name races
    with ``unload_model`` clearing the registry mid-frame.

    If *session* is ``None`` (registry cleared) but a prior call cached maps whose weakref
    is still alive — another worker still holding that session — reuse those maps. That
    covers helpers that allocate output buffers via ``get_ort_io_torch_dtype`` before they
    re-fetch the session. Cache entries are only reused for the same session object, so a
    reload with a different export (FP16 vs FP32) re-introspects.
    """
    if session is None:
        entry = cache.get(model_name)
        if entry is not None:
            ins, outs, session_ref = entry
            if session_ref is not None and session_ref() is not None:
                return ins, outs
            cache.pop(model_name, None)
        raise RuntimeError(f"No ONNX session loaded for {model_name!r}")
    entry = cache.get(model_name)
    if entry is not None:
        ins, outs, session_ref = entry
        cached_session = session_ref() if session_ref is not None else None
        if cached_session is session:
            return ins, outs
    ins = {i.name: _ort_warmup_numpy_dtype_for_input(i) for i in session.get_inputs()}
    outs = {o.name: _ort_warmup_numpy_dtype_for_input(o) for o in session.get_outputs()}
    try:
        session_ref = weakref.ref(session)
    except TypeError:
        session_ref = None
    cache[model_name] = (ins, outs, session_ref)
    return ins, outs


def normalize_ort_bind_device_type(device_type: str) -> str:
    """ORT IOBinding expects ``cuda`` / ``cpu``, not ``cuda:0`` torch device strings."""
    dt = str(device_type).strip().lower()
    if dt.startswith("cuda"):
        return "cuda"
    if dt == "cpu":
        return "cpu"
    return dt


def _numpy_scalar_type_to_torch_dtype(np_scalar_type: type) -> torch.dtype:
    """Map numpy scalar kind (from ``_ort_warmup_numpy_dtype_for_input``) to ``torch.dtype``."""
    if np_scalar_type == np.float16:
        return torch.float16
    if np_scalar_type == np.float64:
        return torch.float64
    if np_scalar_type == np.int64:
        return torch.int64
    if np_scalar_type == np.int32:
        return torch.int32
    if np_scalar_type == np.uint8:
        return torch.uint8
    if np_scalar_type == np.bool_:
        return torch.bool
    return torch.float32
