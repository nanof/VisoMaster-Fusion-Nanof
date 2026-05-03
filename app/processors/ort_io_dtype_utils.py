"""ORT declared I/O dtype helpers (lightweight — no models_processor import chain)."""

from __future__ import annotations

import numpy as np
import torch


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
