"""ORT I/O dtype helpers (imports lightweight ort_io_dtype_utils, not models_processor)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.processors.ort_io_dtype_utils import (
    _numpy_scalar_type_to_torch_dtype,
    _ort_warmup_numpy_dtype_for_input,
)


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
