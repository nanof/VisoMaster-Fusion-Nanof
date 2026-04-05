import numpy as np
import torch

from app.processors.dmdnet_landmarks import (
    _finalize_dmdnet_roi_box,
    get_component_location_tensor,
    landmarks106_to_68_xy,
)


def test_landmarks106_to_68_shape_and_order():
    pts = np.random.randn(106, 2).astype(np.float32)
    out = landmarks106_to_68_xy(pts)
    assert out.shape == (68, 2)
    assert out.dtype == np.float32
    assert np.allclose(out[0], pts[1])


def test_finalize_dmdnet_roi_box_expands_degenerate():
    out = _finalize_dmdnet_roi_box(np.array([100.0, 100.0, 100.0, 100.0]))
    assert out[2] > out[0] and out[3] > out[1]
    assert out[2] - out[0] >= 32
    assert out[3] - out[1] >= 32


def test_get_component_location_tensor_runs():
    lm = np.zeros((68, 2), dtype=np.float32)
    lm[:, 0] = np.linspace(200, 300, 68)
    lm[:, 1] = np.linspace(200, 400, 68)
    t = get_component_location_tensor(lm, device=torch.device("cpu"))
    assert t.shape == (4, 4)
    assert t.dtype == torch.float32
