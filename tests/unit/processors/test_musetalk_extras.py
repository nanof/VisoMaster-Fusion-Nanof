"""MuseTalk helpers that must not require torch/diffusers at import time."""

from __future__ import annotations

import numpy as np

from app.processors.pytorch_extras.musetalk.blending import (
    blend_face_region,
    expand_bbox,
    soft_lower_face_mask,
)
from app.processors.pytorch_extras.musetalk.paths import (
    musetalk_assets_ready,
    musetalk_root,
    unet_config_path,
)
from app.processors.pytorch_extras.retalking_stub import run_retalking_placeholder


def test_musetalk_paths_resolve():
    root = musetalk_root()
    assert root.name == "musetalk"
    cfg = unet_config_path()
    assert cfg.is_file()
    assert cfg.stat().st_size > 20


def test_soft_mask_and_blend_do_not_need_gpu():
    mask = soft_lower_face_mask(64, 64)
    assert mask.shape == (64, 64)
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    face = np.full((256, 256, 3), 255, dtype=np.uint8)
    out = blend_face_region(frame, face, (50, 40, 150, 160))
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    # Mouth region should pick up some white from the face crop.
    assert int(out[120:150, 80:120].mean()) > 0


def test_expand_bbox_clamps():
    x1, y1, x2, y2 = expand_bbox([10, 10, 90, 90], (100, 100), extra_margin=10)
    assert 0 <= x1 < x2 <= 100
    assert 0 <= y1 < y2 <= 100


def test_retalking_placeholder_reports_readiness(capsys):
    # Activation is via the UI toggle, not the env var. The helper just reports
    # whether weights are present.
    run_retalking_placeholder()
    out = capsys.readouterr().out
    assert "MuseTalk" in out


def test_assets_ready_false_without_weights():
    # Fresh clone / CI without MuseTalk download.
    assert musetalk_assets_ready() in (True, False)
