"""MuseTalk helpers that must not require torch/diffusers at import time."""

from __future__ import annotations

import numpy as np

from app.processors.pytorch_extras.musetalk.blending import (
    blend_face_region,
    expand_bbox,
    soft_lower_face_mask,
)
from app.processors.pytorch_extras.musetalk.paths import (
    format_musetalk_import_error,
    format_musetalk_load_error,
    musetalk_assets_error_message,
    musetalk_assets_ready,
    musetalk_deps_error_message,
    musetalk_missing_assets,
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
    # whether weights/deps are present.
    run_retalking_placeholder()
    out = capsys.readouterr().out
    assert "MuseTalk" in out
    assert "Check / Update" in out or "ready" in out.lower()


def test_assets_ready_false_without_weights():
    # Fresh clone / CI without MuseTalk download.
    assert musetalk_assets_ready() in (True, False)


def test_assets_error_message_lists_missing_or_is_empty():
    msg = musetalk_assets_error_message()
    if musetalk_assets_ready():
        assert msg == ""
        assert musetalk_missing_assets() == []
    else:
        assert "Missing:" in msg
        assert "Check / Update Models" in msg
        assert musetalk_missing_assets()


def test_deps_error_message_mentions_launcher():
    msg = musetalk_deps_error_message(["diffusers", "librosa"])
    assert "diffusers" in msg
    assert "librosa" in msg
    assert "Check / Update Dependencies" in msg


def test_format_import_error_maps_known_package():
    err = ImportError("No module named 'diffusers'")
    err.name = "diffusers"
    msg = format_musetalk_import_error(err)
    assert msg is not None
    assert "diffusers" in msg
    assert "Check / Update Dependencies" in msg


def test_format_load_error_oom_hint():
    msg = format_musetalk_load_error(RuntimeError("CUDA out of memory"))
    assert "GPU memory" in msg or "out of" in msg.lower()


def test_format_load_error_generic():
    msg = format_musetalk_load_error(RuntimeError("boom"))
    assert "boom" in msg
    assert msg.startswith("MuseTalk load failed:")
