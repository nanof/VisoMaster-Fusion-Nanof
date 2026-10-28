"""GPU smoke for last-stage upscalers. Run with the portable interpreter + CUDA."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.integration]


def test_flashvsr_tiny_same_size_lq_on_small_frame():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from app.processors.pytorch_extras.pipeline_upscaler.flashvsr_backend import (
        FlashVSRBackend,
        flashvsr_aligned_hw,
    )
    from app.processors.pytorch_extras.pipeline_upscaler.paths import assets_ready

    if not assets_ready("FlashVSR Tiny"):
        pytest.skip("FlashVSR Tiny weights missing")
    tile_h, tile_w = 96, 128
    lq_h, lq_w = flashvsr_aligned_hw(tile_h * 4, tile_w * 4)
    assert (lq_h, lq_w) == (tile_h * 4, tile_w * 4)
    frame = np.random.randint(0, 255, (tile_h, tile_w, 3), dtype=np.uint8)
    backend = FlashVSRBackend()
    try:
        assert backend.load("FlashVSR Tiny", device="cuda"), backend._last_error
        out = backend.upscale_bgr(frame, scale=4, color_fix=False, seed=0)
        assert out.shape == (tile_h * 4, tile_w * 4, 3)
        assert out.dtype == np.uint8
    finally:
        backend.unload()


def test_seedvr2_tiled_vae_small_frame():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from app.processors.pytorch_extras.pipeline_upscaler.paths import assets_ready
    from app.processors.pytorch_extras.pipeline_upscaler.seedvr2_backend import (
        SeedVR2Backend,
        _ensure_vae_tile_sizes,
    )

    if not assets_ready("SeedVR2 3B"):
        pytest.skip("SeedVR2 3B weights missing")
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    backend = SeedVR2Backend()
    try:
        assert backend.load("SeedVR2 3B", device="cuda"), backend._last_error
        _ensure_vae_tile_sizes(backend.runner, True)
        h, w = backend.runner.encode_tile_size
        assert (h, w) == (512, 512)
        out = backend.upscale_bgr(frame, scale=2, color_fix=False, tiled_vae=True, seed=0)
        assert out.shape[0] == 128 and out.shape[1] == 128
        assert out.dtype == np.uint8
    finally:
        backend.unload()


def test_hypir_small_frame_1x():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from app.processors.pytorch_extras.pipeline_upscaler.hypir_backend import (
        HypirBackend,
    )
    from app.processors.pytorch_extras.pipeline_upscaler.paths import assets_ready

    if not assets_ready("HYPIR SD2"):
        pytest.skip("HYPIR weights missing")
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    backend = HypirBackend()
    try:
        assert backend.load("HYPIR SD2", device="cuda"), backend._last_error
        out = backend.upscale_bgr(frame, scale=1, color_fix=False, seed=0)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8
    finally:
        backend.unload()


def test_pmrf_small_frame_2x():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from app.processors.pytorch_extras.pipeline_upscaler.paths import assets_ready
    from app.processors.pytorch_extras.pipeline_upscaler.pmrf_backend import (
        PmrfBackend,
    )

    if not assets_ready("PMRF 2x"):
        pytest.skip("PMRF weights missing")
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    backend = PmrfBackend()
    try:
        assert backend.load("PMRF 2x", device="cuda"), backend._last_error
        out = backend.upscale_bgr(frame, scale=2, color_fix=False, seed=0)
        assert out.shape == (128, 128, 3)
        assert out.dtype == np.uint8
    finally:
        backend.unload()
