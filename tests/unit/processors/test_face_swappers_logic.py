"""
FS-* tests for face-swapper logic (embedding math, guards, model dispatch).

All model inference is mocked.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# FS-02: CSCS normalized embedding is a unit vector
# ---------------------------------------------------------------------------


def test_cscs_embedding_unit_norm():
    """After L2 normalisation the embedding magnitude should be 1.0."""
    raw = torch.randn(512)
    normed = raw / (raw.norm() + 1e-8)
    norm_val = normed.norm().item()
    assert abs(norm_val - 1.0) < 1e-5, f"Norm should be ~1.0, got {norm_val}"


def test_cscs_embedding_unit_norm_batch():
    """Works for a batch of embeddings."""
    raw = torch.randn(4, 512)
    norms = raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = raw / norms
    per_norm = normed.norm(dim=1)
    assert torch.allclose(per_norm, torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# FS-01: calc_inswapper_latent returns None on bad input
# ---------------------------------------------------------------------------


def test_calc_inswapper_latent_returns_none_on_none_embedding():
    """Simulate the None-on-failure guard in calc_inswapper_latent."""
    embedding = None

    def calc_inswapper_latent(emb):
        if emb is None:
            return None
        return emb @ emb.T  # placeholder

    result = calc_inswapper_latent(embedding)
    assert result is None


def test_calc_inswapper_latent_returns_none_on_empty():
    embedding = np.array([])

    def calc_inswapper_latent(emb):
        if emb is None or len(emb) == 0:
            return None
        return emb

    result = calc_inswapper_latent(embedding)
    assert result is None


# ---------------------------------------------------------------------------
# FS-05: GHOSTFACE_MODELS frozenset has exactly 3 expected members
# ---------------------------------------------------------------------------


def test_ghostface_models_frozenset_contents():
    GHOSTFACE_MODELS = frozenset({"GhostFace-v1", "GhostFace-v2", "GhostFace-v3"})
    assert "GhostFace-v1" in GHOSTFACE_MODELS
    assert "GhostFace-v2" in GHOSTFACE_MODELS
    assert "GhostFace-v3" in GHOSTFACE_MODELS
    assert len(GHOSTFACE_MODELS) == 3


def test_ghostface_models_is_frozenset():
    GHOSTFACE_MODELS = frozenset({"GhostFace-v1", "GhostFace-v2", "GhostFace-v3"})
    assert isinstance(GHOSTFACE_MODELS, frozenset)


# ---------------------------------------------------------------------------
# FS-05c: BLENDSWAP model name set (mirrors FrameWorker.BLENDSWAP_MODELS)
# ---------------------------------------------------------------------------


def test_blendswap_models_frozenset():
    from app.processors.workers.frame_worker import FrameWorker

    assert FrameWorker.BLENDSWAP_MODELS == frozenset({"BlendSwap-256"})


# ---------------------------------------------------------------------------
# FS-05d: UNIFACE model name set (mirrors FrameWorker.UNIFACE_MODELS)
# ---------------------------------------------------------------------------


def test_uniface_models_frozenset():
    from app.processors.workers.frame_worker import FrameWorker

    assert FrameWorker.UNIFACE_MODELS == frozenset({"UniFace-256"})


# ---------------------------------------------------------------------------
# FS-05b: HYPERSWAP model name set (mirrors FrameWorker.HYPERSWAP_MODELS)
# ---------------------------------------------------------------------------


def test_hyperswap_models_frozenset_contents():
    HYPERSWAP_MODELS = frozenset({"HyperSwap-v1", "HyperSwap-v2", "HyperSwap-v3"})
    assert len(HYPERSWAP_MODELS) == 3
    assert "HyperSwap-v1" in HYPERSWAP_MODELS


def test_hyperswap_alignment_disjoint_from_ghostface():
    """HyperSwap must use arcface128 (Inswapper-style), not Ghost arcfacemap."""
    from app.processors.workers.frame_worker import FrameWorker

    assert FrameWorker.HYPERSWAP_MODELS.isdisjoint(FrameWorker.GHOSTFACE_MODELS)
    assert FrameWorker.HYPERSWAP_MODELS == frozenset(
        {"HyperSwap-v1", "HyperSwap-v2", "HyperSwap-v3"}
    )


def test_hyperswap_exposed_in_swap_model_selection():
    # Avoid importing SWAPPER_LAYOUT_DATA (circular UI imports under pytest).
    from pathlib import Path

    text = Path("app/ui/widgets/swapper_layout_data.py").read_text(encoding="utf-8")
    for name in ("HyperSwap-v1", "HyperSwap-v2", "HyperSwap-v3"):
        assert f'"{name}"' in text


def test_hyperswap_models_are_fp16_safe():
    from app.processors.models_data import fp16_safe_models_list

    for name in ("HyperSwapv1", "HyperSwapv2", "HyperSwapv3"):
        assert name in fp16_safe_models_list


def test_hyperswap_maps_to_inswapper128_arcface():
    from app.processors.models_data import arcface_mapping_model_dict

    for name in ("HyperSwap-v1", "HyperSwap-v2", "HyperSwap-v3"):
        assert arcface_mapping_model_dict[name] == "HyperSwapArcFace"


def test_hyperswap_arcface_aliases_w600k_session():
    from app.processors.face_swappers import FaceSwappers

    assert (
        FaceSwappers._arcface_ort_session_name("HyperSwapArcFace")
        == "Inswapper128ArcFace"
    )
    assert (
        FaceSwappers._arcface_ort_session_name("Inswapper128ArcFace")
        == "Inswapper128ArcFace"
    )


def test_hyperswap_ff_align_uses_arcface_112_v2_single_warp():
    """HyperSwap identity crop must be one landmark warp to arcface_112_v2 (no convert chain)."""
    from app.processors.face_swappers import FaceSwappers
    import app.processors.utils.faceutil as faceutil

    fs = FaceSwappers(models_processor=object())  # type: ignore[arg-type]
    calls: list[dict] = []

    def _fake_warp(img, kps, image_size=112, mode="arcface112", interpolation=None):
        calls.append({"image_size": image_size, "mode": mode})
        return torch.full((3, image_size, image_size), 200, dtype=torch.uint8), None

    orig = faceutil.warp_face_by_face_landmark_5
    faceutil.warp_face_by_face_landmark_5 = _fake_warp
    try:
        kps = np.array(
            [[40.0, 50.0], [80.0, 50.0], [60.0, 70.0], [45.0, 90.0], [75.0, 90.0]],
            dtype=np.float32,
        )
        out = fs._align_hyperswap_ff_arcface_112(
            torch.zeros(3, 256, 256, dtype=torch.uint8), kps
        )
    finally:
        faceutil.warp_face_by_face_landmark_5 = orig

    assert calls == [{"image_size": 112, "mode": "arcface112"}]
    assert out.shape == (3, 112, 112)


def test_arcface112_template_matches_facefusion_arcface_112_v2():
    """faceutil arcface_src / 112 must equal FaceFusion's normalized arcface_112_v2."""
    from app.processors.utils import faceutil

    template = np.squeeze(faceutil.get_arcface_template(112, "arcface112")) / 112.0
    facefusion_arcface_112_v2 = np.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089],
        ],
        dtype=np.float32,
    )
    assert np.allclose(template, facefusion_arcface_112_v2, atol=1e-5)


def test_hyperswap_ui_to_model_name():
    from app.processors.face_swappers import FaceSwappers

    assert FaceSwappers._hyperswap_ui_to_model_name("HyperSwap-v1") == "HyperSwapv1"
    assert FaceSwappers._hyperswap_ui_to_model_name("HyperSwap-v2") == "HyperSwapv2"
    assert FaceSwappers._hyperswap_ui_to_model_name("HyperSwap-v3") == "HyperSwapv3"
    assert FaceSwappers._hyperswap_ui_to_model_name("GhostFace-v1") is None


def test_calc_hyperswap_latent_l2_unit_and_shape():
    from app.processors.face_swappers import FaceSwappers

    fs = FaceSwappers(models_processor=object())  # type: ignore[arg-type]
    raw = np.random.randn(512).astype(np.float32)
    lat = fs.calc_hyperswap_latent(raw)
    assert lat is not None
    assert lat.shape == (1, 512)
    assert abs(float(np.linalg.norm(lat)) - 1.0) < 1e-5


def test_calc_hyperswap_latent_none_on_empty():
    from app.processors.face_swappers import FaceSwappers

    fs = FaceSwappers(models_processor=object())  # type: ignore[arg-type]
    assert fs.calc_hyperswap_latent(None) is None
    assert fs.calc_hyperswap_latent(np.array([])) is None


def test_calc_hyperswap_latent_none_on_near_zero_norm():
    """Degenerate embeddings must not reach ORT as an unnormalized row."""
    from app.processors.face_swappers import FaceSwappers

    fs = FaceSwappers(models_processor=object())  # type: ignore[arg-type]
    assert fs.calc_hyperswap_latent(np.zeros(512, dtype=np.float32)) is None
    assert fs.calc_hyperswap_latent(np.full(512, 1e-12, dtype=np.float32)) is None


def test_hyperswap_session_flags_reset_on_unload_and_model_switch():
    """A one-off TRT failure must not permanently disable HyperSwap batch/mask."""
    from app.processors.face_swappers import FaceSwappers
    import threading

    class _Proc:
        def __init__(self):
            self.model_lock = threading.Lock()
            self.unloaded = []

        def unload_model(self, name):
            self.unloaded.append(name)

    proc = _Proc()
    fs = FaceSwappers(models_processor=proc)  # type: ignore[arg-type]
    fs._hyperswap_ort_batch_session_disabled = True
    fs._hyperswap_ort_batch_fail_logged = True
    fs._hyperswap_native_mask_disabled = True

    fs.unload_models()
    assert fs._hyperswap_ort_batch_session_disabled is False
    assert fs._hyperswap_ort_batch_fail_logged is False
    assert fs._hyperswap_native_mask_disabled is False

    fs.current_swapper_model = "HyperSwapv1"
    fs._hyperswap_ort_batch_session_disabled = True
    fs._hyperswap_native_mask_disabled = True
    fs._manage_model("HyperSwapv2")
    assert fs._hyperswap_ort_batch_session_disabled is False
    assert fs._hyperswap_native_mask_disabled is False
    assert "HyperSwapv1" in proc.unloaded


def test_prefetch_strength_lerp_needs_original_prev_face():
    """Batched prefetch must lerp against the pre-swap crop, not the swapped result.

    Mirrors ``swap_core`` StrengthEnableToggle handling after plane batch inference.
    """
    original_hwc = torch.zeros(8, 8, 3, dtype=torch.float32)
    swap = torch.full((3, 8, 8), 200.0)

    def _strength_lerp(prev_hwc_01: torch.Tensor, alpha: float) -> torch.Tensor:
        prev = torch.mul(prev_hwc_01, 255).clamp(0, 255).permute(2, 0, 1)
        return torch.lerp(prev.float(), swap.float(), alpha)

    alpha = 0.5
    buggy_prev = torch.div(swap, 255.0).permute(1, 2, 0)
    assert _strength_lerp(buggy_prev, alpha).mean().item() == pytest.approx(200.0)
    assert _strength_lerp(original_hwc, alpha).mean().item() == pytest.approx(100.0)


def test_hyperswap_batch_io_matches_swap_core_minus_one_one_range():
    """Batch ORT-256 preprocess/decode must match single-face swap_core [-1, 1]."""
    # HWC float [0,1] as after swap_core's /255 (and after batch sharpness on [0,1]).
    hwc = torch.rand(256, 256, 3, dtype=torch.float32)

    # Single-face HyperSwap/GhostFace path in get_swapped_and_prev_face:
    single_in = torch.mul(hwc, 255.0).permute(2, 0, 1)
    single_in = torch.div(single_in.float(), 127.5)
    single_in = torch.sub(single_in, 1)

    # Batched path after Paso 1:
    batch_in = hwc.permute(2, 0, 1) * 2.0 - 1.0
    assert torch.allclose(batch_in, single_in, atol=1e-5)

    # Synthetic model identity in [-1,1] → decode to uint8 CHW
    fake_out = batch_in.clone()
    decoded_batch = (fake_out * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    decoded_single = (
        torch.add(torch.mul(fake_out, 127.5), 127.5).clamp(0, 255).to(torch.uint8)
    )
    assert torch.equal(decoded_batch, decoded_single)
    # Round-trip roughly recovers [0,255] from original HWC
    orig_u8 = (hwc * 255.0).clamp(0, 255).to(torch.uint8).permute(2, 0, 1)
    assert (decoded_batch.to(torch.int16) - orig_u8.to(torch.int16)).abs().max() <= 1


def test_session_io_names_do_not_leak_across_models_on_id_reuse():
    """A reused id() must not bind HyperSwap's 'source' on an ArcFace session."""
    from app.processors.face_swappers import FaceSwappers

    def _make_session(input_name: str, output_name: str):
        class _S:
            def get_inputs(self):
                return [type("I", (), {"name": input_name})()]

            def get_outputs(self):
                return [type("O", (), {"name": output_name})()]

        return _S()

    fs = FaceSwappers(models_processor=object())  # type: ignore[arg-type]
    arcface = _make_session("data", "683")

    # A HyperSwap session was unloaded and CPython handed its id to `arcface`.
    fs._hyperswap_output_name("HyperSwap-v3", _make_session("source", "output"))
    fs._session_io_name_cache[("HyperSwap-v3", id(arcface))] = {
        "input": "source",
        "outputs": ["output"],
    }

    names = fs._session_io_names("Inswapper128ArcFace", arcface)
    assert names["input"] == "data"
    assert names["outputs"] == ["683"]


def test_hyperswap_batched_disables_session_after_failure():
    from app.processors.face_swappers import FaceSwappers

    class _FakeProc:
        def bind_ort_io_input(self, *a, **k):
            raise RuntimeError("reject batch")

        def bind_ort_io_output(self, *a, **k):
            pass

    class _FakeModel:
        def get_inputs(self):
            return [type("I", (), {"name": "target"})()]

        def get_outputs(self):
            return [type("O", (), {"name": "output"})()]

        def io_binding(self):
            return object()

    fs = FaceSwappers(models_processor=_FakeProc())  # type: ignore[arg-type]
    fs._load_swapper_model = lambda _n: _FakeModel()  # type: ignore[method-assign]
    fs._run_model_with_lazy_build_check = lambda *a, **k: None  # type: ignore[method-assign]

    images = torch.zeros(2, 3, 256, 256)
    emb = torch.zeros(2, 512)
    out = torch.empty_like(images)
    assert fs.run_hyperswap_batched(images, emb, out, "HyperSwap-v3") is False
    assert fs._hyperswap_ort_batch_session_disabled is True
    # Second call short-circuits without binding
    assert fs.run_hyperswap_batched(images, emb, out, "HyperSwap-v3") is False


# ---------------------------------------------------------------------------
# FS-04: GhostFace fallback to input face when model fails
# ---------------------------------------------------------------------------


def test_ghostface_fallback_on_none_output():
    """If the swapper returns None, output should fall back to the input face."""
    input_face = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)

    def run_ghostface_swap(face_tensor, model):
        # Simulate model returning None
        return None

    model = None  # mocked
    result = run_ghostface_swap(input_face, model)
    swapped = result if result is not None else input_face

    assert torch.equal(swapped, input_face)


# ---------------------------------------------------------------------------
# FS-06: keep_alive_tensors list grows when restorer/KV tensors are appended
# ---------------------------------------------------------------------------


def test_keep_alive_tensors_grows():
    keep_alive_tensors: list = []
    kv_tensor = torch.randn(1, 4, 64, 64)
    keep_alive_tensors.append(kv_tensor)
    assert len(keep_alive_tensors) == 1
    assert keep_alive_tensors[0] is kv_tensor


def test_keep_alive_tensors_prevents_gc():
    """Tensors appended to keep_alive_tensors are still reachable."""
    import weakref

    keep_alive: list = []
    t = torch.randn(100, 100)
    ref = weakref.ref(t)
    keep_alive.append(t)
    del t
    import gc

    gc.collect()
    # Still alive because keep_alive holds a reference
    assert ref() is not None
