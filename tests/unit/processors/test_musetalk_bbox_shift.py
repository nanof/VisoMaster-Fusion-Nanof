"""Two community fixes for MuseTalk's generic mouth, wired end to end.

MuseTalk regenerates the mouth from audio at 256px, so it comes back soft and
identity-poor. The project's own advice is (1) ``bbox_shift``, to trade lip
motion against how much real face survives, and (2) run a face restorer over the
generated mouth to add texture back.

``bbox_shift`` moves the nose-bridge point the crop is built from. Implementing
it as a shift of the model's input mask inside a fixed crop instead put a hard
horizontal seam across the nose, because that mask edge stopped coinciding with
the boundary the blend fades in at. These tests pin both halves of that lesson.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.processors.pytorch_extras.musetalk.framing import (  # noqa: E402
    landmark_crop_bbox,
)
from app.processors.pytorch_extras.musetalk.models import MuseTalkVAE  # noqa: E402
from app.processors.workers.frame_worker import FrameWorker  # noqa: E402

SIDE = 256
FRAME = (720, 1280, 3)


def _vae() -> MuseTalkVAE:
    """A VAE with only the mask machinery set, so no weights are loaded."""
    obj = MuseTalkVAE.__new__(MuseTalkVAE)
    obj._resized_img = SIDE
    obj.device = torch.device("cpu")
    obj._rebuild_mask()
    return obj


def test_the_model_input_mask_is_always_the_exact_lower_half() -> None:
    """The seam bug: this edge must stay where the blend fades in."""
    kept = int((_vae()._mask_tensor[:, 0] > 0.5).sum().item())
    assert kept == SIDE // 2


def test_the_vae_exposes_no_way_to_move_that_boundary() -> None:
    """Guards against reintroducing the shift at the wrong layer."""
    assert not hasattr(_vae(), "set_bbox_shift")


def _landmarks_68(bridge_y: float = 300.0, chin_y: float = 400.0) -> np.ndarray:
    """68 points whose max y is the chin and whose point 29 is the bridge."""
    pts = np.zeros((68, 2), dtype=np.float32)
    pts[:, 0] = np.linspace(500.0, 700.0, 68)
    pts[:, 1] = bridge_y
    pts[29, 1] = bridge_y
    pts[8, 1] = chin_y  # chin: the lowest landmark
    return pts


def _crop(bbox_shift: int) -> tuple[int, int, int, int]:
    framed = landmark_crop_bbox(
        _landmarks_68(), None, FRAME, extra_margin=10, bbox_shift=bbox_shift
    )
    assert framed is not None
    return framed


def test_zero_shift_centres_the_crop_on_the_bridge() -> None:
    x1, y1, x2, y2 = _crop(0)
    # top = bridge - (chin - bridge) = 300 - 100
    assert y1 == 200
    assert y2 == 410  # chin + extra_margin


def test_a_positive_shift_moves_the_crop_top_toward_the_mouth() -> None:
    """Positive drags the bridge down, so the half line lands nearer the mouth."""
    _, y1_zero, _, _ = _crop(0)
    _, y1_pos, _, _ = _crop(9)
    assert y1_pos > y1_zero
    # bridge 309, half 91, top 218
    assert y1_pos == 218


def test_a_negative_shift_keeps_more_of_the_real_face() -> None:
    _, y1_zero, _, _ = _crop(0)
    _, y1_neg, _, _ = _crop(-9)
    assert y1_neg < y1_zero
    assert y1_neg == 182  # bridge 291, half 109


def test_the_shift_leaves_the_chin_and_the_width_alone() -> None:
    """Only the top edge moves, so the mouth does not change scale sideways."""
    base = _crop(0)
    for shift in (-9, 9):
        moved = _crop(shift)
        assert moved[0] == base[0] and moved[2] == base[2]
        assert moved[3] == base[3]


def test_an_absurd_shift_is_refused_rather_than_framed_wrong() -> None:
    """Past the chin there is no half distance left; the caller falls back."""
    assert (
        landmark_crop_bbox(
            _landmarks_68(), None, FRAME, extra_margin=10, bbox_shift=500
        )
        is None
    )


def _fake_worker(apply_facerestorer=None, parameters=None) -> types.SimpleNamespace:
    mp = types.SimpleNamespace(get_effective_torch_device=lambda: torch.device("cpu"))
    if apply_facerestorer is not None:
        mp.apply_facerestorer = apply_facerestorer
    return types.SimpleNamespace(
        models_processor=mp,
        parameters=parameters or {},
        _RESTORER_FIDELITY_DEFAULT=FrameWorker._RESTORER_FIDELITY_DEFAULT,
        _restorer_fidelity_in_use=lambda: FrameWorker._restorer_fidelity_in_use(
            types.SimpleNamespace(
                parameters=parameters or {},
                _RESTORER_FIDELITY_DEFAULT=FrameWorker._RESTORER_FIDELITY_DEFAULT,
            )
        ),
    )


def test_the_mouth_reuses_the_swaps_fidelity_weight() -> None:
    """CodeFormer caches its CUDA graph per fidelity value.

    Passing our own number invalidated the graph the swap had just built, every
    single frame, which flashed the build dialog nonstop.
    """
    seen: list[float] = []

    def fake(_t, _det, _rtype, _blend, fidelity, *_a):  # noqa: ANN001
        seen.append(float(fidelity))
        return torch.full((3, 512, 512), 255.0)

    worker = _fake_worker(
        apply_facerestorer=fake,
        parameters={"face-1": {"FaceFidelityWeightDecimalSlider": 0.55}},
    )
    cb = FrameWorker._musetalk_restore_crop(
        worker, {"MuseTalkRestoreMouthStrengthSlider": 100}
    )
    cb(np.zeros((SIDE, SIDE, 3), np.uint8))
    assert seen == [0.55]


def test_fidelity_falls_back_to_the_app_default() -> None:
    fake_self = types.SimpleNamespace(
        parameters={}, _RESTORER_FIDELITY_DEFAULT=FrameWorker._RESTORER_FIDELITY_DEFAULT
    )
    assert FrameWorker._restorer_fidelity_in_use(fake_self) == pytest.approx(0.9)


def test_unusable_fidelity_does_not_break_the_frame() -> None:
    fake_self = types.SimpleNamespace(
        parameters={"f": {"FaceFidelityWeightDecimalSlider": "abc"}},
        _RESTORER_FIDELITY_DEFAULT=FrameWorker._RESTORER_FIDELITY_DEFAULT,
    )
    assert FrameWorker._restorer_fidelity_in_use(fake_self) == pytest.approx(0.9)


def test_no_restorer_available_means_no_callback() -> None:
    cb = FrameWorker._musetalk_restore_crop(
        _fake_worker(), {"MuseTalkRestoreMouthStrengthSlider": 60}
    )
    assert cb is None


def test_zero_strength_disables_the_pass() -> None:
    cb = FrameWorker._musetalk_restore_crop(
        _fake_worker(apply_facerestorer=lambda *a, **k: None),
        {"MuseTalkRestoreMouthStrengthSlider": 0},
    )
    assert cb is None


def test_the_call_matches_the_models_processor_signature() -> None:
    """``ModelsProcessor.apply_facerestorer`` takes target_kps positionally.

    A stand-in with the loose ``*args`` signature hid a real TypeError here, so
    this double mirrors the wrapper exactly.
    """
    seen: dict = {}

    def wrapper_signature(  # noqa: ANN001
        swapped_face_upscaled,
        restorer_det_type,
        restorer_type,
        restorer_blend,
        fidelity_weight,
        detect_score,
        target_kps,
        slot_id: int = 1,
        dmd_landmarks_68_crop=None,
    ):
        seen.update(det=restorer_det_type, kps=target_kps)
        return torch.full((3, 512, 512), 255.0)

    cb = FrameWorker._musetalk_restore_crop(
        _fake_worker(apply_facerestorer=wrapper_signature),
        {"MuseTalkRestoreMouthStrengthSlider": 100},
    )
    cb(np.zeros((SIDE, SIDE, 3), np.uint8))
    assert seen == {"det": "Original", "kps": None}


def test_the_restorer_is_fed_the_512_crop_it_expects() -> None:
    """In 'Original' mode the restorer does not resize its input.

    GFPGAN assumes the 512 crop the swap path produces, so handing it MuseTalk's
    256 mouth raised "size of tensor a (512) must match tensor b (256)".
    """
    shapes: list[tuple[int, ...]] = []

    def fake(t, *_a):  # noqa: ANN001
        shapes.append(tuple(t.shape))
        return torch.full((3, 512, 512), 255.0)

    cb = FrameWorker._musetalk_restore_crop(
        _fake_worker(apply_facerestorer=fake),
        {"MuseTalkRestoreMouthStrengthSlider": 100},
    )
    out = cb(np.zeros((SIDE, SIDE, 3), np.uint8))
    assert shapes == [(3, 512, 512)]
    assert out.shape == (SIDE, SIDE, 3)  # and comes back at the crop's own size


def test_full_strength_replaces_the_mouth_with_the_restored_one() -> None:
    def fake(_t, det, rtype, blend, fw, ds, kps):  # noqa: ANN001
        assert det == "Original"  # crop is not FFHQ-aligned, so no realign
        assert rtype == "GFPGAN-v1.4"
        return torch.full((3, 512, 512), 255.0)

    cb = FrameWorker._musetalk_restore_crop(
        _fake_worker(apply_facerestorer=fake),
        {"MuseTalkRestoreMouthStrengthSlider": 100},
    )
    out = cb(np.zeros((SIDE, SIDE, 3), np.uint8))
    assert out.shape == (SIDE, SIDE, 3)
    assert out.mean() > 240.0


def test_partial_strength_blends_toward_the_restored_mouth() -> None:
    def fake(_t, *a):  # noqa: ANN001
        return torch.full((3, 512, 512), 255.0)

    cb = FrameWorker._musetalk_restore_crop(
        _fake_worker(apply_facerestorer=fake),
        {"MuseTalkRestoreMouthStrengthSlider": 50},
    )
    out = cb(np.zeros((SIDE, SIDE, 3), np.uint8))
    assert 100.0 < out.mean() < 155.0


def test_a_failed_restore_is_swallowed_by_the_engine_not_the_callback() -> None:
    """The callback itself may raise; the engine wraps it. Here we just ensure a
    ``None`` return from the restorer leaves the crop untouched."""

    def fake(_t, *a):  # noqa: ANN001
        return None

    cb = FrameWorker._musetalk_restore_crop(
        _fake_worker(apply_facerestorer=fake),
        {"MuseTalkRestoreMouthStrengthSlider": 80},
    )
    recon = np.full((SIDE, SIDE, 3), 40, np.uint8)
    assert np.array_equal(cb(recon), recon)
