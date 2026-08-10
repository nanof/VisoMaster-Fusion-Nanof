"""Lip-sync before the swap is what gives the mouth its identity back.

MuseTalk cannot preserve a face's lip shape: the lower half is masked out of its
input and the shape comes from an audio-driven prior, so whichever stage runs last
owns the mouth. Running it first turns it into a driver and lets the swapper impose
identity over the generated pose. These tests pin which stage owns the frame and
that the hand-off between the swap's GPU tensors and lip-sync's BGR arrays is
faithful, channel order included.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.processors.workers.frame_worker import FrameWorker

ON = {"MuseTalkEnableToggle": True}


def _worker() -> FrameWorker:
    return FrameWorker.__new__(FrameWorker)


# --- which stage owns the mouth ---------------------------------------------


def test_before_the_swap_is_the_default() -> None:
    """Including for workspaces saved before the setting existed."""
    assert FrameWorker._musetalk_order({}) == "before"
    assert FrameWorker._musetalk_order({"MuseTalkPipelineOrderSelection": None}) == (
        "before"
    )


def test_the_setting_is_read_from_the_selection() -> None:
    assert (
        FrameWorker._musetalk_order(
            {"MuseTalkPipelineOrderSelection": "After the swap"}
        )
        == "after"
    )
    assert (
        FrameWorker._musetalk_order(
            {"MuseTalkPipelineOrderSelection": "Before the swap"}
        )
        == "before"
    )


def test_an_unrecognised_choice_keeps_the_better_order() -> None:
    assert (
        FrameWorker._musetalk_order({"MuseTalkPipelineOrderSelection": "nonsense"})
        == "before"
    )


def test_only_one_stage_ever_runs_lip_sync() -> None:
    """Both hooks firing would paint the mouth twice and undo the swap's identity."""
    worker = _worker()
    before = {**ON, "MuseTalkPipelineOrderSelection": "Before the swap"}
    after = {**ON, "MuseTalkPipelineOrderSelection": "After the swap"}
    assert not worker._musetalk_runs_after_swap(before)
    assert worker._musetalk_runs_after_swap(after)


def test_hybrid_is_recognised() -> None:
    assert (
        FrameWorker._musetalk_order(
            {"MuseTalkPipelineOrderSelection": "Before + light after (hybrid)"}
        )
        == "hybrid"
    )


def test_hybrid_runs_both_stages() -> None:
    """Its whole point: place the mouth before, re-sharpen it after."""
    worker = _worker()
    hybrid = {**ON, "MuseTalkPipelineOrderSelection": "Before + light after (hybrid)"}
    assert worker._musetalk_runs_after_swap(hybrid)


def test_the_hybrid_after_pass_is_confined_to_the_mouth() -> None:
    """It must force the mouth-only mask; on the jaw it would erase the swap."""
    opts = FrameWorker._musetalk_mask_options(
        {"MuseTalkMouthOnlyToggle": False, "MuseTalkFaceParsingToggle": False},
        hybrid_after=True,
    )
    assert opts["mouth_only"] is True


def test_the_hybrid_after_pass_is_partly_transparent() -> None:
    """Full opacity would overwrite the swap's mouth; the pose is already aligned."""
    opts = FrameWorker._musetalk_mask_options(
        {"MuseTalkHybridAfterStrengthSlider": 40}, hybrid_after=True
    )
    assert opts["mouth_alpha"] == pytest.approx(0.40)


def test_every_other_pass_paints_the_mouth_opaque() -> None:
    """Only the hybrid after-pass steps the opacity down."""
    assert FrameWorker._musetalk_mask_options({})["mouth_alpha"] == 1.0
    assert (
        FrameWorker._musetalk_mask_options({}, hybrid_after=False)["mouth_alpha"] == 1.0
    )


def test_hybrid_after_needs_both_mouth_toggles() -> None:
    worker = _worker()
    assert worker._musetalk_mouth_only_available({**ON})  # both default on
    assert not worker._musetalk_mouth_only_available(
        {**ON, "MuseTalkFaceParsingToggle": False}
    )
    assert not worker._musetalk_mouth_only_available(
        {**ON, "MuseTalkMouthOnlyToggle": False}
    )


def test_vr180_does_not_use_the_equirect_post_swap_hook() -> None:
    """VR lip-sync runs per perspective crop; the equirect hook would lack geometry."""
    worker = _worker()
    control = {
        **ON,
        "MuseTalkPipelineOrderSelection": "Before the swap",
        "VR180ModeEnableToggle": True,
    }
    assert not worker._musetalk_runs_after_swap(control)
    control_after = {
        **ON,
        "MuseTalkPipelineOrderSelection": "After the swap",
        "VR180ModeEnableToggle": True,
    }
    assert not worker._musetalk_runs_after_swap(control_after)


def test_vr_crop_lip_sync_passes_crop_local_geometry(monkeypatch) -> None:
    """Perspective crops must not fall back to empty equirect detections."""
    seen: dict[str, object] = {}

    def fake_apply(
        bgr,
        control,
        *,
        hybrid_after=False,
        bboxes=None,
        landmarks=None,
        kpss_5=None,
        face_index=None,
    ):
        seen["shape"] = bgr.shape
        seen["bboxes"] = bboxes
        seen["landmarks"] = landmarks
        seen["kpss_5"] = kpss_5
        seen["face_index"] = face_index
        out = bgr.copy()
        out[:, :] = (1, 2, 3)
        return out

    worker = _worker()
    monkeypatch.setattr(worker, "_musetalk_apply_bgr", fake_apply, raising=False)
    monkeypatch.setattr(
        worker,
        "_hwc_rgb_uint8_from_chw_tensor",
        lambda t: t.permute(1, 2, 0).cpu().numpy(),
        raising=False,
    )

    crop = torch.zeros((3, 64, 64), dtype=torch.uint8)
    k5 = np.array(
        [[20.0, 20.0], [44.0, 20.0], [32.0, 32.0], [24.0, 44.0], [40.0, 44.0]],
        dtype=np.float32,
    )
    k68 = np.zeros((68, 2), dtype=np.float32)
    k68[:, 0] = np.linspace(16, 48, 68)
    k68[:, 1] = 32.0
    out = worker._musetalk_apply_vr_crop(
        crop, {**ON}, kps_5_on_crop=k5, kps_all_on_crop=k68
    )
    assert seen["face_index"] == 0
    assert seen["shape"] == (64, 64, 3)
    assert len(seen["bboxes"]) == 1
    bb = np.asarray(seen["bboxes"][0])
    assert bb[0] >= 1 and bb[2] <= 63
    assert len(seen["landmarks"]) == 1
    assert np.asarray(seen["landmarks"][0]).shape == (68, 2)
    assert len(seen["kpss_5"]) == 1
    assert out.shape == (3, 64, 64)
    assert tuple(int(v) for v in out[:, 0, 0]) == (3, 2, 1)


def test_vr_crop_lip_sync_stands_aside_when_disabled(monkeypatch) -> None:
    worker = _worker()

    def boom(*_a, **_k):
        raise AssertionError("must not run")

    monkeypatch.setattr(worker, "_musetalk_apply_bgr", boom, raising=False)
    crop = torch.zeros((3, 16, 16), dtype=torch.uint8)
    assert (
        worker._musetalk_apply_vr_crop(
            crop,
            {"MuseTalkEnableToggle": False},
            kps_5_on_crop=None,
            kps_all_on_crop=None,
        )
        is crop
    )
    assert (
        worker._musetalk_apply_vr_crop(
            crop,
            {**ON, "MuseTalkBypassToggle": True},
            kps_5_on_crop=None,
            kps_all_on_crop=None,
        )
        is crop
    )


def test_neither_stage_runs_when_lip_sync_is_off_or_bypassed() -> None:
    worker = _worker()
    assert not worker._musetalk_runs_after_swap({"MuseTalkEnableToggle": False})
    assert not worker._musetalk_runs_after_swap({**ON, "MuseTalkBypassToggle": True})


# --- the tensor / array hand-off --------------------------------------------


def _frame_tensor() -> torch.Tensor:
    """CHW uint8 RGB, with a distinct value per channel to catch a swapped order."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :] = (10, 20, 30)  # R, G, B
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def test_the_frame_reaches_lip_sync_as_bgr(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_apply(bgr, control):
        seen["px"] = tuple(int(v) for v in bgr[0, 0])
        return bgr

    worker = _worker()
    monkeypatch.setattr(worker, "_musetalk_apply_bgr", fake_apply, raising=False)
    worker._musetalk_apply_before_swap(_frame_tensor(), {**ON})
    assert seen["px"] == (30, 20, 10), "the engine works in BGR"


def test_the_result_comes_back_as_rgb_in_place(monkeypatch) -> None:
    def fake_apply(bgr, control):
        out = bgr.copy()
        out[:, :] = (99, 50, 7)  # BGR
        return out

    worker = _worker()
    monkeypatch.setattr(worker, "_musetalk_apply_bgr", fake_apply, raising=False)
    out = worker._musetalk_apply_before_swap(_frame_tensor(), {**ON})
    assert out.shape == (3, 8, 8)
    assert out.dtype == torch.uint8
    assert tuple(int(v) for v in out[:, 0, 0]) == (7, 50, 99), "back to RGB"


def test_an_untouched_frame_skips_the_upload(monkeypatch) -> None:
    """No round trip when lip-sync declined the frame; the tensor is passed on."""
    worker = _worker()
    monkeypatch.setattr(
        worker, "_musetalk_apply_bgr", lambda bgr, control: bgr, raising=False
    )
    tensor = _frame_tensor()
    assert worker._musetalk_apply_before_swap(tensor, {**ON}) is tensor


def test_the_pre_swap_hook_stands_aside_when_it_is_not_its_turn(monkeypatch) -> None:
    worker = _worker()

    def boom(bgr, control):
        raise AssertionError("lip-sync must not run here")

    monkeypatch.setattr(worker, "_musetalk_apply_bgr", boom, raising=False)
    tensor = _frame_tensor()
    for control in (
        {"MuseTalkEnableToggle": False},
        {**ON, "MuseTalkBypassToggle": True},
        {**ON, "MuseTalkPipelineOrderSelection": "After the swap"},
    ):
        assert worker._musetalk_apply_before_swap(tensor, control) is tensor


def test_hybrid_also_runs_the_pre_swap_pass(monkeypatch) -> None:
    """The pre-swap pass is what places the mouth for the after-pass to sharpen."""
    ran = {"n": 0}

    def fake_apply(bgr, control):
        ran["n"] += 1
        return bgr

    worker = _worker()
    monkeypatch.setattr(worker, "_musetalk_apply_bgr", fake_apply, raising=False)
    control = {**ON, "MuseTalkPipelineOrderSelection": "Before + light after (hybrid)"}
    worker._musetalk_apply_before_swap(_frame_tensor(), control)
    assert ran["n"] == 1


def test_the_hybrid_after_pass_stands_down_without_parsing(monkeypatch) -> None:
    """Falling back to the jaw mask here would repaint over the swap's identity."""
    worker = _worker()

    def boom(bgr, control):
        raise AssertionError("the hybrid after-pass must not run without parsing")

    # The mouth mask is unavailable, so the guarded apply must return the frame
    # before ever reaching the engine.
    monkeypatch.setattr(
        worker,
        "_musetalk_apply_bgr",
        FrameWorker._musetalk_apply_bgr.__get__(worker),
        raising=False,
    )
    monkeypatch.setattr(worker, "_musetalk_landmarks", boom, raising=False)
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    control = {**ON, "MuseTalkFaceParsingToggle": False}
    assert worker._musetalk_apply_bgr(bgr, control, hybrid_after=True) is bgr


def test_a_failure_leaves_the_swap_input_alone(monkeypatch) -> None:
    """The swap must still run: a broken mouth is better than a dropped frame."""
    worker = _worker()

    def boom(bgr, control):
        raise RuntimeError("engine gone")

    monkeypatch.setattr(worker, "_musetalk_apply_bgr", boom, raising=False)
    tensor = _frame_tensor()
    assert worker._musetalk_apply_before_swap(tensor, {**ON}) is tensor
    assert worker._musetalk_preswap_err_logged is True
