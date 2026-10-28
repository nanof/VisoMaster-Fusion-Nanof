from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from app.processors.face_masks import FaceMasks
from app.processors.perform_recast import PerformRecast
from app.processors.video_processor import VideoProcessor
from app.processors.workers.frame_worker import FrameWorker


def test_pre_swap_shaping_off_leaves_faces_untouched():
    shaper = MagicMock()
    worker = SimpleNamespace(
        local_control_state_from_feeder={"edit_enabled": True},
        frame_edits=SimpleNamespace(apply_face_shaping_gpu=shaper),
    )
    worker._pre_swap_shaping_requested = (
        lambda parameters: FrameWorker._pre_swap_shaping_requested(worker, parameters)
    )
    faces = ("a", "b", "c", "d")
    out, applied = FrameWorker._shape_original_faces_before_swap(
        worker,
        faces,
        None,
        np.zeros((5, 2), dtype=np.float32),
        {"FaceShapingEnableToggle": True, "FaceShapingPreSwapToggle": False},
        False,
    )
    assert out is faces
    assert applied is False
    shaper.assert_not_called()


def test_pre_swap_shaping_rebuilds_scales_and_reports_applied():
    shaped = torch.zeros(3, 4, 4)
    shaper = MagicMock(return_value=shaped)
    worker = SimpleNamespace(
        local_control_state_from_feeder={"edit_enabled": True},
        frame_edits=SimpleNamespace(apply_face_shaping_gpu=shaper),
        t256=lambda image: image[:, :2, :2],
        t384=lambda image: image[:, :3, :3],
        t128=lambda image: image[:, :1, :1],
    )
    worker._pre_swap_shaping_requested = (
        lambda parameters: FrameWorker._pre_swap_shaping_requested(worker, parameters)
    )
    tform = SimpleNamespace(params=np.eye(3, dtype=np.float32))
    kps = np.zeros((5, 2), dtype=np.float32)
    faces = (torch.ones(3, 4, 4), None, None, None)
    out, applied = FrameWorker._shape_original_faces_before_swap(
        worker,
        faces,
        tform,
        kps,
        {"FaceShapingEnableToggle": True, "FaceShapingPreSwapToggle": True},
        False,
    )
    assert applied is True
    assert out[0] is shaped
    assert out[1].shape[-1] == 3
    assert out[2].shape[-1] == 2
    assert out[3].shape[-1] == 1
    shaper.assert_called_once()


def test_later_shaping_is_skipped_after_pre_swap():
    shaper = MagicMock(return_value="shaped")
    worker = SimpleNamespace(
        local_control_state_from_feeder={"edit_enabled": True},
        frame_edits=SimpleNamespace(apply_face_shaping_gpu=shaper),
    )
    params = {
        "FaceShapingEnableToggle": True,
        "FaceEditorBeforeTypeSelection": "Beginning",
    }
    result = FrameWorker._apply_face_shaping_if_enabled(
        worker, "swap", np.zeros((5, 2)), params, "Beginning", pre_swap_applied=True
    )
    assert result == "swap"
    shaper.assert_not_called()


def _mouth_labels(size: int = 48) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.zeros(size, size, dtype=torch.long)
    labels[16:32, 16:32] = 11
    labels[12:16, 16:32] = 12
    labels[32:36, 16:32] = 13
    image = torch.full((3, size, size), 40, dtype=torch.uint8)
    image[:, 16:32, 16:32] = 200
    return image, labels.clone(), labels.clone()


def test_cavity_blend_zero_and_tongue_off_match_missing_keys():
    masks = FaceMasks(MagicMock())
    image, labels_orig, labels_swap = _mouth_labels()
    explicit = masks._enhance_and_align_original_mouth(
        image,
        image,
        labels_orig,
        labels_swap,
        {
            "MouthOriginalCavityBlendingDecimalSlider": 0.0,
            "RestoreTongueToggle": False,
            "MouthOriginalCavityBlurSlider": 15,
            "MouthOriginalCavityDarkenDecimalSlider": 1.5,
            "MouthParserStretchDecimalSlider": 1.0,
        },
    )
    omitted = masks._enhance_and_align_original_mouth(
        image,
        image,
        labels_orig,
        labels_swap,
        {
            "MouthOriginalCavityBlurSlider": 15,
            "MouthOriginalCavityDarkenDecimalSlider": 1.5,
            "MouthParserStretchDecimalSlider": 1.0,
        },
    )
    assert explicit is not None and omitted is not None
    assert torch.equal(explicit[0], omitted[0])
    assert torch.equal(explicit[1], omitted[1])


def test_playback_segment_snap_and_wrap():
    player = SimpleNamespace(
        main_window=SimpleNamespace(job_marker_pairs=[(10, 20), (40, 55)]),
        is_playing_segments=False,
        _playback_segments=[],
        _playback_segment_index=0,
        _playback_segment_end=None,
    )
    player._valid_playback_marker_pairs = (
        lambda: VideoProcessor._valid_playback_marker_pairs(player)
    )
    assert VideoProcessor._snap_playback_segment_start(player, 0) == 10
    assert player._playback_segment_index == 0
    assert VideoProcessor._snap_playback_segment_start(player, 25) == 40
    assert player._playback_segment_index == 1
    assert VideoProcessor._snap_playback_segment_start(player, 30) == 40
    assert VideoProcessor._advance_playback_segment(player) == 10
    assert player._playback_segment_index == 0


def test_keyframe_sync_falls_back_to_requested_frame(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise RuntimeError("ffprobe missing")

    monkeypatch.setattr(
        "app.processors.video_processor.subprocess.run", _fail
    )
    player = SimpleNamespace(media_path="clip.mp4")
    frame, timestamp = VideoProcessor.resolve_live_preview_start(player, 48, 24.0)
    assert frame == 48
    assert timestamp == 2.0


def test_denoiser_color_match_stays_on_when_toggle_is_missing():
    seen = {}

    def _apply(_image, **kwargs):
        seen.update(kwargs)
        return torch.zeros(3, 4, 4)

    worker = SimpleNamespace(
        models_processor=SimpleNamespace(apply_denoiser_unet=_apply)
    )
    image = torch.zeros(3, 4, 4, dtype=torch.uint8)
    FrameWorker._apply_denoiser_pass(worker, image, {}, "Before", {"layer": 1})
    assert seen["enable_color_match"] is True


def test_brow_indices_do_not_overlap_eyes_or_mouth():
    brows = set(PerformRecast.BROWS_INDICES)
    assert brows.isdisjoint(PerformRecast.EYE_INDICES)
    assert brows.isdisjoint(PerformRecast.MOUTH_INDICES)
