"""
VR-EC-*, VR-PC-*, VR-SE-* tests for app.helpers.vr_utils

All tests run on CPU; no GPU required.
External Equirec2Perspec_vr / Perspec2Equirec_vr are imported as-is — the test
uses a small (90×180) equirectangular so they execute quickly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.helpers.vr_utils import (
    EquirectangularConverter,
    PerspectiveConverter,
    _SOBEL_X_KERNEL,
    _SOBEL_Y_KERNEL,
    _get_sobel_kernels,
)

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_equirect(h: int = 90, w: int = 180) -> np.ndarray:
    """Minimal gradient equirectangular image (HWC, uint8, RGB)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img[:, :, 2] = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))
    return img


def make_crop_tensor(h: int = 64, w: int = 64) -> torch.Tensor:
    """Random uint8 CHW tensor to use as a processed crop."""
    return torch.randint(0, 256, (3, h, w), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Module-level kernel tests
# ---------------------------------------------------------------------------


def test_sobel_kernels_shape():
    assert _SOBEL_X_KERNEL.shape == (1, 1, 3, 3)
    assert _SOBEL_Y_KERNEL.shape == (1, 1, 3, 3)


def test_get_sobel_kernels_moves_to_device():
    kx, ky = _get_sobel_kernels(CPU)
    assert kx.device.type == "cpu"
    assert ky.device.type == "cpu"


# ---------------------------------------------------------------------------
# VR-EC-* EquirectangularConverter
# ---------------------------------------------------------------------------


class TestEquirectangularConverter:
    @pytest.fixture(autouse=True)
    def converter(self):
        img = make_equirect(90, 180)
        self.ec = EquirectangularConverter(img, CPU)

    # VR-EC-01: constructor stores dimensions correctly
    def test_dimensions_stored(self):
        assert self.ec.height == 90
        assert self.ec.width == 180
        assert self.ec.channels == 3

    # VR-EC-01b: tensor has correct shape on CPU
    def test_tensor_shape(self):
        t = self.ec.equirect_tensor_cxhxw_rgb_uint8
        assert t.shape == (3, 90, 180)
        assert t.dtype == torch.uint8

    # VR-EC-04: center bbox → theta≈0, phi≈0
    def test_center_bbox_gives_near_zero_angles(self):
        # Center of a 90×180 image
        bbox = np.array([85.0, 40.0, 95.0, 50.0])  # center ~(90, 45) on 180×90
        theta, phi = self.ec.calculate_theta_phi_from_bbox(bbox)
        assert abs(theta) < 10.0  # roughly centred horizontally
        assert abs(phi) < 10.0  # roughly centred vertically

    # VR-EC-05: left-half bbox → negative theta
    def test_left_bbox_gives_negative_theta(self):
        bbox = np.array([10.0, 35.0, 50.0, 55.0])  # left quarter
        theta, _ = self.ec.calculate_theta_phi_from_bbox(bbox)
        assert theta < 0.0

    # VR-EC-06: right-half bbox → positive theta
    def test_right_bbox_gives_positive_theta(self):
        bbox = np.array([130.0, 35.0, 170.0, 55.0])  # right quarter
        theta, _ = self.ec.calculate_theta_phi_from_bbox(bbox)
        assert theta > 0.0

    # VR-EC-02: get_perspective_crop returns CHW tensor
    def test_get_perspective_crop_shape(self):
        crop = self.ec.get_perspective_crop(FOV=60, THETA=0, PHI=0, height=64, width=64)
        assert isinstance(crop, torch.Tensor)
        assert crop.ndim == 3
        assert crop.shape[0] == 3  # C=3

    # VR-EC-07: same params twice → identical tensors
    def test_get_perspective_crop_deterministic(self):
        c1 = self.ec.get_perspective_crop(FOV=60, THETA=0, PHI=0, height=64, width=64)
        c2 = self.ec.get_perspective_crop(FOV=60, THETA=0, PHI=0, height=64, width=64)
        assert torch.equal(c1, c2)


# ---------------------------------------------------------------------------
# VR-PC-* / VR-SE-* PerspectiveConverter
# ---------------------------------------------------------------------------


class TestPerspectiveConverter:
    @pytest.fixture(autouse=True)
    def converter(self):
        self.h, self.w = 90, 180
        img = make_equirect(self.h, self.w)
        self.pc = PerspectiveConverter(img, CPU)

    # VR-PC-01: stitch modifies target in-place (same storage)
    def test_stitch_modifies_in_place(self):
        target = torch.zeros(3, self.h, self.w, dtype=torch.uint8)
        original_data_ptr = target.data_ptr()
        crop = make_crop_tensor()
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=0.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=True,
        )
        assert target.data_ptr() == original_data_ptr

    # VR-PC-02: is_left_eye=True → right half unchanged
    def test_left_eye_leaves_right_half_unchanged(self):
        half = self.w // 2
        target = torch.zeros(3, self.h, self.w, dtype=torch.uint8)
        right_half_before = target[:, :, half:].clone()
        crop = make_crop_tensor()
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=-90.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=True,
        )
        # Right half must not have changed
        assert torch.equal(target[:, :, half:], right_half_before)

    # VR-PC-03: is_left_eye=False → left half unchanged
    def test_right_eye_leaves_left_half_unchanged(self):
        half = self.w // 2
        target = torch.zeros(3, self.h, self.w, dtype=torch.uint8)
        left_half_before = target[:, :, :half].clone()
        crop = make_crop_tensor()
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=90.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=False,
        )
        assert torch.equal(target[:, :, :half], left_half_before)

    # VR-PC-05: None crop → early return, target unchanged
    def test_none_crop_early_return(self):
        target = torch.ones(3, self.h, self.w, dtype=torch.uint8) * 42
        target_before = target.clone()
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=None,
            theta=0.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=True,
        )
        assert torch.equal(target, target_before)

    # VR-PC-06: empty crop tensor → early return, target unchanged
    def test_empty_crop_early_return(self):
        target = torch.ones(3, self.h, self.w, dtype=torch.uint8) * 42
        target_before = target.clone()
        empty_crop = torch.zeros(0, dtype=torch.uint8)
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=empty_crop,
            theta=0.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=True,
        )
        assert torch.equal(target, target_before)

    # VR-PC-09: output dtype is uint8 after stitch
    def test_output_dtype_uint8(self):
        target = torch.zeros(3, self.h, self.w, dtype=torch.uint8)
        crop = make_crop_tensor()
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=0.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=True,
        )
        assert target.dtype == torch.uint8

    # VR-PC-10: values in [0, 255] after blending
    def test_no_value_overflow(self):
        target = torch.full((3, self.h, self.w), 200, dtype=torch.uint8)
        crop = make_crop_tensor()
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=0.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=True,
        )
        assert target.min().item() >= 0
        assert target.max().item() <= 255

    # VR-PC-07: feathering returns float mask in [0, 1]
    def test_apply_feathering_range(self):
        mask = torch.zeros(1, self.h, self.w, dtype=torch.bool)
        mask[:, 20:70, 40:140] = True
        feathered = self.pc._apply_feathering(
            mask, feather_radius=5, erosion_kernel_size=5
        )
        assert feathered.dtype == torch.float32
        assert feathered.min().item() >= 0.0
        assert feathered.max().item() <= 1.0

    # VR-PC-11: blur cache reuse — same (k, σ) hits the cache
    def test_blur_cache_reuse(self):
        mask = torch.zeros(1, self.h, self.w, dtype=torch.bool)
        mask[:, 10:80, 20:160] = True
        _ = self.pc._apply_feathering(mask, feather_radius=5, erosion_kernel_size=5)
        cache_size_after_first = len(self.pc._blur_cache)
        _ = self.pc._apply_feathering(mask, feather_radius=5, erosion_kernel_size=5)
        assert len(self.pc._blur_cache) == cache_size_after_first  # no new entry


# ---------------------------------------------------------------------------
# VR-SE-* Single-Eye Mode — eye_region_mask construction
# ---------------------------------------------------------------------------


class TestEyeRegionMask:
    """
    Test the eye_region_mask logic directly by inspecting what
    stitch_single_perspective passes down, using a controlled crop that lands
    in a known region of the equirectangular.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.h, self.w = 90, 180
        img = make_equirect(self.h, self.w)
        self.pc = PerspectiveConverter(img, CPU)

    # VR-SE-01: is_left_eye=None → mask covers the full frame width
    def test_none_eye_mask_is_full_frame(self):
        """
        Verify by checking that pixels on BOTH halves can be modified when
        is_left_eye=None, unlike when is_left_eye=True (only left) or False (only right).
        We use a bright crop at theta=0 which lands near the centre seam.
        """
        half = self.w // 2

        # Full-frame mode
        target_full = torch.zeros(3, self.h, self.w, dtype=torch.uint8)
        crop = torch.full((3, 64, 64), 255, dtype=torch.uint8)
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target_full,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=0.0,
            phi=0.0,
            fov=90.0,
            is_left_eye=None,
        )

        # Left-eye-only mode
        target_left = torch.zeros(3, self.h, self.w, dtype=torch.uint8)
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target_left,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=0.0,
            phi=0.0,
            fov=90.0,
            is_left_eye=True,
        )

        # Full-frame mode should produce non-zero pixels on the right half,
        # whereas left-eye-only mode must leave the right half zero.
        # (The crop at theta=0 straddles the seam, so full-frame bleeds right.)
        right_half_full = target_full[:, :, half:].float().sum().item()
        right_half_left = target_left[:, :, half:].float().sum().item()
        assert right_half_full > right_half_left, (
            "Full-frame mode should allow stitching into the right half; "
            "left-eye-only mode should not."
        )

    # VR-SE-02: is_left_eye=True → right half is protected (zero if target was zero)
    def test_true_eye_protects_right_half(self):
        half = self.w // 2
        target = torch.zeros(3, self.h, self.w, dtype=torch.uint8)
        crop = torch.full((3, 64, 64), 200, dtype=torch.uint8)
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=-90.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=True,
        )
        # Right half must remain all-zero
        assert target[:, :, half:].sum().item() == 0

    # VR-SE-03: is_left_eye=False → left half is protected
    def test_false_eye_protects_left_half(self):
        half = self.w // 2
        target = torch.zeros(3, self.h, self.w, dtype=torch.uint8)
        crop = torch.full((3, 64, 64), 200, dtype=torch.uint8)
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=90.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=False,
        )
        assert target[:, :, :half].sum().item() == 0

    # VR-SE-04: mask shape matches equirectangular dimensions
    def test_mask_shape_matches_equirect(self):
        """
        Indirectly verify by ensuring the target tensor is not resized by stitch.
        """
        target = torch.zeros(3, self.h, self.w, dtype=torch.uint8)
        crop = make_crop_tensor(32, 32)
        self.pc.stitch_single_perspective(
            target_equirect_torch_cxhxw_rgb_uint8=target,
            processed_crop_torch_cxhxw_rgb_uint8=crop,
            theta=0.0,
            phi=0.0,
            fov=60.0,
            is_left_eye=None,
        )
        assert target.shape == (3, self.h, self.w)
