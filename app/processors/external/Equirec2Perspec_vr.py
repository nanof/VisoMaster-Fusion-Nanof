import cv2
from collections import OrderedDict
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

_PERSP_GRID_CACHE: OrderedDict = OrderedDict()  # module-level cache — persists across frames
_PERSP_GRID_CACHE_MAX = 256


def _get_cached_grid(cache_key, compute_fn):
    """Get or compute a perspective sampling grid, evicting oldest if cache is full."""
    if cache_key in _PERSP_GRID_CACHE:
        return _PERSP_GRID_CACHE[cache_key]
    if len(_PERSP_GRID_CACHE) >= _PERSP_GRID_CACHE_MAX:
        _PERSP_GRID_CACHE.popitem(last=False)  # evict oldest
    grid = compute_fn()
    _PERSP_GRID_CACHE[cache_key] = grid
    return grid


class Equirectangular:
    def __init__(self, img_tensor_cxhxw_rgb_uint8: torch.Tensor):
        """
        Initializes with an equirectangular image tensor.
        :param img_tensor_cxhxw_rgb_uint8: Torch tensor (C, H, W) in RGB, uint8 format, on GPU.
        """
        if not isinstance(img_tensor_cxhxw_rgb_uint8, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")
        if img_tensor_cxhxw_rgb_uint8.ndim != 3:
            raise ValueError("Input tensor must be 3-dimensional (C, H, W).")

        self._img_tensor_cxhxw_rgb_float = img_tensor_cxhxw_rgb_uint8.float() / 255.0 # Normalize to [0,1]
        self.device = img_tensor_cxhxw_rgb_uint8.device
        self._channels, self._height, self._width = self._img_tensor_cxhxw_rgb_float.shape

    def GetPerspective(self, FOV: float, THETA: float, PHI: float, height: int, width: int) -> torch.Tensor:
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        # Returns: Perspective crop as Torch tensor (C, H, W) in RGB, uint8 format, on GPU.

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        # Cache key uses only (FOV, height, width) — the perspective plane grid (persp_xx,
        # persp_yy, w_len, h_len) depends only on FOV and crop dimensions, NOT on THETA/PHI.
        # THETA/PHI only affect the rotation matrices computed below, which are cheap (2×
        # cv2.Rodrigues calls) and are NOT cached here.  Including THETA/PHI in the key
        # was a bug: it caused a cache miss for every unique viewing direction even when the
        # FOV and size were identical, wasting GPU memory (each entry holds two H×W tensors)
        # and CPU time on repeated meshgrid computations.
        cache_key = (FOV, height, width)
        if cache_key in _PERSP_GRID_CACHE:
            persp_xx, persp_yy, w_len, h_len = _PERSP_GRID_CACHE[cache_key]
        else:
            wFOV = FOV
            hFOV = float(height) / float(width) * wFOV
            w_len = torch.tan(torch.deg2rad(torch.tensor(wFOV / 2.0, device=self.device)))
            h_len = torch.tan(torch.deg2rad(torch.tensor(hFOV / 2.0, device=self.device)))

            # Create perspective grid
            persp_x_coords = torch.linspace(-w_len, w_len, width, device=self.device, dtype=torch.float32)
            persp_y_coords = torch.linspace(-h_len, h_len, height, device=self.device, dtype=torch.float32)
            persp_yy, persp_xx = torch.meshgrid(persp_y_coords, persp_x_coords, indexing='ij')
            if len(_PERSP_GRID_CACHE) >= _PERSP_GRID_CACHE_MAX:
                _PERSP_GRID_CACHE.popitem(last=False)  # evict oldest
            _PERSP_GRID_CACHE[cache_key] = (persp_xx, persp_yy, w_len, h_len)

        # Points in 3D space on the perspective image plane (camera looking along X-axis)
        x_3d = torch.ones_like(persp_xx)
        y_3d = persp_xx
        z_3d = -persp_yy # Negative because image y is typically top-to-bottom, z is up in 3D

        # Normalize to unit vectors
        D = torch.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
        xyz_persp_norm = torch.stack((x_3d/D, y_3d/D, z_3d/D), dim=2) # H, W, 3

        # Rotation matrices
        y_axis_np = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis_np = np.array([0.0, 0.0, 1.0], np.float32)

        # 1. Yaw around Z-axis
        R1_np, _ = cv2.Rodrigues(z_axis_np * np.radians(THETA))
        # 2. Pitch around new Y-axis
        R2_np, _ = cv2.Rodrigues(np.dot(R1_np, y_axis_np) * np.radians(-PHI))
        R1_torch = torch.from_numpy(R1_np).float().to(self.device)
        R2_torch = torch.from_numpy(R2_np).float().to(self.device)

        # Rotate the 3D points
        # (H, W, 3) -> (H*W, 3) -> (3, H*W) for matmul
        xyz_flat = xyz_persp_norm.reshape(-1, 3).T
        # Apply rotations: R = R2 @ R1
        # Rotated_xyz = R @ xyz_persp_norm (if xyz_persp_norm is column vectors)
        # Here, we transform points from perspective camera space to world space, then to equirectangular.
        # The original code implies rotations to align the perspective view within the equirectangular sphere.
        # So, we rotate the perspective rays.
        rotated_xyz_flat = R2_torch @ R1_torch @ xyz_flat
        rotated_xyz = rotated_xyz_flat.T.reshape(height, width, 3) # H, W, 3

        # Convert Cartesian to spherical coordinates (longitude, latitude)
        # x_eq = rotated_xyz[..., 0], y_eq = rotated_xyz[..., 1], z_eq = rotated_xyz[..., 2]
        lon_rad = torch.atan2(rotated_xyz[..., 1], rotated_xyz[..., 0]) # Longitude
        # Bug 1 fix: clamp to [-1, 1] before asin to avoid NaN from float rounding at poles
        lat_rad = torch.asin(torch.clamp(rotated_xyz[..., 2], -1.0, 1.0))  # Latitude

        # Convert spherical to equirectangular pixel coordinates
        lon_px = (lon_rad / torch.pi) * equ_cx + equ_cx # Map [-pi, pi] to [0, equ_w-1]
        lat_px = (-lat_rad / (torch.pi / 2.0)) * equ_cy + equ_cy # Map [-pi/2, pi/2] to [0, equ_h-1] (lat is inverted)

        # Pre-sanitise pixel coords before normalising for grid_sample.
        # Longitude: wrap circularly so the 0°/360° seam maps cleanly.
        # fmod handles the case where atan2 returns exactly ±π, producing lon_px == equ_w.
        lon_px = torch.fmod(lon_px, equ_w)
        # Latitude: clamp at the poles so float rounding near ±90° never exceeds image bounds.
        lat_px = torch.clamp(lat_px, 0.0, equ_h - 1.0)

        # Create grid for grid_sample (expects N, H_out, W_out, 2) with (x, y) in [-1, 1]
        grid_x = (lon_px / (equ_w - 1)) * 2.0 - 1.0
        grid_y = (lat_px / (equ_h - 1)) * 2.0 - 1.0
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0) # 1, H_out, W_out, 2

        # Sample from the equirectangular image.
        # padding_mode='border' replicates edge pixels for any residual out-of-bound coords
        # after the fmod/clamp above, avoiding the black-pixel artefacts that 'zeros' caused.
        persp_float = F.grid_sample(self._img_tensor_cxhxw_rgb_float.unsqueeze(0), grid,
                                    mode='bilinear', padding_mode='border', align_corners=True)

        # Use round_() before byte() for nearest-integer rounding (matches P2E convention).
        # Plain .byte() truncates toward zero, causing a systematic -0.5 LSB bias on
        # pixel values that would otherwise round up (e.g. 127.9 → 127 instead of 128).
        persp_uint8 = torch.clamp(persp_float.squeeze(0) * 255.0, 0, 255).round_().byte()
        return persp_uint8

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height
