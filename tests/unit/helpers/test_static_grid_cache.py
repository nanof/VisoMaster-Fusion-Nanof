"""Bounded LRU for get_grid_for_pasting static grids."""

from __future__ import annotations

import torch
from skimage import transform as trans

from app.helpers import miscellaneous as misc


def test_static_grid_cache_is_bounded_lru():
    misc.clear_static_grid_cache()
    tform = trans.SimilarityTransform(scale=1.0, rotation=0.0, translation=(0.0, 0.0))
    device = torch.device("cpu")
    original_max = misc._STATIC_GRID_CACHE_MAX
    misc._STATIC_GRID_CACHE_MAX = 2
    try:
        for size in (8, 16, 24):
            misc.get_grid_for_pasting(tform, size, size, size, size, device)
        assert len(misc._static_grid_cache) == 2
        keys = list(misc._static_grid_cache)
        assert keys[0][0] == 16
        assert keys[1][0] == 24
    finally:
        misc._STATIC_GRID_CACHE_MAX = original_max
        misc.clear_static_grid_cache()
