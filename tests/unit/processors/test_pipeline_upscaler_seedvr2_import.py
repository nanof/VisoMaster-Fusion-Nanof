"""SeedVR2 import isolation: no vendor ``src`` on sys.path, six finder patched."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from app.processors.pytorch_extras.pipeline_upscaler.paths import vendor_seedvr2_root
from app.processors.pytorch_extras.pipeline_upscaler.seedvr2_backend import (
    _drop_polluted_top_level_src,
    _patch_six_meta_importers,
    _prepare_import_env,
    _remove_vendor_from_sys_path,
)


def test_prepare_import_env_does_not_insert_vendor_root():
    root = str(vendor_seedvr2_root())
    previous = list(sys.path)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        _prepare_import_env()
        vendor = vendor_seedvr2_root().resolve()
        assert vendor not in {Path(p).resolve() for p in sys.path if p}
    finally:
        sys.path[:] = previous


def test_remove_vendor_from_sys_path():
    root = str(vendor_seedvr2_root())
    previous = list(sys.path)
    sys.path.insert(0, root)
    try:
        _remove_vendor_from_sys_path()
        assert root not in sys.path
    finally:
        sys.path[:] = previous


def test_drop_polluted_top_level_src():
    previous = sys.modules.get("src")
    child_name = "src.core.generation_utils"
    previous_child = sys.modules.get(child_name)
    stub = types.ModuleType("src")
    stub.__file__ = str(vendor_seedvr2_root() / "src" / "__init__.py")
    stub.__path__ = [str(vendor_seedvr2_root() / "src")]
    sys.modules["src"] = stub
    sys.modules[child_name] = types.ModuleType(child_name)
    try:
        _drop_polluted_top_level_src()
        assert "src" not in sys.modules
        assert child_name not in sys.modules
    finally:
        if previous is None:
            sys.modules.pop("src", None)
        else:
            sys.modules["src"] = previous
        if previous_child is None:
            sys.modules.pop(child_name, None)
        else:
            sys.modules[child_name] = previous_child


def test_patch_six_meta_importers_adds_path():
    class _SixMetaPathImporter:
        pass

    importer = _SixMetaPathImporter()
    sys.meta_path.append(importer)
    try:
        _patch_six_meta_importers()
        assert importer._path == []
        assert _SixMetaPathImporter._path == []
    finally:
        sys.meta_path.remove(importer)


def test_prepare_import_env_keeps_six_moves_urllib():
    import six.moves.urllib as urllib_moves

    previous = list(sys.meta_path)
    try:
        _prepare_import_env()
        import six.moves.urllib as urllib_moves_after

        assert urllib_moves_after is not None
        assert urllib_moves.parse is not None
        importers = [i for i in sys.meta_path if type(i).__name__ == "_SixMetaPathImporter"]
        assert importers
        assert all(hasattr(i, "_path") for i in importers)
    finally:
        sys.meta_path[:] = previous


def test_ensure_vae_tile_sizes_fills_none():
    from types import SimpleNamespace

    from app.processors.pytorch_extras.pipeline_upscaler.seedvr2_backend import (
        _ensure_vae_tile_sizes,
    )

    runner = SimpleNamespace(
        encode_tiled=False,
        decode_tiled=False,
        encode_tile_size=None,
        decode_tile_size=None,
        encode_tile_overlap=None,
        decode_tile_overlap=None,
    )
    _ensure_vae_tile_sizes(runner, True)
    assert runner.encode_tiled is True
    assert runner.decode_tiled is True
    assert runner.encode_tile_size == (512, 512)
    assert runner.decode_tile_size == (512, 512)
    assert runner.encode_tile_overlap == (64, 64)
    assert runner.decode_tile_overlap == (64, 64)
    h, w = runner.encode_tile_size
    # This is the unpack that crashed with None.
    assert h == 512 and w == 512
