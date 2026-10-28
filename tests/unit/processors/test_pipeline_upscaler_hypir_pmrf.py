"""Vendor import smoke for HYPIR / PMRF (no weights)."""

from __future__ import annotations


def test_hypir_and_pmrf_listed_as_pipeline_types():
    from app.processors.pytorch_extras.pipeline_upscaler.scale import (
        PIPELINE_UPSCALER_TYPE_HYPIR,
        PIPELINE_UPSCALER_TYPE_PMRF,
        PIPELINE_UPSCALER_TYPES,
    )

    assert PIPELINE_UPSCALER_TYPE_HYPIR in PIPELINE_UPSCALER_TYPES
    assert PIPELINE_UPSCALER_TYPE_PMRF in PIPELINE_UPSCALER_TYPES


def test_missing_assets_unknown_type():
    from app.processors.pytorch_extras.pipeline_upscaler.paths import missing_assets

    assert missing_assets("not-a-model") == ["unknown type: not-a-model"]


def test_pmrf_create_arch_import():
    from app.processors.external.pmrf.utils.create_arch import create_arch

    assert callable(create_arch)


def test_pmrf_swinir_import_without_timm():
    from app.processors.external.pmrf.arch.swinir.swinir import SwinIR

    assert SwinIR is not None


def test_pmrf_vendor_files_exist():
    from app.processors.pytorch_extras.pipeline_upscaler.paths import vendor_pmrf_root

    root = vendor_pmrf_root()
    assert (root / "lightning_models" / "mmse_rectified_flow.py").is_file()
    assert (root / "LICENSE").is_file()


def test_hypir_base_does_not_import_unvendored_tiled_vae():
    from app.processors.pytorch_extras.pipeline_upscaler.paths import vendor_hypir_root

    text = (vendor_hypir_root() / "HYPIR" / "enhancer" / "base.py").read_text(
        encoding="utf-8"
    )
    assert "tiled_vae" not in text


def test_hypir_vendor_files_exist():
    from app.processors.pytorch_extras.pipeline_upscaler.paths import vendor_hypir_root

    root = vendor_hypir_root()
    assert (root / "HYPIR" / "enhancer" / "sd2.py").is_file()
    assert (root / "HYPIR" / "enhancer" / "base.py").is_file()
    assert (root / "LICENSE").is_file()


def test_hypir_sd2_mirrors_are_public_not_gated_stabilityai():
    from app.processors.pytorch_extras.pipeline_upscaler.download import _HF_SD2_REPOS

    assert _HF_SD2_REPOS
    assert all("stabilityai/" not in repo for repo in _HF_SD2_REPOS)
    assert "Manojb/stable-diffusion-2-1-base" in _HF_SD2_REPOS
