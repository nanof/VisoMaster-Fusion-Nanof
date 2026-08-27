"""Schema tests for Secondary Swapper widgets in SWAPPER_LAYOUT_DATA (upstream #337)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _stub_module(name: str) -> MagicMock:
    mod = MagicMock()
    mod.__name__ = name
    mod.__spec__ = None
    return mod


for _mod_name in [
    "PySide6",
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "app.ui.widgets.actions.control_actions",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _stub_module(_mod_name)

import app.ui.widgets.actions  # noqa: E402,F401

from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA  # noqa: E402

SWAPPER = SWAPPER_LAYOUT_DATA["Swapper"]
STRENGTH = SWAPPER_LAYOUT_DATA["Swap strength and likeness"]

SECONDARY_WIDGETS = [
    "SecondarySwapperEnableToggle",
    "SecondarySwapModelSelection",
    "SecondarySwapperResSelection",
    "SecondarySwapperBlendAmountSlider",
]


def test_secondary_swapper_widgets_exist():
    for name in SECONDARY_WIDGETS:
        assert name in SWAPPER, f"{name} missing from Swapper layout"


def test_secondary_swapper_defaults_off():
    assert SWAPPER["SecondarySwapperEnableToggle"]["default"] is False


def test_secondary_children_gated_on_toggle():
    for name in (
        "SecondarySwapModelSelection",
        "SecondarySwapperResSelection",
        "SecondarySwapperBlendAmountSlider",
    ):
        entry = SWAPPER[name]
        assert entry.get("parentToggle") == "SecondarySwapperEnableToggle"
        assert entry.get("requiredToggleValue") is True


def test_secondary_model_options_are_compatible_arcface_family():
    opts = SWAPPER["SecondarySwapModelSelection"]["options"]
    assert opts == [
        "Inswapper128",
        "AlphaFace",
        "InStyleSwapper256 Version A",
        "InStyleSwapper256 Version B",
        "InStyleSwapper256 Version C",
    ]
    assert SWAPPER["SecondarySwapModelSelection"]["default"] in opts


def test_secondary_strength_requires_both_toggles():
    entry = STRENGTH["SecondaryStrengthAmountSlider"]
    parents = [p.strip() for p in str(entry["parentToggle"]).split("&")]
    assert "StrengthEnableToggle" in parents
    assert "SecondarySwapperEnableToggle" in parents
    assert entry.get("requiredToggleValue") is True


def test_primary_strength_label_distinguishes_dual_swap():
    assert STRENGTH["StrengthAmountSlider"]["label"] == "Primary Strength"


def test_secondary_parent_toggles_reference_existing_keys():
    known = set()
    for _cat, widgets in SWAPPER_LAYOUT_DATA.items():
        known.update(widgets)
    for name in (
        *SECONDARY_WIDGETS,
        "SecondaryStrengthAmountSlider",
    ):
        entry = SWAPPER.get(name) or STRENGTH.get(name)
        assert entry is not None, name
        if "parentToggle" not in entry:
            continue
        for parent in str(entry["parentToggle"]).replace("|", "&").split("&"):
            parent = parent.strip()
            assert parent in known, f"{name}.parentToggle={parent!r} is missing"
