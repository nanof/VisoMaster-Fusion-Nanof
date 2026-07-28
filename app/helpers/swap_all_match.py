"""Helpers for Swap-all-by-index / Swap-all-by-random control modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def swap_all_match_active(control: Mapping[str, object]) -> bool:
    """True when either index or random swap-all matching is enabled."""
    if control.get("SwapOnlyBestMatchEnableToggle", False):
        return False
    return bool(
        control.get("SequentialTargetMatchEnableToggle", False)
        or control.get("RandomTargetMatchEnableToggle", False)
    )


def swap_all_assignment_mode(control: Mapping[str, object]) -> str:
    """Return ``\"random\"`` or ``\"index\"`` for new-face input assignment."""
    if control.get("RandomTargetMatchEnableToggle", False) and not control.get(
        "SequentialTargetMatchEnableToggle", False
    ):
        return "random"
    return "index"


def pinned_checked_input_indices(main_window: "MainWindow") -> set[int]:
    """Checked input-face indices marked fixed for random reshuffles."""
    pinned: set[int] = set()
    idx = 0
    for _fid, btn in main_window.input_faces.items():
        if not btn.isChecked():
            continue
        if getattr(btn, "random_fixed", False):
            pinned.add(idx)
        idx += 1
    return pinned


def pinned_indices_from_checked(checked_inputs_ordered: list) -> set[int]:
    """Pinned indices within an already-ordered checked-input list."""
    return {
        i
        for i, btn in enumerate(checked_inputs_ordered)
        if getattr(btn, "random_fixed", False)
    }
