"""Helpers for Swap-all-by-index / Swap-all-by-random control modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from app.helpers import qt_lifecycle

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


def checked_input_face_buttons(main_window: "MainWindow") -> list:
    """Ordered checked input-face cards.

    Cards destroyed by a list clear can still sit in ``input_faces``; reading
    them raises from shiboken, which would kill the calling worker thread, so
    they are treated as unchecked.
    """
    return [
        btn
        for btn in list(main_window.input_faces.values())
        if qt_lifecycle.is_checked(btn)
    ]


def pinned_checked_input_indices(main_window: "MainWindow") -> set[int]:
    """Checked input-face indices marked fixed for random reshuffles."""
    return {
        idx
        for idx, btn in enumerate(checked_input_face_buttons(main_window))
        if getattr(btn, "random_fixed", False)
    }


def pinned_indices_from_checked(checked_inputs_ordered: list) -> set[int]:
    """Pinned indices within an already-ordered checked-input list."""
    return {
        i
        for i, btn in enumerate(checked_inputs_ordered)
        if getattr(btn, "random_fixed", False)
    }


def collapse_checked_inputs_to_first(main_window: "MainWindow") -> None:
    """Keep only the first checked input face selected and assigned.

    Used when leaving swap-all modes: the multi-check pool is for per-detection
    assignment, not for blending into ``assigned_input_faces``.
    """
    from app.ui.widgets.actions import common_actions as common_widget_actions

    checked = checked_input_face_buttons(main_window)
    first = checked[0] if checked else None

    for btn in qt_lifecycle.alive_values(main_window.input_faces):
        qt_lifecycle.set_checked(btn, btn is first)

    for target_face in list(getattr(main_window, "target_faces", {}).values()):
        target_face.assigned_input_faces.clear()
        if first is not None:
            target_face.assigned_input_faces[first.face_id] = first.embedding_store
        target_face.calculate_assigned_input_embedding()

    common_widget_actions.refresh_frame(main_window)
