"""Guards for Qt widgets whose C++ side may already be destroyed.

Card buttons become C++-owned once handed to ``QListWidget.setItemWidget``, so
clearing or taking items destroys them while the Python wrappers can still be
referenced by ``MainWindow`` dicts or read by the processing threads. Any call
on such a wrapper raises shiboken's ``RuntimeError: Internal C++ object
(...) already deleted``, which kills worker threads that only wanted to read a
checkbox state.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:  # shiboken6 ships with PySide6; stay importable in headless test envs.
    from shiboken6 import isValid as _shiboken_is_valid
except ImportError:  # pragma: no cover
    _shiboken_is_valid = None


def is_alive(widget: Any) -> bool:
    """True when ``widget`` still has a live C++ counterpart.

    Objects that are not shiboken wrappers (plain Python widgets, test doubles)
    count as alive.
    """
    if widget is None:
        return False
    if _shiboken_is_valid is not None:
        try:
            return bool(_shiboken_is_valid(widget))
        except TypeError:
            return True
    try:
        widget.objectName()
    except RuntimeError:
        return False
    except AttributeError:
        return True
    return True


def is_checked(widget: Any) -> bool:
    """``isChecked()`` reporting False for destroyed widgets instead of raising."""
    try:
        return bool(widget.isChecked())
    except RuntimeError:
        return False


def set_checked(widget: Any, checked: bool) -> bool:
    """``setChecked()`` that is a no-op on destroyed widgets."""
    try:
        widget.setChecked(bool(checked))
    except RuntimeError:
        return False
    return True


def delete_later(widget: Any) -> None:
    """Schedule deletion only when the C++ object is still around."""
    if is_alive(widget):
        widget.deleteLater()


def prune_dead(buttons: Dict[Any, Any]) -> List[Any]:
    """Drop destroyed widgets from ``buttons`` and return the removed keys."""
    dead_keys = [key for key, widget in list(buttons.items()) if not is_alive(widget)]
    for key in dead_keys:
        buttons.pop(key, None)
    return dead_keys


def alive_values(buttons: Dict[Any, Any]) -> List[Any]:
    """Snapshot of the still-usable widgets in ``buttons``, pruning the rest."""
    prune_dead(buttons)
    return list(buttons.values())
