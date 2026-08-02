"""Guards against reading input-face cards after Qt destroyed their C++ side."""

from types import SimpleNamespace

from app.helpers import qt_lifecycle, swap_all_match


def _destroyed_card():
    """A checkable card whose C++ object is destroyed by clearing its list."""
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    list_widget = QtWidgets.QListWidget()
    card = QtWidgets.QPushButton()
    card.setCheckable(True)
    card.setChecked(True)
    list_widget.setItemWidget(QtWidgets.QListWidgetItem(list_widget), card)

    list_widget.clear()
    # Item widgets are released through a deferred delete, so flush it here.
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    return card


def test_guards_report_destroyed_card_instead_of_raising():
    card = _destroyed_card()

    assert qt_lifecycle.is_alive(card) is False
    assert qt_lifecycle.is_checked(card) is False
    assert qt_lifecycle.set_checked(card, False) is False
    qt_lifecycle.delete_later(card)


def test_prune_dead_drops_destroyed_cards_only():
    from PySide6 import QtWidgets

    dead = _destroyed_card()
    alive = QtWidgets.QPushButton()
    buttons = {"dead": dead, "alive": alive}

    assert qt_lifecycle.prune_dead(buttons) == ["dead"]
    assert buttons == {"alive": alive}
    assert qt_lifecycle.alive_values(buttons) == [alive]


def test_is_alive_treats_plain_objects_as_usable():
    assert qt_lifecycle.is_alive(SimpleNamespace()) is True
    assert qt_lifecycle.is_alive(None) is False


def test_checked_input_face_buttons_skips_destroyed_cards():
    from PySide6 import QtWidgets

    alive = QtWidgets.QPushButton()
    alive.setCheckable(True)
    alive.setChecked(True)
    main_window = SimpleNamespace(
        input_faces={"dead": _destroyed_card(), "alive": alive}
    )

    assert swap_all_match.checked_input_face_buttons(main_window) == [alive]
