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


def test_checked_input_face_buttons_follows_list_widget_order():
    from PySide6 import QtWidgets

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    faces_list = QtWidgets.QListWidget()
    fav_list = QtWidgets.QListWidget()

    def _add(list_widget, name, checked):
        btn = QtWidgets.QPushButton(name)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.list_widget = list_widget
        item = QtWidgets.QListWidgetItem(list_widget)
        list_widget.setItemWidget(item, btn)
        return btn

    # Dict insertion order deliberately differs from visual order.
    fav_b = _add(fav_list, "fav_b", True)
    face_a = _add(faces_list, "face_a", True)
    fav_a = _add(fav_list, "fav_a", True)
    # Visual fav order is fav_b then fav_a (add order). Faces list: face_a.
    main_window = SimpleNamespace(
        input_faces={"x": fav_b, "y": face_a, "z": fav_a},
        inputFacesList=faces_list,
        inputFacesFavoritesList=fav_list,
    )
    ordered = swap_all_match.checked_input_face_buttons(main_window)
    assert [b.text() for b in ordered] == ["face_a", "fav_b", "fav_a"]


def test_clear_checked_inputs_outside_list():
    from PySide6 import QtWidgets

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    faces_list = QtWidgets.QListWidget()
    fav_list = QtWidgets.QListWidget()

    def _add(list_widget, checked=True):
        btn = QtWidgets.QPushButton()
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.list_widget = list_widget
        item = QtWidgets.QListWidgetItem(list_widget)
        list_widget.setItemWidget(item, btn)
        return btn

    face = _add(faces_list)
    fav = _add(fav_list)
    main_window = SimpleNamespace(input_faces={"f": face, "v": fav})
    swap_all_match.clear_checked_inputs_outside_list(main_window, fav_list, keep=fav)
    assert face.isChecked() is False
    assert fav.isChecked() is True
