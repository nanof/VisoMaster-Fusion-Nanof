from types import SimpleNamespace

from app.ui.widgets.actions.control_actions import handle_ff_auto_quality_toggle


def _quality_widget():
    widget = SimpleNamespace(
        enabled=True,
        row_widget=SimpleNamespace(enabled=True),
        label_widget=SimpleNamespace(enabled=True),
        reset_default_button=SimpleNamespace(enabled=True),
        line_edit=SimpleNamespace(enabled=True),
    )

    def set_enabled(value):
        widget.enabled = value

    widget.setEnabled = set_enabled
    for child_name in (
        "row_widget",
        "label_widget",
        "reset_default_button",
        "line_edit",
    ):
        child = getattr(widget, child_name)

        def _set(value, target=child):
            target.enabled = value

        child.setEnabled = _set
    return widget


def test_auto_quality_toggle_disables_manual_slider():
    widget = _quality_widget()
    window = SimpleNamespace(parameter_widgets={"FFQualitySlider": widget})
    handle_ff_auto_quality_toggle(window, True)
    assert widget.enabled is False
    assert widget.row_widget.enabled is False
    assert widget.label_widget.enabled is False
    assert widget.reset_default_button.enabled is False
    assert widget.line_edit.enabled is False


def test_auto_quality_toggle_restores_manual_slider():
    widget = _quality_widget()
    widget.enabled = False
    window = SimpleNamespace(parameter_widgets={"FFQualitySlider": widget})
    handle_ff_auto_quality_toggle(window, False)
    assert widget.enabled is True
    assert widget.line_edit.enabled is True
