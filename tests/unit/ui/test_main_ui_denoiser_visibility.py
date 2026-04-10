from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.ui.main_ui import (
    DENOISER_MODE_FULL_RESTORE,
    DENOISER_MODE_SINGLE_STEP,
    MainWindow,
)


def _make_parameter_widget():
    widget = MagicMock()
    widget.row_widget = MagicMock()
    widget.label_widget = MagicMock()
    widget.reset_default_button = MagicMock()
    widget.line_edit = MagicMock()
    return widget


def test_single_step_denoiser_mode_collapses_ddim_row_wrappers():
    single_step_widget = _make_parameter_widget()
    ddim_steps_widget = _make_parameter_widget()
    cfg_scale_widget = _make_parameter_widget()
    main_window = SimpleNamespace(
        parameter_widgets={
            "DenoiserSingleStepTimestepSliderBefore": single_step_widget,
            "DenoiserDDIMStepsSliderBefore": ddim_steps_widget,
            "DenoiserCFGScaleDecimalSliderBefore": cfg_scale_widget,
        }
    )

    MainWindow.update_denoiser_controls_visibility_for_pass(
        main_window, "Before", DENOISER_MODE_SINGLE_STEP
    )

    single_step_widget.row_widget.setVisible.assert_called_once_with(True)
    ddim_steps_widget.row_widget.setVisible.assert_called_once_with(False)
    cfg_scale_widget.row_widget.setVisible.assert_called_once_with(False)


def test_full_restore_denoiser_mode_collapses_single_step_row_wrapper():
    single_step_widget = _make_parameter_widget()
    ddim_steps_widget = _make_parameter_widget()
    cfg_scale_widget = _make_parameter_widget()
    main_window = SimpleNamespace(
        parameter_widgets={
            "DenoiserSingleStepTimestepSliderAfter": single_step_widget,
            "DenoiserDDIMStepsSliderAfter": ddim_steps_widget,
            "DenoiserCFGScaleDecimalSliderAfter": cfg_scale_widget,
        }
    )

    MainWindow.update_denoiser_controls_visibility_for_pass(
        main_window, "After", DENOISER_MODE_FULL_RESTORE
    )

    single_step_widget.row_widget.setVisible.assert_called_once_with(False)
    ddim_steps_widget.row_widget.setVisible.assert_called_once_with(True)
    cfg_scale_widget.row_widget.setVisible.assert_called_once_with(True)
