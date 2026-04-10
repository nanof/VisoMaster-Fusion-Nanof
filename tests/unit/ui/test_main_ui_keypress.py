from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock


def _module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _load_main_ui_module():
    qt_widgets = _module(
        "PySide6.QtWidgets",
        QMainWindow=type("QMainWindow", (), {}),
        QListWidget=type("QListWidget", (), {}),
        QWidget=type("QWidget", (), {}),
    )
    qt_widgets.__getattr__ = lambda attr: type(attr, (), {})
    qt_core = _module(
        "PySide6.QtCore",
        Qt=SimpleNamespace(
            Key_Escape=1,
            Key_F11=2,
            Key_T=3,
            Key_V=4,
            Key_C=5,
            Key_D=6,
            Key_A=7,
            Key_Z=8,
            Key_Space=9,
        ),
        Signal=lambda *args, **kwargs: object(),
        Slot=lambda *args, **kwargs: lambda func: func,
    )
    qt_core.__getattr__ = lambda attr: type(attr, (), {})
    qt_gui = _module("PySide6.QtGui")
    qt_gui.__getattr__ = lambda attr: type(attr, (), {})
    pyside6 = _module(
        "PySide6",
        QtWidgets=qt_widgets,
        QtCore=qt_core,
        QtGui=qt_gui,
    )

    widget_components = _module(
        "app.ui.widgets.widget_components",
        ToggleButton=type("ToggleButton", (), {}),
        SelectionBox=type("SelectionBox", (), {}),
        ParameterDecimalSlider=type("ParameterDecimalSlider", (), {}),
        ParameterSlider=type("ParameterSlider", (), {}),
        ParameterLineEdit=type("ParameterLineEdit", (), {}),
        TargetFaceCardButton=type("TargetFaceCardButton", (), {}),
        InputFaceCardButton=type("InputFaceCardButton", (), {}),
        EmbeddingCardButton=type("EmbeddingCardButton", (), {}),
        TargetMediaCardButton=type("TargetMediaCardButton", (), {}),
    )
    widget_components.__getattr__ = lambda attr: type(attr, (), {})

    stub_modules = {
        "PySide6": pyside6,
        "PySide6.QtWidgets": qt_widgets,
        "PySide6.QtCore": qt_core,
        "PySide6.QtGui": qt_gui,
        "torch": _module("torch"),
        "app.ui.core.main_window": _module(
            "app.ui.core.main_window", Ui_MainWindow=type("Ui_MainWindow", (), {})
        ),
        "app.ui.widgets.actions.common_actions": _module(
            "app.ui.widgets.actions.common_actions"
        ),
        "app.ui.widgets.actions.card_actions": _module(
            "app.ui.widgets.actions.card_actions"
        ),
        "app.ui.widgets.actions.layout_actions": _module(
            "app.ui.widgets.actions.layout_actions"
        ),
        "app.ui.widgets.actions.video_control_actions": _module(
            "app.ui.widgets.actions.video_control_actions",
            view_fullscreen=MagicMock(),
            toggle_theatre_mode=MagicMock(),
            advance_video_slider_by_n_frames=MagicMock(),
            rewind_video_slider_by_n_frames=MagicMock(),
        ),
        "app.ui.widgets.actions.filter_actions": _module(
            "app.ui.widgets.actions.filter_actions"
        ),
        "app.ui.widgets.actions.save_load_actions": _module(
            "app.ui.widgets.actions.save_load_actions"
        ),
        "app.ui.widgets.actions.list_view_actions": _module(
            "app.ui.widgets.actions.list_view_actions"
        ),
        "app.ui.widgets.actions.graphics_view_actions": _module(
            "app.ui.widgets.actions.graphics_view_actions"
        ),
        "app.ui.widgets.actions.job_manager_actions": _module(
            "app.ui.widgets.actions.job_manager_actions"
        ),
        "app.ui.widgets.actions.preset_actions": _module(
            "app.ui.widgets.actions.preset_actions"
        ),
        "app.ui.widgets.advanced_embedding_editor": _module(
            "app.ui.widgets.advanced_embedding_editor",
            EmbeddingGUI=type("EmbeddingGUI", (), {}),
        ),
        "app.ui.widgets.actions.control_actions": _module(
            "app.ui.widgets.actions.control_actions"
        ),
        "app.processors.video_processor": _module(
            "app.processors.video_processor",
            VideoProcessor=type("VideoProcessor", (), {}),
        ),
        "app.processors.models_processor": _module(
            "app.processors.models_processor",
            ModelsProcessor=type("ModelsProcessor", (), {}),
        ),
        "app.ui.widgets.widget_components": widget_components,
        "app.ui.widgets.event_filters": _module(
            "app.ui.widgets.event_filters",
            GraphicsViewEventFilter=type("GraphicsViewEventFilter", (), {}),
            VideoSeekSliderEventFilter=type("VideoSeekSliderEventFilter", (), {}),
            videoSeekSliderLineEditEventFilter=type(
                "videoSeekSliderLineEditEventFilter", (), {}
            ),
            ListWidgetEventFilter=type("ListWidgetEventFilter", (), {}),
        ),
        "app.ui.widgets.ui_workers": _module("app.ui.widgets.ui_workers"),
        "app.ui.widgets.common_layout_data": _module(
            "app.ui.widgets.common_layout_data", COMMON_LAYOUT_DATA={}
        ),
        "app.ui.widgets.denoiser_layout_data": _module(
            "app.ui.widgets.denoiser_layout_data", DENOISER_LAYOUT_DATA={}
        ),
        "app.ui.widgets.swapper_layout_data": _module(
            "app.ui.widgets.swapper_layout_data",
            SWAPPER_LAYOUT_DATA={},
            MASK_SHOW_DEFAULT="default",
            MASK_SHOW_OPTIONS=[],
        ),
        "app.ui.widgets.settings_layout_data": _module(
            "app.ui.widgets.settings_layout_data", SETTINGS_LAYOUT_DATA={}
        ),
        "app.ui.widgets.face_editor_layout_data": _module(
            "app.ui.widgets.face_editor_layout_data", FACE_EDITOR_LAYOUT_DATA={}
        ),
        "app.helpers.app_metadata": _module(
            "app.helpers.app_metadata",
            get_app_display_metadata=lambda *_args, **_kwargs: SimpleNamespace(
                window_title="VisoMaster"
            ),
        ),
        "app.helpers.miscellaneous": _module(
            "app.helpers.miscellaneous",
            DFMModelManager=type("DFMModelManager", (), {}),
            ParametersDict=dict,
            ThumbnailManager=type("ThumbnailManager", (), {}),
        ),
        "app.helpers.typing_helper": _module(
            "app.helpers.typing_helper",
            FacesParametersTypes=dict,
            ParametersTypes=dict,
            ControlTypes=dict,
            MarkerTypes=dict,
        ),
        "app.processors.models_data": _module(
            "app.processors.models_data", models_dir="models"
        ),
    }

    saved_modules = {
        name: sys.modules.get(name) for name in [*stub_modules, "app.ui.main_ui"]
    }

    try:
        for name, module in stub_modules.items():
            sys.modules[name] = module
        sys.modules.pop("app.ui.main_ui", None)
        return importlib.import_module("app.ui.main_ui")
    finally:
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_escape_exits_theatre_mode_directly_when_combined_mode_is_enabled():
    main_ui = _load_main_ui_module()
    theatre_toggle = MagicMock()
    fullscreen_toggle = MagicMock()
    main_ui.video_control_actions.toggle_theatre_mode = theatre_toggle
    main_ui.video_control_actions.view_fullscreen = fullscreen_toggle

    main_window = SimpleNamespace(
        control={"TheatreModeUsesFullscreenToggle": True},
        is_theatre_mode=True,
        isFullScreen=lambda: True,
    )
    event = SimpleNamespace(key=lambda: main_ui.QtCore.Qt.Key_Escape)

    main_ui.MainWindow.keyPressEvent(main_window, event)

    theatre_toggle.assert_called_once_with(main_window)
    fullscreen_toggle.assert_not_called()


def test_escape_keeps_existing_fullscreen_behavior_when_combined_mode_is_disabled():
    main_ui = _load_main_ui_module()
    theatre_toggle = MagicMock()
    fullscreen_toggle = MagicMock()
    main_ui.video_control_actions.toggle_theatre_mode = theatre_toggle
    main_ui.video_control_actions.view_fullscreen = fullscreen_toggle

    main_window = SimpleNamespace(
        control={"TheatreModeUsesFullscreenToggle": False},
        is_theatre_mode=True,
        isFullScreen=lambda: True,
    )
    event = SimpleNamespace(key=lambda: main_ui.QtCore.Qt.Key_Escape)

    main_ui.MainWindow.keyPressEvent(main_window, event)

    fullscreen_toggle.assert_called_once_with(main_window)
    theatre_toggle.assert_not_called()
