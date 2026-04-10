"""Essentials tab: shortcuts bound to existing controls/parameters.

- `bind_control` + `data_type="parameter"`: same value as Face Swap / Restorers (per face).
- `bind_control` + `data_type="control"`: same value as Settings (global).
- `mirror_checkable_button`: mirrors a checkable `QPushButton` on the main window.
"""

from typing import Any

from app.ui.widgets.actions import control_actions

ESSENTIALS_PARAMETER_LAYOUT_DATA: Any = {
    "Current face": {
        "Essentials_SwapFacesMirrorToggle": {
            "level": 1,
            "label": "Swap on/off",
            "default": False,
            "mirror_checkable_button": "swapfacesButton",
            "help": "Same as the toolbar “Swap Faces” button: enable or disable swapping.",
        },
        "Essentials_OccluderEnableMirrorToggle": {
            "level": 1,
            "label": "Occlusion mask",
            "default": False,
            "bind_control": "OccluderEnableToggle",
            "help": "Same as Face Swap → Masks.",
            "exec_function": control_actions.handle_face_mask_state_change,
            "exec_function_args": ["OccluderEnableToggle"],
        },
        "Essentials_DFLXSegEnableMirrorToggle": {
            "level": 1,
            "label": "DFL XSeg mask",
            "default": False,
            "bind_control": "DFLXSegEnableToggle",
            "help": "Same as Face Swap → Masks.",
            "exec_function": control_actions.handle_face_mask_state_change,
            "exec_function_args": ["DFLXSegEnableToggle"],
        },
        "Essentials_FaceRestorerEnableMirrorToggle": {
            "level": 1,
            "label": "Enable face restorer",
            "default": False,
            "bind_control": "FaceRestorerEnableToggle",
            "help": "Same as Restorers → Face Restorer.",
            "exec_function": control_actions.handle_restorer_state_change,
            "exec_function_args": ["FaceRestorerEnableToggle"],
        },
        "Essentials_FaceExpressionEnableMirrorToggle": {
            "level": 1,
            "label": "Enable face expression restorer",
            "default": False,
            "bind_control": "FaceExpressionEnableBothToggle",
            "help": "Same as Restorers → Face expressions.",
            "exec_function": control_actions.handle_face_expression_toggle_change,
            "exec_function_args": ["FaceExpressionEnableBothToggle"],
        },
    },
}

ESSENTIALS_CONTROL_LAYOUT_DATA: Any = {
    "Preview & post": {
        "Essentials_PreviewFsr1MirrorToggle": {
            "level": 1,
            "label": "FSR 1 preview (OpenGL)",
            "default": False,
            "bind_control": "PreviewFsr1EnableToggle",
            "help": "Same as Settings → Video Playback Settings.",
            "exec_function": control_actions.handle_preview_fsr1_toggle,
            "exec_function_args": ["PreviewFsr1EnableToggle"],
        },
        "Essentials_PreviewNisMirrorToggle": {
            "level": 1,
            "label": "NIS preview (OpenGL)",
            "default": False,
            "bind_control": "PreviewNisEnableToggle",
            "help": "Same as Settings → Video Playback Settings.",
            "exec_function": control_actions.handle_preview_nis_toggle,
            "exec_function_args": ["PreviewNisEnableToggle"],
        },
        "Essentials_FrameEnhancerMirrorToggle": {
            "level": 1,
            "label": "Enable frame enhancer",
            "default": False,
            "bind_control": "FrameEnhancerEnableToggle",
            "help": "Same as Settings → Frame Enhancer.",
            "exec_function": control_actions.handle_frame_enhancer_state_change,
            "exec_function_args": ["FrameEnhancerEnableToggle"],
        },
    },
}
