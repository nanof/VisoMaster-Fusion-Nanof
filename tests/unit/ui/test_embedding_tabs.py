"""Embedding tabs, dismissed target faces, and clear-all folder retention."""

from types import SimpleNamespace

import numpy

from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import list_view_actions


class _DummySignal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _PathLabel:
    def __init__(self, text=""):
        self.value = text
        self.tooltip = text

    def setText(self, text):
        self.value = str(text)

    def setToolTip(self, tooltip):
        self.tooltip = str(tooltip)


def test_removed_face_uses_similarity_scale():
    class _Processor:
        def findCosineDistance(self, old, new):
            return 80.0 if old is new else 10.0

    face = object()
    processor = _Processor()
    assert card_actions._matches_removed_face(processor, face, [face])
    assert not card_actions._matches_removed_face(processor, face, [object()])


def test_remember_removed_face_copies_embedding():
    window = SimpleNamespace()
    original = numpy.array([1.0, 2.0], dtype=numpy.float32)
    card_actions._remember_removed_face_embeddings(window, {"arcface": original})
    stored = window._removed_target_face_embeddings[0]
    original[0] = 9.0
    assert stored[0] == 1.0


def test_clear_all_target_media_keeps_folder_path(monkeypatch):
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    monkeypatch.setattr(
        "app.ui.widgets.actions.video_control_actions.block_if_issue_scan_active",
        lambda *_args, **_kwargs: False,
    )

    class _Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.control = {
                "SkipClearConfirmationToggle": True,
                "AutoLoadTargetFolderToggle": False,
            }
            button = SimpleNamespace(
                media_path="C:/media/clip.mp4",
                remove_target_media_from_list=lambda: None,
            )
            self.target_videos = {"clip": button}
            self.target_faces = {}
            self.selected_video_button = button
            self.last_target_media_folder_path = "C:/kept-folder"
            self.labelTargetVideosPath = _PathLabel("C:/kept-folder")
            self._target_folder_ignored_paths = {"C:/already-ignored.mp4"}
            self.placeholder_update_signal = _DummySignal()
            self.targetVideosList = QtWidgets.QListWidget(self)
            self.video_processor = SimpleNamespace(
                stop_processing=lambda: None,
                media_path="C:/media/clip.mp4",
                file_type="video",
                current_frame=None,
                media_capture=None,
                _clear_single_frame_preview_caches=lambda: None,
            )
            self.is_batch_processing = True
            self.video_loader_worker = None
            self.app_display_metadata = SimpleNamespace(window_title="VisoMaster")

    window = _Window()
    assert list_view_actions.clear_all_target_media(window) is True
    assert window.last_target_media_folder_path == "C:/kept-folder"
    assert window.labelTargetVideosPath.value == "C:/kept-folder"
    assert "C:/already-ignored.mp4" in window._target_folder_ignored_paths
    assert any(
        path.replace("\\", "/").endswith("media/clip.mp4")
        for path in window._target_folder_ignored_paths
    )
    list_view_actions.stop_filter_workers(window)


def test_embedding_tabs_add_close_and_snapshot():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    host = QtWidgets.QWidget()
    layout = QtWidgets.QGridLayout(host)
    embeddings_list = QtWidgets.QListWidget()
    embeddings_list.setFixedHeight(112)
    layout.addWidget(embeddings_list, 1, 2)

    window = SimpleNamespace(
        inputEmbeddingsList=embeddings_list,
        merged_embeddings={},
        loaded_embedding_filename="C:/embeds/first.json",
        target_faces={},
        _closing=False,
        _embedding_filter_sequence=0,
        control={},
    )
    list_view_actions.setup_embedding_tabs(window)
    assert window.embeddingTabs.count() == 1
    assert window.embeddingTabs.tabText(0) == "Embeddings"

    list_view_actions.add_embedding_tab(window, title="Second")
    assert window.embeddingTabs.count() == 2
    assert window.inputEmbeddingsList is not embeddings_list

    snapshot = list_view_actions.get_embedding_tabs_state(window)
    assert len(snapshot["tabs"]) == 2
    assert snapshot["active_index"] == 1

    list_view_actions._on_embedding_tab_close(window, 1)
    assert window.embeddingTabs.count() == 1
    list_view_actions._on_embedding_tab_close(window, 0)
    assert window.embeddingTabs.count() == 1

    list_view_actions.stop_filter_workers(window)
    QtWidgets.QApplication.processEvents()
