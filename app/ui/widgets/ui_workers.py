import uuid
from functools import partial
from typing import TYPE_CHECKING, Dict
import traceback
import os

import torch
import numpy
from PySide6 import QtCore as qtc
from PySide6.QtGui import QPixmap

from app.helpers import miscellaneous as misc_helpers
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


class TargetMediaLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(
        str, QPixmap, str, str
    )  # Signal with media path and QPixmap and file_type, media_id
    webcam_thumbnail_ready = qtc.Signal(str, QPixmap, str, str, int, int)
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(
        self,
        main_window: "MainWindow",
        folder_name=False,
        files_list=None,
        media_ids=None,
        webcam_mode=False,
        parent=None,
    ):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.media_ids = media_ids or []
        self.webcam_mode = webcam_mode
        self._running = True  # Flag to control the running state

    def run(self):
        if self.folder_name:
            self.load_videos_and_images_from_folder(self.folder_name)
        if self.files_list:
            self.load_videos_and_images_from_files_list(self.files_list)
        if self.webcam_mode:
            self.load_webcams()
        self.finished.emit()

    def load_videos_and_images_from_folder(self, folder_name):
        # Initially hide the placeholder text
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, True
        )
        video_files = misc_helpers.get_video_files(
            folder_name, self.main_window.control["TargetMediaFolderRecursiveToggle"]
        )
        image_files = misc_helpers.get_image_files(
            folder_name, self.main_window.control["TargetMediaFolderRecursiveToggle"]
        )

        i = 0
        media_files = video_files + image_files
        # Sorting the list
        media_files.sort(key=lambda x: os.path.basename(str(x)).lower())

        for media_file in media_files:
            if not self._running:  # Check if the thread is still running
                break
            media_file_path = os.path.join(folder_name, media_file)
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = common_widget_actions.extract_frame_as_pixmap(
                self.main_window, media_file_path, file_type
            )
            if self.media_ids:
                media_id = self.media_ids[i]
            else:
                media_id = str(uuid.uuid1().int)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type, media_id)
            i += 1
        # Show/Hide the placeholder text based on the number of items in ListWidget
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, False
        )

    def load_videos_and_images_from_files_list(self, files_list):
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, True
        )
        media_files = files_list
        # Sorting the list
        media_files.sort(key=lambda x: os.path.basename(str(x)).lower())

        i = 0
        for media_file_path in media_files:
            if not self._running:  # Check if the thread is still running
                break
            if not os.path.exists(media_file_path):
                continue
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = common_widget_actions.extract_frame_as_pixmap(
                self.main_window, media_file_path, file_type=file_type
            )
            if self.media_ids:
                media_id = self.media_ids[i]
            else:
                media_id = str(uuid.uuid1().int)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type, media_id)
            i += 1
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, False
        )

    def load_webcams(
        self,
    ):
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, True
        )
        camera_backend = CAMERA_BACKENDS[
            self.main_window.control["WebcamBackendSelection"]
        ]
        for i in range(int(self.main_window.control["WebcamMaxNoSelection"])):
            try:
                pixmap = common_widget_actions.extract_frame_as_pixmap(
                    self.main_window,
                    media_file_path=f"Webcam {i}",
                    file_type="webcam",
                    webcam_index=i,
                    webcam_backend=camera_backend,
                )
                media_id = str(uuid.uuid1().int)

                if pixmap:
                    # Emit the signal to update GUI
                    self.webcam_thumbnail_ready.emit(
                        f"Webcam {i}", pixmap, "webcam", media_id, i, camera_backend
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                traceback.print_exc()
        self.main_window.placeholder_update_signal.emit(
            self.main_window.targetVideosList, False
        )

    def stop(self):
        """Stop the thread by setting the running flag to False."""
        self._running = False
        self.quit()
        self.wait(1000)
        if self.isRunning():
            self.terminate()


class InputFacesLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, numpy.ndarray, object, QPixmap, str)
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(
        self,
        main_window: "MainWindow",
        media_path=False,
        folder_name=False,
        files_list=None,
        face_ids=None,
        parent=None,
    ):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.face_ids = face_ids or []
        self._running = True  # Flag to control the running state

    def run(self):
        """
        Main worker thread execution. Loads models first, then processes files.
        """
        try:
            # Proceed with file processing now that models are ready.
            if self.folder_name or self.files_list:
                self.main_window.placeholder_update_signal.emit(
                    self.main_window.inputFacesList, True
                )
                self.load_faces(self.folder_name, self.files_list)
                self.main_window.placeholder_update_signal.emit(
                    self.main_window.inputFacesList, False
                )
        except Exception as e:
            print(f"[ERROR] Error in InputFacesLoaderWorker: {e}")
            traceback.print_exc()
        finally:
            self.finished.emit()

    def load_faces(self, folder_name=False, files_list=None):
        control = self.main_window.control.copy()
        files_list = files_list or []

        # OPTIMIZED: Pair the file paths with their correct IDs before any processing
        # This prevents ID shifting if an image fails, and avoids destructive sorting.
        paired_files_ids = []

        if folder_name:
            image_files = misc_helpers.get_image_files(
                self.folder_name,
                self.main_window.control["InputFacesFolderRecursiveToggle"],
            )
            image_files.sort()  # Safe to sort here, IDs are generated fresh
            for path in image_files:
                paired_files_ids.append(
                    (os.path.join(folder_name, path), str(uuid.uuid1().int))
                )
        elif files_list:
            # DO NOT SORT if loading from a workspace, keep original saved order
            for idx, path in enumerate(files_list):
                f_id = self.face_ids[idx] if self.face_ids else str(uuid.uuid1().int)
                paired_files_ids.append((path, f_id))

        for image_file_path, face_id in paired_files_ids:
            if not self._running:  # Check if the thread is still running
                break
            if not misc_helpers.is_image_file(image_file_path):
                continue

            frame = misc_helpers.read_image_file(image_file_path)
            if frame is None:
                continue

            # Frame must be in RGB format
            frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

            img = torch.from_numpy(frame.astype("uint8")).to(
                self.main_window.models_processor.device
            )
            img = img.permute(2, 0, 1)
            _, kpss_5, _ = self.main_window.models_processor.run_detect(
                img,
                control["DetectorModelSelection"],
                max_num=1,
                score=control["DetectorScoreSlider"] / 100.0,
                input_size=(512, 512),
                use_landmark_detection=control["LandmarkDetectToggle"],
                landmark_detect_mode=control["LandmarkDetectModelSelection"],
                landmark_score=control["LandmarkDetectScoreSlider"] / 100.0,
                from_points=control["DetectFromPointsToggle"],
                rotation_angles=[0]
                if not control["AutoRotationToggle"]
                else [0, 90, 180, 270],
            )

            if kpss_5 is None or len(kpss_5) == 0:
                continue

            face_kps = kpss_5[0]
            if face_kps.any():
                # Calculate embedding ONLY for the selected recognition model
                selected_recognition_model = control["RecognitionModelSelection"]
                face_emb, cropped_img = (
                    self.main_window.models_processor.run_recognize_direct(
                        img,
                        face_kps,
                        control["SimilarityTypeSelection"],
                        selected_recognition_model,  # Use selected model
                    )
                )

                if face_emb is None:  # Check if recognition failed
                    continue

                cropped_img_np = cropped_img.cpu().numpy()
                # Swap channels from RGB to BGR for pixmap creation
                face_img = numpy.ascontiguousarray(cropped_img_np[..., ::-1])
                pixmap = common_widget_actions.get_pixmap_from_frame(
                    self.main_window, face_img
                )

                embedding_store: Dict[str, numpy.ndarray] = {
                    selected_recognition_model: face_emb,
                    "kps_5": face_kps,
                }

                self.thumbnail_ready.emit(
                    image_file_path, face_img, embedding_store, pixmap, face_id
                )

        torch.cuda.empty_cache()

    def stop(self):
        """Stop the thread by setting the running flag to False."""
        self._running = False
        self.quit()
        self.wait(1000)
        if self.isRunning():
            self.terminate()


class FilterWorker(qtc.QThread):
    filtered_results = qtc.Signal(list, int)  # (visible_indices, snapshot_size)

    def __init__(
        self, main_window: "MainWindow", search_text="", filter_list="target_videos"
    ):
        super().__init__()
        self.main_window = main_window
        self.search_text = search_text
        self.filter_list = filter_list
        # Snapshot attributes set by filter_actions before start() is called.
        # Initialised to safe empty defaults so the worker never accesses Qt widgets.
        self.items_snapshot: list = []
        self.include_file_types: list = []
        self.filter_list_widget = self.get_list_widget()
        self.filtered_results.connect(
            partial(
                filter_actions.update_filtered_list,
                main_window,
                self.filter_list_widget,
            )
        )

    def get_list_widget(
        self,
    ):
        list_widget = False
        if self.filter_list == "target_videos":
            list_widget = self.main_window.targetVideosList
        elif self.filter_list == "input_faces":
            list_widget = self.main_window.inputFacesList
        elif self.filter_list == "merged_embeddings":
            list_widget = self.main_window.inputEmbeddingsList
        return list_widget

    def run(
        self,
    ):
        if self.filter_list == "target_videos":
            self.filter_target_videos()
        elif self.filter_list == "input_faces":
            self.filter_input_faces()
        elif self.filter_list == "merged_embeddings":
            self.filter_merged_embeddings()

    def filter_target_videos(self):
        # Operates only on pre-captured plain Python data — no Qt widget access.
        search_text = self.search_text
        include_file_types = self.include_file_types

        visible_indices = []
        for index, media_path, file_type in self.items_snapshot:
            if (not search_text or search_text in media_path.lower()) and (
                file_type in include_file_types
            ):
                visible_indices.append(index)

        self.filtered_results.emit(visible_indices, len(self.items_snapshot))

    def filter_input_faces(self):
        # Operates only on pre-captured plain Python data — no Qt widget access.
        search_text = self.search_text

        visible_indices = []
        for index, media_path in self.items_snapshot:
            if not search_text or search_text in media_path.lower():
                visible_indices.append(index)

        self.filtered_results.emit(visible_indices, len(self.items_snapshot))

    def filter_merged_embeddings(self):
        # Operates only on pre-captured plain Python data — no Qt widget access.
        search_text = self.search_text

        visible_indices = []
        for index, embedding_name in self.items_snapshot:
            if not search_text or search_text in embedding_name.lower():
                visible_indices.append(index)

        self.filtered_results.emit(visible_indices, len(self.items_snapshot))

    def stop_thread(self):
        self.quit()
        self.wait()
