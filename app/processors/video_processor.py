import threading
import queue
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Tuple, Optional, cast, List
import time
import subprocess
from pathlib import Path
import os
import gc
from functools import partial
import shutil
import uuid
from datetime import datetime
import cv2
import psutil
import numpy
import torch
import pyvirtualcam
import math
import copy
from PySide6.QtCore import QObject, QTimer, Signal, Slot

# Internal project imports
from app.processors.workers.frame_worker import FrameWorker, _env_flag
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS
import app.helpers.miscellaneous as misc_helpers
from app.helpers.screen_capture import (
    create_screen_capture_from_control,
    mss_available,
)
from app.helpers.typing_helper import (
    ControlTypes,
    FacesParametersTypes,
    ParametersTypes,
)

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

IssueScanTargetEmbeddings = dict[str, dict[str, numpy.ndarray]]
IssueScanTargetSnapshot = dict[str, dict[str, Any]]


def _bbox_iou_xyxy(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """IoU for two boxes [x1,y1,x2,y2] in pixel space."""
    a = numpy.asarray(a, dtype=numpy.float64).reshape(-1)
    b = numpy.asarray(b, dtype=numpy.float64).reshape(-1)
    if a.shape[0] < 4 or b.shape[0] < 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    aw = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    bw = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = aw + bw - inter + 1e-6
    return float(inter / union)


LIVE_STREAM_FILE_TYPES = frozenset({"webcam", "screen"})

TAIL_TOLERANCE = 30  # BUG-07: 10 was too tight — codec trailing B-frames can cause read
# failures in the last ~10 frames on H.264/H.265 content, dropping valid end frames.
MAX_CONSECUTIVE_ERRORS = (
    300  # Stop reading after this many consecutive frame read failures
)

# Audio-Video Sync: Always use segmented extraction when frames are skipped (perfect sync)
# Simple extraction used when no frames are skipped (no sync issues)


class Av1ScrubPreviewEmitter(QObject):
    """Delivers FFmpeg scrub previews from a background thread to the GUI thread."""

    frame_ready = Signal(int, object)


class VideoProcessor(QObject):
    """
    Manages all video, image, and webcam processing pipelines.

    This class handles:
    - Reading frames from media (video, image, webcam).
    - Dispatching frames to worker threads (FrameWorker) for processing.
    - Managing the display metronome (QTimer) for smooth playback/recording.
    - Handling default and multi-segment recording via FFmpeg.
    - Controlling the virtual camera (pyvirtualcam) output.
    - Managing audio playback (ffplay) during preview.

    Thread Safety:
    - Critical: Handles `cuda streams` and TensorRT synchronization.
    - Uses `state_lock` to safeguard parameter updates during playback.
    """

    # --- Signals ---
    # Removed QPixmap to ensure thread safety. GUI thread will handle conversion.
    frame_processed_signal = Signal(int, numpy.ndarray, object)
    webcam_frame_processed_signal = Signal(numpy.ndarray, object)
    single_frame_processed_signal = Signal(int, numpy.ndarray, object)
    processing_started_signal = Signal()  # Unified signal for any processing start
    processing_stopped_signal = Signal()  # Unified signal for any processing stop
    processing_heartbeat_signal = Signal()  # Emits periodically to show liveness

    def __init__(self, main_window: "MainWindow", num_threads=2):
        """
        Initialises the VideoProcessor.

        Sets up all media-state, processing-flag, subprocess, metronome, frame-display,
        and multi-segment recording attributes.  Connects internal worker signals to
        their display/storage slots.

        Args:
            main_window: The application's MainWindow, used to access UI widgets,
                         controls, and the models processor.
            num_threads: Number of persistent FrameWorker pool threads to create for
                         parallel frame processing.
        """
        super().__init__()
        self.main_window = main_window

        self.state_lock = threading.Lock()  # Lock for feeder state
        self.feeder_parameters: FacesParametersTypes | None = None
        self.feeder_control: ControlTypes | None = None

        # --- Worker Thread Management ---
        self.num_threads = num_threads
        self.preroll_target = max(
            20, self.num_threads * 2
        )  # Target number of frames before playback starts
        self.max_display_buffer_size = (
            self.preroll_target * 4
        )  # Max frames allowed "in flight" (queued + being displayed)
        self.max_frames_to_display_size = 8  # VP-22: Hard cap on frames_to_display dict

        # This queue will hold tasks: (frame_number, frame_rgb_data, params, control) or None (poison pill)
        # Pool tasks: (frame#, rgb, params, control, bboxes, kpss_5, kpss, feeder_perf)
        self.frame_queue: queue.Queue[Any] = queue.Queue(
            maxsize=self.max_display_buffer_size
        )
        # This list will hold our *persistent* worker threads
        self.worker_threads: List[threading.Thread] = []
        # Single-frame (scrubbing) worker — tracked so a new seek can stop the old one
        # before starting a fresh worker, preventing concurrent model inference crashes.
        self._current_single_frame_worker: "FrameWorker | None" = None

        # --- Media State ---
        self.media_capture: cv2.VideoCapture | None = None
        self.file_type: str | None = None  # "video", "image", "webcam", or "screen"
        self.fps = 0.0  # Target FPS for playback or recording
        self.media_path: str | None = None
        self.media_rotation: int = 0
        self.current_frame_number = 0  # The *next* frame to be read/processed
        self.max_frame_number = 0
        # AV1 (av01): lighter, imprecise scrub while dragging the seek slider.
        self.is_av1_codec: bool = False
        self._av1_scrub_preview_last_t: float = 0.0
        self._av1_scrub_queue: "queue.Queue[int]" = queue.Queue(maxsize=1)
        self._av1_scrub_session: int = 0
        self._av1_scrub_worker_lock = threading.Lock()
        self._av1_scrub_worker_running: bool = False
        self._av1_scrub_emitter = Av1ScrubPreviewEmitter(self)
        self._av1_scrub_emitter.frame_ready.connect(
            lambda fn, bgr: video_control_actions.apply_av1_scrub_preview_frame(
                self.main_window, fn, bgr
            )
        )
        self.current_frame: Optional[numpy.ndarray] = (
            None  # The most recently read/processed frame
        )

        # --- Sequential Detection State ---
        self.last_detected_faces: list[dict] = []
        self._smoothed_kps: dict[int, numpy.ndarray] = {}
        self._smoothed_dense_kps: dict[int, numpy.ndarray] = {}

        # --- Processing State Flags ---
        self.processing = False  # MASTER flag: True if playback, recording, or webcam stream is active
        self.recording: bool = False  # True if "default-style" recording is active
        self.is_processing_segments: bool = (
            False  # True if "multi-segment" recording is active
        )
        self.triggered_by_job_manager: bool = False  # For multi-segment job integration

        # --- Subprocesses ---
        self.virtcam: pyvirtualcam.Camera | None = None
        self.recording_sp: subprocess.Popen | None = (
            None  # FFmpeg process for both recording styles
        )
        self.ffplay_sound_sp: subprocess.Popen | None = (
            None  # ffplay process for live audio
        )

        # --- Metronome and Timing ---
        self.processing_start_frame: int = (
            0  # The frame number where processing started
        )
        self.last_display_schedule_time_sec: float = (
            0.0  # Used by metronome to prevent drift
        )
        self.target_delay_sec: float = 1.0 / 30.0  # Time between frames for metronome
        self.preroll_timer = QTimer(self)
        self.feeder_thread: threading.Thread | None = (
            None  # The dedicated thread that reads frames and "feeds" the workers
        )
        self.playback_started: bool = False
        self.heartbeat_frame_counter: int = 0  # Counter for heartbeat signal

        # --- Performance Timing ---
        self.start_time = 0.0
        self.end_time = 0.0
        self.playback_display_start_time = (
            0.0  # Time when frames *actually* started displaying
        )
        self.playback_frames_displayed: int = 0  # Frames painted in display_next_frame (preview FPS)
        self.play_start_time = 0.0  # Used by default style for audio segmenting
        self.play_end_time = 0.0  # Used by default style for audio segmenting

        # Playback-only: wall-clock master frame index (keeps preview in sync with ffplay)
        self._playback_use_wall_clock: bool = False
        self._playback_clock_t0: float = 0.0
        self._playback_clock_anchor_frame: int = 0
        # Optional: derive wall-clock target from ffplay start + file fps * atempo (preview + live sound)
        self._wall_clock_use_audio_file_rate: bool = False
        self._audio_sync_wall_t0: float = 0.0
        self._audio_sync_anchor_fn: int = 0
        self._audio_sync_fps_file: float = 0.0
        self._audio_sync_rate: float = 1.0
        self._audio_sync_last_seek_monotonic: float = 0.0

        # Adding Cuda Streams for thread safety
        self.feeder_stream = (
            None  # torch.cuda.Stream() if torch.cuda.is_available() else None
        )

        # Main thread writes, feeder thread reads: skip GPU face detect when preview is idle.
        self._feeder_ui_swap_enabled: bool = True
        self._feeder_ui_edit_enabled: bool = True
        self._feeder_ui_face_compare: bool = False
        self._feeder_ui_face_mask: bool = False

        # --- Default Recording State ---
        self.temp_file: str = ""  # Temporary video file (without audio)
        # Counters for accurate duration calculation
        self.frames_written: int = 0  # Number of frames successfully sent to FFmpeg
        self.last_displayed_frame: int | None = (
            None  # Last frame number that was displayed/written
        )

        # --- Frame Skip Tracking ---
        self.skipped_frames: set[int] = (
            set()
        )  # Track which frames were skipped during recording/segment processing
        self.consecutive_read_errors: int = 0  # Count consecutive read failures
        self.max_consecutive_errors: int = (
            MAX_CONSECUTIVE_ERRORS  # Stop after this many consecutive errors
        )
        self.total_skipped_frames: int = 0  # Counter for skipped frames
        self.stopped_by_error_limit: bool = (
            False  # Track if processing stopped due to error limit
        )
        self.manual_dropped_skip_count: int = 0
        self.read_error_skip_count: int = 0

        # --- Multi-Segment Recording State ---
        self.segments_to_process: List[Tuple[int, int]] = []
        self.current_segment_index: int = -1
        self.temp_segment_files: List[str] = []
        self.current_segment_end_frame: int | None = None
        self.segment_temp_dir: str | None = None

        # --- Utility Timers ---
        self.gpu_memory_update_timer = QTimer()
        self.gpu_memory_update_timer.timeout.connect(
            partial(common_widget_actions.update_gpu_memory_progressbar, main_window)
        )

        # --- Frame Display/Storage ---
        self.next_frame_to_display = 0  # The next frame number the UI should display
        # Changed to store ONLY numpy arrays to prevent VRAM memory bloat
        self.frames_to_display: Dict[int, numpy.ndarray] = {}  # Processed video frames
        # Fallback frame cached during slider seek preview so process_current_frame()
        # can use it when the near-EOF re-read fails (OpenCV seek unreliability).
        self._seek_cached_frame: Optional[Tuple[int, numpy.ndarray]] = None
        # Playback scrub: UI sets latest target; feeder applies seek without stop_processing().
        self._interactive_playback_seek_pending: int | None = None
        # Feeder: seek target to combine with next read_frame (same lock; see manual drop skip).
        self._feeder_deferred_seek_read: int | None = None

        # Consecutive-frame ArcFace cache: key (frame_num, face_idx) — video pool only.
        self._recognition_cache_by_frame: "OrderedDict[tuple[int, int], dict[str, Any]]" = (
            OrderedDict()
        )
        self._recognition_cache_max: int = 256
        self._recognition_cache_lock = threading.Lock()
        self.webcam_frames_to_display: queue.Queue[
            Tuple[numpy.ndarray, Any]
        ] = queue.Queue()  # (processed BGR frame, pipeline profile or None)

        self.frames_pipeline_profile: Dict[int, Any] = {}

        # --- Signal Connections ---
        self.frame_processed_signal.connect(self.store_frame_to_display)
        self.webcam_frame_processed_signal.connect(self.store_webcam_frame_to_display)
        self.single_frame_processed_signal.connect(self.display_current_frame)
        self.single_frame_processed_signal.connect(self.store_frame_to_display)

    @Slot(int, numpy.ndarray, object)
    def store_frame_to_display(self, frame_number, frame, profile=None):
        """Slot to store a processed video/image frame from a worker."""

        # Drop stale frames arriving late from slower threads if we already scrubbed or played past them.
        # This prevents RAM bloat and keeps the metronome buffer clean.
        if self.file_type == "video" and frame_number < self.next_frame_to_display:
            return

        self.frames_to_display[frame_number] = frame
        if profile is not None:
            self.frames_pipeline_profile[frame_number] = profile
        else:
            self.frames_pipeline_profile.pop(frame_number, None)
        # VP-22: Evict stale frames (already past next_frame_to_display) when the
        # buffer exceeds the soft cap. NEVER evict frames that the metronome still
        # needs — doing so causes a permanent stall.
        while len(self.frames_to_display) > self.max_frames_to_display_size:
            oldest = min(self.frames_to_display)
            if oldest >= self.next_frame_to_display:
                # All stored frames are still needed; cannot evict safely.
                break
            arr = self.frames_to_display.pop(oldest)
            del arr
            self.frames_pipeline_profile.pop(oldest, None)

    @Slot(numpy.ndarray, object)
    def store_webcam_frame_to_display(self, frame, profile=None):
        """
        Slot to store a processed webcam frame from a worker.
        For live webcam, we only want the *latest* frame.
        """
        # Clear all pending (old) frames from the queue
        while not self.webcam_frames_to_display.empty():
            try:
                self.webcam_frames_to_display.get_nowait()
            except queue.Empty:
                break

        # Put the new, latest frame in the now-empty queue
        self.webcam_frames_to_display.put((frame, profile))

    @Slot(int, numpy.ndarray, object)
    def display_current_frame(self, frame_number, frame, profile=None):
        """
        Slot to display a single, specific frame.
        Used after seeking or loading new media. NOT part of the metronome loop.
        """

        # During fast scrubbing with AI workers enabled, an older thread might finish processing
        # a frame AFTER the user has already seeked to a newer frame.
        # We must reject these "ghost" frames to prevent the UI from jumping backward.
        if self.file_type == "video" and frame_number != self.next_frame_to_display:
            return

        # Create QPixmap Just-In-Time strictly inside the GUI Thread
        pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, frame)

        if self.main_window.loading_new_media:
            graphics_view_actions.update_graphics_view(
                self.main_window, pixmap, frame_number, reset_fit=True
            )
            self.main_window.loading_new_media = False
        else:
            graphics_view_actions.update_graphics_view(
                self.main_window, pixmap, frame_number
            )
        self.current_frame = frame
        common_widget_actions.update_gpu_memory_progressbar(self.main_window)
        graphics_view_actions.update_pipeline_profile_overlay(self.main_window, profile)

    def _start_metronome(self, target_fps: float, is_first_start: bool = True):
        """
        Unified metronome starter.
        This function configures and starts the metronome loop for all processing types.

        :param target_fps: The target FPS. Use > 9000 for max speed (recording).
        :param is_first_start: True if this is the very first start (e.g., not a new segment).
        """

        # Determine timer interval
        if target_fps <= 0:
            target_fps = 30.0  # Fallback

        if target_fps > 9000:  # Convention for "max speed"
            self.target_delay_sec = 0.005
        else:
            self.target_delay_sec = 1.0 / target_fps

        self._playback_use_wall_clock = (
            self.file_type == "video"
            and not self.recording
            and not self.is_processing_segments
            and target_fps <= 9000
        )

        # Start utility timers and emit signal
        self.gpu_memory_update_timer.start(5000)

        if is_first_start:
            self.processing_started_signal.emit()  # Emit unified signal
            # Record the time when the display *actually* starts
            self.playback_display_start_time = time.perf_counter()
            if self._playback_use_wall_clock:
                want_audio_master = (
                    self.main_window.control.get(
                        "VideoPlaybackAudioSyncPreviewToggle", False
                    )
                    and self.main_window.liveSoundButton.isChecked()
                    and not self.recording
                    and self.file_type == "video"
                    and self._audio_sync_wall_t0 > 0.0
                )
                if want_audio_master:
                    # Re-read slider so stop/play/seek cannot leave a stale anchor vs ffplay -ss.
                    anchor_ui = self._timeline_frame_from_ui()
                    self._audio_sync_anchor_fn = anchor_ui
                    self._playback_clock_t0 = self._audio_sync_wall_t0
                    self._playback_clock_anchor_frame = anchor_ui
                    self._wall_clock_use_audio_file_rate = True
                    with self.state_lock:
                        self.next_frame_to_display = anchor_ui
                    print(
                        "[INFO] Preview: audio-master wall clock (target frame from ffplay timeline)."
                    )
                else:
                    self._wall_clock_use_audio_file_rate = False
                    self._playback_clock_t0 = time.perf_counter()
                    self._playback_clock_anchor_frame = self._timeline_frame_from_ui()
                    with self.state_lock:
                        self.next_frame_to_display = self._playback_clock_anchor_frame
            else:
                self._playback_clock_t0 = 0.0
                self._wall_clock_use_audio_file_rate = False

        # Start the metronome loop
        self.last_display_schedule_time_sec = time.perf_counter()
        self.heartbeat_frame_counter = 0  # Reset heartbeat counter
        self.display_next_frame()  # Start the loop

    def _ensure_precise_metronome_timer(self) -> None:
        """Lazily create the single-shot PreciseTimer used by :meth:`display_next_frame`."""
        if hasattr(self, "precise_metronome"):
            return
        from PySide6.QtCore import Qt, QTimer

        self.precise_metronome = QTimer(self)
        self.precise_metronome.setTimerType(Qt.TimerType.PreciseTimer)
        self.precise_metronome.setSingleShot(True)
        self.precise_metronome.timeout.connect(self.display_next_frame)

    def _arm_display_metronome_retry_ms(self, retry_ms: int) -> None:
        """Re-enter display soon when no frame was shown (decode/worker slightly behind)."""
        if not self.processing:
            return
        self._ensure_precise_metronome_timer()
        self.precise_metronome.start(max(1, int(retry_ms)))

    def _arm_display_metronome_after_frame_shown(self) -> None:
        """Advance the cadence clock only after a frame was actually painted."""
        if not self.processing:
            return
        self._ensure_precise_metronome_timer()
        now_sec = time.perf_counter()
        self.last_display_schedule_time_sec += self.target_delay_sec
        if self.last_display_schedule_time_sec < now_sec:
            self.last_display_schedule_time_sec = now_sec + 0.001
        wait_time_sec = self.last_display_schedule_time_sec - now_sec
        wait_ms = int(wait_time_sec * 1000)
        if wait_ms <= 0:
            wait_ms = 1
        self.precise_metronome.start(wait_ms)

    def _check_preroll_and_start_playback(self):
        """
        Called by preroll_timer.
        Checks if the display buffer is full enough to start playback.
        """
        if not self.processing:
            self.preroll_timer.stop()
            return

        # If playback has already started, stop this timer and exit.
        if self.playback_started:
            self.preroll_timer.stop()
            return

        # Check if the buffer is filled
        if len(self.frames_to_display) >= self.preroll_target:
            self.preroll_timer.stop()
            self.playback_started = True
            print(
                f"[INFO] Preroll buffer filled ({len(self.frames_to_display)} frames). Starting playback components..."
            )

            # Call the dedicated playback start function
            self._start_synchronized_playback()

        else:
            # Not ready yet, keep waiting
            print(
                f"[INFO] Buffering... {len(self.frames_to_display)} / {self.preroll_target}"
            )

    def _feeder_loop(self):
        """
        This function runs in a separate thread (self.feeder_thread).
        Its only job is to read frames from the source and send them to the workers.
        """
        print(
            f"[INFO] Feeder thread started (Mode: {self.file_type}, Segments: {self.is_processing_segments})."
        )

        # Determine which feed logic to use
        try:
            if self.file_type in LIVE_STREAM_FILE_TYPES:
                self._feed_webcam()
            elif (
                self.file_type == "video"
            ):  # Handles both standard video and segment video
                self._feed_video_loop()
            else:
                print(
                    f"[ERROR] Feeder thread: Unknown mode (file_type: {self.file_type})."
                )

        except Exception as e:
            print(f"[ERROR] Unhandled exception in feeder thread: {e}")
            # Ensure processing loops terminate so the application does not hang.
            self.processing = False
            self.is_processing_segments = False

        print("[INFO] Feeder thread finished.")

    def _get_target_input_height(self) -> Optional[int]:
        """
        Helper to determine the target input height if global resize is enabled.
        Returns None if resizing is disabled or invalid.
        """
        resize_enabled = self.main_window.control.get("GlobalInputResizeToggle", False)

        if not resize_enabled:
            return None

        try:
            # Get the selected resolution string (e.g., "720p")
            size_str = self.main_window.control.get(
                "GlobalInputResizeSizeSelection", "720p"
            )
            # Extract the number (e.g., 720)
            return int(str(size_str).replace("p", ""))
        except Exception as e:
            print(
                f"[WARN] Could not parse global input resolution, defaulting to original size. Error: {e}"
            )
            return None

    def sync_feeder_ui_face_flags_from_main_window(self) -> None:
        """Call from the Qt main thread only. Updates flags read by the feeder detection fast-path."""
        mw = self.main_window
        self._feeder_ui_swap_enabled = mw.swapfacesButton.isChecked()
        self._feeder_ui_edit_enabled = mw.editFacesButton.isChecked()
        self._feeder_ui_face_compare = mw.faceCompareCheckBox.isChecked()
        self._feeder_ui_face_mask = mw.faceMaskCheckBox.isChecked()

    def clear_recognition_embedding_cache(self) -> None:
        with self._recognition_cache_lock:
            self._recognition_cache_by_frame.clear()

    def try_reuse_recognition_embedding(
        self,
        frame_num: int,
        face_idx: int,
        bbox: numpy.ndarray,
        kps_5: numpy.ndarray,
        model: str,
        sim_type: str,
    ) -> numpy.ndarray | None:
        """Reuse embedding from (F-1, same face index) when bbox/kps are stable."""
        if frame_num <= 0:
            return None
        prev_key = (frame_num - 1, face_idx)
        with self._recognition_cache_lock:
            prev = self._recognition_cache_by_frame.get(prev_key)
            if prev is None:
                return None
            if prev["model"] != model or prev["sim"] != sim_type:
                return None
            if _bbox_iou_xyxy(bbox, prev["bbox"]) < 0.88:
                return None
            if numpy.max(numpy.abs(kps_5.astype(numpy.float32) - prev["kps_5"])) > 4.0:
                return None
            return cast(numpy.ndarray, prev["emb"])

    def store_recognition_embedding(
        self,
        frame_num: int,
        face_idx: int,
        bbox: numpy.ndarray,
        kps_5: numpy.ndarray,
        emb: numpy.ndarray,
        model: str,
        sim_type: str,
    ) -> None:
        if frame_num <= 0:
            return
        emb_c = numpy.asarray(emb, dtype=numpy.float32).copy()
        row = {
            "bbox": numpy.asarray(bbox, dtype=numpy.float32).copy(),
            "kps_5": numpy.asarray(kps_5, dtype=numpy.float32).copy(),
            "emb": emb_c,
            "model": model,
            "sim": sim_type,
        }
        store_key = (frame_num, face_idx)
        with self._recognition_cache_lock:
            self._recognition_cache_by_frame[store_key] = row
            self._recognition_cache_by_frame.move_to_end(store_key)
            while len(self._recognition_cache_by_frame) > self._recognition_cache_max:
                self._recognition_cache_by_frame.popitem(last=False)

    def _clear_sequential_detection_feed_state(self) -> None:
        """Drop ByteTrack / EMA state after a timeline jump or new playback anchor.

        _run_sequential_detection reuses last_detected_faces when
        current_frame_number % FaceDetectionIntervalSlider != 0. If the timeline
        moved (scrub, audio-sync catch-up, reopen) but this state was not cleared,
        the next frame can inherit bbox/keypoints from the wrong image — no faces,
        broken swap (Find Faces bypasses the feeder and still works).
        """
        self.last_detected_faces.clear()
        self._smoothed_kps.clear()

    def _sequential_detection_required(
        self,
        local_control_for_worker: dict,
        local_params_for_worker: dict | None,
    ) -> bool:
        """True if the feeder must run GPU face detection for this frame."""
        if self._feeder_ui_swap_enabled or self._feeder_ui_edit_enabled:
            return True
        if self._feeder_ui_face_compare or self._feeder_ui_face_mask:
            return True
        if local_control_for_worker.get(
            "FaceEditorEnableToggle", False
        ) or local_control_for_worker.get("FaceExpressionEnableBothToggle", False):
            return True
        if local_control_for_worker.get("ModeEnableToggle", False):
            return True
        if local_params_for_worker:
            for face_params in local_params_for_worker.values():
                if not isinstance(face_params, dict):
                    continue
                if (
                    face_params.get("FaceEditorEnableToggle", False)
                    or face_params.get("FaceExpressionEnableBothToggle", False)
                    or face_params.get("AutoMouthExpressionEnableToggle", False)
                    or face_params.get("FaceMakeupEnableToggle", False)
                    or face_params.get("HairMakeupEnableToggle", False)
                    or face_params.get("EyeBrowsMakeupEnableToggle", False)
                    or face_params.get("LipsMakeupEnableToggle", False)
                ):
                    return True
        return False

    def _run_sequential_detection(
        self,
        frame_rgb: numpy.ndarray,
        local_control_for_worker: dict,
        local_params_for_worker: dict | None = None,
        frame_tensor: torch.Tensor | None = None,
        *,
        force_detection: bool = False,
    ):
        """
        Runs face detection sequentially in the feeder thread to guarantee
        flawless Temporal EMA smoothing and tracking (ByteTrack).
        Includes a rigorous Sanitization Shield to prevent dtype('O') crashes.
        """
        # VR180 requires specialized spherical detection in the FrameWorker, skip sequential here.
        if local_control_for_worker.get("VR180ModeEnableToggle", False):
            return None, None, None

        if not force_detection and not self._sequential_detection_required(
            local_control_for_worker, local_params_for_worker
        ):
            self.last_detected_faces = []
            return (
                numpy.empty((0, 4), dtype=numpy.float32),
                numpy.empty((0, 5, 2), dtype=numpy.float32),
                numpy.empty((0, 68, 2), dtype=numpy.float32),
            )

        import contextlib

        # VRAM Optimization: Get the current global stream instead of a custom one.
        local_stream = (
            torch.cuda.current_stream() if torch.cuda.is_available() else None
        )
        stream_context = (
            torch.cuda.stream(local_stream)
            if local_stream
            else contextlib.nullcontext()
        )

        with stream_context:
            use_landmark = local_control_for_worker.get("LandmarkDetectToggle", True)
            landmark_mode = local_control_for_worker.get(
                "LandmarkDetectModelSelection", "203"
            )
            from_points = local_control_for_worker.get("DetectFromPointsToggle", False)

            # Check if LivePortrait or Makeup features are enabled (they strictly require 203 landmarks)
            requires_203 = False

            # 1. Check global control fallback
            if local_control_for_worker.get(
                "FaceEditorEnableToggle", False
            ) or local_control_for_worker.get("FaceExpressionEnableBothToggle", False):
                requires_203 = True

            # 2. Check per-face parameters (where these settings actually live)
            if not requires_203 and local_params_for_worker:
                for face_id, face_params in local_params_for_worker.items():
                    if isinstance(face_params, dict):
                        if (
                            face_params.get("FaceEditorEnableToggle", False)
                            or face_params.get("FaceExpressionEnableBothToggle", False)
                            or face_params.get("AutoMouthExpressionEnableToggle", False)
                            or face_params.get("FaceMakeupEnableToggle", False)
                            or face_params.get("HairMakeupEnableToggle", False)
                            or face_params.get("EyeBrowsMakeupEnableToggle", False)
                            or face_params.get("LipsMakeupEnableToggle", False)
                        ):
                            requires_203 = True
                            break

            if requires_203:
                # Force 203 and from_points to ensure LivePortrait does not crash
                use_landmark = True
                landmark_mode = "203"
                from_points = True
            elif local_control_for_worker.get(
                "edit_enabled", True
            ) or local_control_for_worker.get("swap_enabled", True):
                # Standard swap/edit still needs landmarks for basic face alignment,
                # but we respect the user's choice for the landmark model and from_points.
                use_landmark = True

            detection_interval = int(
                local_control_for_worker.get("FaceDetectionIntervalSlider", 1)
            )
            previous_faces_arg = None

            if (
                len(self.last_detected_faces) > 0
                and self.current_frame_number % detection_interval != 0
            ):
                previous_faces_arg = self.last_detected_faces
            device = self.main_window.models_processor.device

            owns_frame_tensor = frame_tensor is None
            if frame_tensor is None:
                # OPTIMISATION PCIe : non_blocking=True frees CPU immediately
                frame_tensor = (
                    torch.from_numpy(frame_rgb)
                    .to(device, non_blocking=True)
                    .permute(2, 0, 1)  # Convert [H, W, C] -> [C, H, W]
                )

            # 1. Run Detection
            bboxes, kpss_5, kpss = self.main_window.models_processor.run_detect(
                frame_tensor,
                local_control_for_worker.get("DetectorModelSelection", "RetinaFace"),
                max_num=int(local_control_for_worker.get("MaxFacesToDetectSlider", 20)),
                score=local_control_for_worker.get("DetectorScoreSlider", 50) / 100.0,
                input_size=misc_helpers.detector_input_size_from_control(
                    local_control_for_worker
                ),
                use_landmark_detection=use_landmark,
                landmark_detect_mode=landmark_mode,
                landmark_score=local_control_for_worker.get(
                    "LandmarkDetectScoreSlider", 50
                )
                / 100.0,
                from_points=from_points,
                rotation_angles=[0]
                if not local_control_for_worker.get("AutoRotationToggle", False)
                else [0, 90, 180, 270],
                use_mean_eyes=local_control_for_worker.get(
                    "LandmarkMeanEyesToggle", False
                ),
                previous_detections=previous_faces_arg,
            )
            # Free up VRAM immediately since the tensor is no longer needed in this thread
            if owns_frame_tensor:
                del frame_tensor

            # CUDA Stream Sync: Safely wait for GPU on the current stream
            if local_stream:
                local_stream.synchronize()

        # TensorRT and ONNX reuse memory buffers for maximum performance.
        # If we do not copy these arrays, the feeder thread will overwrite the memory
        # of the frames waiting in the queue, causing erratic swap loss!
        if isinstance(bboxes, numpy.ndarray):
            bboxes = bboxes.copy()
        if isinstance(kpss_5, numpy.ndarray):
            kpss_5 = kpss_5.copy()
        if isinstance(kpss, numpy.ndarray):
            kpss = kpss.copy()
        # Ensure 'bboxes' and keypoints are strictly valid float32 arrays and not 'object'.
        if isinstance(bboxes, numpy.ndarray):
            if bboxes.dtype == object:
                try:
                    bboxes = bboxes.astype(numpy.float32)
                except Exception:
                    # If conversion fails (ragged array), purge the corrupted detections
                    bboxes = numpy.empty((0, 4), dtype=numpy.float32)

            # Filter out NaNs, Infs, and validate shape integrity
            if bboxes.size > 0 and bboxes.ndim == 2 and bboxes.shape[1] == 4:
                valid_mask = numpy.isfinite(bboxes).all(axis=1)

                # If there are any corrupted rows, filter them across all arrays
                if not valid_mask.all():
                    bboxes = bboxes[valid_mask]
                    # Safely align kpss_5 if dimensions match
                    if isinstance(kpss_5, numpy.ndarray) and kpss_5.shape[0] == len(
                        valid_mask
                    ):
                        kpss_5 = kpss_5[valid_mask]
                    # Safely align dense kpss if dimensions match
                    if isinstance(kpss, numpy.ndarray) and kpss.shape[0] == len(
                        valid_mask
                    ):
                        kpss = kpss[valid_mask]
            else:
                bboxes = numpy.empty((0, 4), dtype=numpy.float32)
        else:
            bboxes = numpy.empty((0, 4), dtype=numpy.float32)
        # If the sanitization purged the bboxes, we MUST purge the keypoints too.
        # Otherwise, the FrameWorker receives mismatched arrays and skips the face.
        if bboxes.shape[0] == 0:
            if isinstance(kpss_5, numpy.ndarray):
                kpss_5 = numpy.empty((0, 5, 2), dtype=numpy.float32)
            if isinstance(kpss, numpy.ndarray):
                kpss = numpy.empty((0, 68, 2), dtype=numpy.float32)

        # 2. Update tracking state for the next sequential frame
        detected_for_state = []
        # Updated condition to use the sanitized bboxes array instead of checking isinstance again
        if bboxes.shape[0] > 0:
            for i in range(len(bboxes)):
                detected_for_state.append({"bbox": bboxes[i], "score": 1.0})
        self.last_detected_faces = detected_for_state

        # Get toggle value to enable smoothing
        is_smoothing_enabled = local_control_for_worker.get(
            "KPSSmoothingEnableToggle", True
        )

        # 3. Apply Sequential EMA smoothing perfectly in order
        if is_smoothing_enabled:
            img_h_for_kps, img_w_for_kps = frame_rgb.shape[0], frame_rgb.shape[1]

            if isinstance(kpss_5, numpy.ndarray) and kpss_5.shape[0] > 0:
                kpss_5 = kpss_5.copy()
                n_faces = kpss_5.shape[0]
                new_smoothed_kps = {}
                new_smoothed_dense_kps = {}

                has_dense_kps = isinstance(kpss, numpy.ndarray) and kpss.shape[0] > 0
                if has_dense_kps:
                    kpss = kpss.copy()

                for _i in range(n_faces):
                    _raw = kpss_5[_i]

                    if (
                        _raw is None
                        or _raw.size == 0
                        or numpy.any(numpy.isnan(_raw))
                        or numpy.any(numpy.isinf(_raw))
                    ):
                        continue
                    if (
                        numpy.any(_raw[:, 0] < 0)
                        or numpy.any(_raw[:, 0] >= img_w_for_kps)
                        or numpy.any(_raw[:, 1] < 0)
                        or numpy.any(_raw[:, 1] >= img_h_for_kps)
                    ):
                        continue

                    _centroid_raw = numpy.mean(_raw, axis=0)
                    _best_match_key = None
                    _min_dist = float("inf")

                    # Match current face to previous faces spatially
                    for _k, _prev_kps in self._smoothed_kps.items():
                        _centroid_prev = numpy.mean(_prev_kps, axis=0)
                        _dist = numpy.linalg.norm(_centroid_raw - _centroid_prev)
                        if _dist < 50.0 and _dist < _min_dist:
                            _min_dist = float(_dist)
                            _best_match_key = _k

                    if _best_match_key is not None:
                        # Adaptive Alpha (Smart EMA) using slider value
                        base_alpha = (
                            local_control_for_worker.get("KPSEmaAlphaSlider", 35)
                            / 100.0
                        )

                        movement_factor = min(1.0, _min_dist / 15.0)
                        dynamic_alpha = base_alpha + movement_factor * (
                            1.0 - base_alpha
                        )

                        # Smoothing on KPS 5
                        new_smoothed_kps[_i] = (
                            dynamic_alpha * _raw
                            + (1.0 - dynamic_alpha)
                            * self._smoothed_kps[_best_match_key]
                        )
                        del self._smoothed_kps[_best_match_key]

                        # Smoothing on Dense KPS
                        if has_dense_kps:
                            if _best_match_key in self._smoothed_dense_kps:
                                new_smoothed_dense_kps[_i] = (
                                    dynamic_alpha * kpss[_i]
                                    + (1.0 - dynamic_alpha)
                                    * self._smoothed_dense_kps[_best_match_key]
                                )
                                del self._smoothed_dense_kps[_best_match_key]
                            else:
                                new_smoothed_dense_kps[_i] = kpss[_i].copy()
                    else:
                        new_smoothed_kps[_i] = _raw.copy()
                        if has_dense_kps:
                            new_smoothed_dense_kps[_i] = kpss[_i].copy()

                    kpss_5[_i] = new_smoothed_kps[_i]
                    if has_dense_kps:
                        kpss[_i] = new_smoothed_dense_kps[_i]

                self._smoothed_kps = new_smoothed_kps
                self._smoothed_dense_kps = new_smoothed_dense_kps
        else:
            # Safe Clear if disabled
            self._smoothed_kps.clear()
            self._smoothed_dense_kps.clear()

        return bboxes, kpss_5, kpss

    def _feed_video_loop(self):
        """
        Unified feeder logic for standard video playback AND segment recording.
        Reads frames as long as processing is active and within the limits.
        Now supports skipping unreadable or manually dropped frames instead of stopping.
        """

        # Determine the mode at startup
        is_segment_mode = self.is_processing_segments

        # The feeder's state is initialized in process_video()
        # We just need to track the last marker
        last_marker_data = None

        # Determine the stop condition (control variable)
        def stop_flag_check():
            return self.is_processing_segments if is_segment_mode else self.processing

        print(
            f"[INFO] Feeder: Starting video loop (Mode: {'Segment' if is_segment_mode else 'Standard'})."
        )

        # Reset skip tracking at start
        self.consecutive_read_errors = 0
        self.skipped_frames.clear()
        self.total_skipped_frames = 0
        self.manual_dropped_skip_count = 0
        self.read_error_skip_count = 0

        # VP-19: Cache target input height outside the loop; only re-read on detected change.
        cached_resize_toggle = self.main_window.control.get(
            "GlobalInputResizeToggle", False
        )
        cached_target_height = self._get_target_input_height()

        while stop_flag_check():
            try:
                seek_before_read: int | None = None
                if self._feeder_deferred_seek_read is not None:
                    seek_before_read = self._feeder_deferred_seek_read
                    self._feeder_deferred_seek_read = None

                # 0. Guard: feeder_parameters must be initialised before we can process.
                if self.feeder_parameters is None:
                    time.sleep(0.005)
                    continue

                # 0b. Interactive timeline scrub during playback (no full pipeline teardown).
                pending_interactive_seek: int | None = None
                with self.state_lock:
                    if self._interactive_playback_seek_pending is not None:
                        pending_interactive_seek = self._interactive_playback_seek_pending
                        self._interactive_playback_seek_pending = None
                if pending_interactive_seek is not None:
                    with self.state_lock:
                        self.frames_to_display.clear()
                        self.frames_pipeline_profile.clear()
                        self.current_frame_number = pending_interactive_seek
                        self.next_frame_to_display = pending_interactive_seek
                    with self.frame_queue.mutex:
                        self.frame_queue.queue.clear()
                    seek_before_read = pending_interactive_seek
                    self.consecutive_read_errors = 0
                    self._audio_sync_last_seek_monotonic = time.perf_counter()
                    self._clear_sequential_detection_feed_state()

                # 1. Mode-specific stop logic
                if is_segment_mode:
                    if self.current_segment_end_frame is None:
                        time.sleep(0.01)  # Wait for the segment to be configured
                        continue
                    if self.current_frame_number > self.current_segment_end_frame:
                        # This segment is finished, the feeder's job is done.
                        print(
                            f"[INFO] Feeder: Reached end of segment {self.current_segment_index + 1}. Stopping feed."
                        )
                        break
                else:  # Standard mode
                    if self.current_frame_number > self.max_frame_number:
                        break  # End of video

                # 2. Buffer control
                # VP-22: Enforce hard cap on frames_to_display to bound memory usage.
                if len(self.frames_to_display) >= self.max_frames_to_display_size:
                    time.sleep(0.005)  # Wait 5ms (display dict full)
                    continue
                in_flight_frames = (
                    len(self.frames_to_display) + self.frame_queue.qsize()
                )
                if in_flight_frames >= self.max_display_buffer_size:
                    time.sleep(0.005)  # Wait 5ms (buffer full)
                    continue

                # 2b. Audio-master preview: seek read position toward the ffplay timeline
                # when processing cannot keep up; otherwise preview stays in slow motion while
                # audio runs at real time.
                if (
                    seek_before_read is None
                    and not is_segment_mode
                    and not self.recording
                    and self._wall_clock_use_audio_file_rate
                    and self._playback_use_wall_clock
                ):
                    now_seek = time.perf_counter()
                    min_lag, slices, max_step, min_interval = (
                        self._get_audio_sync_feeder_tuning()
                    )
                    if now_seek - self._audio_sync_last_seek_monotonic >= min_interval:
                        target_eff = self._advance_past_skipped_for_display(
                            self._expected_frame_from_wall_clock()
                        )
                        cur_fn = int(self.current_frame_number)
                        lag = int(target_eff) - cur_fn
                        if lag >= min_lag:
                            # Spread correction: move only a fraction of the lag per seek (capped).
                            raw_step = max(1, (lag + slices - 1) // slices)
                            step = min(raw_step, max_step)
                            jump_to = min(
                                cur_fn + step,
                                int(target_eff),
                                int(self.max_frame_number),
                            )
                            if jump_to > cur_fn:
                                self._audio_sync_last_seek_monotonic = now_seek
                                with self.state_lock:
                                    self.current_frame_number = jump_to
                                    # Do not bump next_frame_to_display here: the display loop
                                    # picks the best frame <= wall-clock target; forcing it ahead
                                    # of decoded frames made store_frame_to_display reject work and
                                    # starve the UI when catch-up retried without a read (issue: loop
                                    # of seek+clear+continue never reached read_frame/enqueue).
                                    self.frames_to_display.clear()
                                    self.frames_pipeline_profile.clear()
                                seek_before_read = jump_to
                                with self.frame_queue.mutex:
                                    self.frame_queue.queue.clear()
                                self._clear_sequential_detection_feed_state()
                                print(
                                    f"[INFO] Feeder: audio-sync catch-up +{jump_to - cur_fn}f "
                                    f"({cur_fn}→{jump_to}, lag≈{lag}, target={target_eff}).",
                                    flush=True,
                                )
                                # Fall through: must read/enqueue at least one frame this iteration
                                # so workers can refill frames_to_display; otherwise lag stays high
                                # and we only seek+clear in a tight loop (frozen preview).

                if (
                    is_segment_mode or self.recording
                ) and self.current_frame_number in self.main_window.dropped_frames:
                    self._mark_skipped_frame(self.current_frame_number, "manual_drop")
                    self.current_frame_number += 1
                    self._feeder_deferred_seek_read = self.current_frame_number
                    continue

                # 3. Determine Input Resolution (Global Resize)
                # VP-19: Use cached value; only re-read when the toggle changes.
                current_resize_toggle = self.main_window.control.get(
                    "GlobalInputResizeToggle", False
                )
                if current_resize_toggle != cached_resize_toggle:
                    cached_resize_toggle = current_resize_toggle
                    cached_target_height = self._get_target_input_height()
                target_height = cached_target_height

                _perf_stages_feed = (
                    _env_flag("VISIOMASTER_PERF_STAGES")
                    or bool(
                        self.main_window.control.get(
                            "PipelineProfileOverlayEnableToggle", False
                        )
                    )
                )
                _t_feed_read0 = time.perf_counter()
                ret, frame_bgr = misc_helpers.read_frame(
                    self.media_capture,
                    self.media_rotation,
                    preview_target_height=target_height,
                    seek_to_frame_first=seek_before_read,
                )
                _t_feed_after_read = time.perf_counter()
                if not ret:
                    fn = self.current_frame_number

                    # 1) Segment mode: read failure near segment end -> treat as segment EOF/stop
                    if (
                        self.is_processing_segments
                        and self.current_segment_end_frame is not None
                    ):
                        if fn >= self.current_segment_end_frame - TAIL_TOLERANCE:
                            with self.state_lock:
                                # Advance past the segment end to trigger display_next_frame()'s segment-end branch
                                self.next_frame_to_display = (
                                    self.current_segment_end_frame + 1
                                )
                                # Optional: also advance the feeder's own frame counter to avoid other logic misinterpreting state
                                self.current_frame_number = (
                                    self.current_segment_end_frame + 1
                                )
                            print(
                                f"[INFO] Feeder: Treat read failure near segment tail as EOF (frame={fn})."
                            )
                            break

                    # 2) Standard mode: read failure near file end -> treat as EOF
                    if (
                        not is_segment_mode
                        and fn >= self.max_frame_number - TAIL_TOLERANCE
                    ):
                        print(
                            f"[INFO] Feeder: Read failure near file end (frame={fn}/{self.max_frame_number}), treating as EOF."
                        )
                        # Advance next_frame_to_display past max to trigger finalization
                        with self.state_lock:
                            self.next_frame_to_display = self.max_frame_number + 1
                        self.processing = False
                        break

                    # 3) Standard mode: unified read-failure skip logic (no longer
                    # depends on potentially inaccurate max_frame_number). Skip the
                    # unreadable frame and continue, but stop if too many
                    # consecutive failures suggest we actually reached EOF.
                    self.consecutive_read_errors += 1
                    self._mark_skipped_frame(self.current_frame_number, "read_error")

                    # Check if too many consecutive errors (likely reached actual EOF)
                    if self.consecutive_read_errors > self.max_consecutive_errors:
                        print(
                            f"[INFO] Feeder: Too many consecutive read errors ({self.consecutive_read_errors}), likely reached EOF. Stopping."
                        )
                        self.stopped_by_error_limit = True
                        # Advance next_frame_to_display past max to trigger finalization
                        with self.state_lock:
                            self.next_frame_to_display = self.max_frame_number + 1
                        if is_segment_mode:
                            self.is_processing_segments = False
                        else:
                            self.processing = False
                        break

                    # Log skip and move to next frame
                    print(
                        f"[WARN] Feeder: Skipping unreadable frame {self.current_frame_number} "
                        f"(Total skipped: {self.total_skipped_frames}, Consecutive read errors: {self.consecutive_read_errors})."
                    )
                    self.current_frame_number += 1
                    self._feeder_deferred_seek_read = self.current_frame_number
                    continue  # Skip this frame and try the next one

                # Successfully read a frame, reset consecutive error counter
                self.consecutive_read_errors = 0

                frame_num_to_process = self.current_frame_number

                # Get marker data *only* for the exact frame
                marker_data = self.main_window.markers.get(frame_num_to_process)

                local_params_for_worker: FacesParametersTypes
                local_control_for_worker: ControlTypes

                # Lock the state while reading/writing
                with self.state_lock:
                    if marker_data and marker_data != last_marker_data:
                        # This frame IS a marker, update the feeder's state
                        print(
                            f"[INFO] Frame {frame_num_to_process} is a marker. Updating feeder state."
                        )

                        self.feeder_parameters = copy.deepcopy(
                            marker_data["parameters"]
                        )

                        # Reset controls to default first
                        self.feeder_control = {}
                        for (
                            widget_name,
                            widget,
                        ) in self.main_window.parameter_widgets.items():
                            if widget_name in self.main_window.control:
                                self.feeder_control[widget_name] = widget.default_value

                        if "control" in marker_data and isinstance(
                            marker_data["control"], dict
                        ):
                            self.feeder_control.update(
                                cast(ControlTypes, marker_data["control"]).copy()
                            )

                        last_marker_data = marker_data

                    # Use the (potentially updated) feeder state
                    # We MUST send copies, as the worker will use them in parallel
                    local_params_for_worker = copy.deepcopy(self.feeder_parameters)
                    local_control_for_worker = copy.deepcopy(self.feeder_control)

                _t_feed_before_rgb = time.perf_counter()
                frame_rgb = misc_helpers.bgr_uint8_to_rgb_contiguous(frame_bgr)
                _t_feed_after_rgb = time.perf_counter()

                # --- Inject Sequential Detection ---
                _t_seq_det = time.perf_counter()
                bboxes, kpss_5, kpss = self._run_sequential_detection(
                    frame_rgb, local_control_for_worker, local_params_for_worker
                )
                feeder_perf = None
                if _perf_stages_feed:
                    _sync_det = (
                        self.main_window.models_processor.device == "cuda"
                        and torch.cuda.is_available()
                        and (
                            _env_flag("VISIOMASTER_PERF_STAGES")
                            or bool(
                                self.main_window.control.get(
                                    "PipelineProfileGpuSyncToggle", False
                                )
                            )
                        )
                    )
                    if _sync_det:
                        torch.cuda.synchronize()
                    _det_ms = (time.perf_counter() - _t_seq_det) * 1000.0
                    _read_ms = (_t_feed_after_read - _t_feed_read0) * 1000.0
                    _state_ms = (_t_feed_before_rgb - _t_feed_after_read) * 1000.0
                    _rgb_ms = (_t_feed_after_rgb - _t_feed_before_rgb) * 1000.0
                    feeder_perf = {
                        "read_frame_ms": _read_ms,
                        "feeder_state_ms": _state_ms,
                        "rgb_pack_ms": _rgb_ms,
                        "sequential_detect_ms": _det_ms,
                    }
                    if _env_flag("VISIOMASTER_PERF_STAGES"):
                        print(
                            f"[PERF-STAGES] frame={frame_num_to_process} role=feeder "
                            f"read_frame_ms={_read_ms:.2f} feeder_state_ms={_state_ms:.2f} "
                            f"rgb_pack_ms={_rgb_ms:.2f} sequential_detect_ms={_det_ms:.2f}",
                            flush=True,
                        )

                # The worker will use the feeder's state *from this exact moment*
                task = (
                    frame_num_to_process,
                    frame_rgb,
                    local_params_for_worker,
                    local_control_for_worker,
                    bboxes,
                    kpss_5,
                    kpss,
                    feeder_perf,
                )

                # Put the task in the queue for the worker pool
                self.frame_queue.put(task)

                # DO NOT START A WORKER HERE
                self.current_frame_number += 1

            except Exception as e:
                print(
                    f"[ERROR] Error in _feed_video_loop (Mode: {'Segment' if is_segment_mode else 'Standard'}): {e}"
                )
                if is_segment_mode:
                    self.is_processing_segments = False
                else:
                    self.processing = False  # Stop the loop
                # Send poison pills to unblock all waiting worker threads immediately.
                for _ in self.worker_threads:
                    try:
                        # Use block=False instead of false timeout
                        self.frame_queue.put(None, block=False)
                    except queue.Full:
                        pass

        # Log summary of skipped frames at end
        if self.total_skipped_frames > 0:
            print(
                f"[INFO] Feeder loop finished. Total frames skipped: {self.total_skipped_frames}"
            )
            print(
                f"[INFO] Skip reasons: manual dropped frames={self.manual_dropped_skip_count}, read errors={self.read_error_skip_count}"
            )
            print(
                f"[INFO] Skipped frame numbers: {sorted(list(self.skipped_frames)[:100])}{'...' if len(self.skipped_frames) > 100 else ''}"
            )

    def _feed_webcam(self):
        """Feeder logic for webcam streaming."""
        while self.processing:
            try:
                in_flight_frames = (
                    len(self.webcam_frames_to_display.queue) + self.frame_queue.qsize()
                )
                if in_flight_frames >= self.max_display_buffer_size:
                    time.sleep(0.005)  # Wait 5ms (buffer full)
                    continue

                _perf_stages_wcam = (
                    _env_flag("VISIOMASTER_PERF_STAGES")
                    or bool(
                        self.main_window.control.get(
                            "PipelineProfileOverlayEnableToggle", False
                        )
                    )
                )
                _t_w_read0 = time.perf_counter()
                ret, frame_bgr = misc_helpers.read_frame(
                    self.media_capture, 0, preview_target_height=None
                )
                _t_w_after_read = time.perf_counter()
                if not ret:
                    print("[WARN] Feeder: Failed to read live capture frame.")
                    continue  # Try again

                _t_w_before_rgb = time.perf_counter()
                frame_rgb = misc_helpers.bgr_uint8_to_rgb_contiguous(frame_bgr)
                _t_w_after_rgb = time.perf_counter()

                # The worker pool expects a task.
                # For live capture (webcam/screen), read the *current* global parameters
                _t_w_before_params = time.perf_counter()
                with self.main_window.models_processor.model_lock:
                    local_params_for_worker = self.main_window.parameters.copy()
                    local_control_for_worker = self.main_window.control.copy()
                _t_w_after_params = time.perf_counter()

                # --- Inject Sequential Detection ---
                _t_seq_det_w = time.perf_counter()
                bboxes, kpss_5, kpss = self._run_sequential_detection(
                    frame_rgb, local_control_for_worker, local_params_for_worker
                )
                feeder_perf_w = None
                if _perf_stages_wcam:
                    _sync_w = (
                        self.main_window.models_processor.device == "cuda"
                        and torch.cuda.is_available()
                        and (
                            _env_flag("VISIOMASTER_PERF_STAGES")
                            or bool(
                                self.main_window.control.get(
                                    "PipelineProfileGpuSyncToggle", False
                                )
                            )
                        )
                    )
                    if _sync_w:
                        torch.cuda.synchronize()
                    _det_ms_w = (time.perf_counter() - _t_seq_det_w) * 1000.0
                    _read_ms_w = (_t_w_after_read - _t_w_read0) * 1000.0
                    _rgb_ms_w = (_t_w_after_rgb - _t_w_before_rgb) * 1000.0
                    _param_ms_w = (_t_w_after_params - _t_w_before_params) * 1000.0
                    feeder_perf_w = {
                        "read_frame_ms": _read_ms_w,
                        "rgb_pack_ms": _rgb_ms_w,
                        "feeder_params_lock_ms": _param_ms_w,
                        "sequential_detect_ms": _det_ms_w,
                    }
                    if _env_flag("VISIOMASTER_PERF_STAGES"):
                        print(
                            f"[PERF-STAGES] frame=live role=feeder "
                            f"read_frame_ms={_read_ms_w:.2f} rgb_pack_ms={_rgb_ms_w:.2f} "
                            f"feeder_params_lock_ms={_param_ms_w:.2f} "
                            f"sequential_detect_ms={_det_ms_w:.2f}",
                            flush=True,
                        )

                task = (
                    0,  # frame_number is always 0 for live streams
                    frame_rgb,
                    local_params_for_worker,
                    local_control_for_worker,
                    bboxes,
                    kpss_5,
                    kpss,
                    feeder_perf_w,
                )

                # Put the task in the queue for the worker pool
                self.frame_queue.put(task)
                # Live streams must advance this counter each fed frame: _run_sequential_detection
                # uses current_frame_number % FaceDetectionIntervalSlider to choose full detect vs
                # ByteTrack carry-over. It never advanced here (unlike _feed_video_loop), so after
                # video playback the index could stay odd/even-stuck and skip real detection on
                # every screen/webcam frame — swap/find-faces diverge (find faces bypasses feeder).
                self.current_frame_number += 1

            except Exception as e:
                print(f"[ERROR] Error in _feed_webcam loop: {e}")
                self.processing = False

    def _mark_skipped_frame(self, frame_number: int, reason: str) -> None:
        """Track skipped-frame reasons for later audio-rebuild diagnostics."""
        self.skipped_frames.add(frame_number)
        self.total_skipped_frames += 1

        if reason == "manual_drop":
            self.manual_dropped_skip_count += 1
        elif reason == "read_error":
            self.read_error_skip_count += 1

    def _get_audio_sync_feeder_tuning(self) -> tuple[int, int, int, float]:
        """Catch-up tuning from settings (Video Playback → audio sync child controls)."""
        c = self.main_window.control
        try:
            min_lag = int(c.get("VideoPlaybackAudioSyncMinLagSlider", 8))
        except (TypeError, ValueError):
            min_lag = 8
        try:
            slices = int(c.get("VideoPlaybackAudioSyncCatchupSlicesSlider", 5))
        except (TypeError, ValueError):
            slices = 5
        try:
            max_step = int(c.get("VideoPlaybackAudioSyncMaxStepSlider", 12))
        except (TypeError, ValueError):
            max_step = 12
        try:
            interval = float(
                c.get("VideoPlaybackAudioSyncMinSeekIntervalDecimalSlider", 0.04)
            )
        except (TypeError, ValueError):
            interval = 0.04

        min_lag = max(2, min(60, min_lag))
        slices = max(2, min(15, slices))
        max_step = max(1, min(90, max_step))
        interval = max(0.01, min(0.30, interval))
        return min_lag, slices, max_step, interval

    def _timeline_frame_from_ui(self) -> int:
        """Playback timeline index from the seek slider (clamped). Used to keep audio/ffplay aligned after seek/stop/play."""
        if self.file_type != "video":
            return int(self.next_frame_to_display)
        try:
            fn = int(self.main_window.videoSeekSlider.value())
        except (TypeError, ValueError):
            fn = int(self.next_frame_to_display)
        try:
            max_fn = int(self.max_frame_number)
        except (TypeError, ValueError):
            max_fn = fn
        if max_fn < 0:
            max_fn = fn
        return max(0, min(fn, max_fn))

    def _expected_frame_from_wall_clock(self) -> int:
        """Frame index the wall clock says we should have reached (playback preview only)."""
        if not self._playback_use_wall_clock or self._playback_clock_t0 <= 0.0:
            return self.next_frame_to_display

        elapsed = time.perf_counter() - self._playback_clock_t0

        if self._wall_clock_use_audio_file_rate:
            r = max(0.5, float(self._audio_sync_rate))
            f = float(self._audio_sync_fps_file)
            if f <= 0:
                f = float(self.fps) if self.fps > 0 else 30.0
            fn = self._playback_clock_anchor_frame + int(elapsed * r * f)
        else:
            if self.fps <= 0:
                return self.next_frame_to_display
            fn = self._playback_clock_anchor_frame + int(elapsed * float(self.fps))

        return max(0, min(fn, self.max_frame_number))

    def _advance_past_skipped_for_display(self, fn: int) -> int:
        while fn in self.skipped_frames and fn <= self.max_frame_number:
            fn += 1
        return fn

    def display_next_frame(self):
        """
        The core metronome loop.
        This function is called repeatedly via QTimer.singleShot.
        """

        # 0. Check for end-of-media FIRST (before processing flag check)
        # This ensures we finalize even if feeder stopped due to errors
        is_playback_loop_enabled = self.main_window.control["VideoPlaybackLoopToggle"]
        should_stop_playback = False
        should_finalize_default_recording = False

        if self.file_type == "video":
            if self.is_processing_segments:
                # --- Segment Recording Stop Logic ---
                if (
                    self.current_segment_end_frame is not None
                    and self.next_frame_to_display > self.current_segment_end_frame
                ):
                    print(
                        f"[INFO] Segment {self.current_segment_index + 1} end frame ({self.current_segment_end_frame}) reached."
                    )
                    self.stop_current_segment()  # Segment logic handles its own stop
                    return
            elif self.next_frame_to_display > self.max_frame_number:
                # --- Default Playback/Recording Stop Logic ---
                print("[INFO] End of media reached.")
                if self.recording:
                    should_finalize_default_recording = True
                elif is_playback_loop_enabled:
                    self.next_frame_to_display = 1
                    self.main_window.videoSeekSlider.blockSignals(True)
                    self.main_window.videoSeekSlider.setValue(
                        self.next_frame_to_display
                    )
                    self.main_window.videoSeekSlider.blockSignals(False)
                    should_stop_playback = True
                else:
                    should_stop_playback = True

            if should_finalize_default_recording:
                self._finalize_default_style_recording()
                return
            elif should_stop_playback:
                self.stop_processing()
                if is_playback_loop_enabled:
                    self.process_video()
                return

        # 1. Stop check (after end-of-media check)
        if not self.processing:  # General check (if stop_processing was called)
            return

        # --- Metronome scheduling (VP-PERF-01): arm the *next* tick only after we
        # successfully paint a frame (see end of this method). If we returned early
        # because the buffer is not ready, use a short retry without advancing the
        # cadence clock — otherwise we "lose" display slots and cap FPS below nominal.
        _retry_ms = max(2, min(12, int(self.target_delay_sec * 250)))

        # --- 6. Get the frame to display (if ready) ---
        frame = None
        frame_number_to_display = 0  # Used for UI update
        profile_for_overlay = None

        if self.file_type in LIVE_STREAM_FILE_TYPES:
            # --- Live stream (webcam / screen) — queue of latest processed frames ---
            if self.webcam_frames_to_display.empty():
                self._arm_display_metronome_retry_ms(_retry_ms)
                return  # Frame not ready, skip display
            _w_item = self.webcam_frames_to_display.get()
            if isinstance(_w_item, tuple) and len(_w_item) >= 2:
                frame, profile_for_overlay = _w_item[0], _w_item[1]
            else:
                frame = _w_item  # type: ignore[assignment]
            frame_number_to_display = 0  # Not relevant for live streams

        else:
            # --- Video/Image Logic (Dictionary) ---
            if self.file_type == "video" and self._playback_use_wall_clock:
                target = self._advance_past_skipped_for_display(
                    self._expected_frame_from_wall_clock()
                )

                low = self.next_frame_to_display
                while low in self.skipped_frames and low <= self.max_frame_number:
                    low += 1
                self.next_frame_to_display = low

                if low > target:
                    self._arm_display_metronome_retry_ms(_retry_ms)
                    return

                candidates = [
                    k
                    for k in self.frames_to_display
                    if low <= k <= target and k not in self.skipped_frames
                ]
                if not candidates:
                    self._arm_display_metronome_retry_ms(_retry_ms)
                    return

                best = max(candidates)
                if best > low:
                    for k in list(self.frames_to_display.keys()):
                        if low <= k < best:
                            arr = self.frames_to_display.pop(k, None)
                            if arr is not None:
                                del arr
                            self.frames_pipeline_profile.pop(k, None)
                    self.next_frame_to_display = best
                    if best - low >= 10:
                        print(
                            f"[INFO] Display: Wall-clock catch-up, skipping to frame {best} "
                            f"(target≤{target}, had been at {low})"
                        )

                frame_number_to_display = self.next_frame_to_display
            else:
                frame_number_to_display = self.next_frame_to_display

                # Skip frames that were corrupted/skipped during processing
                # Find the next non-skipped frame to display
                original_frame = frame_number_to_display
                while (
                    frame_number_to_display in self.skipped_frames
                    and frame_number_to_display <= self.max_frame_number
                ):
                    frame_number_to_display += 1

                # Update next_frame_to_display to skip all consecutive skipped frames
                if frame_number_to_display > original_frame:
                    skipped_count = frame_number_to_display - original_frame
                    print(
                        f"[INFO] Display: Advancing past {skipped_count} skipped frame(s), jumping to frame {frame_number_to_display}"
                    )
                    self.next_frame_to_display = frame_number_to_display

            if frame_number_to_display not in self.frames_to_display:
                # Frame not ready.
                self._arm_display_metronome_retry_ms(_retry_ms)
                return
            frame = self.frames_to_display.pop(frame_number_to_display)
            profile_for_overlay = self.frames_pipeline_profile.pop(
                frame_number_to_display, None
            )

        # --- 7. Frame is ready: Process and Display ---
        self.current_frame = frame  # Update current frame state

        # Emit a signal every 500 frames to notify JobProcessor we are still alive
        if self.file_type not in LIVE_STREAM_FILE_TYPES:  # Don't spam on live streams
            self.heartbeat_frame_counter += 1
            if self.heartbeat_frame_counter >= 500:
                self.heartbeat_frame_counter = 0
                self.processing_heartbeat_signal.emit()

        # Send to Virtual Cam
        self.send_frame_to_virtualcam(frame)

        # Write to FFmpeg
        if self.is_processing_segments or self.recording:
            if (
                self.recording_sp
                and self.recording_sp.stdin
                and not self.recording_sp.stdin.closed
            ):
                try:
                    self.recording_sp.stdin.write(frame.tobytes())
                    # update counters for duration calculation
                    self.frames_written += 1
                    self.last_displayed_frame = frame_number_to_display
                except OSError as e:
                    log_prefix = (
                        f"segment {self.current_segment_index + 1}"
                        if self.is_processing_segments
                        else "recording"
                    )
                    print(
                        f"[WARN] Error writing frame {frame_number_to_display} to FFmpeg stdin during {log_prefix}: {e}"
                    )
            else:
                log_prefix = (
                    f"segment {self.current_segment_index + 1}"
                    if self.is_processing_segments
                    else "recording"
                )
                print(
                    f"[WARN] FFmpeg stdin not available for {log_prefix} when trying to write frame {frame_number_to_display}."
                )

        # Update UI
        if self.file_type not in LIVE_STREAM_FILE_TYPES:
            # This is the metronome tick.
            if frame_number_to_display in self.main_window.markers:
                # Acquire lock to safely modify parameters and controls
                with self.main_window.models_processor.model_lock:
                    # 1. Load data from marker into main_window.parameters/control
                    video_control_actions.update_parameters_and_control_from_marker(
                        self.main_window, frame_number_to_display
                    )

                    # 2. Update all UI widgets to reflect the new state
                    video_control_actions.update_widget_values_from_markers(
                        self.main_window, frame_number_to_display
                    )

        # CREATE QPIXMAP JUST-IN-TIME (GUI Thread)
        _perf_disp = os.environ.get("VISIOMASTER_PERF_DISPLAY", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        _t_disp0 = time.perf_counter() if _perf_disp else 0.0
        pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, frame)

        graphics_view_actions.update_graphics_view(
            self.main_window, pixmap, frame_number_to_display
        )
        graphics_view_actions.update_pipeline_profile_overlay(
            self.main_window, profile_for_overlay
        )
        if _perf_disp:
            print(
                f"[PERF-DISPLAY] frame={frame_number_to_display} "
                f"ui_ms={(time.perf_counter() - _t_disp0) * 1000.0:.2f}",
                flush=True,
            )

        self.playback_frames_displayed += 1

        # --- 8. Clean up and Increment ---
        if self.file_type not in LIVE_STREAM_FILE_TYPES:
            # Increment for next frame
            self.next_frame_to_display += 1

        # Audio-master preview: if still behind the ffplay timeline, repaint soon so we
        # can consume newer frames from the buffer without waiting a full metronome step.
        if (
            self._wall_clock_use_audio_file_rate
            and self.file_type == "video"
            and self._playback_use_wall_clock
            and not self.recording
        ):
            tgt_follow = self._advance_past_skipped_for_display(
                self._expected_frame_from_wall_clock()
            )
            if tgt_follow - self.next_frame_to_display >= 2:
                self._arm_display_metronome_retry_ms(max(1, min(6, _retry_ms // 2)))
                return

        # Advance cadence and schedule the next display tick only after painting.
        self._arm_display_metronome_after_frame_shown()

    def send_frame_to_virtualcam(self, frame: numpy.ndarray):
        """
        OPTIMIZED: Sends the given frame to the pyvirtualcam device.
        Removed sleep_until_next_frame() to prevent blocking the Main GUI Thread.
        The UI metronome (QTimer) already handles perfect timing and synchronization.
        """
        if self.main_window.control["SendVirtCamFramesEnableToggle"] and self.virtcam:
            height, width, _ = frame.shape
            if self.virtcam.height != height or self.virtcam.width != width:
                # Resolution changed (e.g. source swap / restorer output differs).
                # Avoid hammering OBS with rapid close/reopen cycles — schedule a
                # single deferred restart so the driver gets adequate settling time.
                # We skip this frame rather than sending one with the wrong size.
                print(
                    f"[INFO] VirtCam resolution changed "
                    f"({self.virtcam.width}x{self.virtcam.height} → {width}x{height}). "
                    f"Restarting virtual camera…"
                )
                self.enable_virtualcam()
                return  # Frame already consumed; next tick will send at the new size.

            # Need to check again if virtcam was successfully re-enabled
            if self.virtcam:
                try:
                    self.virtcam.send(frame)
                    # REMOVED: self.virtcam.sleep_until_next_frame()
                    # It forces the UI thread to freeze and fights the metronome.
                except Exception as e:
                    print(f"[WARN] Failed sending frame to virtualcam: {e}")

    def set_number_of_threads(self, value):
        """Updates the thread count for the *next* worker pool."""
        if not value:
            value = 1
        # Stop processing if it's running, to apply the new count on next start
        if self.processing or self.is_processing_segments:
            print(
                f"[INFO] Setting thread count to {value}. Stopping active processing."
            )
            self.stop_processing()
        else:
            print(f"[INFO] Max Threads set as {value}. Will be applied on next run.")

        self.main_window.models_processor.set_number_of_threads(value)
        self.num_threads = value

    def process_video(self):
        """
        Start video processing.
        This can be either simple playback OR "default-style" recording.
        """

        # 1. Guards
        if self.processing or self.is_processing_segments:
            print(
                "[INFO] Processing already in progress (play or segment). Ignoring start request."
            )
            # Reset recording flag so a caller that set it before this guard fires
            # does not leave the application in a state where recording=True but
            # nothing is actually recording.
            if self.recording and not self.is_processing_segments:
                self.recording = False
                video_control_actions.reset_media_buttons(self.main_window)
            return

        if self.file_type != "video":
            print("[WARN] Process video: Only applicable for video files.")
            return

        if not (self.media_capture and self.media_capture.isOpened()):
            # Attempt lazy reopen — the capture may have been released during finalization
            # of a previous recording and the OS file handle not yet fully freed.
            if self.file_type == "video" and self.media_path:
                print(
                    "[INFO] media_capture not open on process_video() entry; attempting reopen..."
                )
                current_slider_pos = self.main_window.videoSeekSlider.value()
                if self._reopen_video_capture(current_slider_pos):
                    print("[INFO] media_capture reopened successfully.")
                else:
                    self.media_capture = None

            if not (self.media_capture and self.media_capture.isOpened()):
                print("[ERROR] Unable to open the video source.")
                self.processing = False
                self.recording = False
                self.is_processing_segments = False
                video_control_actions.reset_media_buttons(self.main_window)
                return

        # 2. Determine target FPS (after guards so media_capture is confirmed open)
        if self.main_window.control["VideoPlaybackCustomFpsToggle"]:
            # Custom FPS mode is enabled
            self.fps = self.main_window.control["VideoPlaybackCustomFpsSlider"]
            _fps_src = "custom slider"
        else:
            # Custom FPS mode is DISABLED, use original
            self.fps = self.media_capture.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
            _fps_src = "container (OpenCV CAP_PROP_FPS)"

        if not self.recording:
            print(
                f"[INFO] Display metronome pacing: {self.fps:.3f} fps "
                f"({1000.0 / float(self.fps):.2f} ms/frame) — source: {_fps_src}. "
                "If preview feels slow or fast vs. the real clip, enable "
                "'VideoPlaybackCustomFpsToggle' and set the target fps."
            )

        mode = "recording (default-style)" if self.recording else "playback"
        print(f"[INFO] Starting video {mode} processing setup...")

        # 3. Set State Flags
        self.processing = True  # General flag ON
        self.is_processing_segments = False
        self.playback_started = False
        self.stopped_by_error_limit = False  # Reset error limit flag for new processing

        # Initialize feeder state with the current UI global state
        with self.state_lock:
            self.feeder_parameters = copy.deepcopy(self.main_window.parameters)
            self.feeder_control = copy.deepcopy(self.main_window.control)

        self.sync_feeder_ui_face_flags_from_main_window()
        self.clear_recognition_embedding_cache()

        # Seed global PyTorch/CUDA RNG once per video session from the denoiser seed
        # slider. This ensures reproducible denoiser output for the whole video without
        # resetting the seed on every frame (which would break multi-threaded workers).
        _denoiser_seed = int(
            self.main_window.control.get("DenoiserBaseSeedSlider", 220)
        )
        torch.manual_seed(_denoiser_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(_denoiser_seed)

        # Check if this recording was initiated by the Job Manager
        job_mgr_flag = getattr(self.main_window, "job_manager_initiated_record", False)
        if self.recording and job_mgr_flag:
            self.triggered_by_job_manager = True
            print("[INFO] Detected default-style recording initiated by Job Manager.")
        else:
            self.triggered_by_job_manager = False
        try:
            self.main_window.job_manager_initiated_record = False
        except Exception:
            pass

        # 4. Setup Recording (if applicable)
        if self.recording:
            # Disable UI elements
            if not self.main_window.control["KeepControlsToggle"]:
                layout_actions.disable_all_parameters_and_control_widget(
                    self.main_window
                )

        # 6a. Reset Timers and Containers
        self.start_time = time.perf_counter()
        self.playback_frames_displayed = 0
        self.frames_to_display.clear()
        self.frames_pipeline_profile.clear()

        # 6b. START WORKER POOL
        print(f"[INFO] Starting {self.num_threads} persistent worker thread(s)...")
        # Ensure old workers are cleared (from a previous run)
        self.join_and_clear_threads()
        self.worker_threads = []
        # Clear any stale tasks or poison pills left from the previous session.
        # join_and_clear_threads() returns early when worker_threads is empty,
        # so pills from workers that exited via stop_event (not pill consumption)
        # can remain in the queue and kill new workers immediately.
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
            self.frame_queue.all_tasks_done.notify_all()
            self.frame_queue.not_full.notify_all()
        for i in range(self.num_threads):
            worker = FrameWorker(
                frame_queue=self.frame_queue,  # Pass the task queue
                main_window=self.main_window,
                worker_id=i,
            )
            worker.start()
            self.worker_threads.append(worker)

        # --- 7. AUDIO/VIDEO SYNC LOGIC ---

        # 7a. Get the target frame
        actual_start_frame = self.main_window.videoSeekSlider.value()
        print(f"[INFO] Sync: Seeking directly to frame {actual_start_frame}...")

        # 7b–7c. Seek + read under one capture lock (avoids libavcodec races vs other threads).
        target_height = self._get_target_input_height()

        print(
            f"[INFO] Sync: Reading frame {actual_start_frame} using locked helper (Target Height: {target_height})..."
        )
        ret, frame_bgr = misc_helpers.read_frame(
            self.media_capture,
            self.media_rotation,
            preview_target_height=target_height,
            seek_to_frame_first=actual_start_frame,
        )
        print(f"[INFO] Sync: Initial read complete (Result: {ret}).")

        if not ret:
            fallback_frame = int(
                misc_helpers.capture_get_prop(
                    self.media_capture, cv2.CAP_PROP_POS_FRAMES
                )
            )
            fallback_frame_to_try = max(0, fallback_frame - 1)
            print(
                f"[WARN] Failed initial read for frame {actual_start_frame}. Retrying from frame {fallback_frame_to_try}."
            )
            if fallback_frame_to_try == actual_start_frame:
                print("[ERROR] Fallback frame is the same. Cannot proceed.")
                self.stop_processing()
                return
            print(
                f"[INFO] Sync: Retrying read for frame {fallback_frame_to_try} using locked helper..."
            )
            ret, frame_bgr = misc_helpers.read_frame(
                self.media_capture,
                self.media_rotation,
                preview_target_height=target_height,
                seek_to_frame_first=fallback_frame_to_try,
            )
            print(f"[INFO] Sync: Retry read complete (Result: {ret}).")
            if not ret:
                print(
                    f"[ERROR] Capture failed definitively near frame {actual_start_frame}."
                )
                self.stop_processing()
                return
            actual_start_frame = (
                fallback_frame_to_try  # Use the frame we successfully read
            )

        # 7d. Frame is valid - Store for potential FFmpeg init
        frame_rgb = misc_helpers.bgr_uint8_to_rgb_contiguous(frame_bgr)
        self.current_frame = frame_rgb  # Store for FFmpeg dimensions

        # DELAYED FFMPEG CREATION
        if self.recording:
            if not self.create_ffmpeg_subprocess(output_filename=None):
                print("[ERROR] Failed to start FFmpeg for default-style recording.")
                self.stop_processing()  # Abort the start
                return

        # !!! CRITICAL: Reset position AGAIN so the feeder reads this frame too !!!
        print(
            f"[INFO] Sync: Resetting position to frame {actual_start_frame} for feeder thread..."
        )
        misc_helpers.seek_frame(self.media_capture, actual_start_frame)
        print("[INFO] Sync: Position reset complete.")

        # 7e. Update counters
        self.next_frame_to_display = (
            actual_start_frame  # Display starts here once buffered
        )
        self.processing_start_frame = actual_start_frame
        self.current_frame_number = (
            actual_start_frame  # Feeder reads this frame first when it starts
        )
        self._clear_sequential_detection_feed_state()
        self._audio_sync_last_seek_monotonic = 0.0

        # Calculate play_start_time
        self.play_start_time = (
            float(actual_start_frame / float(self.fps)) if self.fps > 0 else 0.0
        )
        if self.recording:
            print(
                f"[INFO] Recording audio start time set to: {self.play_start_time:.3f}s (Frame: {actual_start_frame})"
            )

        # 7f. Update the slider
        self.main_window.videoSeekSlider.blockSignals(True)
        self.main_window.videoSeekSlider.setValue(actual_start_frame)
        self.main_window.videoSeekSlider.blockSignals(False)

        # --- 8. STARTING THE FEEDER THREAD AND METRONOME ---
        # VP-34: Initialize timing BEFORE starting the metronome to ensure immediate execution.
        self.last_display_schedule_time_sec = time.perf_counter()

        print(
            f"[INFO] Starting feeder thread (Mode: video, Recording: {self.recording})..."
        )
        self.feeder_thread = threading.Thread(target=self._feeder_loop, daemon=True)
        self.feeder_thread.start()

        if self.recording:
            # Recording: start the display metronome immediately
            print("[INFO] Recording mode: Starting metronome immediately.")
            self._start_metronome(9999.0, is_first_start=True)
        else:
            if self.main_window.control.get("VideoPlaybackBufferingToggle", False):
                # Playback: start the preroll monitor
                print(
                    f"[INFO] Playback mode: Waiting for preroll buffer (target: {self.preroll_target} frames)..."
                )

                # Ensure the connection is clean
                try:
                    self.preroll_timer.timeout.disconnect(
                        self._check_preroll_and_start_playback
                    )
                except RuntimeError:
                    pass  # Disconnection failed, which is normal the first time

                self.preroll_timer.timeout.connect(
                    self._check_preroll_and_start_playback
                )
                self.preroll_timer.start(100)
            else:
                # Recording: start the display metronome immediately
                print("[INFO] Playback mode.")
                self._start_synchronized_playback()

    def start_frame_worker(
        self, frame_number, frame, is_single_frame=False, synchronous=False
    ):
        """
        Starts a one-shot FrameWorker for a *single frame*.
        This is NOT used by the video pool.
        """
        # Stop any previous single-frame worker before starting a new one.
        # Without this, fast scrubbing spawns concurrent workers that share the same
        # model sessions — TRT inference is not thread-safe and crashes under concurrent
        # calls.  VR180 workers are especially vulnerable because they run for several
        # seconds (multiple face detections + landmark detection + stitching per frame).
        prev = self._current_single_frame_worker
        if prev is not None and prev.is_alive():
            prev.stop_event.set()
            prev.join(timeout=3.0)
            if prev.is_alive():
                print("[WARN] Previous single-frame worker did not finish within 3 s.")
        self._current_single_frame_worker = None

        worker = FrameWorker(
            frame=frame,  # Pass frame directly
            main_window=self.main_window,
            frame_number=frame_number,
            frame_queue=None,  # No queue for single frame
            is_single_frame=is_single_frame,
            worker_id=-1,  # Indicates single-frame mode
        )

        if synchronous:
            worker.run()
            return worker
        else:
            # Run in a *new* thread (asynchronous).
            self._current_single_frame_worker = worker
            worker.start()
            return worker

    def process_current_frame(self, synchronous: bool = False):
        """
        Process the single, currently selected frame (e.g., after seek or for image).
        This is a one-shot operation, not part of the metronome.
        """
        if self.processing or self.is_processing_segments:
            print("[INFO] Stopping active processing to process single frame.")
            if not self.stop_processing():
                print("[WARN] Could not stop active processing cleanly.")

        # Seed global PyTorch/CUDA RNG from the denoiser seed slider before every
        # single-frame preview. This ensures the seed slider change visibly affects
        # the denoised preview output.
        _denoiser_seed = int(
            self.main_window.control.get("DenoiserBaseSeedSlider", 220)
        )
        torch.manual_seed(_denoiser_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(_denoiser_seed)

        # Set frame number for processing
        if self.file_type == "video":
            self.current_frame_number = self.main_window.videoSeekSlider.value()
        elif self.file_type == "image" or self.file_type in LIVE_STREAM_FILE_TYPES:
            self.current_frame_number = 0

        self.next_frame_to_display = self.current_frame_number

        frame_to_process = None
        read_successful = False

        # --- Determine Input Resolution (Global Resize) ---
        target_height = self._get_target_input_height()

        # --- Read the frame based on file type ---
        if self.file_type == "video" and self.media_capture:
            # Apply target_height for VIDEO (seek + read atomically under capture lock)
            ret, frame_bgr = misc_helpers.read_frame(
                self.media_capture,
                self.media_rotation,
                preview_target_height=target_height,
                seek_to_frame_first=self.current_frame_number,
            )

            if ret and frame_bgr is not None:
                frame_to_process = numpy.ascontiguousarray(
                    frame_bgr[..., ::-1]
                )  # BGR to RGB
                read_successful = True
                misc_helpers.seek_frame(self.media_capture, self.current_frame_number)
            else:
                fn = self.current_frame_number
                max_fn = self.max_frame_number
                # Fallback: use the raw frame cached during the last slider seek preview.
                # OpenCV seeks near EOF are unreliable; the slider already read this
                # frame successfully so we can reuse it to avoid a silent no-op.
                if (
                    self._seek_cached_frame is not None
                    and self._seek_cached_frame[0] == fn
                    and self._seek_cached_frame[1] is not None
                ):
                    cached_frame_bgr = self._seek_cached_frame[1]
                    # Apply GlobalInputResize if needed (preview was read at native res)
                    if (
                        target_height is not None
                        and cached_frame_bgr.shape[0] > target_height
                    ):
                        h, w = cached_frame_bgr.shape[:2]
                        scale = target_height / h
                        cached_frame_bgr = cv2.resize(
                            cached_frame_bgr,
                            (int(w * scale), target_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    frame_to_process = cached_frame_bgr[..., ::-1]  # BGR to RGB
                    read_successful = True
                    misc_helpers.seek_frame(self.media_capture, fn)
                    print(
                        f"[INFO] Using cached slider frame {fn} as fallback for single processing."
                    )
                elif fn >= max_fn - TAIL_TOLERANCE:
                    print(
                        f"[INFO] EOF reached at frame {fn} (reported max={max_fn}), stopping gracefully."
                    )
                    self.current_frame_number = max_fn + 1
                    return None
                else:
                    print(
                        f"[ERROR] Cannot read frame {self.current_frame_number} for single processing!"
                    )
                    self.main_window.last_seek_read_failed = True

        elif self.file_type == "image":
            frame_bgr = misc_helpers.read_image_file(self.media_path)
            if frame_bgr is not None:
                # Apply target_height for IMAGE (Manual resize)
                if target_height is not None and frame_bgr.shape[0] > target_height:
                    h, w = frame_bgr.shape[:2]
                    scale = target_height / h
                    new_w = int(w * scale)
                    frame_bgr = cv2.resize(
                        frame_bgr, (new_w, target_height), interpolation=cv2.INTER_AREA
                    )

                frame_to_process = numpy.ascontiguousarray(
                    frame_bgr[..., ::-1]
                )  # BGR to RGB
                read_successful = True
            else:
                print("[ERROR] Unable to read image file for processing.")

        elif self.file_type in LIVE_STREAM_FILE_TYPES and self.media_capture:
            # DO NOT apply target_height for live capture (native resolution)
            ret, frame_bgr = misc_helpers.read_frame(
                self.media_capture, 0, preview_target_height=None
            )
            if ret and frame_bgr is not None:
                frame_to_process = numpy.ascontiguousarray(
                    frame_bgr[..., ::-1]
                )  # BGR to RGB
                read_successful = True
            else:
                print("[ERROR] Unable to read live capture frame for processing!")

        # --- Process if read was successful ---
        if read_successful and frame_to_process is not None:
            return self.start_frame_worker(
                self.current_frame_number,
                frame_to_process,
                is_single_frame=True,
                synchronous=synchronous,
            )

        return None

    def stop_processing(self) -> bool:
        """
        General Stop / Abort Function.
        This is the master function to stop *any* active processing
        (playback, recording, segments, webcam).

        Returns:
            True if any active processing was stopped or a broken capture was recovered.
        """
        # Step 0: Capture current state for return value and cleanup logic
        was_active = self.processing or self.is_processing_segments
        was_recording_default_style = self.recording
        was_processing_segments = self.is_processing_segments

        # VP-34: Check if capture is missing/broken while idle. If so, fix it.
        if not was_active:
            if self.file_type == "video" and self.media_path:
                if not self.media_capture or not self.media_capture.isOpened():
                    print(
                        "[INFO] stop_processing: Capture missing/closed while idle. Recovering..."
                    )
                    self._reopen_video_capture(self.main_window.videoSeekSlider.value())
                    video_control_actions.reset_media_buttons(self.main_window)
                    return True
            video_control_actions.reset_media_buttons(self.main_window)
            return False  # Nothing was active and capture seems OK

        print("[INFO] Aborting active processing...")
        if was_active:
            graphics_view_actions.reset_playback_fps_preview_session(self.main_window)

        # 1. Reset flags FIRST to stop all loops immediately.
        # VP-29: Set recording=False early to prevent further frames from being
        # dispatched to FFmpeg by concurrent worker threads.
        self.processing = False
        self.is_processing_segments = False
        self.recording = False
        self.triggered_by_job_manager = False
        self._playback_use_wall_clock = False
        self._playback_clock_t0 = 0.0
        self._playback_clock_anchor_frame = 0
        self._wall_clock_use_audio_file_rate = False
        self._audio_sync_wall_t0 = 0.0
        self._audio_sync_anchor_fn = 0
        self._audio_sync_fps_file = 0.0
        self._audio_sync_rate = 1.0
        self._audio_sync_last_seek_monotonic = 0.0
        self._interactive_playback_seek_pending = None
        self._feeder_deferred_seek_read = None

        # 2. Stop utility timers and audio
        self.gpu_memory_update_timer.stop()
        self.preroll_timer.stop()
        pm = getattr(self, "precise_metronome", None)
        if pm is not None:
            pm.stop()
        self.stop_live_sound()

        # Face tracker defaults (use thread-safe reset)
        self.main_window.models_processor.face_detectors.reset_tracker()

        # 3a. Release the capture object to unblock the feeder.
        # The feeder calls read_frame() in a loop; releasing here causes the next read
        # to fail immediately, driving the feeder's EOF branch and exit.
        print("[INFO] Releasing media capture to unblock feeder thread...")
        if self.media_capture:
            misc_helpers.release_capture(self.media_capture)
            self.media_capture = None

        # 3b. Wait for the feeder thread to fully exit.
        print("[INFO] Waiting for feeder thread to complete...")
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=3.0)
            if self.feeder_thread.is_alive():
                print("[WARN] Feeder thread did not join gracefully within 3s timeout.")
        self.feeder_thread = None
        print("[INFO] Feeder thread joined.")

        # 3c. Clear display buffers and join worker threads.
        # VP-24: We clear the queue and then send poison pills to wake workers
        # blocked on queue.get().
        self.frames_to_display.clear()
        self.frames_pipeline_profile.clear()
        self.clear_recognition_embedding_cache()
        self._seek_cached_frame = None  # release seek-preview frame (~6–25 MB at HD/4K)
        self.webcam_frames_to_display.queue.clear()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        print("[INFO] Waiting for worker threads to complete...")
        self.join_and_clear_threads()
        print("[INFO] Worker threads joined.")

        # 5. Stop and cleanup FFmpeg subprocess
        if self.recording_sp:
            print("[INFO] Closing and waiting for active FFmpeg subprocess...")
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(f"[WARN] Error closing ffmpeg stdin during abort: {e}")
            try:
                self.recording_sp.wait(timeout=5)
                print("[INFO] FFmpeg subprocess terminated.")
            except subprocess.TimeoutExpired:
                print("[WARN] FFmpeg subprocess did not terminate gracefully, killing.")
                self.recording_sp.kill()
                self.recording_sp.wait()
            except Exception as e:
                print(f"[ERROR] Error waiting for FFmpeg subprocess: {e}")
            self.recording_sp = None

        # 6. Cleanup temp files based on stopped mode.
        if was_processing_segments:
            print("[INFO] Cleaning up segment temporary directory due to abort.")
            self._cleanup_temp_dir()
        elif was_recording_default_style:
            print("[INFO] Cleaning up default-style temporary file due to abort.")
            if self.temp_file and os.path.exists(self.temp_file):
                try:
                    os.remove(self.temp_file)
                    print(f"[INFO] Removed temporary file: {self.temp_file}")
                except OSError as e:
                    print(
                        f"[WARN] Could not remove temp file {self.temp_file} during abort: {e}"
                    )
            self.temp_file = ""

        # 7. Reset segment state
        self.segments_to_process = []
        self.current_segment_index = -1
        self.temp_segment_files = []
        self.current_segment_end_frame = None
        self.last_detected_faces.clear()
        self._smoothed_kps.clear()

        # 8. RE-OPEN media capture IMMEDIATELY.
        # VP-34: This is critical. By ensuring media_capture is re-opened before
        # returning, we ensure that on_change_video_seek_slider() (which calls
        # stop_processing() first) can still read a frame for the preview.
        if self.file_type == "video" and self.media_path:
            current_slider_pos = self.main_window.videoSeekSlider.value()
            if self._reopen_video_capture(current_slider_pos):
                print(
                    f"[INFO] Video capture re-opened and seeked to {current_slider_pos} after stop."
                )
            else:
                print("[WARN] Failed to re-open media capture after active stop.")
        elif self.file_type == "webcam":
            # For webcam, re-opening essentially prepares it for the next 'Play' click.
            try:
                webcam_index = int(
                    self.main_window.control.get("WebcamDeviceSelection", 0)
                )
                self.media_capture = cv2.VideoCapture(webcam_index)
                if not self.media_capture.isOpened():
                    print("[WARN] Failed to re-open webcam capture after stop.")
                    self.media_capture = None
            except Exception as e:
                print(f"[WARN] Error re-opening webcam capture: {e}")
                self.media_capture = None
        elif self.file_type == "screen":
            try:
                if mss_available():
                    self.media_capture = create_screen_capture_from_control(
                        self.main_window.control
                    )
                else:
                    print("[WARN] mss not available; cannot re-open screen capture.")
                    self.media_capture = None
            except Exception as e:
                print(f"[WARN] Error re-opening screen capture: {e}")
                self.media_capture = None

        # 9. Final cleanup and UI reset
        layout_actions.enable_all_parameters_and_control_widget(self.main_window)
        video_control_actions.reset_media_buttons(self.main_window)

        print("[INFO] Clearing GPU Cache and running garbage collection.")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARN] Error clearing Torch cache: {e}")
        gc.collect()

        try:
            self.disable_virtualcam()
        except Exception:
            pass

        # compute end metrics using helper
        self.play_end_time, end_frame_for_calc, frames_actually_processed, duration = (
            self._compute_play_end()
        )
        if duration is not None:
            print(
                f"[INFO] Probed temp video duration during abort: {duration:.3f}s (recorded clip length), "
                f"play_end_time set to {self.play_end_time:.3f}s [media time]."
            )
        else:
            print(
                f"[INFO] Calculated recording end time (frame estimate) during abort: {self.play_end_time:.3f}s (based on frame {end_frame_for_calc})"
            )

        # 11. Final Timing and Logging
        self.end_time = time.perf_counter()
        processing_time_sec = self.end_time - self.start_time

        try:
            start_frame_num = getattr(
                self, "processing_start_frame", end_frame_for_calc
            )
            num_frames_processed = end_frame_for_calc - start_frame_num
            if num_frames_processed < 0:
                num_frames_processed = 0
        except Exception:
            num_frames_processed = 0

        self._log_processing_summary(processing_time_sec, num_frames_processed)
        self.playback_display_start_time = 0.0
        self.processing_stopped_signal.emit()

        return True  # Processing was stopped

    def join_and_clear_threads(self):
        """
        Stops and waits for all pool worker threads to finish.
        This function's *only* job is to set events, send pills, and join.
        It does NOT clear the queue.
        """
        active_threads = self.worker_threads
        if not active_threads:
            return  # Nothing to do

        print(f"[INFO] Signaling {len(active_threads)} active worker(s) to stop...")

        # 1. Set stop event for all workers in the pool
        for thread in active_threads:
            if hasattr(thread, "stop_event") and not thread.stop_event.is_set():
                try:
                    thread.stop_event.set()
                except Exception as e:
                    print(
                        f"[WARN] Error setting stop_event on thread {thread.name}: {e}"
                    )

        # 2. Wake up any workers blocked on queue.get() by sending a "poison pill" (None).
        # VP-24: Clear the queue first so pills are never lost when the queue is full,
        # then put one pill per worker unconditionally.
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        for _ in active_threads:
            try:
                self.frame_queue.put(None, block=False)
            except queue.Full:
                # Should not happen after the clear above, but guard anyway.
                pass
            except Exception as e:
                print(f"[WARN] Error putting poison pill in queue: {e}")

        # 3. Join all threads
        for thread in active_threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        print(f"[WARN] Thread {thread.name} did not join gracefully.")
            except Exception as e:
                print(f"[WARN] Error joining thread {thread.name}: {e}")

        # 4. Clear the worker list
        self.worker_threads.clear()

        # 5. Release GPU memory held by the now-dead workers (kernel tensors,
        #    FrameEnhancers/FrameEdits helpers, etc.).  CPython's reference-counting
        #    will free them eventually, but calling GC + empty_cache here ensures
        #    VRAM is reclaimed before the next session allocates new workers.
        import gc as _gc

        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reopen_video_capture(self, seek_frame: int = 0) -> bool:
        """
        Private helper to robustly re-open the video capture.
        Performs up to 3 attempts with a test read to ensure the capture is
        actually functional (not just 'open' according to OpenCV).
        """
        if not self.media_path:
            return False

        for attempt in range(3):
            try:
                print(f"[INFO] Re-opening video capture (attempt {attempt + 1})...")
                # First ensure any existing capture is released
                if self.media_capture:
                    misc_helpers.release_capture(self.media_capture)
                    self.media_capture = None

                self.media_capture = cv2.VideoCapture(self.media_path)
                # Explicitly enable OpenCV's auto-rotation to let it handle metadata natively
                if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                    self.media_capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
                if self.media_capture and self.media_capture.isOpened():
                    # PERFORM TEST READ: essential on Windows to detect silent handle failures
                    ret, _ = misc_helpers.read_frame(
                        self.media_capture,
                        self.media_rotation,
                        seek_to_frame_first=seek_frame,
                    )
                    if ret:
                        # Success! Reset counters and seek back to the target frame.
                        self.current_frame_number = seek_frame
                        self.next_frame_to_display = seek_frame
                        self._clear_sequential_detection_feed_state()
                        misc_helpers.seek_frame(self.media_capture, seek_frame)
                        self.refresh_video_codec_flags()
                        self.reset_av1_scrub_pipeline()
                        print(
                            f"[INFO] Video capture re-opened and verified at frame {seek_frame}."
                        )
                        return True
                    else:
                        print(
                            f"[WARN] Attempt {attempt + 1}: Capture is open but read() failed."
                        )
                        seek_frame = max(0, seek_frame - 1)
                else:
                    print(
                        f"[WARN] Attempt {attempt + 1}: VideoCapture.isOpened() is False."
                    )
            except Exception as e:
                print(f"[WARN] Attempt {attempt + 1}: Exception during re-open: {e}")

            # Cleanup before retry
            if self.media_capture:
                misc_helpers.release_capture(self.media_capture)
                self.media_capture = None
            time.sleep(0.2)

        print("[ERROR] Failed to re-open functional video capture after 3 attempts.")
        return False

    def refresh_video_codec_flags(self) -> None:
        """Set ``is_av1_codec`` from the open capture (for scrub heuristics)."""
        self.is_av1_codec = False
        if self.file_type != "video" or not self.media_capture:
            return
        try:
            fourcc = misc_helpers.capture_get_prop(
                self.media_capture, cv2.CAP_PROP_FOURCC
            )
            tag = misc_helpers.cv_fourcc_to_tag(fourcc)
        except Exception:
            return
        self.is_av1_codec = misc_helpers.is_av1_fourcc_tag(tag)

    def reset_av1_scrub_pipeline(self) -> None:
        """Invalidate in-flight AV1 scrub decodes (new clip or capture replaced)."""
        self._av1_scrub_session += 1
        try:
            while True:
                self._av1_scrub_queue.get_nowait()
        except queue.Empty:
            pass

    def enqueue_av1_scrub_preview(self, frame_num: int) -> None:
        """Queue latest frame index for background FFmpeg preview (coalesces under load)."""
        if not self.media_path or self.file_type != "video":
            return
        try:
            self._av1_scrub_queue.put_nowait(frame_num)
        except queue.Full:
            try:
                self._av1_scrub_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._av1_scrub_queue.put_nowait(frame_num)
            except queue.Full:
                pass

        with self._av1_scrub_worker_lock:
            if not self._av1_scrub_worker_running:
                self._av1_scrub_worker_running = True
                threading.Thread(
                    target=self._av1_scrub_worker_loop, daemon=True
                ).start()

    def _av1_scrub_worker_loop(self) -> None:
        while True:
            try:
                fn = self._av1_scrub_queue.get(timeout=2.0)
            except queue.Empty:
                with self._av1_scrub_worker_lock:
                    self._av1_scrub_worker_running = False
                return
            while True:
                try:
                    fn = self._av1_scrub_queue.get_nowait()
                except queue.Empty:
                    break

            sess = self._av1_scrub_session
            path = self.media_path
            fps = float(self.fps or 0.0)
            if not path:
                continue

            ok, bgr = misc_helpers.read_video_frame_ffmpeg_input_seek(
                path, fn, fps, max_height=480
            )
            if sess != self._av1_scrub_session:
                continue
            self._av1_scrub_emitter.frame_ready.emit(fn, bgr if ok else None)

    # --- Utility Methods ---

    def _format_duration(self, total_seconds: float) -> str:
        """
        Converts a duration in seconds to a human-readable string (e.g., 1h 15m 30.55s).

        :param total_seconds: The duration in seconds.
        :return: A formatted string.
        """
        try:
            total_seconds = float(total_seconds)

            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60

            parts = []
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0 or (hours > 0 and seconds == 0):
                parts.append(f"{minutes}m")

            # Always show seconds
            if hours > 0 or minutes > 0:
                # Show 2 decimal places if we also show hours/minutes
                parts.append(f"{seconds:05.2f}s")
            else:
                # Show 3 decimal places if it's only seconds
                parts.append(f"{seconds:.3f}s")

            return " ".join(parts)
        except Exception:
            # Fallback in case of an error
            return f"{total_seconds:.3f} seconds"

    def _apply_job_timestamp_to_output_name(
        self,
        was_triggered_by_job: bool,
        job_name: Optional[str],
        use_job_name: bool,
        output_file_name: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Appends the standard output timestamp to job-driven names."""
        if not was_triggered_by_job:
            return job_name, output_file_name

        timestamp = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
        if use_job_name and job_name:
            job_name = f"{job_name}_{timestamp}"
        elif output_file_name:
            output_file_name = f"{output_file_name}_{timestamp}"

        return job_name, output_file_name

    def _log_processing_summary(
        self, processing_time_sec: float, num_frames_processed: int
    ):
        """
        Calculates and prints the final processing time and average FPS.
        Uses the actual display duration for FPS calculation if playback occurred.
        """

        # 1. Print formatted duration (overall processing time)
        formatted_duration = self._format_duration(processing_time_sec)
        print(f"\n[INFO] Processing completed in {formatted_duration}")

        # 2. Calculate and print FPS (based on actual display time)
        display_duration_sec = 0.0
        # Check if playback actually started displaying frames
        if (
            self.playback_display_start_time > 0
            and self.end_time > self.playback_display_start_time
        ):
            display_duration_sec = self.end_time - self.playback_display_start_time
            print(
                f"[INFO] (Actual display duration: {self._format_duration(display_duration_sec)})"
            )
        else:
            # Playback might have stopped during preroll or it was a recording-only task
            # Use the overall time, but mention it includes setup/buffering
            display_duration_sec = processing_time_sec
            if (
                self.start_time != self.playback_display_start_time
            ):  # Check if display never started
                print(
                    "[INFO] (Note: FPS calculation includes initial buffering/setup time)"
                )

        try:
            # Prefer frames actually painted (preview metronome); timeline span inflates FPS
            # when wall-clock catch-up skips many indices in one tick.
            frames_for_fps = (
                self.playback_frames_displayed
                if self.playback_frames_displayed > 0
                else num_frames_processed
            )
            if (
                display_duration_sec > 0.01 and frames_for_fps > 0
            ):  # Use a small threshold for duration
                avg_fps = frames_for_fps / display_duration_sec
                label = (
                    "Average Display FPS (frames painted)"
                    if self.playback_frames_displayed > 0
                    else "Average Display FPS (timeline span)"
                )
                print(f"[INFO] {label}: {avg_fps:.2f}\n")
            elif frames_for_fps > 0:
                print(
                    "[WARN] Display duration too short to calculate meaningful FPS.\n"
                )
            else:
                print(
                    "[WARN] No frames were displayed or duration was zero, cannot calculate FPS.\n"
                )
        except Exception as e:
            print(f"[WARN] Could not calculate average FPS: {e}\n")

    # --- FFmpeg and Finalization ---

    def create_ffmpeg_subprocess(self, output_filename: str):
        """
        Creates the FFmpeg subprocess for recording.
        This is a merged function used by both default-style and multi-segment recording.

        :param output_filename: The direct output path. If None, it's default-style
                                recording and a temp file will be generated.
        """
        control = self.main_window.control.copy()
        is_segment = output_filename is not None

        # 1. Guards
        if (
            not isinstance(self.current_frame, numpy.ndarray)
            or self.current_frame.size == 0
        ):
            print("[ERROR] Current frame invalid. Cannot get dimensions.")
            return False
        if not self.media_path or not Path(self.media_path).is_file():
            print("[ERROR] Original media path invalid.")
            return False
        if self.fps <= 0:
            print("[ERROR] Invalid FPS.")
            return False

        start_time_sec = 0.0
        end_time_sec = 0.0

        if is_segment:
            if self.current_segment_index < 0 or self.current_segment_index >= len(
                self.segments_to_process
            ):
                print(f"[ERROR] Invalid segment index {self.current_segment_index}.")
                return False
            start_frame, end_frame = self.segments_to_process[
                self.current_segment_index
            ]
            start_time_sec = start_frame / self.fps
            end_time_sec = end_frame / self.fps

        # 2. Frame Dimensions
        frame_height, frame_width, _ = self.current_frame.shape
        # VP-28: Apply enhancer dimension scaling for BOTH segment and default recording modes.
        if control["FrameEnhancerEnableToggle"]:
            if control["FrameEnhancerTypeSelection"] in (
                "RealEsrgan-x2-Plus",
                "BSRGan-x2",
            ):
                frame_height = frame_height * 2
                frame_width = frame_width * 2
            elif control["FrameEnhancerTypeSelection"] in (
                "RealEsrgan-x4-Plus",
                "BSRGan-x4",
                "UltraSharp-x4",
                "UltraMix-x4",
                "RealEsr-General-x4v3",
            ):
                frame_height = frame_height * 4
                frame_width = frame_width * 4

        # Calculate downscale dimensions
        frame_height_down = frame_height
        frame_width_down = frame_width
        if control["FrameEnhancerDownToggle"]:
            if frame_width != 1920 or frame_height != 1080:
                frame_width_down_mult = frame_width / 1920
                # VP-27: Force even dimensions — most video codecs (h264/hevc) require
                # width and height to be multiples of 2.
                frame_height_down = math.ceil(frame_height / frame_width_down_mult) & ~1
                frame_width_down = 1920
            else:
                print("[WARN] Already 1920*1080")

        # 3. Output File Path and Logging
        if is_segment:
            segment_num = self.current_segment_index + 1
            print(
                f"[INFO] Creating FFmpeg (Segment {segment_num}): Video Dim={frame_width}x{frame_height}, FPS={self.fps}, Output='{output_filename}'"
            )
            print(
                f"[INFO] Audio Segment: Start={start_time_sec:.3f}s, End={end_time_sec:.3f}s (Frames {start_frame}-{end_frame})"
            )

            if Path(output_filename).is_file():
                try:
                    os.remove(output_filename)
                except OSError as e:
                    print(
                        f"[WARN] Could not remove existing segment file {output_filename}: {e}"
                    )
        else:
            # Default-style: create a unique temp file
            date_and_time = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
            try:
                base_temp_dir = os.path.join(os.getcwd(), "temp_files", "default")
                os.makedirs(base_temp_dir, exist_ok=True)

                # Clean up orphaned temp files from previous crashed sessions.
                # These are left behind when the application exits uncleanly during
                # a recording.  Only remove files older than 24 hours to avoid
                # accidentally deleting files from a recording that is still active
                # in another instance.
                try:
                    _cutoff = time.time() - 86400  # 24 hours
                    for _stale in Path(base_temp_dir).glob("temp_output_*.mp4"):
                        try:
                            if _stale.stat().st_mtime < _cutoff:
                                _stale.unlink()
                                print(f"[INFO] Removed stale temp file: {_stale.name}")
                        except OSError:
                            pass

                    _stale_audio_dir = Path(base_temp_dir) / "temp_audio"
                    if _stale_audio_dir.is_dir():
                        for _stale_audio_file in _stale_audio_dir.iterdir():
                            try:
                                if _stale_audio_file.stat().st_mtime < _cutoff:
                                    if _stale_audio_file.is_dir():
                                        shutil.rmtree(
                                            _stale_audio_file, ignore_errors=True
                                        )
                                    else:
                                        _stale_audio_file.unlink()
                                    print(
                                        f"[INFO] Removed stale temp audio artifact: {_stale_audio_file.name}"
                                    )
                            except OSError:
                                pass

                        try:
                            next(_stale_audio_dir.iterdir())
                        except StopIteration:
                            try:
                                _stale_audio_dir.rmdir()
                                print("[INFO] Removed empty stale temp audio directory")
                            except OSError:
                                pass
                except Exception:
                    pass  # Non-critical; never block recording startup

                self.temp_file = os.path.join(
                    base_temp_dir, f"temp_output_{date_and_time}.mp4"
                )
                print(f"[INFO] Default temp file will be created at: {self.temp_file}")
            except Exception as e:
                print(f"[ERROR] Failed to create temporary directory/file path: {e}")
                self.temp_file = f"temp_output_{date_and_time}.mp4"
                print(
                    f"[WARN] Falling back to local directory for temp file: {self.temp_file}"
                )

            print(
                f"[INFO] Creating FFmpeg : Video Dim={frame_width}x{frame_height}, FPS={self.fps}, Temp Output='{self.temp_file}'"
            )

            if Path(self.temp_file).is_file():
                try:
                    os.remove(self.temp_file)
                except OSError as e:
                    print(
                        f"[WARN] Could not remove existing temp file {self.temp_file}: {e}"
                    )

        # 4. Build FFmpeg Arguments
        hdrpreset = control["FFPresetsHDRSelection"]
        sdrpreset = control["FFPresetsSDRSelection"]
        ffquality = control["FFQualitySlider"]
        ffspatial = int(control["FFSpatialAQToggle"])
        fftemporal = int(control["FFTemporalAQToggle"])

        # Base args: read raw video from stdin.
        # VP-12: Frames written to stdin are in BGR24 byte order.
        # FrameWorker returns numpy arrays in BGR channel order (OpenCV convention).
        # display_next_frame writes frame.tobytes() directly, so the pixel format
        # passed to FFmpeg MUST remain "bgr24" to match the raw bytes.
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",  # The processed frame from FrameWorker is BGR
            "-s",
            f"{frame_width}x{frame_height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",  # Read from stdin
        ]

        if is_segment:
            # For segments, add the audio source and time limits
            args.extend(
                [
                    "-ss",
                    str(start_time_sec),
                    "-to",
                    str(end_time_sec),
                    "-i",
                    self.media_path,
                    "-map",
                    "0:v:0",  # Map video from stdin
                    "-map",
                    "1:a:0?",  # Map audio from media_path (if exists)
                    "-c:a",
                    "aac",
                    "-shortest",
                ]
            )

        # Video codec args
        if control["HDREncodeToggle"]:
            # HDR uses X265
            args.extend(
                [
                    "-c:v",
                    "libx265",
                    "-profile:v",
                    "main10",
                    "-preset",
                    str(hdrpreset),
                    "-pix_fmt",
                    "yuv420p10le",
                    "-x265-params",
                    f"crf={ffquality}:vbv-bufsize=10000:vbv-maxrate=10000:selective-sao=0:no-sao=1:strong-intra-smoothing=0:rect=0:aq-mode={ffspatial}:t-aq={fftemporal}:hdr-opt=1:repeat-headers=1:colorprim=bt2020:range=limited:transfer=smpte2084:colormatrix=bt2020nc:range=limited:master-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)':max-cll=1000,400",
                ]
            )
        else:
            # NVENC for SDR
            args.extend(
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    str(sdrpreset),
                    "-profile:v",
                    "main10",
                    "-cq",
                    str(ffquality),
                    "-pix_fmt",
                    "yuv420p10le",
                    "-colorspace",
                    "rgb",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-spatial-aq",
                    str(ffspatial),
                    "-temporal-aq",
                    str(fftemporal),
                    "-tier",
                    "high",
                    "-tag:v",
                    "hvc1",
                ]
            )

        target_matrix = "bt2020nc" if control["HDREncodeToggle"] else "bt709"
        scale_params = f"in_range=pc:out_range=tv:out_color_matrix={target_matrix}"

        if control["FrameEnhancerDownToggle"]:
            args.extend(
                [
                    "-vf",
                    f"scale={frame_width_down}x{frame_height_down}:{scale_params}:flags=lanczos+accurate_rnd+full_chroma_int",
                ]
            )
        else:
            args.extend(
                [
                    "-vf",
                    f"scale={scale_params}",
                ]
            )

        # Output file
        if is_segment:
            args.extend([output_filename])
        else:
            args.extend([self.temp_file])

        # 5. Start Subprocess
        try:
            self.recording_sp = subprocess.Popen(
                args, stdin=subprocess.PIPE, bufsize=-1
            )
            # reset write counters each time we start a new FFmpeg session
            self.frames_written = 0
            self.last_displayed_frame = None
            return True
        except FileNotFoundError:
            print(
                "[ERROR] FFmpeg command not found. Ensure FFmpeg is installed and in system PATH."
            )
            self.main_window.display_messagebox_signal.emit(
                "FFmpeg Error", "FFmpeg command not found.", self.main_window
            )
            return False
        except Exception as e:
            print(f"[ERROR] Failed to start FFmpeg subprocess : {e}")
            if is_segment:
                self.main_window.display_messagebox_signal.emit(
                    "FFmpeg Error",
                    f"Failed to start FFmpeg for segment {segment_num}:\n{e}",
                    self.main_window,
                )
            else:
                self.main_window.display_messagebox_signal.emit(
                    "FFmpeg Error", f"Failed to start FFmpeg:\n{e}", self.main_window
                )
            return False

    def _identify_frame_segments(self, actual_end_frame: int) -> List[Tuple[int, int]]:
        """
        Identify all continuous segments of successfully processed frames.
        Returns a list of (start_frame, end_frame) tuples, using absolute frame
        numbers from the original media.  When recording began partway through
        the source the first segment needs to start at ``self.processing_start_frame``
        rather than zero.

        Args:
            actual_end_frame: The actual last frame that was recorded (0-based,
                              absolute index in source)

        Example: if recording started at frame 100 and frames 150, 175 were
        skipped in a 200-frame video:
        Returns: [(100, 149), (151, 174), (176, 199)]
        """
        # Determine the first frame we actually processed (may be >0 if we
        # sought).  Default to 0 for standard playback/recordings that start
        # at the beginning.
        start_frame = getattr(self, "processing_start_frame", 0) or 0

        if not self.skipped_frames:
            # No skipped frames - single segment from start_frame to end
            return [(start_frame, actual_end_frame)]

        # Sort skipped frames and ignore any that occur before start_frame
        sorted_skipped = [f for f in sorted(self.skipped_frames) if f >= start_frame]
        segments = []
        segment_start = start_frame

        for skipped_frame in sorted_skipped:
            if skipped_frame > segment_start:
                # Frames from segment_start to skipped_frame-1 are successful
                segment_end = skipped_frame - 1
                if segment_start <= segment_end:
                    segments.append((segment_start, segment_end))
            # Next segment starts after the skipped frame
            segment_start = skipped_frame + 1

        # Add final segment if there are frames after the last skipped frame
        if segment_start <= actual_end_frame:
            segments.append((segment_start, actual_end_frame))

        # summary only; detailed segment listings are rarely needed and can
        # clutter the console.  If fuller diagnostics are required the
        # developer can re-enable by inspecting `self.skipped_frames` directly.
        print(f"[INFO] Identified {len(segments)} continuous frame segment(s)")

        return segments

    def _get_issue_scan_ranges(self) -> List[Tuple[int, int]]:
        """Return the frame ranges that a scan should inspect."""
        max_frame = int(self.max_frame_number)
        scan_ranges: List[Tuple[int, int]] = []
        open_start_frame: Optional[int] = None

        for start_frame, end_frame in self.main_window.job_marker_pairs:
            if start_frame is None:
                continue
            normalized_start = int(start_frame)
            if end_frame is None:
                open_start_frame = normalized_start
                continue

            normalized_end = int(end_frame)
            if normalized_end >= normalized_start:
                scan_ranges.append((normalized_start, normalized_end))

        if open_start_frame is not None and open_start_frame <= max_frame:
            scan_ranges.append((open_start_frame, max_frame))

        if scan_ranges:
            return misc_helpers.normalize_issue_scan_ranges(scan_ranges)

        return [(0, max_frame)]

    def describe_issue_scan_scope(
        self, scan_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> str:
        """Return a short human-readable description of the current scan scope."""
        scan_ranges = scan_ranges or self._get_issue_scan_ranges()
        max_frame = int(self.max_frame_number)
        if not getattr(self.main_window, "job_marker_pairs", []):
            return "Scanning full clip"
        if scan_ranges == [(0, max_frame)]:
            return "Scanning full clip"

        open_start_frames = [
            int(start_frame)
            for start_frame, end_frame in self.main_window.job_marker_pairs
            if start_frame is not None and end_frame is None
        ]
        has_open_start = bool(open_start_frames)
        open_start_frame = min(open_start_frames) if open_start_frames else None

        if len(scan_ranges) == 1:
            start_frame, end_frame = scan_ranges[0]
            if (
                has_open_start
                and end_frame == max_frame
                and open_start_frame is not None
            ):
                if start_frame < open_start_frame:
                    return f"Scanning 1 marked range and record start frame {open_start_frame} to end"
                if open_start_frame > 0:
                    return f"Scanning from record start frame {open_start_frame}"
            return "Scanning 1 marked range"

        effective_complete_segments = len(scan_ranges)
        effective_open_start_frame: Optional[int] = None
        if (
            has_open_start
            and scan_ranges[-1][1] == max_frame
            and open_start_frame is not None
        ):
            effective_open_start_frame = open_start_frame
            effective_complete_segments -= 1

        if effective_complete_segments and effective_open_start_frame is not None:
            range_label = "range" if effective_complete_segments == 1 else "ranges"
            return (
                f"Scanning {effective_complete_segments} marked {range_label} "
                f"and record start frame {effective_open_start_frame} to end"
            )
        if effective_complete_segments:
            range_label = "range" if effective_complete_segments == 1 else "ranges"
            return f"Scanning {effective_complete_segments} marked {range_label}"
        if effective_open_start_frame is not None:
            return f"Scanning from record start frame {effective_open_start_frame}"
        return "Scanning full clip"

    @staticmethod
    def _compute_longest_issue_run(issue_frames: list[int]) -> int:
        longest_issue_run = 0
        current_run = 0
        previous_frame = None
        for frame_number in sorted(set(issue_frames)):
            if previous_frame is not None and frame_number == previous_frame + 1:
                current_run += 1
            else:
                current_run = 1
            longest_issue_run = max(longest_issue_run, current_run)
            previous_frame = frame_number
        return longest_issue_run

    def _resolve_scan_state_for_frame(
        self,
        frame_number: int,
        base_control: ControlTypes,
        base_params: FacesParametersTypes,
        target_faces_snapshot: Optional[dict] = None,
        control_defaults_snapshot: Optional[ControlTypes] = None,
    ) -> tuple[ControlTypes, FacesParametersTypes]:
        """Resolve the effective control/parameter state for a scan frame.

        This mirrors playback/render marker semantics: if a marker exists at or
        before the frame, its parameter/control payload becomes the active state
        for that frame; otherwise the scan-start state remains active.
        """
        marker_data = video_control_actions._get_marker_data_for_position(  # type: ignore[attr-defined]
            self.main_window, frame_number
        )
        if not marker_data:
            return (
                cast(ControlTypes, copy.deepcopy(base_control)),
                cast(FacesParametersTypes, copy.deepcopy(base_params)),
            )

        local_params = cast(
            FacesParametersTypes, copy.deepcopy(marker_data.get("parameters", {}))
        )
        local_control: ControlTypes = cast(ControlTypes, {})
        local_control.update(
            cast(
                ControlTypes,
                copy.deepcopy(
                    control_defaults_snapshot
                    if control_defaults_snapshot is not None
                    else {}
                ),
            )
        )

        control_data = marker_data.get("control")
        if isinstance(control_data, dict):
            local_control.update(cast(ControlTypes, control_data).copy())

        # Mirror the playback helper behavior by ensuring every current target
        # face has a parameter dict, falling back to defaults when missing.
        active_target_faces = (
            target_faces_snapshot
            if target_faces_snapshot is not None
            else self.main_window.target_faces
        )
        for face_id in active_target_faces.keys():
            face_id_str = str(face_id)
            if face_id_str not in local_params:
                local_params[face_id_str] = cast(
                    ParametersTypes,
                    copy.deepcopy(self.main_window.default_parameters.data),
                )

        return local_control, local_params

    def _build_issue_scan_state_segments(
        self,
        scan_ranges: List[Tuple[int, int]],
        base_control: ControlTypes,
        base_params: FacesParametersTypes,
        target_faces_snapshot: dict,
        control_defaults_snapshot: Optional[ControlTypes] = None,
    ) -> list[tuple[int, int, ControlTypes, FacesParametersTypes]]:
        """Group scan ranges into marker-stable segments."""
        marker_positions = sorted(
            int(frame_number)
            for frame_number in getattr(self.main_window, "markers", {}).keys()
        )
        segments: list[tuple[int, int, ControlTypes, FacesParametersTypes]] = []

        for start_frame, end_frame in scan_ranges:
            range_markers = [
                marker_frame
                for marker_frame in marker_positions
                if start_frame < marker_frame <= end_frame
            ]
            segment_start = start_frame
            local_control, local_params = self._resolve_scan_state_for_frame(
                start_frame,
                base_control,
                base_params,
                target_faces_snapshot,
                control_defaults_snapshot,
            )

            for next_marker_frame in range_markers + [end_frame + 1]:
                segment_end = next_marker_frame - 1
                if segment_end >= segment_start:
                    segments.append(
                        (segment_start, segment_end, local_control, local_params)
                    )
                if next_marker_frame <= end_frame:
                    segment_start = next_marker_frame
                    local_control, local_params = self._resolve_scan_state_for_frame(
                        next_marker_frame,
                        base_control,
                        base_params,
                        target_faces_snapshot,
                        control_defaults_snapshot,
                    )

        return segments

    def _reset_issue_scan_sequential_state(self) -> None:
        """Clear scan-local sequential detection state at tracking boundaries."""
        self.last_detected_faces = []
        self._smoothed_kps = {}
        self._smoothed_dense_kps = {}

    def _prepare_issue_scan_match_context(
        self,
        local_control: ControlTypes,
        local_params: FacesParametersTypes,
        target_faces_snapshot: IssueScanTargetSnapshot,
    ) -> dict[str, Any]:
        """Precompute target embeddings and thresholds for a stable scan segment."""
        recognition_model = str(
            local_control.get("RecognitionModelSelection", "arcface_128")
        )
        similarity_type = str(local_control.get("SimilarityTypeSelection", "Opal"))
        default_params = dict(self.main_window.default_parameters.data)
        prepared_targets: list[tuple[str, float, numpy.ndarray]] = []

        for target_id, target_face_snapshot in target_faces_snapshot.items():
            face_id_str = str(target_face_snapshot.get("face_id", target_id))
            face_specific_params = misc_helpers.copy_mapping_data(
                local_params.get(face_id_str)
            )
            params_pd = misc_helpers.ParametersDict(
                face_specific_params, default_params
            )
            target_embeddings = cast(
                IssueScanTargetEmbeddings,
                target_face_snapshot.get("embeddings_by_model", {}),
            )
            target_embedding = target_embeddings.get(recognition_model, {}).get(
                similarity_type
            )
            if (
                not isinstance(target_embedding, numpy.ndarray)
                or target_embedding.size == 0
            ):
                continue
            prepared_targets.append(
                (
                    face_id_str,
                    float(params_pd["SimilarityThresholdSlider"]),
                    target_embedding,
                )
            )

        return {
            "recognition_model": recognition_model,
            "similarity_type": similarity_type,
            "prepared_targets": prepared_targets,
        }

    def _find_best_target_match_for_scan(
        self,
        detected_embedding: numpy.ndarray,
        prepared_targets: list[tuple[str, float, numpy.ndarray]],
    ) -> str | None:
        """Return the best target face using a precomputed scan match context."""
        best_target = None
        highest_sim = -1.0

        for target_face_id, threshold, target_embedding in prepared_targets:
            sim = self.main_window.models_processor.findCosineDistance(
                detected_embedding, target_embedding
            )
            if sim >= threshold and sim > highest_sim:
                highest_sim = sim
                best_target = target_face_id

        return best_target

    def _build_issue_scan_target_embedding(
        self,
        target_face: Any,
        recognition_model: str,
        similarity_type: str,
    ) -> numpy.ndarray:
        cropped_face = getattr(target_face, "cropped_face", None)
        if not isinstance(cropped_face, numpy.ndarray) or cropped_face.size == 0:
            return numpy.array([])
        image = numpy.ascontiguousarray(cropped_face)
        image_uint8 = (
            image if image.dtype == numpy.uint8 else image.astype("uint8", copy=False)
        )
        image_tensor = (
            torch.from_numpy(image_uint8)
            .to(self.main_window.models_processor.device, non_blocking=True)
            .permute(2, 0, 1)
        )
        height, width = image_uint8.shape[:2]
        full_face_kps = numpy.array(
            [
                [0.3 * width, 0.35 * height],
                [0.7 * width, 0.35 * height],
                [0.5 * width, 0.55 * height],
                [0.35 * width, 0.75 * height],
                [0.65 * width, 0.75 * height],
            ],
            dtype=numpy.float32,
        )
        face_emb, _ = self.main_window.models_processor.run_recognize_direct(
            image_tensor,
            full_face_kps,
            similarity_type,
            recognition_model,
        )
        return face_emb if isinstance(face_emb, numpy.ndarray) else numpy.array([])

    def prepare_issue_scan_target_faces_snapshot(
        self,
        scan_ranges: list[tuple[int, int]],
        base_control: ControlTypes,
        base_params: FacesParametersTypes,
        control_defaults_snapshot: Optional[ControlTypes] = None,
    ) -> IssueScanTargetSnapshot:
        """Build a worker-safe target-face snapshot for issue scans."""
        live_target_faces = dict(self.main_window.target_faces)
        if not live_target_faces:
            return {}

        scan_segments = self._build_issue_scan_state_segments(
            scan_ranges,
            base_control,
            base_params,
            live_target_faces,
            control_defaults_snapshot,
        )
        required_embedding_modes = {
            (
                str(local_control.get("RecognitionModelSelection", "arcface_128")),
                str(local_control.get("SimilarityTypeSelection", "Opal")),
            )
            for _start_frame, _end_frame, local_control, _local_params in scan_segments
        }
        if not required_embedding_modes:
            required_embedding_modes = {("arcface_128", "Opal")}

        target_faces_snapshot: IssueScanTargetSnapshot = {}
        for target_id, target_face in live_target_faces.items():
            embeddings_by_model: IssueScanTargetEmbeddings = {}
            for recognition_model, similarity_type in sorted(required_embedding_modes):
                model_embeddings = embeddings_by_model.setdefault(recognition_model, {})
                model_embeddings[similarity_type] = (
                    self._build_issue_scan_target_embedding(
                        target_face,
                        recognition_model,
                        similarity_type,
                    )
                )

            target_faces_snapshot[str(target_id)] = {
                "face_id": str(getattr(target_face, "face_id", target_id)),
                "embeddings_by_model": embeddings_by_model,
            }

        return target_faces_snapshot

    def scan_issue_frames(
        self,
        progress_callback=None,
        issue_found_callback=None,
        is_cancelled=None,
        scan_ranges: Optional[List[Tuple[int, int]]] = None,
        target_height: Optional[int] = None,
        base_control: Optional[dict] = None,
        base_params: Optional[dict] = None,
        target_faces_snapshot: Optional[IssueScanTargetSnapshot] = None,
        control_defaults_snapshot: Optional[dict] = None,
        reset_frame_number: Optional[int] = None,
    ) -> Optional[dict]:
        """Run a full-frame detection scan and return issue-frame results."""
        capture = cv2.VideoCapture(self.media_path)
        if not capture or not capture.isOpened():
            raise RuntimeError("Could not open the selected video for scanning.")

        scan_ranges = scan_ranges or self._get_issue_scan_ranges()
        dropped_frames_snapshot = {
            int(frame) for frame in getattr(self.main_window, "dropped_frames", set())
        }
        total_frames = misc_helpers.count_issue_scan_frames(
            scan_ranges, dropped_frames_snapshot
        )
        target_height = (
            target_height
            if target_height is not None
            else self._get_target_input_height()
        )
        base_control = cast(
            ControlTypes,
            copy.deepcopy(
                base_control if base_control is not None else self.main_window.control
            ),
        )
        base_params = cast(
            FacesParametersTypes,
            copy.deepcopy(
                base_params if base_params is not None else self.main_window.parameters
            ),
        )
        if target_faces_snapshot is None:
            target_faces_snapshot = self.prepare_issue_scan_target_faces_snapshot(
                scan_ranges,
                base_control,
                base_params,
                cast(Optional[ControlTypes], control_defaults_snapshot),
            )
        else:
            target_faces_snapshot = cast(
                IssueScanTargetSnapshot,
                dict(target_faces_snapshot),
            )
        previous_last_detected_faces = copy.deepcopy(self.last_detected_faces)
        previous_smoothed_kps = copy.deepcopy(self._smoothed_kps)
        previous_smoothed_dense_kps = copy.deepcopy(self._smoothed_dense_kps)
        total_frames_scanned = 0
        tracking_enabled = False
        issue_frames_by_face: dict[str, set[int]] = {
            str(face_id): set() for face_id in target_faces_snapshot.keys()
        }

        try:
            self._reset_issue_scan_sequential_state()
            scan_segments = self._build_issue_scan_state_segments(
                scan_ranges,
                base_control,
                base_params,
                target_faces_snapshot,
                cast(Optional[ControlTypes], control_defaults_snapshot),
            )
            tracking_enabled = any(
                bool(local_control.get("FaceTrackingEnableToggle", False))
                for _start_frame, _end_frame, local_control, _local_params in scan_segments
            )
            if tracking_enabled:
                self.main_window.models_processor.face_detectors.reset_tracker()
            previous_segment_tracking_enabled: Optional[bool] = None

            def emit_progress(frame_number: int) -> None:
                if progress_callback:
                    progress_callback(total_frames_scanned, total_frames, frame_number)

            def emit_issue(face_id: str, frame_number: int) -> None:
                normalized_face_id = str(face_id)
                face_frames = issue_frames_by_face.setdefault(normalized_face_id, set())
                normalized_frame = int(frame_number)
                if normalized_frame in face_frames:
                    return
                face_frames.add(normalized_frame)
                if issue_found_callback:
                    issue_found_callback(normalized_face_id, normalized_frame)

            def build_result(cancelled: bool) -> dict[str, Any]:
                faces_with_issues = sum(
                    1 for frames in issue_frames_by_face.values() if frames
                )
                return {
                    "issue_frames_by_face": {
                        face_id: sorted(frames)
                        for face_id, frames in issue_frames_by_face.items()
                    },
                    "frames_scanned": total_frames_scanned,
                    "faces_with_issues": faces_with_issues,
                    "cancelled": cancelled,
                }

            for start_frame, end_frame, local_control, local_params in scan_segments:
                current_segment_tracking_enabled = bool(
                    local_control.get("FaceTrackingEnableToggle", False)
                )
                if (
                    current_segment_tracking_enabled
                    and previous_segment_tracking_enabled is False
                ):
                    self.main_window.models_processor.face_detectors.reset_tracker()
                    self._reset_issue_scan_sequential_state()
                match_context = self._prepare_issue_scan_match_context(
                    local_control, local_params, target_faces_snapshot
                )
                misc_helpers.seek_frame(capture, start_frame)
                self.current_frame_number = start_frame
                frame_number = start_frame

                while frame_number <= end_frame:
                    if is_cancelled and is_cancelled():
                        return build_result(True)
                    if frame_number in dropped_frames_snapshot:
                        next_frame = frame_number + 1
                        while (
                            next_frame <= end_frame
                            and next_frame in dropped_frames_snapshot
                        ):
                            next_frame += 1
                        self.current_frame_number = next_frame
                        misc_helpers.seek_frame(capture, self.current_frame_number)
                        frame_number = next_frame
                        continue

                    ret, frame_bgr = misc_helpers.read_frame(
                        capture,
                        self.media_rotation,
                        preview_target_height=target_height,
                    )
                    if not ret or not isinstance(frame_bgr, numpy.ndarray):
                        for face_id in issue_frames_by_face:
                            emit_issue(face_id, frame_number)
                        self.current_frame_number = frame_number + 1
                        misc_helpers.seek_frame(capture, self.current_frame_number)
                        total_frames_scanned += 1
                        emit_progress(frame_number)
                        frame_number += 1
                        continue

                    frame_rgb = misc_helpers.bgr_uint8_to_rgb_contiguous(frame_bgr)
                    frame_rgb_uint8 = (
                        frame_rgb
                        if frame_rgb.dtype == numpy.uint8
                        else frame_rgb.astype("uint8", copy=False)
                    )
                    frame_tensor = (
                        torch.from_numpy(frame_rgb_uint8)
                        .to(self.main_window.models_processor.device, non_blocking=True)
                        .permute(2, 0, 1)
                    )
                    self.current_frame_number = frame_number
                    bboxes, kpss_5, _ = self._run_sequential_detection(
                        frame_rgb,
                        local_control,
                        local_params,
                        frame_tensor=frame_tensor,
                        force_detection=True,
                    )
                    detected_embeddings: list[numpy.ndarray] = []
                    if (
                        isinstance(bboxes, numpy.ndarray)
                        and bboxes.shape[0] > 0
                        and isinstance(kpss_5, numpy.ndarray)
                        and kpss_5.shape[0] > 0
                    ):
                        max_faces = min(bboxes.shape[0], kpss_5.shape[0])
                        recognition_model = match_context["recognition_model"]
                        similarity_type = match_context["similarity_type"]
                        for face_index in range(max_faces):
                            face_kps = kpss_5[face_index]
                            face_bbox = bboxes[face_index]
                            if not misc_helpers.is_detected_face_eligible_for_matching(
                                face_kps,
                                face_bbox,
                                FrameWorker._MIN_FACE_PIXELS,
                            ):
                                continue
                            face_emb, _ = (
                                self.main_window.models_processor.run_recognize_direct(
                                    frame_tensor,
                                    face_kps,
                                    similarity_type,
                                    recognition_model,
                                )
                            )
                            if (
                                isinstance(face_emb, numpy.ndarray)
                                and face_emb.size > 0
                            ):
                                detected_embeddings.append(face_emb)
                    del frame_tensor

                    matched_face_ids: set[str] = set()
                    prepared_targets = match_context["prepared_targets"]
                    for detected_embedding in detected_embeddings:
                        best_target_face_id = self._find_best_target_match_for_scan(
                            detected_embedding, prepared_targets
                        )
                        if best_target_face_id is not None:
                            matched_face_ids.add(best_target_face_id)

                    for face_id in issue_frames_by_face:
                        if face_id not in matched_face_ids:
                            emit_issue(face_id, frame_number)
                    total_frames_scanned += 1
                    emit_progress(frame_number)
                    frame_number += 1
                previous_segment_tracking_enabled = current_segment_tracking_enabled

            return build_result(False)
        finally:
            self.last_detected_faces = previous_last_detected_faces
            self._smoothed_kps = previous_smoothed_kps
            self._smoothed_dense_kps = previous_smoothed_dense_kps
            if tracking_enabled:
                self.main_window.models_processor.face_detectors.reset_tracker()
            self.current_frame_number = (
                reset_frame_number
                if reset_frame_number is not None
                else int(self.main_window.videoSeekSlider.value())
            )
            misc_helpers.release_capture(capture)

    def _extract_audio_segments(
        self, segments: List[Tuple[int, int]], temp_audio_dir: str
    ) -> Tuple[bool, List[str]]:
        """
        Extract audio from the original media for each frame segment.

        Returns: (success: bool, audio_files: List[str])
            - success: True if all segments extracted successfully
            - audio_files: List of paths to extracted audio files
        """
        audio_files = []

        for idx, (start_frame, end_frame) in enumerate(segments):
            # Convert frame numbers to time (seconds)
            start_time = start_frame / self.fps if self.fps > 0 else 0
            # end_time is exclusive (one frame after the last frame we want)
            end_time = (end_frame + 1) / self.fps if self.fps > 0 else 0

            # Skip empty segments (should not happen with our segment identification, but safety check)
            if start_time >= end_time:
                print(
                    f"[WARN] Skipping empty audio segment {idx + 1} (start_time={start_time:.3f}s >= end_time={end_time:.3f}s)"
                )
                continue

            # Use a containerized AAC output rather than raw ADTS .aac.
            # Raw AAC concatenation is brittle on some skipped-frame rebuilds,
            # especially for MKV-derived inputs with awkward timestamps.
            audio_file = os.path.join(temp_audio_dir, f"audio_segment_{idx:04d}.m4a")
            audio_files.append(audio_file)

            # Always normalize skipped-frame rebuild audio to AAC-in-M4A.
            # This keeps the concat/remux path codec-agnostic for any source
            # audio format that FFmpeg can decode from the input media.
            media_path: str = self.media_path  # type: ignore[assignment]
            args: list[str] = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-err_detect",
                "ignore_err",
                "-i",
                media_path,
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-vn",
                "-map",
                "0:a:0?",
                "-af",
                "aresample=async=1:first_pts=0",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-y",
                audio_file,
            ]

            try:
                print(
                    f"[INFO] Extracting audio segment {idx + 1}/{len(segments)}: {start_time:.3f}s → {end_time:.3f}s"
                )
                subprocess.run(args, check=True, capture_output=True, text=True)

                # Validate output; if it's not valid, retry once with the same
                # normalized extraction settings to rule out a transient failure.
                if not self._validate_audio_file(audio_file):
                    print(
                        f"[WARN] Validation failed for segment {idx + 1}, retrying extraction once"
                    )
                    re_args: list[str] = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "warning",
                        "-err_detect",
                        "ignore_err",
                        "-i",
                        media_path,
                        "-ss",
                        str(start_time),
                        "-to",
                        str(end_time),
                        "-vn",
                        "-map",
                        "0:a:0?",
                        "-af",
                        "aresample=async=1:first_pts=0",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-y",
                        audio_file,
                    ]
                    try:
                        subprocess.run(
                            re_args, check=True, capture_output=True, text=True
                        )
                    except subprocess.CalledProcessError as e2:
                        print(
                            f"[ERROR] Retry extraction failed for segment {idx + 1}: {e2}"
                        )
                        print(f"[ERROR] FFmpeg stderr: {e2.stderr}")
                        for audio in audio_files:
                            try:
                                os.remove(audio)
                            except OSError:
                                pass
                        return False, []
                    if not self._validate_audio_file(audio_file):
                        print(
                            f"[ERROR] Retried segment {idx + 1} is still invalid after validation"
                        )
                        for audio in audio_files:
                            try:
                                os.remove(audio)
                            except OSError:
                                pass
                        return False, []

                print(f"[INFO] Segment {idx + 1} extracted successfully")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to extract audio segment {idx + 1}: {e}")
                print(f"[ERROR] FFmpeg stderr: {e.stderr}")
                print(f"[ERROR] FFmpeg command: {' '.join(args)}")
                # Cleanup partial files
                for audio in audio_files:
                    try:
                        os.remove(audio)
                    except OSError:
                        pass
                return False, []
            except FileNotFoundError:
                print("[ERROR] FFmpeg not found. Cannot extract audio segments.")
                return False, []

        print(f"[INFO] All {len(segments)} audio segment(s) extracted successfully")
        return True, audio_files

    def _validate_audio_file(self, audio_file_path: str) -> bool:
        """
        Validate that an audio file can be properly decoded by FFmpeg.
        Returns True if audio is valid, False if corrupted.
        """
        if not os.path.exists(audio_file_path):
            print(f"[ERROR] Audio file does not exist: {audio_file_path}")
            return False

        try:
            # Try to probe the audio file with ffprobe
            args = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                audio_file_path,
            ]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                print(f"[WARN] ffprobe failed for {audio_file_path}: {result.stderr}")
                return False

            # Check if we got valid JSON output
            import json

            probe_data = json.loads(result.stdout)

            # Check if there's an audio stream
            audio_streams = [
                s
                for s in probe_data.get("streams", [])
                if s.get("codec_type") == "audio"
            ]
            if not audio_streams:
                print(f"[WARN] No audio stream found in {audio_file_path}")
                return False

            # Check duration
            format_info = probe_data.get("format", {})
            duration = format_info.get("duration")
            if duration is None or float(duration) <= 0:
                print(f"[WARN] Invalid or zero duration in {audio_file_path}")
                return False

            print(f"[INFO] Audio validation passed: {duration}s duration")
            return True

        except subprocess.TimeoutExpired:
            print(f"[WARN] Audio validation timed out for {audio_file_path}")
            return False
        except json.JSONDecodeError:
            print(f"[WARN] Invalid ffprobe output for {audio_file_path}")
            return False
        except Exception as e:
            print(f"[WARN] Audio validation failed for {audio_file_path}: {e}")
            return False

    def _probe_video_duration(self, file_path: str) -> float | None:
        """
        Return the duration (in seconds) of the video file at `file_path` using
        ffprobe.  If probing fails for any reason the function returns None.
        """
        if not file_path or not os.path.isfile(file_path):
            return None
        try:
            args = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None
            duration_str = result.stdout.strip()
            return float(duration_str) if duration_str else None
        except Exception as e:
            print(f"[WARN] Failed to probe video duration for {file_path}: {e}")
            return None

    def _compute_play_end(self) -> Tuple[float, int, int, float | None]:
        """Compute timing values used when finalizing a recording.

        Returns a tuple of:
          (play_end_time, end_frame_for_calc, frames_actually_processed, duration_probed)

        ``duration_probed`` is the length of the temp video file if probing
        succeeded, otherwise ``None``.  ``play_end_time`` is always an absolute
        timestamp in the original media timeline (i.e. includes ``play_start_time``).
        """
        end_frame = min(self.next_frame_to_display, self.max_frame_number + 1)
        frames_processed = end_frame - self.total_skipped_frames

        duration = None
        if self.temp_file and Path(self.temp_file).is_file():
            duration = self._probe_video_duration(self.temp_file)

        if duration is not None:
            play_end = self.play_start_time + duration
        elif self.frames_written > 0 and self.fps > 0:
            play_end = self.play_start_time + (self.frames_written / float(self.fps))
        else:
            play_end = float(end_frame / float(self.fps)) if self.fps > 0 else 0.0

        return play_end, end_frame, frames_processed, duration

    def _concatenate_audio_segments(
        self, audio_files: List[str], temp_audio_dir: str
    ) -> Optional[str]:
        """
        Concatenate multiple audio files into a single audio file using FFmpeg concat demuxer.

        Returns: Path to concatenated audio file, or None if failed
        """

        if not audio_files:
            print("[ERROR] No audio segments to concatenate")
            return None

        if len(audio_files) == 1:
            # Only one segment, return it directly
            print("[INFO] Only one audio segment, no concatenation needed")
            return audio_files[0]

        # Create concat manifest file
        concat_file = os.path.join(temp_audio_dir, "concat_manifest.txt")
        try:
            with open(concat_file, "w") as f:
                for audio_file in audio_files:
                    # FFmpeg concat demuxer expects absolute paths
                    abs_path = os.path.abspath(audio_file)
                    formatted_path = abs_path.replace("\\", "/")
                    f.write(f"file '{formatted_path}'\n")
            print(f"[INFO] Created concat manifest with {len(audio_files)} segments")
        except OSError as e:
            print(f"[ERROR] Failed to create concat manifest: {e}")
            return None

        output_audio = os.path.join(temp_audio_dir, "audio_concatenated.m4a")

        # FFmpeg concat demuxer command
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",  # Allow absolute filenames
            "-i",
            concat_file,
            "-vn",
            # Re-encode once here to flatten the segment timestamps into a
            # single monotonic audio stream before the final mux.
            "-af",
            "aresample=async=1:first_pts=0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-y",
            output_audio,
        ]

        try:
            print(f"[INFO] Concatenating {len(audio_files)} audio segment(s)...")
            subprocess.run(args, check=True)
            print("[INFO] ✓ Successfully concatenated audio segments")
            return output_audio
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to concatenate audio segments: {e}")
            print(f"[ERROR] FFmpeg command: {' '.join(args)}")
            return None
        except FileNotFoundError:
            print("[ERROR] FFmpeg not found. Cannot concatenate audio.")
            return None

    def _write_video_only_output(self, source_video: str, output_video: str) -> bool:
        """Fallback writer: produce a playable video-only output when audio handling fails."""
        if not source_video or not os.path.exists(source_video):
            print(f"[ERROR] Video-only fallback source missing: {source_video}")
            return False

        if output_video and os.path.exists(output_video):
            try:
                os.remove(output_video)
            except OSError:
                pass

        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            source_video,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-y",
            output_video,
        ]

        try:
            subprocess.run(args, check=True)
            print(
                f"[WARN] Audio processing failed; emitted video-only output: {output_video}"
            )
            return True
        except Exception as e:
            print(f"[ERROR] Video-only remux fallback failed: {e}")
            return False

    def _concatenate_segments_video_only(
        self, list_file_path: str, final_file_path: str
    ) -> bool:
        """Fallback concatenation for segment mode when audio concat fails."""
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file_path,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-y",
            final_file_path,
        ]

        try:
            subprocess.run(args, check=True)
            print(
                f"[WARN] Segment audio concat failed; emitted video-only output: {final_file_path}"
            )
            return True
        except Exception as e:
            print(f"[ERROR] Segment video-only fallback concat failed: {e}")
            return False

    def _attempt_segment_video_only_fallback(
        self, list_file_path: str, final_file_path: str, failure_message: str
    ) -> bool:
        """Try segment video-only concat fallback and show UI error if it fails."""
        print("[WARN] Attempting segment video-only fallback concatenation...")
        if self._concatenate_segments_video_only(list_file_path, final_file_path):
            return True

        self.main_window.display_messagebox_signal.emit(
            "Recording Error",
            failure_message,
            self.main_window,
        )
        return False

    def _rebuild_segment_audio_if_needed(self, segment_num: int) -> None:
        """Rebuild current segment audio from kept frame ranges when frames were skipped."""
        if not (
            self.total_skipped_frames > 0
            and self.temp_segment_files
            and self.current_segment_index >= 0
            and self.current_segment_index < len(self.segments_to_process)
        ):
            return

        current_segment_path = self.temp_segment_files[-1]
        if not (
            os.path.exists(current_segment_path)
            and os.path.getsize(current_segment_path) > 0
            and self.segment_temp_dir
        ):
            return

        start_frame, end_frame = self.segments_to_process[self.current_segment_index]
        actual_end_frame = (
            self.last_displayed_frame
            if self.last_displayed_frame is not None
            else end_frame
        )

        if actual_end_frame < start_frame:
            print(
                f"[WARN] Segment {segment_num}: invalid frame range for audio correction ({start_frame}..{actual_end_frame})."
            )
            return

        temp_audio_dir = os.path.join(
            self.segment_temp_dir,
            f"segment_audio_{self.current_segment_index:03d}_{uuid.uuid4().hex}",
        )
        os.makedirs(temp_audio_dir, exist_ok=True)

        previous_start_frame = getattr(self, "processing_start_frame", 0)
        try:
            self.processing_start_frame = start_frame
            keep_segments = self._identify_frame_segments(actual_end_frame)
        finally:
            self.processing_start_frame = previous_start_frame

        try:
            print(
                f"[INFO] Segment {segment_num}: rebuilding audio for skipped frames "
                f"(manual dropped={self.manual_dropped_skip_count}, read errors={self.read_error_skip_count})."
            )
            audio_ok, audio_files = self._extract_audio_segments(
                keep_segments, temp_audio_dir
            )
            if not (audio_ok and audio_files):
                print(
                    f"[WARN] Segment {segment_num}: audio extraction failed during skip correction, keeping original segment audio."
                )
                return

            corrected_audio = self._concatenate_audio_segments(
                audio_files, temp_audio_dir
            )
            if not corrected_audio:
                print(
                    f"[WARN] Segment {segment_num}: corrected audio concatenation failed, keeping original segment audio."
                )
                return

            remuxed_segment_path = os.path.join(
                self.segment_temp_dir,
                f"segment_{self.current_segment_index:03d}_synced.mp4",
            )
            args = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                current_segment_path,
                "-i",
                corrected_audio,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                "-y",
                remuxed_segment_path,
            ]
            subprocess.run(args, check=True)
            os.replace(remuxed_segment_path, current_segment_path)
            print(
                f"[INFO] Segment {segment_num}: rebuilt audio after skipping {self.total_skipped_frames} frame(s)."
            )
        except Exception as e:
            print(
                f"[WARN] Segment {segment_num}: failed to rebuild synced audio ({e}), keeping original segment audio."
            )
        finally:
            shutil.rmtree(temp_audio_dir, ignore_errors=True)

    def _finalize_default_style_recording(self):
        """Finalizes a successful default-style recording (adds audio, cleans up)."""
        print("[INFO] Finalizing default-style recording...")
        temp_audio_dir: str | None = None

        # Check if processing stopped due to error limit
        if self.stopped_by_error_limit:
            print(
                f"[WARN] Recording stopped due to excessive consecutive read errors ({self.consecutive_read_errors}). "
                f"Output will be saved with '_incomplete' suffix. Total skipped frames: {self.total_skipped_frames}."
            )

        try:
            self.processing = False  # Stop metronome

            # 1. Stop timers and any residual audio subprocess
            self.gpu_memory_update_timer.stop()
            self.preroll_timer.stop()
            self.stop_live_sound()

            # 2. Release capture early to unblock the feeder.
            print("[INFO] Releasing media capture to unblock feeder thread...")
            if self.media_capture:
                misc_helpers.release_capture(self.media_capture)
                self.media_capture = None

            # 3. Wait for the feeder thread to exit fully.
            print("[INFO] Waiting for feeder thread to complete...")
            if self.feeder_thread and self.feeder_thread.is_alive():
                self.feeder_thread.join(timeout=3.0)
                if self.feeder_thread.is_alive():
                    print(
                        "[WARN] Feeder thread did not exit cleanly during finalization."
                    )
            self.feeder_thread = None
            print("[INFO] Feeder thread joined.")

            # 4. Clear buffers and join worker threads.
            self.frames_to_display.clear()
            self.frames_pipeline_profile.clear()
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
            print("[INFO] Waiting for final worker threads...")
            self.join_and_clear_threads()
            print("[INFO] Worker threads joined.")

            # 6. Finalize FFmpeg (close stdin, wait for file to be written)
            if self.recording_sp:
                if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                    try:
                        print("[INFO] Closing FFmpeg stdin...")
                        self.recording_sp.stdin.close()
                    except OSError as e:
                        print(
                            f"[WARN] Error closing FFmpeg stdin during finalization: {e}"
                        )
                # VP-29: Mark recording stopped early.
                self.recording = False
                print("[INFO] Waiting for FFmpeg subprocess to finish writing...")
                try:
                    self.recording_sp.wait(timeout=10)
                    print("[INFO] FFmpeg subprocess finished.")
                except subprocess.TimeoutExpired:
                    print(
                        "[WARN] FFmpeg subprocess timed out during finalization, killing."
                    )
                    self.recording_sp.kill()
                    self.recording_sp.wait()
                except Exception as e:
                    print(
                        f"[ERROR] Error waiting for FFmpeg subprocess during finalization: {e}"
                    )
                self.recording_sp = None

            # 7. Calculate audio segment times
            end_frame_for_calc = min(
                self.next_frame_to_display, self.max_frame_number + 1
            )
            # Use frames actually written to FFmpeg for robust A/V timing.
            actual_frames_processed = max(0, int(self.frames_written))
            self.play_end_time = (
                self.play_start_time + float(actual_frames_processed / float(self.fps))
                if self.fps > 0
                else self.play_start_time
            )
            print(
                f"[INFO] Calculated recording end time: {self.play_end_time:.3f}s "
                f"(Frame {end_frame_for_calc}, skipped {self.total_skipped_frames}, "
                f"actual {actual_frames_processed})"
            )

            # 8. Audio Merging
            if self.play_end_time <= self.play_start_time:
                print("[WARN] Recording produced no frames. Skipping audio merge.")
                if self.temp_file and os.path.exists(self.temp_file):
                    try:
                        os.remove(self.temp_file)
                    except OSError:
                        pass
                self.temp_file = ""
            elif (
                self.temp_file
                and os.path.exists(self.temp_file)
                and os.path.getsize(self.temp_file) > 0
            ):
                # 5a. Determine final output path
                was_triggered_by_job = getattr(self, "triggered_by_job_manager", False)
                job_name = (
                    getattr(self.main_window, "current_job_name", None)
                    if was_triggered_by_job
                    else None
                )
                use_job_name = (
                    getattr(self.main_window, "use_job_name_for_output", False)
                    if was_triggered_by_job
                    else False
                )
                output_file_name = (
                    getattr(self.main_window, "output_file_name", None)
                    if was_triggered_by_job
                    else None
                )

                job_name, output_file_name = self._apply_job_timestamp_to_output_name(
                    was_triggered_by_job,
                    job_name,
                    use_job_name,
                    output_file_name,
                )

                final_file_path = misc_helpers.get_output_file_path(
                    self.media_path,
                    self.main_window.control["OutputMediaFolder"],
                    job_name=job_name,
                    use_job_name_for_output=use_job_name,
                    output_file_name=output_file_name,
                )

                # Add suffix if stopped due to error limit
                if self.stopped_by_error_limit:
                    path_obj = Path(final_file_path)
                    final_file_path = str(
                        path_obj.parent / f"{path_obj.stem}_incomplete{path_obj.suffix}"
                    )
                    print(
                        f"[WARN] Output marked as incomplete due to excessive read errors: {final_file_path}"
                    )

                output_dir = os.path.dirname(final_file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                if Path(final_file_path).is_file():
                    try:
                        os.remove(final_file_path)
                    except OSError:
                        pass

                # 5b. Run FFmpeg audio merge command
                print("[INFO] Adding audio (default-style merge)...")
                try:
                    if self.total_skipped_frames > 0:
                        print(
                            "[INFO] Rebuilding audio because frames were skipped "
                            f"(manual dropped={self.manual_dropped_skip_count}, read errors={self.read_error_skip_count})."
                        )
                        temp_audio_root = os.path.join(
                            os.path.dirname(self.temp_file), "temp_audio"
                        )
                        temp_audio_dir = os.path.join(
                            temp_audio_root,
                            f"{Path(self.temp_file).stem}_{uuid.uuid4().hex}",
                        )
                        os.makedirs(temp_audio_dir, exist_ok=True)

                        # Convert skipped frame map into keep-ranges, then extract and concat audio.
                        start_frame_for_calc = (
                            getattr(self, "processing_start_frame", 0) or 0
                        )
                        actual_end_frame = (
                            self.last_displayed_frame
                            if self.last_displayed_frame is not None
                            else end_frame_for_calc - 1
                        )
                        if actual_end_frame < start_frame_for_calc:
                            raise RuntimeError(
                                f"invalid frame boundaries: start={start_frame_for_calc}, end={actual_end_frame}"
                            )
                        segments = self._identify_frame_segments(actual_end_frame)
                        audio_ok, audio_files = self._extract_audio_segments(
                            segments, temp_audio_dir
                        )
                        if not audio_ok or not audio_files:
                            raise RuntimeError("failed to extract segmented audio")

                        final_audio_path = self._concatenate_audio_segments(
                            audio_files, temp_audio_dir
                        )
                        if not final_audio_path:
                            raise RuntimeError("failed to concatenate segmented audio")

                        args = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            self.temp_file,
                            "-i",
                            final_audio_path,
                            "-c:v",
                            "copy",
                            "-c:a",
                            "copy",
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0",
                            "-shortest",
                            final_file_path,
                        ]
                    else:
                        args = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            self.temp_file,
                            "-ss",
                            str(self.play_start_time),
                            "-to",
                            str(self.play_end_time),
                            "-i",
                            self.media_path,
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0?",
                            "-shortest",
                            # REMOVED: "-af", "aresample=async=1000" (Breaks CFR sync and incompatible with -c:a copy)
                            final_file_path,
                        ]

                    subprocess.run(args, check=True)
                    print(
                        f"[INFO] --- Successfully created final video: {final_file_path} ---"
                    )
                except Exception as e:
                    print(f"[ERROR] Audio merge failed: {e}")
                    if self.temp_file and os.path.exists(self.temp_file):
                        print(
                            "[WARN] Falling back to video-only output for default-style recording."
                        )
                        if not self._write_video_only_output(
                            self.temp_file, final_file_path
                        ):
                            self.main_window.display_messagebox_signal.emit(
                                "Recording Error",
                                f"Audio merge failed and video-only fallback also failed:\n{e}",
                                self.main_window,
                            )
                finally:
                    if self.temp_file and os.path.exists(self.temp_file):
                        try:
                            os.remove(self.temp_file)
                        except OSError:
                            pass
                    self.temp_file = ""
                    if temp_audio_dir and os.path.isdir(temp_audio_dir):
                        try:
                            shutil.rmtree(temp_audio_dir, ignore_errors=True)
                        except OSError:
                            pass
                    temp_audio_dir = None

            # 6. Final Timing and Logging
            self.end_time = time.perf_counter()
            processing_time_sec = self.end_time - self.start_time
            try:
                start_frame_num = getattr(
                    self, "processing_start_frame", end_frame_for_calc
                )
                num_frames_processed = end_frame_for_calc - start_frame_num
                if num_frames_processed < 0:
                    num_frames_processed = 0
            except Exception:
                num_frames_processed = 0
            self._log_processing_summary(processing_time_sec, num_frames_processed)

            # AutoSave workspace if enabled
            if self.main_window.control.get("AutoSaveWorkspaceToggle"):
                json_file_path = misc_helpers.get_output_file_path(
                    self.media_path, self.main_window.control["OutputMediaFolder"]
                )
                json_file_path += ".json"
                save_load_actions.save_current_workspace(
                    self.main_window, json_file_path
                )

            # 8b. Reopen media capture AFTER FFmpeg audio merge.
            if self.file_type == "video" and self.media_path:
                reset_frame = getattr(self, "processing_start_frame", 0)
                if self._reopen_video_capture(reset_frame):
                    self.main_window.videoSeekSlider.blockSignals(True)
                    self.main_window.videoSeekSlider.setValue(reset_frame)
                    self.main_window.videoSeekSlider.blockSignals(False)
                else:
                    print("[WARN] Failed to re-open media capture after recording.")

        except Exception as e:
            print(f"[ERROR] Exception during _finalize_default_style_recording: {e}")

        finally:
            # 10. Reset State and UI
            self.recording = False
            self.processing = False
            self.is_processing_segments = False

            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)

            print("[INFO] Clearing GPU Cache.")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

            try:
                self.disable_virtualcam()
            except Exception:
                pass

            if (
                self.main_window.control.get("OpenOutputToggle")
                and not self.triggered_by_job_manager
            ):
                try:
                    list_view_actions.open_output_media_folder(self.main_window)
                except Exception:
                    pass

            print("[INFO] Default-style recording finalized.")
            self.processing_stopped_signal.emit()

    # --- Virtual Camera Methods ---

    def enable_virtualcam(self, backend=False):
        """Starts the pyvirtualcam device."""

        # Guard: Only run if the user has actually enabled the virtual cam
        if not self.main_window.control.get("SendVirtCamFramesEnableToggle", False):
            # Ensure it's also disabled if the toggle is off
            self.disable_virtualcam()
            return

        if not self.media_capture and not isinstance(self.current_frame, numpy.ndarray):
            print("[WARN] Cannot enable virtual camera without media loaded.")
            return

        frame_height, frame_width = 0, 0
        current_fps = self.fps if self.fps > 0 else 30

        if (
            isinstance(self.current_frame, numpy.ndarray)
            and self.current_frame.ndim == 3
        ):
            frame_height, frame_width, _ = self.current_frame.shape
        elif self.media_capture and self.media_capture.isOpened():
            frame_height = int(
                misc_helpers.capture_get_prop(
                    self.media_capture, cv2.CAP_PROP_FRAME_HEIGHT
                )
            )
            frame_width = int(
                misc_helpers.capture_get_prop(
                    self.media_capture, cv2.CAP_PROP_FRAME_WIDTH
                )
            )
            if current_fps == 30:
                cap_fps_v = misc_helpers.capture_get_prop(
                    self.media_capture, cv2.CAP_PROP_FPS
                )
                current_fps = cap_fps_v if cap_fps_v > 0 else 30

        if frame_width <= 0 or frame_height <= 0:
            print(
                f"[ERROR] Cannot enable virtual camera: Invalid dimensions ({frame_width}x{frame_height})."
            )
            return

        self.disable_virtualcam()  # Close existing cam first

        # OBS Virtual Camera (and some other backends) uses a Windows kernel-mode
        # virtual device.  If a new pyvirtualcam.Camera() is opened immediately
        # after close(), the driver has not yet fully released the handle and the
        # new connection is silently ignored by OBS — producing the symptom where
        # the virtual cam appears to stop and cannot be reactivated without
        # switching to another cam and back.  A short settling delay eliminates
        # this race condition.
        time.sleep(0.15)

        backend_to_use = backend or self.main_window.control["VirtCamBackendSelection"]
        print(
            f"[INFO] Enabling virtual camera: {frame_width}x{frame_height} @ {int(current_fps)}fps, Backend: {backend_to_use}, Format: BGR"
        )

        for attempt in range(2):
            try:
                self.virtcam = pyvirtualcam.Camera(
                    width=frame_width,
                    height=frame_height,
                    fps=int(current_fps),
                    backend=backend_to_use,
                    fmt=pyvirtualcam.PixelFormat.BGR,  # Processed frame is BGR
                )
                print(f"[INFO] Virtual camera '{self.virtcam.device}' started.")
                break  # success — exit retry loop
            except Exception as e:
                if attempt == 0:
                    # First attempt failed (driver may still be releasing the handle).
                    # Wait longer and try once more before giving up.
                    print(
                        f"[WARN] Virtual camera open failed (attempt 1): {e}. Retrying in 500 ms…"
                    )
                    time.sleep(0.5)
                else:
                    print(f"[ERROR] Failed to enable virtual camera: {e}")
                    self.virtcam = None

    def disable_virtualcam(self):
        """Stops the pyvirtualcam device."""
        if self.virtcam:
            print(f"[INFO] Disabling virtual camera '{self.virtcam.device}'.")
            try:
                self.virtcam.close()
            except Exception as e:
                print(f"[WARN] Error closing virtual camera: {e}")
            self.virtcam = None

    # --- Multi-Segment Recording Methods ---

    def start_multi_segment_recording(
        self, segments: list[tuple[int, int]], triggered_by_job_manager: bool = False
    ):
        """
        Initializes and starts a multi-segment recording job.

        :param segments: A list of (start_frame, end_frame) tuples.
        :param triggered_by_job_manager: Flag for Job Manager integration.
        """

        # 1. Guards
        if self.processing or self.is_processing_segments:
            print(
                "[WARN] Attempted to start segment recording while already processing."
            )
            return

        if self.file_type != "video":
            print("[ERROR] Multi-segment recording only supported for video files.")
            return
        if not segments:
            print("[ERROR] No segments provided for multi-segment recording.")
            return
        if not (self.media_capture and self.media_capture.isOpened()):
            print("[ERROR] Video source not open for multi-segment recording.")
            return

        print("[INFO] --- Initializing multi-segment recording... ---")

        # 2. Set State Flags
        self.is_processing_segments = True
        self.recording = False
        self.processing = True  # Master flag
        self.triggered_by_job_manager = triggered_by_job_manager
        self.stopped_by_error_limit = False  # Reset error limit flag for new processing
        # Ensure all elements in 'segments' are strictly tuples of integers.
        sanitized_segments = []
        for seg in segments:
            try:
                # Convert list to tuple and ensure elements are ints
                sanitized_segments.append((int(seg[0]), int(seg[1])))
            except (IndexError, TypeError, ValueError) as e:
                print(f"[WARN] Ignoring malformed segment {seg}: {e}")

        self.segments_to_process = sorted(sanitized_segments)
        self.current_segment_index = -1
        self.temp_segment_files = []
        self.segment_temp_dir = None

        # 3. Disable UI
        if not self.main_window.control["KeepControlsToggle"]:
            layout_actions.disable_all_parameters_and_control_widget(self.main_window)

        # 4. Create Temp Directory
        try:
            base_temp_dir = os.path.join(os.getcwd(), "temp_files", "segments")
            os.makedirs(base_temp_dir, exist_ok=True)
            unique_id = uuid.uuid4()
            self.segment_temp_dir = os.path.join(base_temp_dir, f"run_{unique_id}")
            os.makedirs(self.segment_temp_dir, exist_ok=True)
            print(
                f"[INFO] Created temporary directory for segments: {self.segment_temp_dir}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to create temporary directory: {e}")
            self.main_window.display_messagebox_signal.emit(
                "File System Error",
                f"Failed to create temporary directory:\n{e}",
                self.main_window,
            )
            self.stop_processing()
            return

        # 5. Start Process
        self.start_time = time.perf_counter()
        self.playback_frames_displayed = 0

        # 6. Start the first segment
        self.process_next_segment()

    def process_next_segment(self):
        """
        Sets up and starts processing for the *next* segment in the list.
        This function is called iteratively by stop_current_segment.
        """

        # 1. Increment segment index
        self.current_segment_index += 1
        segment_num = self.current_segment_index + 1

        # 2. Check if all segments are done
        if self.current_segment_index >= len(self.segments_to_process):
            print("[INFO] All segments processed.")
            self.finalize_segment_concatenation()
            return

        # 3. Get segment details
        start_frame, end_frame = self.segments_to_process[self.current_segment_index]
        print(
            f"[INFO] --- Starting Segment {segment_num}/{len(self.segments_to_process)} (Frames: {start_frame} - {end_frame}) ---"
        )
        self.current_segment_end_frame = end_frame

        if not self.media_capture or not self.media_capture.isOpened():
            print(
                f"[ERROR] Media capture not available for seeking to segment {segment_num}."
            )
            self.stop_processing()
            return

        # 4. Seek to the start frame of the segment
        print(f"[INFO] Seeking to start frame {start_frame}...")
        misc_helpers.seek_frame(self.media_capture, start_frame)

        # --- CRITICAL CHANGE: Apply Global Resize here too ---
        target_height = self._get_target_input_height()
        # -----------------------------------------------------

        ret, frame_bgr = misc_helpers.read_frame(
            self.media_capture,
            self.media_rotation,
            preview_target_height=target_height,  # <--- Used to be None
        )
        if ret:
            self.current_frame = numpy.ascontiguousarray(
                frame_bgr[..., ::-1]
            )  # BGR to RGB
            # Must re-set position, as read() advances it
            misc_helpers.seek_frame(self.media_capture, start_frame)
            self.current_frame_number = start_frame
            self.next_frame_to_display = start_frame
            # Update slider for visual feedback
            self.main_window.videoSeekSlider.blockSignals(True)
            self.main_window.videoSeekSlider.setValue(start_frame)
            self.main_window.videoSeekSlider.blockSignals(False)
        else:
            print(
                f"[ERROR] Could not read frame {start_frame} at start of segment {segment_num}. Aborting."
            )
            self.stop_processing()
            return

        # 5. Clear containers AND START WORKER POOL for the new segment
        self.frames_to_display.clear()
        self.frames_pipeline_profile.clear()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        print(
            f"[INFO] Starting {self.num_threads} persistent worker thread(s) for segment..."
        )
        # Ensure old workers are cleaned up (if present)
        self.join_and_clear_threads()
        self.worker_threads = []
        for i in range(self.num_threads):
            worker = FrameWorker(
                frame_queue=self.frame_queue,  # Pass the task queue
                main_window=self.main_window,
                worker_id=i,
            )
            worker.start()
            self.worker_threads.append(worker)

        # 6. Setup FFmpeg subprocess for this segment
        # create_ffmpeg_subprocess uses self.current_frame.shape, so it will automatically
        # pick up the resized dimensions we set in step 4.
        temp_segment_filename = f"segment_{self.current_segment_index:03d}.mp4"
        temp_segment_path = os.path.join(self.segment_temp_dir, temp_segment_filename)
        self.temp_segment_files.append(temp_segment_path)

        if not self.create_ffmpeg_subprocess(output_filename=temp_segment_path):
            print(
                f"[ERROR] Failed to create ffmpeg subprocess for segment {segment_num}. Aborting."
            )
            self.stop_processing()
            return

        # 7. Synchronously process the first frame of the segment
        # VP-15: Use synchronous=True so the first frame is fully processed and the
        # single_frame_processed_signal has fired before the metronome starts.
        # This prevents the metronome from ticking before any frame is in frames_to_display.
        current_start_frame = self.current_frame_number
        print(
            f"[INFO] Sync: Synchronously processing first frame {current_start_frame} of segment..."
        )
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        self.start_frame_worker(
            current_start_frame,
            self.current_frame,
            is_single_frame=True,
            synchronous=True,
        )

        # 8. Update counters
        # self.current_frame_number was set to start_frame (e.g., 100)
        # We must increment it so the *next* read is correct (e.g., 101)
        self.current_frame_number += 1

        # 9. Start Metronome ET Feeder
        target_fps = 9999.0  # Always max speed for segments
        is_first = self.current_segment_index == 0

        # Start the feeder thread
        with self.state_lock:
            self.feeder_parameters = self.main_window.parameters.copy()
            self.feeder_control = self.main_window.control.copy()
        self.sync_feeder_ui_face_flags_from_main_window()
        self.clear_recognition_embedding_cache()
        print(
            f"[INFO] Starting feeder thread (Mode: segment {self.current_segment_index})..."
        )
        self.feeder_thread = threading.Thread(target=self._feeder_loop, daemon=True)
        self.feeder_thread.start()

        # Start the display metronome
        self._start_metronome(target_fps, is_first_start=is_first)

    def stop_current_segment(self):
        """
        Stops processing the *current* segment, finalizes its file,
        and triggers the next segment or final concatenation.
        """
        if not self.is_processing_segments:
            print("[WARN] stop_current_segment called but not processing segments.")
            return

        segment_num = self.current_segment_index + 1
        print(f"[INFO] --- Stopping Segment {segment_num} --- ")

        # 1. Stop timers
        self.gpu_memory_update_timer.stop()

        # 2a. Wait for the feeder thread
        print(f"[INFO] Waiting for feeder thread from segment {segment_num}...")
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=2.0)

            # VP-26: If the join timed out, abort rather than proceeding with two live feeders.
            if self.feeder_thread.is_alive():
                print(
                    f"[ERROR] Feeder thread from segment {segment_num} did not join within timeout. Aborting segment processing."
                )
                self.feeder_thread = None
                self.stop_processing()
                return
            else:
                print("[INFO] Feeder thread joined.")

        else:
            # This case is normal if the feeder finished its work very quickly
            print("[INFO] Feeder thread was already finished.")

        self.feeder_thread = None

        # 2b. Wait for workers
        print(f"[INFO] Waiting for workers from segment {segment_num}...")
        self.join_and_clear_threads()
        print("[INFO] Workers joined.")
        self.frames_to_display.clear()
        self.frames_pipeline_profile.clear()

        # 3. Finalize FFmpeg for this segment
        if self.recording_sp:
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    print(f"[INFO] Closing FFmpeg stdin for segment {segment_num}...")
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(
                        f"[WARN] Error closing FFmpeg stdin for segment {segment_num}: {e}"
                    )
            print(
                f"[INFO] Waiting for FFmpeg subprocess (segment {segment_num}) to finish writing..."
            )
            try:
                self.recording_sp.wait(timeout=10)
                print(f"[INFO] FFmpeg subprocess (segment {segment_num}) finished.")
            except subprocess.TimeoutExpired:
                print(
                    f"[WARN] FFmpeg subprocess (segment {segment_num}) timed out, killing."
                )
                self.recording_sp.kill()
                self.recording_sp.wait()
            except Exception as e:
                print(
                    f"[ERROR] Error waiting for FFmpeg subprocess (segment {segment_num}): {e}"
                )
            self.recording_sp = None
        else:
            print(
                f"[WARN] No active FFmpeg subprocess found when stopping segment {segment_num}."
            )

        if self.temp_segment_files and not os.path.exists(self.temp_segment_files[-1]):
            print(
                f"[ERROR] Segment file '{self.temp_segment_files[-1]}' not found after processing segment {segment_num}."
            )

        # If frames were skipped in this segment, rebuild segment audio
        # from valid frame ranges so concatenated output stays in sync.
        self._rebuild_segment_audio_if_needed(segment_num)

        # 4. Process the *next* segment
        self.process_next_segment()

    def finalize_segment_concatenation(self):
        """Concatenates all valid temporary segment files into the final output file."""
        print("[INFO] --- Finalizing concatenation of segments... ---")

        # Check if processing stopped due to error limit
        if self.stopped_by_error_limit:
            print(
                f"[WARN] Segment recording stopped due to excessive consecutive read errors ({self.consecutive_read_errors}). "
                f"Output will be saved with '_incomplete' suffix. Total skipped frames: {self.total_skipped_frames}."
            )

        # Failsafe: If this is called while an ffmpeg process is still running
        if self.recording_sp:
            segment_num = self.current_segment_index + 1
            print(
                f"[INFO] Finalizing: Stopping active FFmpeg process for segment {segment_num}..."
            )
            if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
                try:
                    self.recording_sp.stdin.close()
                except OSError as e:
                    print(
                        f"[WARN] Error closing FFmpeg stdin during early finalization: {e}"
                    )
            try:
                self.recording_sp.wait(timeout=10)
                print(
                    f"[INFO] FFmpeg subprocess (segment {segment_num}) finished writing."
                )
            except subprocess.TimeoutExpired:
                print(
                    f"[WARN] FFmpeg subprocess (segment {segment_num}) timed out, killing."
                )
                self.recording_sp.kill()
                self.recording_sp.wait()
            except Exception as e:
                print(f"[ERROR] Error waiting for FFmpeg subprocess: {e}")
            self.recording_sp = None

        was_triggered_by_job = self.triggered_by_job_manager

        # 1. Reset state flags
        self.processing = False
        self.is_processing_segments = False
        self.recording = False

        # 2. Find all valid (non-empty) segment files
        valid_segment_files = [
            f
            for f in self.temp_segment_files
            if f and os.path.exists(f) and os.path.getsize(f) > 0
        ]

        if not valid_segment_files:
            print("[WARN] No valid temporary segment files found to concatenate.")
            self._cleanup_temp_dir()
            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)
            self.segments_to_process = []
            self.current_segment_index = -1
            self.temp_segment_files = []
            self.triggered_by_job_manager = False
            return

        # 3. Determine final output path
        job_name = (
            getattr(self.main_window, "current_job_name", None)
            if was_triggered_by_job
            else None
        )
        use_job_name = (
            getattr(self.main_window, "use_job_name_for_output", False)
            if was_triggered_by_job
            else False
        )
        output_file_name = (
            getattr(self.main_window, "output_file_name", None)
            if was_triggered_by_job
            else None
        )

        job_name, output_file_name = self._apply_job_timestamp_to_output_name(
            was_triggered_by_job,
            job_name,
            use_job_name,
            output_file_name,
        )

        final_file_path = misc_helpers.get_output_file_path(
            self.media_path,
            self.main_window.control["OutputMediaFolder"],
            job_name=job_name,
            use_job_name_for_output=use_job_name,
            output_file_name=output_file_name,
        )

        # Add suffix if stopped due to error limit
        if self.stopped_by_error_limit:
            path_obj = Path(final_file_path)
            final_file_path = str(
                path_obj.parent / f"{path_obj.stem}_incomplete{path_obj.suffix}"
            )
            print(
                f"[WARN] Output marked as incomplete due to excessive read errors: {final_file_path}"
            )

        output_dir = os.path.dirname(final_file_path)

        # Check if output_dir is not an empty string before creating it
        if output_dir and not os.path.exists(output_dir):
            try:
                # Added exist_ok=True for thread-safety
                os.makedirs(output_dir, exist_ok=True)
                print(f"[INFO] Created output directory: {output_dir}")
            except OSError as e:
                print(f"[ERROR] Failed to create output directory {output_dir}: {e}")
                self.main_window.display_messagebox_signal.emit(
                    "File Error",
                    f"Could not create output directory:\n{output_dir}\n\n{e}",
                    self.main_window,
                )
                self._cleanup_temp_dir()
                layout_actions.enable_all_parameters_and_control_widget(
                    self.main_window
                )
                video_control_actions.reset_media_buttons(self.main_window)
                return

        if Path(final_file_path).is_file():
            print(f"[INFO] Removing existing final file: {final_file_path}")
            try:
                os.remove(final_file_path)
            except OSError as e:
                print(f"[ERROR] Failed to remove existing file {final_file_path}: {e}")
                self.main_window.display_messagebox_signal.emit(
                    "File Error",
                    f"Could not delete existing file:\n{final_file_path}\n\n{e}",
                    self.main_window,
                )
                self._cleanup_temp_dir()
                layout_actions.enable_all_parameters_and_control_widget(
                    self.main_window
                )
                video_control_actions.reset_media_buttons(self.main_window)
                return

        # 4. Create FFmpeg list file
        list_file_path = os.path.join(self.segment_temp_dir, "mylist.txt")
        concatenation_successful = False
        concat_args = []  # VP-33: initialise before try so except blocks can reference it safely
        try:
            print(f"[INFO] Creating ffmpeg list file: {list_file_path}")
            with open(list_file_path, "w", encoding="utf-8") as f_list:
                for segment_path in valid_segment_files:
                    abs_path = os.path.abspath(segment_path)
                    # FFmpeg concat requires forward slashes, even on Windows
                    formatted_path = abs_path.replace("\\", "/")
                    f_list.write(f"file '{formatted_path}'" + os.linesep)

            # 5. Run final concatenation command
            print(
                f"[INFO] Concatenating {len(valid_segment_files)} valid segments into {final_file_path}..."
            )
            concat_args = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file_path,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                # REMOVED: "-af", "aresample=async=1000" (Breaks CFR sync and incompatible with -c:a copy)
                final_file_path,
            ]
            subprocess.run(concat_args, check=True)
            concatenation_successful = True
            log_prefix = "Job Manager: " if was_triggered_by_job else ""
            print(
                f"[INFO] --- {log_prefix}Successfully created final video: {final_file_path} ---"
            )

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg command failed during final concatenation: {e}")
            print(f"FFmpeg arguments: {' '.join(concat_args)}")
            if self._attempt_segment_video_only_fallback(
                list_file_path,
                final_file_path,
                f"FFmpeg command failed during concatenation:\n{e}\nCould not create final video.",
            ):
                concatenation_successful = True
        except FileNotFoundError:
            print("[ERROR] FFmpeg not found. Ensure it's in your system PATH.")
            self.main_window.display_messagebox_signal.emit(
                "Recording Error", "FFmpeg not found.", self.main_window
            )
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during finalization: {e}")
            if self._attempt_segment_video_only_fallback(
                list_file_path,
                final_file_path,
                f"An unexpected error occurred:\n{e}",
            ):
                concatenation_successful = True

        finally:
            # 6. Cleanup
            self._cleanup_temp_dir()

            # 7. Reset state
            self.segments_to_process = []
            self.current_segment_index = -1
            self.temp_segment_files = []
            self.current_segment_end_frame = None
            self.triggered_by_job_manager = False
            print("[INFO] Clearing frame queue of residual pills...")
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()

            # 8. Final timing
            self.end_time = time.perf_counter()
            processing_time_sec = self.end_time - self.start_time
            formatted_duration = self._format_duration(
                processing_time_sec
            )  # Use the new helper

            if concatenation_successful:
                print(
                    f"[INFO] Total segment processing and concatenation finished in {formatted_duration}"
                )
            else:
                print(
                    f"[WARN] Segment processing/concatenation failed after {formatted_duration}."
                )

            # 9. Final cleanup and UI reset
            print(
                "[INFO] Clearing GPU Cache and running garbage collection post-concatenation."
            )
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception as e:
                print(f"[WARN] Error clearing Torch cache: {e}")
            gc.collect()

            # Reset media capture
            if self.file_type == "video" and self.media_path:
                current_slider_pos = self.main_window.videoSeekSlider.value()
                if self._reopen_video_capture(current_slider_pos):
                    print("[INFO] Video capture re-opened and seeked.")
                else:
                    print("[WARN] Failed to re-open media capture after segments.")
            elif self.file_type == "video":
                print("[WARN] media_path not set, cannot re-open video capture.")

            layout_actions.enable_all_parameters_and_control_widget(self.main_window)
            video_control_actions.reset_media_buttons(self.main_window)
            print("[INFO] Multi-segment processing flow finished.")

            if self.main_window.control["OpenOutputToggle"]:
                try:
                    list_view_actions.open_output_media_folder(self.main_window)
                except Exception:
                    pass

            # Emit signal to notify JobProcessor that processing has finished SUCCESSFULLY
            print("[INFO] Emitting processing_stopped_signal (multi-segment success).")
            self.processing_stopped_signal.emit()

    def _cleanup_temp_dir(self):
        """Safely removes the temporary directory used for segments."""
        if self.segment_temp_dir and os.path.exists(self.segment_temp_dir):
            try:
                print(
                    f"[INFO] Cleaning up temporary segment directory: {self.segment_temp_dir}"
                )
                shutil.rmtree(self.segment_temp_dir, ignore_errors=True)
            except Exception as e:
                print(
                    f"[WARN] Failed to delete temporary directory {self.segment_temp_dir}: {e}"
                )
        self.segment_temp_dir = None

    # --- Audio Methods ---

    def start_live_sound(self):
        """Starts ffplay subprocess to play audio synced to the current frame."""
        # VP-13: Guard against a None media_capture (e.g. called after stop_processing).
        if not self.media_capture:
            print("[WARN] start_live_sound: media_capture is None, cannot start audio.")
            return

        if self.ffplay_sound_sp:
            self.stop_live_sound()

        anchor_fn = self._timeline_frame_from_ui()
        cap_fps = misc_helpers.capture_get_prop(self.media_capture, cv2.CAP_PROP_FPS)
        try:
            cap_fps_f = float(cap_fps)
        except (TypeError, ValueError):
            cap_fps_f = 0.0
        if cap_fps_f <= 0:
            cap_fps_f = float(self.fps) if self.fps > 0 else 30.0
        seek_time = anchor_fn / cap_fps_f

        # Adjust audio speed if custom FPS is used
        fpsdiv = 1.0
        if (
            self.main_window.control["VideoPlaybackCustomFpsToggle"]
            and not self.recording
        ):
            fpsorig = misc_helpers.capture_get_prop(
                self.media_capture, cv2.CAP_PROP_FPS
            )
            fpscust = self.main_window.control["VideoPlaybackCustomFpsSlider"]
            if fpsorig > 0 and fpscust > 0:
                fpsdiv = fpscust / fpsorig
        if fpsdiv < 0.5:
            fpsdiv = 0.5  # Don't allow less than 0.5x speed

        fps_file = float(
            misc_helpers.capture_get_prop(self.media_capture, cv2.CAP_PROP_FPS) or 0.0
        )
        if fps_file <= 0:
            fps_file = float(self.fps) if self.fps > 0 else 30.0
        self._audio_sync_wall_t0 = time.perf_counter()
        self._audio_sync_anchor_fn = anchor_fn
        self._audio_sync_fps_file = fps_file
        self._audio_sync_rate = float(fpsdiv)

        _did_audio_realign = False
        with self.state_lock:
            self.next_frame_to_display = anchor_fn
            cur = int(self.current_frame_number)
            if cur != anchor_fn:
                self.current_frame_number = anchor_fn
                try:
                    misc_helpers.seek_frame(self.media_capture, anchor_fn)
                except Exception as e:
                    print(
                        f"[WARN] audio-sync: could not realign capture to frame {anchor_fn}: {e}"
                    )
                else:
                    _did_audio_realign = True
        if _did_audio_realign:
            self._clear_sequential_detection_feed_state()
        self._audio_sync_last_seek_monotonic = 0.0

        args = [
            "ffplay",
            "-vn",  # No video
            "-nodisp",
            "-stats",
            "-loglevel",
            "quiet",
            "-sync",
            "audio",
            "-af",
            f"volume={self.main_window.control['LiveSoundVolumeDecimalSlider']}, atempo={fpsdiv}",
            "-i",  # Specify the input...
            self.media_path,
            "-ss",  # ... THEN specify the seek time for a precise seek
            str(seek_time),
        ]

        self.ffplay_sound_sp = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

    def _start_synchronized_playback(self):
        """
        Starts the playback components (audio and video) in a synchronized manner.
        Called once the preroll buffer is filled.
        """
        # 1. Start audio (ffplay) *first*
        if self.main_window.liveSoundButton.isChecked() and not self.recording:
            print("[INFO] Starting audio subprocess (ffplay)...")
            self.start_live_sound()

            # 2. Start video (metronome) AFTER a delay
            # This is to allow ffplay time to initialize.
            AUDIO_STARTUP_LATENCY_MS = (
                self.main_window.control.get("LiveSoundDelayDecimalSlider") * 1000
            )
            print(
                f"[INFO] Waiting {AUDIO_STARTUP_LATENCY_MS}ms for audio to initialize..."
            )

            # Use the function with the clarified name
            QTimer.singleShot(
                int(AUDIO_STARTUP_LATENCY_MS),
                self._start_video_metronome_after_audio_delay,
            )

        else:
            # No audio, start video immediately
            print("[INFO] No audio. Starting video metronome immediately.")
            self._start_metronome(self.fps, is_first_start=True)

    def _start_video_metronome_after_audio_delay(self):
        """
        Slot for QTimer.singleShot.
        Starts the video metronome *after* the audio initialization delay has passed.
        """
        if not self.processing:  # Check in case the user stopped processing
            return
        print("[INFO] Audio startup delay complete. Starting video metronome.")
        if (
            self.main_window.control.get("VideoPlaybackAudioSyncPreviewToggle", False)
            and self.main_window.liveSoundButton.isChecked()
            and self._audio_sync_wall_t0 > 0.0
        ):
            anchor_fn = self._timeline_frame_from_ui()
            self._audio_sync_anchor_fn = anchor_fn
            self._audio_sync_last_seek_monotonic = 0.0
            with self.state_lock:
                self.next_frame_to_display = anchor_fn
        self._start_metronome(self.fps, is_first_start=True)

    def stop_live_sound(self):
        """Stops the ffplay audio subprocess."""
        if self.ffplay_sound_sp:
            parent_pid = self.ffplay_sound_sp.pid
            try:
                # Kill parent and any child processes
                try:
                    parent_proc = psutil.Process(parent_pid)
                    children = parent_proc.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass

                self.ffplay_sound_sp.terminate()
                try:
                    self.ffplay_sound_sp.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.ffplay_sound_sp.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                print(f"[WARN] Error stopping live sound: {e}")

            self.ffplay_sound_sp = None

    # --- Webcam Methods ---

    def process_webcam(self):
        """Starts the webcam stream using the unified metronome and User Settings."""
        if self.processing:
            print("[WARN] Processing already active, cannot start webcam.")
            return
        if self.file_type != "webcam":
            print("[WARN] Process_webcam: Only applicable for webcam input.")
            return

        # 1. Retrieve User Settings from the UI Control Dictionary
        try:
            # Device Index
            webcam_index = int(self.main_window.control.get("WebcamDeviceSelection", 0))

            # Resolution (String like "1920x1080")
            res_str = self.main_window.control.get("WebcamMaxResSelection", "1280x720")
            target_width, target_height = map(int, res_str.split("x"))

            # Backend (String like "DirectShow") -> Mapped to cv2 Constant
            backend_name = self.main_window.control.get(
                "WebcamBackendSelection", "Default"
            )
            backend_id = CAMERA_BACKENDS.get(backend_name, cv2.CAP_ANY)

            # FPS (String like "30")
            target_fps = int(self.main_window.control.get("WebCamMaxFPSSelection", 30))

        except Exception as e:
            print(
                f"[ERROR] Error parsing webcam settings: {e}. Falling back to defaults."
            )
            webcam_index = 0
            target_width, target_height = 1280, 720
            backend_id = cv2.CAP_ANY
            target_fps = 30

        print(
            f"[INFO] Init Webcam: Device={webcam_index}, Backend={backend_name}, Target={target_width}x{target_height} @ {target_fps}fps"
        )

        # 2. Initialize VideoCapture with the selected Backend
        try:
            self.media_capture = cv2.VideoCapture(webcam_index, backend_id)
        except Exception as e:
            print(f"[ERROR] Failed to init webcam with backend {backend_name}: {e}")
            self.media_capture = cv2.VideoCapture(webcam_index)

        if not (self.media_capture and self.media_capture.isOpened()):
            print("[ERROR] Unable to open webcam source.")
            video_control_actions.reset_media_buttons(self.main_window)
            return

        # 3. Apply Configuration
        try:
            # Force MJPG to allow high framerate at high res (saves USB bandwidth)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.media_capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        self.media_capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.media_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self.media_capture.set(cv2.CAP_PROP_FPS, target_fps)

        # 4. Verify actual resolution obtained
        actual_w = self.media_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(
            f"[INFO] Webcam initialized at: {int(actual_w)}x{int(actual_h)} (Requested: {target_width}x{target_height})"
        )

        # Warn if the camera refused the resolution
        if int(actual_w) != target_width or int(actual_h) != target_height:
            print(
                f"[WARN] Camera did not accept requested resolution. Using {int(actual_w)}x{int(actual_h)}."
            )
            if int(actual_w) == 640 and backend_name != "DirectShow":
                print(
                    "[TIP] Try changing 'Webcam Backend' to 'DirectShow' in Settings to unlock HD."
                )

        print("[INFO] Starting webcam processing setup...")

        # 5. Set State Flags
        self.processing = True
        self.is_processing_segments = False
        self.recording = False
        self.sync_feeder_ui_face_flags_from_main_window()
        self.clear_recognition_embedding_cache()

        # 6. Clear Containers
        self.frames_to_display.clear()
        self.frames_pipeline_profile.clear()
        self.webcam_frames_to_display.queue.clear()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        # Feeder detection interval / smoothing state must not carry over from video sessions.
        self.current_frame_number = 0
        self.last_detected_faces.clear()
        self._smoothed_kps.clear()

        # 7. Start Metronome ET Feeder
        fps = self.media_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.fps = fps

        print(f"[INFO] Webcam target FPS: {self.fps}")

        self.join_and_clear_threads()
        self.worker_threads = []
        for i in range(self.num_threads):
            worker = FrameWorker(
                frame_queue=self.frame_queue,  # Pass the task queue
                main_window=self.main_window,
                worker_id=i,
            )
            worker.start()
            self.worker_threads.append(worker)

        # Start the feeder thread
        print("[INFO] Starting feeder thread (Mode: webcam)...")
        self.feeder_thread = threading.Thread(target=self._feeder_loop, daemon=True)
        self.feeder_thread.start()

        # Start the display metronome
        self._start_metronome(self.fps, is_first_start=True)

    def process_screen(self):
        """Starts desktop screen capture as a live stream (same pipeline as webcam)."""
        if self.processing:
            print("[WARN] Processing already active, cannot start screen capture.")
            return
        if self.file_type != "screen":
            print("[WARN] process_screen: Only applicable for screen input.")
            return
        if not mss_available():
            print("[ERROR] mss is not installed; cannot start screen capture.")
            video_control_actions.reset_media_buttons(self.main_window)
            return
        try:
            self.media_capture = create_screen_capture_from_control(
                self.main_window.control
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize screen capture: {e}")
            video_control_actions.reset_media_buttons(self.main_window)
            self.media_capture = None
            return

        ok, _probe = misc_helpers.read_frame(self.media_capture, 0)
        if not ok:
            print("[ERROR] Screen capture test read failed.")
            misc_helpers.release_capture(self.media_capture)
            self.media_capture = None
            video_control_actions.reset_media_buttons(self.main_window)
            return

        print("[INFO] Starting screen capture processing setup...")

        self.processing = True
        self.is_processing_segments = False
        self.recording = False
        self.sync_feeder_ui_face_flags_from_main_window()
        self.clear_recognition_embedding_cache()

        self.frames_to_display.clear()
        self.frames_pipeline_profile.clear()
        self.webcam_frames_to_display.queue.clear()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        # Same as process_webcam: sequential detection uses current_frame_number; reset so
        # interval/ByteTrack logic matches each live session (fixes screen swap after video).
        self.current_frame_number = 0
        self.last_detected_faces.clear()
        self._smoothed_kps.clear()

        fps = float(self.media_capture.get(cv2.CAP_PROP_FPS) or 0)
        if fps <= 0:
            fps = 30.0
        self.fps = fps
        print(f"[INFO] Screen capture target FPS: {self.fps}")

        self.join_and_clear_threads()
        self.worker_threads = []
        for i in range(self.num_threads):
            worker = FrameWorker(
                frame_queue=self.frame_queue,
                main_window=self.main_window,
                worker_id=i,
            )
            worker.start()
            self.worker_threads.append(worker)

        print("[INFO] Starting feeder thread (Mode: screen)...")
        self.feeder_thread = threading.Thread(target=self._feeder_loop, daemon=True)
        self.feeder_thread.start()

        self._start_metronome(self.fps, is_first_start=True)
