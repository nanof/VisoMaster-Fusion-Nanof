"""
Desktop screen capture as a frame source (BGR numpy), using mss.

Monitor indices follow mss: 0 = all monitors combined; 1 = first physical monitor, etc.
The target-media card uses :data:`SCREEN_CAPTURE_MEDIA_LABEL` (not a filesystem path).
"""

from __future__ import annotations

SCREEN_CAPTURE_MEDIA_LABEL = "Screen capture"

import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import mss
except ImportError as e:  # pragma: no cover
    mss = None  # type: ignore[assignment]
    _MSS_IMPORT_ERROR = e
else:
    _MSS_IMPORT_ERROR = None


def mss_available() -> bool:
    return mss is not None


def mss_import_error() -> Optional[BaseException]:
    return _MSS_IMPORT_ERROR


def _parse_region_rect(s: str) -> Optional[Dict[str, int]]:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    if len(parts) != 4:
        return None
    try:
        left, top, width, height = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return {"left": left, "top": top, "width": width, "height": height}


def screen_capture_settings_from_control(control: dict) -> Tuple[int, Optional[Dict[str, int]], float]:
    """Returns (mss_monitor_index, region_dict_or_none, target_fps)."""
    try:
        mon = int(control.get("ScreenCaptureMonitorSelection", "1"))
    except (TypeError, ValueError):
        mon = 1
    mon = max(0, min(mon, 16))

    mode = str(control.get("ScreenCaptureRegionModeSelection", "Full monitor"))
    region: Optional[Dict[str, int]] = None
    if mode.strip().lower().startswith("custom"):
        rect_s = str(
            control.get("ScreenCaptureRegionRectText")
            or control.get("ScreenCaptureRegionRect", "")
        ).strip()
        region = _parse_region_rect(rect_s)

    try:
        fps = float(control.get("ScreenCaptureMaxFPSSelection", "30"))
    except (TypeError, ValueError):
        fps = 30.0
    if fps <= 0:
        fps = 30.0
    return mon, region, fps


class ScreenCaptureSource:
    """OpenCV-like capture: isOpened(), read(), get(), release(). Thread-safe reads.

    mss uses per-thread handles on Windows (``srcdc``). A single ``mss.mss()`` instance
    must not be shared between the GUI thread and the feeder thread — each ``read()``
    opens a short-lived ``mss.mss()`` in the calling thread.
    """

    def __init__(self, monitor_index: int, region: Optional[Dict[str, int]], target_fps: float):
        if mss is None:
            raise RuntimeError(
                "mss is not installed; add it to requirements and pip install mss"
            ) from _MSS_IMPORT_ERROR
        self._lock = threading.Lock()
        self._monitor_index = monitor_index
        self._region = region
        self._target_fps = float(target_fps)
        self._opened = True
        self._last_w = 0
        self._last_h = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._opened:
                return False, None
            try:
                # Fresh MSS per call = correct thread-local GDI state on Windows.
                with mss.mss() as sct:
                    if self._region is not None:
                        shot = sct.grab(self._region)
                    else:
                        if self._monitor_index >= len(sct.monitors):
                            mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                        else:
                            mon = sct.monitors[self._monitor_index]
                        shot = sct.grab(mon)
                    img = np.asarray(shot, dtype=np.uint8)
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    self._last_h, self._last_w = frame.shape[0], frame.shape[1]
                    return True, frame
            except Exception as e:
                print(f"[ERROR] Screen capture read failed: {e}")
                return False, None

    def get(self, prop_id: int) -> float:
        with self._lock:
            if prop_id == cv2.CAP_PROP_FPS:
                return float(self._target_fps)
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self._last_w)
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self._last_h)
            return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return False

    def release(self) -> None:
        with self._lock:
            self._opened = False


def create_screen_capture_from_control(control: dict) -> ScreenCaptureSource:
    mon, region, fps = screen_capture_settings_from_control(control)
    return ScreenCaptureSource(mon, region, fps)


def grab_one_frame_bgr(control: dict) -> Tuple[bool, Optional[np.ndarray]]:
    """Single grab for thumbnails / preview init."""
    try:
        src = create_screen_capture_from_control(control)
    except RuntimeError:
        return False, None
    try:
        return src.read()
    finally:
        src.release()
