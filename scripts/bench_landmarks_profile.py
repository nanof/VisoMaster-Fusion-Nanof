"""Profile landmark cost through the real app path (ModelsProcessor).

Answers the matrix:
  MuseTalk on/off × landmark 68 vs 203(+LP conflict)

MuseTalk itself does **not** re-run landmarks (``FrameWorker._musetalk_landmarks``
reuses feeder/worker densos). Enabling MuseTalk only **forces** mode ``68``.
The interesting costs are therefore:
  - detect-only (kps_5)
  - + FaceLandmark68   ← MuseTalk-required path
  - + FaceLandmark203  ← LivePortrait / editor / expression path
  - 203 then 68        ← conflict when LP needs 203 and MuseTalk forced 68

Usage (repo root, GPU):

    portable-files\\python\\python.exe -u scripts/bench_landmarks_profile.py

Env:
    WARMUP / ITERS   (default 8 / 40)
    VIDEO_PATH       optional video; first frame with a face used if detection hits
    PROVIDER         CUDA (default) | TensorRT | CPU
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WARMUP = int(os.environ.get("WARMUP", 8))
ITERS = int(os.environ.get("ITERS", 40))
PROVIDER = os.environ.get("PROVIDER", "CUDA")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "").strip()


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    return statistics.median(samples) * 1000.0, statistics.mean(samples) * 1000.0


def _bench(fn, *, warmup: int = WARMUP, iters: int = ITERS) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    _sync()
    samples: list[float] = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return _ms(samples)


def _make_stub_main_window():
    from PySide6.QtCore import QObject, Signal

    class _Signals(QObject):
        model_loading_signal = Signal()
        model_loaded_signal = Signal()

    sig = _Signals()
    control = {
        "ModelWarmupOnLoadToggle": False,
        "KeepModelsAliveToggle": True,
        "ModelUnloadGraceSecondsSlider": 0,
        "ModelEvictIdleMinutesSlider": 0,
        "MaxDFMModelsSlider": 1,
        "SwapModelSelection": "",
        "DetectorModelSelection": "RetinaFace",
        "LandmarkDetectToggle": True,
        "LandmarkDetectModelSelection": "68",
        "LandmarkDetectScoreSlider": 50,
        "LandmarkMeanEyesToggle": False,
        "DetectFromPointsToggle": True,
        "MaxFacesToDetectSlider": 1,
        "DetectorScoreSlider": 50,
        "AutoRotationToggle": False,
    }
    return types.SimpleNamespace(
        control=control,
        model_loading_signal=sig.model_loading_signal,
        model_loaded_signal=sig.model_loaded_signal,
    )


def _load_frame_rgb() -> np.ndarray:
    if VIDEO_PATH:
        import cv2

        cap = cv2.VideoCapture(VIDEO_PATH)
        ok, bgr = cap.read()
        cap.release()
        if not ok or bgr is None:
            raise RuntimeError(f"Could not read frame from VIDEO_PATH={VIDEO_PATH}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Synthetic 720p frame — timing of crop+ORT does not need a real face.
    rng = np.random.default_rng(0)
    img = rng.integers(40, 200, size=(720, 1280, 3), dtype=np.uint8)
    # Soft face-ish blob so warp crops are non-empty / non-constant.
    yy, xx = np.ogrid[180:420, 480:720]
    img[180:420, 480:720] = (
        0.65 * img[180:420, 480:720] + 0.35 * np.full_like(img[180:420, 480:720], 180)
    ).astype(np.uint8)
    del yy, xx
    return img


def main() -> int:
    from PySide6.QtWidgets import QApplication
    import torch

    if not QApplication.instance():
        _app = QApplication(sys.argv)  # noqa: F841

    if not torch.cuda.is_available() and PROVIDER != "CPU":
        print("SKIP: CUDA not available (set PROVIDER=CPU to force CPU ORT)")
        return 2

    device = "cuda" if torch.cuda.is_available() and PROVIDER != "CPU" else "cpu"
    print(f"provider={PROVIDER} device={device} WARMUP={WARMUP} ITERS={ITERS}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    from app.processors.models_processor import ModelsProcessor

    mw = _make_stub_main_window()
    mp = ModelsProcessor(mw, device=device)
    mp.provider_name = PROVIDER if PROVIDER in ("CUDA", "TensorRT", "CPU") else "CUDA"
    mp.providers = mp._build_providers_for_name(mp.provider_name)

    frame_rgb = _load_frame_rgb()
    frame_t = (
        torch.from_numpy(frame_rgb)
        .to(device=device, dtype=torch.uint8, non_blocking=True)
        .permute(2, 0, 1)
        .contiguous()
    )
    h, w = frame_rgb.shape[:2]
    print(f"frame={w}x{h} source={'video' if VIDEO_PATH else 'synthetic'}")

    # --- detect once to get a real bbox/kps_5 when possible ---
    bboxes, kpss_5, _ = mp.run_detect(
        frame_t,
        "RetinaFace",
        max_num=1,
        score=0.3,
        input_size=(512, 512),
        use_landmark_detection=False,
        from_points=False,
        rotation_angles=[0],
    )
    if isinstance(bboxes, np.ndarray) and bboxes.shape[0] > 0:
        bbox = np.asarray(bboxes[0], dtype=np.float32)
        kps5 = np.asarray(kpss_5[0], dtype=np.float32)
        print(f"detect hit: bbox={bbox.tolist()}")
    else:
        # Fallback synthetic geometry (still exercises warp+ORT).
        cx, cy = w * 0.5, h * 0.45
        s = min(w, h) * 0.18
        bbox = np.array([cx - s, cy - s, cx + s, cy + s], dtype=np.float32)
        kps5 = np.array(
            [
                [cx - s * 0.35, cy - s * 0.15],
                [cx + s * 0.35, cy - s * 0.15],
                [cx, cy + s * 0.05],
                [cx - s * 0.25, cy + s * 0.45],
                [cx + s * 0.25, cy + s * 0.45],
            ],
            dtype=np.float32,
        )
        print("detect miss: using synthetic bbox/kps_5 (ORT timing still valid)")

    # Warm-load landmark models used below.
    for mode in ("68", "203"):
        mp.run_detect_landmark(
            frame_t, bbox, kps5, detect_mode=mode, score=0.5, from_points=True
        )

    rows: list[tuple[str, str, float, float]] = []

    def add(scenario: str, note: str, fn):
        med, mean = _bench(fn)
        rows.append((scenario, note, med, mean))
        print(f"{scenario:28s}  median={med:7.2f} ms  mean={mean:7.2f} ms  ({note})")

    add(
        "A_detect_only",
        "MuseTalk off / landmarks off",
        lambda: mp.run_detect(
            frame_t,
            "RetinaFace",
            max_num=1,
            score=0.3,
            input_size=(512, 512),
            use_landmark_detection=False,
            from_points=False,
            rotation_angles=[0],
        ),
    )

    add(
        "B_landmark_68",
        "MuseTalk on (forced 68) or UI=68",
        lambda: mp.run_detect_landmark(
            frame_t, bbox, kps5, detect_mode="68", score=0.5, from_points=True
        ),
    )

    add(
        "C_landmark_203",
        "LP / editor / expression (203)",
        lambda: mp.run_detect_landmark(
            frame_t, bbox, kps5, detect_mode="203", score=0.5, from_points=True
        ),
    )

    def both():
        mp.run_detect_landmark(
            frame_t, bbox, kps5, detect_mode="203", score=0.5, from_points=True
        )
        mp.run_detect_landmark(
            frame_t, bbox, kps5, detect_mode="68", score=0.5, from_points=True
        )

    add(
        "D_203_then_68",
        "conflict: requires_203 + MuseTalk forces 68",
        both,
    )

    add(
        "E_detect_plus_68",
        "feeder-like: detect + MuseTalk landmarks",
        lambda: (
            mp.run_detect(
                frame_t,
                "RetinaFace",
                max_num=1,
                score=0.3,
                input_size=(512, 512),
                use_landmark_detection=False,
                from_points=False,
                rotation_angles=[0],
            ),
            mp.run_detect_landmark(
                frame_t, bbox, kps5, detect_mode="68", score=0.5, from_points=True
            ),
        ),
    )

    add(
        "F_detect_plus_203",
        "feeder-like: detect + LP landmarks",
        lambda: (
            mp.run_detect(
                frame_t,
                "RetinaFace",
                max_num=1,
                score=0.3,
                input_size=(512, 512),
                use_landmark_detection=False,
                from_points=False,
                rotation_angles=[0],
            ),
            mp.run_detect_landmark(
                frame_t, bbox, kps5, detect_mode="203", score=0.5, from_points=True
            ),
        ),
    )

    print()
    print("=== matrix (median ms / face / frame) ===")
    print(f"{'scenario':28s} {'median_ms':>10} {'mean_ms':>10}  note")
    for name, note, med, mean in rows:
        print(f"{name:28s} {med:10.2f} {mean:10.2f}  {note}")

    by = {n: m for n, _, m, _ in rows}
    print()
    print("=== derived ===")
    if "B_landmark_68" in by and "C_landmark_203" in by:
        print(
            f"203 vs 68: {by['C_landmark_203']/by['B_landmark_68']:.2f}× "
            f"({by['C_landmark_203']:.2f} / {by['B_landmark_68']:.2f} ms)"
        )
    if "D_203_then_68" in by and "C_landmark_203" in by:
        extra = by["D_203_then_68"] - by["C_landmark_203"]
        print(
            f"extra 68 after 203 (conflict tax): {extra:.2f} ms "
            f"(D={by['D_203_then_68']:.2f} − C={by['C_landmark_203']:.2f})"
        )
    print(
        "MuseTalk on vs off (landmarks only): +0 ms beyond forcing mode 68 — "
        "engine reuses precomputed densos; MuseTalk GPU cost is musetalk_preswap, not landmarks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
