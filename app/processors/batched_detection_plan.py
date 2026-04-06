"""
Design scaffold for future **batched detector forward** (post-MVP).

The live pipeline runs ``_run_sequential_detection`` in strict frame order because
ByteTrack, ``FaceDetectionIntervalSlider``, and EMA state depend on temporal order.
A batched speedup must keep that order for **post-processing** while only batching
the neural backbone (e.g. RetinaFace) over K consecutive frames.

Intended flow (not wired):

1. The detection thread accumulates up to ``K`` decoded RGB frames with identical
   detector resize geometry.
2. One ORT/TRT run with input shape ``[K, 3, H, W]`` (dynamic batch profile).
3. Split raw scores/boxes per batch index; for ``i = 0..K-1`` in order, run
   existing NMS / ByteTrack / landmark scheduling exactly as today for frame
   ``logical_fn + i``.

Do **not** reorder tasks enqueued to ``frame_queue``; only parallelize work
inside the detector session. VR180 skips sequential detection — any batch path
must remain disabled or separately validated for VR.

See also: ``VideoProcessor._detection_pipeline_loop`` and
``FaceDetectors.run_detect`` in ``face_detectors.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BatchedDetectorPlanSpec:
    """Upper bound K for detector micro-batch (future UI or env)."""

    max_batch: int = 2


def planned_stack_detector_batch(
    frames_rgb: Sequence[Any],
    spec: BatchedDetectorPlanSpec,
) -> tuple[int, list[Any]]:
    """
    Placeholder for stacking up to ``spec.max_batch`` frames for one detector run.

    Real implementation must enforce identical ``H, W`` after detector letterbox
    and respect ``max_batch`` and VRAM limits.
    """
    n = min(len(frames_rgb), max(1, spec.max_batch))
    return n, list(frames_rgb[:n])


def planned_sequential_postprocess_contract() -> str:
    """Human-readable invariant for reviewers and future implementers."""
    return (
        "Run ByteTrack.update, interval skips, and landmark passes in ascending "
        "frame index; batched forward is an implementation detail of the detector only."
    )
