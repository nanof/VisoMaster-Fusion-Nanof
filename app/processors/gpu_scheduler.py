"""Multi-GPU load scheduler and live metrics.

This module contains the pieces that drive proportional frame routing across
multiple logical GPUs (plus the optional CPU-ORT slot):

- ``WeightedScheduler``: deterministic Deficit Round Robin (DRR) over integer
  weights, producing an interleaved sequence (e.g. weights ``{0: 3, 1: 1}``
  emit ``0, 0, 0, 1, 0, 0, 0, 1, ...``). Thread-safe; ``update_weights`` and
  ``set_targets`` are atomic.
- ``GpuLoadMetrics``: small thread-safe collector of per-GPU frame timings and
  a rolling FPS. Used by ``FrameWorker`` to feed the UI and the optional auto
  re-weighting logic.
- ``calibrate_weights_from_timings``: stateless helper that turns a map of
  ``{gpu_id: ms_per_frame}`` into integer DRR weights roughly proportional to
  ``1 / ms`` (faster GPU gets a higher weight).

Design goals:

- No PySide/Qt imports here: this module is imported by core (video_processor
  and frame_worker) and by tests that should run headless.
- Deterministic output for given inputs (DRR, not randomized) so tests stay
  reproducible and logs stay easy to read.
- Cheap: all operations are O(len(weights)) or O(1); scheduler decisions
  happen per frame in the detection thread.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


_MIN_WEIGHT = 1
_MAX_WEIGHT = 1024


def _sanitize_weights(
    weights: Mapping[int, int],
    targets: Iterable[int],
) -> Dict[int, int]:
    """Clamp weights to ``[1, 1024]`` and fill missing targets with weight=1.

    Any target that is not present in ``weights`` receives a default of 1 so
    that every routing target gets at least one slot per DRR cycle. Unknown
    ids in ``weights`` (not in ``targets``) are dropped so the scheduler can
    never emit a gpu id the caller did not allow.
    """
    out: Dict[int, int] = {}
    tset = {int(x) for x in targets}
    for gid in tset:
        raw = int(weights.get(gid, 1)) if weights else 1
        out[gid] = max(_MIN_WEIGHT, min(_MAX_WEIGHT, raw))
    return out


class WeightedScheduler:
    """Deterministic Weighted Round-Robin (DRR) over integer weights.

    The scheduler keeps one integer "deficit" per target. On each call to
    ``next_gpu()``:

    1. All deficits are incremented by their weight.
    2. The target with the highest deficit is picked (ties broken by the
       smallest id to keep output stable).
    3. Its deficit is decremented by the sum of all weights (so its average
       share equals ``weight / sum(weights)``).

    This produces an interleaved stream (no long runs of the same GPU) and is
    a good match for a frame pipeline where we want smooth filling of the
    per-GPU subqueues. For weights ``{0: 3, 1: 1}`` it emits
    ``0, 0, 0, 1, 0, 0, 0, 1, ...`` deterministically.

    ``update_weights`` and ``set_targets`` are thread-safe; the caller (UI
    action or auto-tune) may mutate configuration while ``next_gpu`` is being
    called from the detection thread.
    """

    def __init__(
        self,
        targets: Optional[Iterable[int]] = None,
        weights: Optional[Mapping[int, int]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._targets: List[int] = []
        self._weights: Dict[int, int] = {}
        self._deficits: Dict[int, int] = {}
        self._cycle: int = 0
        self.set_targets(list(targets or [0]), weights or {})

    # -- configuration ---------------------------------------------------
    def set_targets(
        self,
        targets: Iterable[int],
        weights: Optional[Mapping[int, int]] = None,
    ) -> None:
        with self._lock:
            clean_targets = sorted({int(x) for x in targets})
            if not clean_targets:
                clean_targets = [0]
            clean_weights = _sanitize_weights(weights or self._weights, clean_targets)
            self._targets = clean_targets
            self._weights = clean_weights
            self._deficits = {gid: 0 for gid in clean_targets}
            self._cycle = 0

    def update_weights(self, weights: Mapping[int, int]) -> None:
        with self._lock:
            self._weights = _sanitize_weights(weights, self._targets)
            self._deficits = {gid: 0 for gid in self._targets}
            self._cycle = 0

    def get_weights(self) -> Dict[int, int]:
        with self._lock:
            return dict(self._weights)

    def get_targets(self) -> List[int]:
        with self._lock:
            return list(self._targets)

    # -- scheduling ------------------------------------------------------
    def next_gpu(self) -> int:
        """Return the next logical GPU id for a frame."""
        with self._lock:
            if not self._targets:
                return 0
            total = sum(self._weights.get(t, 1) for t in self._targets)
            if total <= 0:
                return self._targets[0]
            for gid in self._targets:
                self._deficits[gid] = self._deficits.get(gid, 0) + int(
                    self._weights.get(gid, 1)
                )
            # Pick the highest deficit, ties broken by smallest id.
            best_id = self._targets[0]
            best_def = self._deficits[best_id]
            for gid in self._targets[1:]:
                d = self._deficits[gid]
                if d > best_def or (d == best_def and gid < best_id):
                    best_id = gid
                    best_def = d
            self._deficits[best_id] -= total
            self._cycle += 1
            return int(best_id)

    def peek_sequence(self, length: int) -> List[int]:
        """Return the next ``length`` picks without mutating external state."""
        snapshot = _SchedulerSnapshot.from_scheduler(self)
        return snapshot.run(length)


class _SchedulerSnapshot:
    """Stateful copy used for peek/tests without touching the live scheduler."""

    def __init__(self, targets: List[int], weights: Dict[int, int]) -> None:
        self.targets = list(targets)
        self.weights = dict(weights)
        self.deficits = {g: 0 for g in self.targets}

    @classmethod
    def from_scheduler(cls, sched: WeightedScheduler) -> "_SchedulerSnapshot":
        return cls(sched.get_targets(), sched.get_weights())

    def run(self, length: int) -> List[int]:
        if not self.targets or length <= 0:
            return []
        total = sum(self.weights.get(t, 1) for t in self.targets)
        if total <= 0:
            return [self.targets[0]] * length
        out: List[int] = []
        for _ in range(length):
            for gid in self.targets:
                self.deficits[gid] += int(self.weights.get(gid, 1))
            best_id = self.targets[0]
            best_def = self.deficits[best_id]
            for gid in self.targets[1:]:
                d = self.deficits[gid]
                if d > best_def or (d == best_def and gid < best_id):
                    best_id = gid
                    best_def = d
            self.deficits[best_id] -= total
            out.append(best_id)
        return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class GpuLoadMetrics:
    """Thread-safe rolling FPS/latency collector keyed by logical GPU id.

    ``record(gpu_id, elapsed_ms)`` is called by ``FrameWorker`` after each
    frame. ``snapshot()`` returns a stable dict consumable by the UI and the
    optional auto re-weighter.

    The rolling windows are bounded by ``max_samples`` per GPU (default 64)
    to keep memory/CPU constant. FPS is computed as ``count / wall_seconds``
    over the window — so it also reflects idle time on a slow GPU, which is
    what we want for "real perceived throughput".
    """

    def __init__(self, max_samples: int = 64) -> None:
        self._lock = threading.Lock()
        self._max_samples = max(8, int(max_samples))
        self._samples: Dict[int, deque] = {}
        self._totals: Dict[int, int] = {}

    def reset(self, gpu_ids: Optional[Iterable[int]] = None) -> None:
        with self._lock:
            if gpu_ids is None:
                self._samples.clear()
                self._totals.clear()
                return
            for g in gpu_ids:
                g = int(g)
                self._samples.pop(g, None)
                self._totals.pop(g, None)

    def record(self, gpu_id: int, elapsed_ms: float) -> None:
        ms = float(elapsed_ms)
        if ms <= 0.0 or ms > 1_000_000.0:
            return
        now = time.perf_counter()
        with self._lock:
            buf = self._samples.get(gpu_id)
            if buf is None:
                buf = deque(maxlen=self._max_samples)
                self._samples[gpu_id] = buf
            buf.append((now, ms))
            self._totals[gpu_id] = int(self._totals.get(gpu_id, 0)) + 1

    def snapshot(self) -> Dict[int, Dict[str, float]]:
        """Return ``{gpu_id: {'fps': ..., 'avg_ms': ..., 'ema_ms': ..., 'count': ...}}``."""
        now = time.perf_counter()
        with self._lock:
            out: Dict[int, Dict[str, float]] = {}
            for gid, buf in self._samples.items():
                n = len(buf)
                if n == 0:
                    continue
                ms_sum = 0.0
                ema = 0.0
                alpha = 0.25
                first_t = buf[0][0]
                for i, (_, ms) in enumerate(buf):
                    ms_sum += ms
                    ema = ms if i == 0 else (alpha * ms + (1.0 - alpha) * ema)
                avg_ms = ms_sum / n
                wall = max(1e-6, now - first_t)
                fps = n / wall
                out[int(gid)] = {
                    "fps": float(fps),
                    "avg_ms": float(avg_ms),
                    "ema_ms": float(ema),
                    "count": float(self._totals.get(gid, n)),
                    "window": float(n),
                }
            return out


# ---------------------------------------------------------------------------
# Auto-weight calibration
# ---------------------------------------------------------------------------


def calibrate_weights_from_timings(
    ms_per_frame: Mapping[int, float],
    *,
    max_weight: int = 8,
    min_weight: int = 1,
) -> Dict[int, int]:
    """Derive DRR weights from measured per-GPU ``ms_per_frame``.

    Rules:

    - Faster GPU (smaller ms) gets a proportionally higher weight.
    - The fastest GPU is normalized to ``max_weight``; others scale down by
      the ratio ``fastest_ms / their_ms``.
    - Results are clamped to ``[min_weight, max_weight]`` and rounded to int.
    - If a GPU reports ``<= 0`` ms (unknown / not measured), it receives
      weight=1 and we do not try to infer from partial data.

    This gives deterministic and predictable output (no divisions in code
    paths that could blow up), and it is tunable via ``max_weight``: raising
    it gives finer granularity for big speed gaps.
    """
    out: Dict[int, int] = {}
    valid = {int(g): float(ms) for g, ms in ms_per_frame.items() if float(ms) > 0.0}
    if not valid:
        return {int(g): 1 for g in ms_per_frame}
    fastest_ms = min(valid.values())
    if fastest_ms <= 0.0:
        return {int(g): 1 for g in ms_per_frame}
    for gid, ms in ms_per_frame.items():
        gid_i = int(gid)
        if float(ms) <= 0.0:
            out[gid_i] = 1
            continue
        ratio = fastest_ms / float(ms)
        w = int(round(ratio * max_weight))
        out[gid_i] = max(int(min_weight), min(int(max_weight), w))
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


LOAD_BALANCING_MODES: Tuple[str, ...] = (
    "round_robin",
    "weighted_manual",
    "weighted_auto",
    "hybrid",
)


def normalize_mode(mode: Optional[str]) -> str:
    """Validate/normalize a load-balancing mode string, with safe fallback."""
    if not mode:
        return "round_robin"
    m = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    if m in LOAD_BALANCING_MODES:
        return m
    aliases = {
        "rr": "round_robin",
        "manual": "weighted_manual",
        "auto": "weighted_auto",
        "weighted": "weighted_manual",
    }
    return aliases.get(m, "round_robin")


def mode_needs_weights(mode: str) -> bool:
    return normalize_mode(mode) in ("weighted_manual", "weighted_auto", "hybrid")


def mode_enables_stealing(mode: str) -> bool:
    return normalize_mode(mode) == "hybrid"


def mode_enables_autotune(mode: str) -> bool:
    return normalize_mode(mode) in ("weighted_auto", "hybrid")


# ---------------------------------------------------------------------------
# Thread distribution helpers
# ---------------------------------------------------------------------------


def distribute_threads_by_weights(
    total_threads: int,
    weights: Mapping[int, int],
    targets: Iterable[int],
    *,
    min_per_gpu: int = 1,
) -> Dict[int, int]:
    """Split ``total_threads`` across ``targets`` proportional to ``weights``.

    Every target receives at least ``min_per_gpu`` thread so every logical
    device has a dedicated worker (otherwise a GPU that got rounded to 0
    would only process stolen frames, which misses the whole point of the
    affinity).
    """
    clean_targets = sorted({int(t) for t in targets})
    if not clean_targets:
        return {}
    n_targets = len(clean_targets)
    total_threads = max(n_targets * max(1, int(min_per_gpu)), int(total_threads))
    clean_weights = _sanitize_weights(weights or {}, clean_targets)
    wsum = sum(clean_weights[g] for g in clean_targets)
    if wsum <= 0:
        base = total_threads // n_targets
        rem = total_threads - base * n_targets
        out = {g: base for g in clean_targets}
        for i in range(rem):
            out[clean_targets[i % n_targets]] += 1
        return out
    raw = {g: (total_threads * clean_weights[g]) / wsum for g in clean_targets}
    # Largest-remainder method to avoid losing/gaining threads to rounding.
    base = {g: max(int(min_per_gpu), int(raw[g])) for g in clean_targets}
    assigned = sum(base.values())
    remainder = total_threads - assigned
    if remainder > 0:
        frac = sorted(
            clean_targets,
            key=lambda g: (raw[g] - int(raw[g])),
            reverse=True,
        )
        i = 0
        while remainder > 0:
            base[frac[i % n_targets]] += 1
            remainder -= 1
            i += 1
    elif remainder < 0:
        # Too many due to min_per_gpu floors — shave from the GPUs with the
        # smallest fractional share first, but never below ``min_per_gpu``.
        shave = sorted(
            clean_targets,
            key=lambda g: (base[g], (raw[g] - int(raw[g]))),
        )
        i = 0
        guard = 0
        while remainder < 0 and guard < 10_000:
            g = shave[i % n_targets]
            if base[g] > int(min_per_gpu):
                base[g] -= 1
                remainder += 1
            i += 1
            guard += 1
    return base


__all__ = [
    "WeightedScheduler",
    "GpuLoadMetrics",
    "calibrate_weights_from_timings",
    "distribute_threads_by_weights",
    "normalize_mode",
    "mode_needs_weights",
    "mode_enables_stealing",
    "mode_enables_autotune",
    "LOAD_BALANCING_MODES",
]
