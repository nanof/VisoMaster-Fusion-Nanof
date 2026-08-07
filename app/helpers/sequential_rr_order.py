"""Spatial ordering for Swap all by index (round-robin) temporal alignment."""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np

IoUFn = Callable[[np.ndarray, np.ndarray], float]


def pick_new_input_index(
    n_in: int,
    *,
    assignment_mode: str,
    spatial_fallback: int,
    used: set[int] | None = None,
    rng: random.Random | None = None,
) -> int:
    """Choose an input index for a newly seen face.

    * ``index``: prefer an unused input, else ``spatial_fallback % n_in``.
    * ``random``: uniform pick among unused inputs; if none left, among all.
    """
    if n_in <= 0:
        return 0
    if assignment_mode == "random":
        picker = rng if rng is not None else random
        if used is not None:
            available = [j for j in range(n_in) if j not in used]
            if available:
                return int(picker.choice(available))
        return int(picker.randrange(n_in))
    if used is not None:
        cand = next((j for j in range(n_in) if j not in used), None)
        if cand is not None:
            return int(cand)
    return int(spatial_fallback) % n_in


def rr_dedupe_assignments(
    assign: list[int],
    n_in: int,
    *,
    assignment_mode: str = "index",
    keep_priority: list[int] | None = None,
    pinned: set[int] | frozenset[int] | None = None,
    rng: random.Random | None = None,
) -> list[int]:
    """Remove repeated inputs while unused inputs remain.

    Temporal matching (memory slots, ByteTrack locks, ghost slots) can hand the
    same input to two faces: two remembered slots may carry the same index, and
    sticky locks are resolved per face without a frame-wide view. Repeats are
    only acceptable once every input is taken (more detections than inputs).

    ``keep_priority`` ranks who keeps a contested input (lower wins); faces
    matched temporally should rank above newly assigned ones so the reshuffle
    moves the new face and the stable one is left untouched. Inputs in
    ``pinned`` always win the contest.
    """
    n = len(assign)
    if n_in <= 0:
        return [0] * n
    out = [int(a) % n_in for a in assign]
    if n <= 1 or n_in == 1:
        return out

    pinned_norm = {int(p) % n_in for p in (pinned or ()) if int(p) >= 0}
    prio = (
        [int(p) for p in keep_priority]
        if keep_priority is not None and len(keep_priority) == n
        else [0] * n
    )
    order = sorted(
        range(n),
        key=lambda i: (0 if out[i] in pinned_norm else 1, prio[i], i),
    )

    used: set[int] = set()
    conflicts: list[int] = []
    for i in order:
        if out[i] in used:
            conflicts.append(i)
        else:
            used.add(out[i])
    if not conflicts:
        return out

    picker = rng if rng is not None else random
    for i in sorted(conflicts):
        free = [j for j in range(n_in) if j not in used]
        if not free:
            break
        pick = int(picker.choice(free)) if assignment_mode == "random" else int(free[0])
        out[i] = pick
        used.add(pick)
    return out


def rr_spatial_order_key(
    bbox: np.ndarray,
    list_index: int,
    track_id: int,
) -> tuple[float, float, int]:
    """Sort key: left→right, top→bottom, then stable tie-break.

    Without a tie-break, two faces with nearly equal center-x can alternate order
    frame-to-frame when the detector jitters, which reshuffles input assignments.
    ByteTrack id breaks ties when valid; otherwise the stable face list index does.
    """
    bb = np.asarray(bbox, dtype=np.float64)
    cx = float((bb[0] + bb[2]) * 0.5)
    cy = float((bb[1] + bb[3]) * 0.5)
    tid = int(track_id)
    tb = tid if tid >= 0 else int(list_index)
    return (cx, cy, tb)


def rr_spatial_sort_indices(boxes: list[np.ndarray]) -> list[int]:
    """Stable left→right order using only bbox geometry (no detector list index)."""
    n = len(boxes)

    def _key(ci: int) -> tuple[float, float, int]:
        bb = np.asarray(boxes[ci], dtype=np.float64)
        return (
            float((bb[0] + bb[2]) * 0.5),
            float((bb[1] + bb[3]) * 0.5),
            ci,
        )

    return sorted(range(n), key=_key)


def rr_centroid_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ba = np.asarray(box_a, dtype=np.float64)
    bb = np.asarray(box_b, dtype=np.float64)
    ax = float((ba[0] + ba[2]) * 0.5)
    ay = float((ba[1] + ba[3]) * 0.5)
    bx = float((bb[0] + bb[2]) * 0.5)
    by = float((bb[1] + bb[3]) * 0.5)
    return float(math.hypot(ax - bx, ay - by))


def rr_greedy_assign_from_memory(
    curr_boxes: list[np.ndarray],
    mem: list[tuple[np.ndarray, int, int]],
    n_in: int,
    centroid_max: float,
    iou_fn: IoUFn,
    *,
    iou_floor: float = 0.08,
    centroid_soft_factor: float = 1.35,
    assignment_mode: str = "index",
    rng: random.Random | None = None,
) -> tuple[list[int], list[bool]]:
    """Greedy IoU then centroid match vs memory slots; stable fallbacks for leftovers.

    Memory should be ordered with most recent / previous-frame slots first.
    """
    if n_in <= 0:
        return [], []

    prev_boxes = [np.asarray(pb, dtype=np.float64).copy() for pb, _, _ in mem]
    prev_inp = [int(ix) for _, ix, _ in mem]
    n_curr = len(curr_boxes)
    curr_assign: list[int | None] = [None] * n_curr
    spatially_matched = [False] * n_curr

    scored: list[tuple[int, float, int, int]] = []
    for ci in range(n_curr):
        cb = curr_boxes[ci]
        for mj in range(len(mem)):
            mb = prev_boxes[mj]
            iou_v = float(iou_fn(cb, mb))
            if iou_v >= iou_floor:
                scored.append((0, -iou_v, ci, mj))
            else:
                dist_v = rr_centroid_distance(cb, mb)
                if dist_v <= centroid_max:
                    scored.append((1, dist_v, ci, mj))

    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    assigned_c: set[int] = set()
    assigned_m: set[int] = set()
    for _tier, _sec, ci, mj in scored:
        if ci in assigned_c or mj in assigned_m:
            continue
        assigned_c.add(ci)
        assigned_m.add(mj)
        curr_assign[ci] = prev_inp[mj]
        spatially_matched[ci] = True

    used_inp_rr: set[int] = set()
    for ci in range(n_curr):
        if curr_assign[ci] is not None:
            used_inp_rr.add(int(curr_assign[ci]) % n_in)

    soft_centroid = float(centroid_max) * float(centroid_soft_factor)
    spatial_rank = {
        ci: rank
        for rank, ci in enumerate(rr_spatial_sort_indices(curr_boxes))
    }

    for ci in range(n_curr):
        if curr_assign[ci] is not None:
            continue
        cb = curr_boxes[ci]
        best_mj: int | None = None
        best_dist = soft_centroid + 1.0
        for mj in range(len(mem)):
            if mj in assigned_m:
                continue
            dist_v = rr_centroid_distance(cb, prev_boxes[mj])
            if dist_v < best_dist:
                best_dist = dist_v
                best_mj = mj
        if best_mj is not None and best_dist <= soft_centroid:
            curr_assign[ci] = prev_inp[best_mj]
            spatially_matched[ci] = True
            assigned_m.add(best_mj)
            used_inp_rr.add(int(prev_inp[best_mj]) % n_in)
            continue

        cand = pick_new_input_index(
            n_in,
            assignment_mode=assignment_mode,
            spatial_fallback=int(spatial_rank[ci]),
            used=used_inp_rr,
            rng=rng,
        )
        curr_assign[ci] = cand
        used_inp_rr.add(int(cand) % n_in)

    out = [int(curr_assign[ci]) % n_in for ci in range(n_curr)]
    return out, spatially_matched
