"""Swap-all-by-index temporal stabilizer (shared, ordered detection-thread state)."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from app.helpers.sequential_rr_order import (
    rr_greedy_assign_from_memory,
    rr_spatial_order_key,
    rr_spatial_sort_indices,
)

IoUFn = Callable[[np.ndarray, np.ndarray], float]


class SequentialRotateStabilizer:
    """Keeps input-face indices stable across frames for Swap all by index.

    Must run on a single ordered thread (video detection pipeline). Pool workers only
    consume precomputed ``_rr_input_idx`` values from the feeder task.
    """

    _SLOT_TTL_FRAMES: int = 90
    _SEEK_GAP_RESET_FRAMES: int = 150
    _MEMORY_MAX_SLOTS: int = 24
    _CENTROID_DIST_FRAC: float = 0.22
    _CENTROID_DIST_MIN_PX: float = 96.0
    _NO_TRACK_TTL_FRAMES: int = 24
    _NO_TRACK_MEMORY_MAX: int = 6
    _NO_TRACK_CENTROID_FRAC: float = 0.12
    _NO_TRACK_IOU_FLOOR: float = 0.14
    _NO_TRACK_PREV_IOU_FLOOR: float = 0.10

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._memory_slots: list[tuple[np.ndarray, int, int]] = []
        self._prev_frame_slots: list[tuple[np.ndarray, int]] = []
        self._track_to_input: dict[int, int] = {}
        self._track_last_seen: dict[int, int] = {}
        self._spatial_slots: dict[int, tuple[np.ndarray, int, int]] = {}
        self._next_spatial_slot_id: int = 0
        self._stabilize_last_fn: int = -999999
        self._stabilize_last_n_inputs: int = -1

    @staticmethod
    def _centroid_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ba = np.asarray(box_a, dtype=np.float64)
        bb = np.asarray(box_b, dtype=np.float64)
        ax = float((ba[0] + ba[2]) * 0.5)
        ay = float((ba[1] + ba[3]) * 0.5)
        bx = float((bb[0] + bb[2]) * 0.5)
        by = float((bb[1] + bb[3]) * 0.5)
        return float(math.hypot(ax - bx, ay - by))

    def _prune_expired_memory(self, frame_number: int, *, ttl: int) -> None:
        self._memory_slots = [
            (bb, ix, ls)
            for bb, ix, ls in self._memory_slots
            if frame_number - int(ls) <= ttl
        ]

    def _prune_stale_tracks(self, frame_number: int) -> None:
        ttl = self._SLOT_TTL_FRAMES
        stale = [
            t
            for t, ls in self._track_last_seen.items()
            if frame_number - int(ls) > ttl
        ]
        for t in stale:
            self._track_last_seen.pop(t, None)
            self._track_to_input.pop(t, None)

    def _prune_spatial_slots(self, frame_number: int) -> None:
        ttl = self._NO_TRACK_TTL_FRAMES
        stale = [
            sid
            for sid, (_, _, ls) in self._spatial_slots.items()
            if frame_number - int(ls) > ttl
        ]
        for sid in stale:
            self._spatial_slots.pop(sid, None)

    def _cap_memory_slots(
        self,
        slots: list[tuple[np.ndarray, int, int]],
        *,
        max_slots: int | None = None,
    ) -> list[tuple[np.ndarray, int, int]]:
        cap = self._MEMORY_MAX_SLOTS if max_slots is None else int(max_slots)
        if len(slots) <= cap:
            return slots
        return sorted(slots, key=lambda s: -int(s[2]))[:cap]

    def _build_memory_for_assign(
        self,
        frame_number: int,
        *,
        no_track_mode: bool,
    ) -> list[tuple[np.ndarray, int, int]]:
        prev = [
            (
                np.asarray(bb, dtype=np.float64).copy(),
                int(ix),
                2_000_000_000,
            )
            for bb, ix in self._prev_frame_slots
        ]
        if no_track_mode:
            ttl = self._NO_TRACK_TTL_FRAMES
            recent = [
                s
                for s in self._memory_slots
                if frame_number - int(s[2]) <= ttl
            ]
            return prev + sorted(recent, key=lambda s: -int(s[2]))
        return prev + sorted(self._memory_slots, key=lambda s: -int(s[2]))

    def _assign_no_track_spatial_slots(
        self,
        det_faces: list[dict],
        order: list[int],
        n_in: int,
        iou_fn: IoUFn,
    ) -> list[int]:
        """Slot-based matching: one persistent slot per physical face (no ByteTrack)."""
        assign: dict[int, int] = {}
        matched_slots: set[int] = set()

        scored: list[tuple[float, int, int]] = []
        for fi in order:
            bb = np.asarray(det_faces[fi]["bbox"], dtype=np.float64)
            for sid, (sbb, _sinp, _ls) in self._spatial_slots.items():
                iou_v = float(iou_fn(bb, sbb))
                if iou_v >= self._NO_TRACK_IOU_FLOOR:
                    scored.append((-iou_v, fi, sid))
        scored.sort()

        for _neg_iou, fi, sid in scored:
            if fi in assign or sid in matched_slots:
                continue
            sinp = int(self._spatial_slots[sid][1]) % n_in
            assign[fi] = sinp
            matched_slots.add(sid)

        for fi in order:
            if fi in assign:
                continue
            bb = np.asarray(det_faces[fi]["bbox"], dtype=np.float64)
            best_iou = self._NO_TRACK_PREV_IOU_FLOOR
            best_inp: int | None = None
            for pbb, pinp in self._prev_frame_slots:
                iou_v = float(iou_fn(bb, pbb))
                if iou_v > best_iou:
                    best_iou = iou_v
                    best_inp = int(pinp) % n_in
            if best_inp is not None:
                assign[fi] = best_inp

        rank_boxes = [np.asarray(det_faces[fi]["bbox"], dtype=np.float64) for fi in order]
        spatial_rank = {
            order[oi]: rank for rank, oi in enumerate(rr_spatial_sort_indices(rank_boxes))
        }
        used_inp: set[int] = {int(v) % n_in for v in assign.values()}
        for fi in order:
            if fi in assign:
                continue
            cand = next((j for j in range(n_in) if j not in used_inp), None)
            if cand is None:
                cand = int(spatial_rank.get(fi, 0)) % n_in
            assign[fi] = int(cand) % n_in
            used_inp.add(int(assign[fi]) % n_in)

        return [int(assign[fi]) % n_in for fi in order]

    def apply(
        self,
        det_faces: list[dict],
        checked_inputs_ordered: list[Any],
        frame_number: int,
        frame_wh: tuple[int, int],
        iou_fn: IoUFn,
        *,
        memory_without_tracking: bool = True,
    ) -> None:
        n_in = len(checked_inputs_ordered)
        if n_in == 0:
            return

        if frame_number < 0:
            self.reset()
            return

        last_fn = self._stabilize_last_fn
        if last_fn >= 0:
            if frame_number + 1 < last_fn:
                self.reset()
            elif frame_number - last_fn > self._SEEK_GAP_RESET_FRAMES:
                self.reset()

        if self._stabilize_last_n_inputs >= 0 and n_in != self._stabilize_last_n_inputs:
            self.reset()

        self._stabilize_last_fn = frame_number
        self._stabilize_last_n_inputs = n_in

        self._prune_expired_memory(
            frame_number,
            ttl=(
                self._NO_TRACK_TTL_FRAMES
                if memory_without_tracking
                else self._SLOT_TTL_FRAMES
            ),
        )
        self._prune_stale_tracks(frame_number)
        self._prune_spatial_slots(frame_number)

        if not det_faces:
            return

        raw_boxes = [
            np.asarray(face["bbox"], dtype=np.float64).copy() for face in det_faces
        ]

        all_tracks_ok = True
        ordered_tids: list[int] = []
        for face in det_faces:
            tid = int(face.get("track_id", -1))
            ordered_tids.append(tid)
            if tid < 0:
                all_tracks_ok = False

        use_tracks = (
            all_tracks_ok
            and ordered_tids
            and len(set(ordered_tids)) == len(ordered_tids)
        )

        if use_tracks:

            def _sort_key(fi: int) -> tuple[float, float, int]:
                return rr_spatial_order_key(
                    det_faces[fi]["bbox"],
                    fi,
                    int(det_faces[fi].get("track_id", -1)),
                )

            order = sorted(range(len(det_faces)), key=_sort_key)
        else:
            order = rr_spatial_sort_indices(raw_boxes)

        curr_boxes = [raw_boxes[fi] for fi in order]
        n_curr = len(order)
        img_w, img_h = int(frame_wh[0]), int(frame_wh[1])
        no_track_mode = bool(memory_without_tracking and not use_tracks)
        centroid_frac = (
            self._NO_TRACK_CENTROID_FRAC
            if no_track_mode
            else self._CENTROID_DIST_FRAC
        )
        centroid_max = max(
            self._CENTROID_DIST_MIN_PX,
            centroid_frac * float(min(img_w, img_h)),
        )

        if no_track_mode:
            base_assign = self._assign_no_track_spatial_slots(
                det_faces, order, n_in, iou_fn
            )
            spatially_matched = [True] * n_curr
        elif use_tracks or memory_without_tracking:
            mem_for_assign = self._build_memory_for_assign(
                frame_number, no_track_mode=no_track_mode
            )
            base_assign, spatially_matched = rr_greedy_assign_from_memory(
                curr_boxes,
                mem_for_assign,
                n_in,
                centroid_max,
                iou_fn,
                iou_floor=0.10 if no_track_mode else 0.08,
            )
        else:
            spatial_rank = {
                ci: rank
                for rank, ci in enumerate(rr_spatial_sort_indices(curr_boxes))
            }
            base_assign = [int(spatial_rank[ci]) % n_in for ci in range(n_curr)]
            spatially_matched = [False] * n_curr

        if use_tracks:
            final_assign = list(base_assign)
            for ci, fi in enumerate(order):
                tid = int(det_faces[fi]["track_id"])
                self._track_last_seen[tid] = frame_number
                mem_inp = int(base_assign[ci]) % n_in
                if tid not in self._track_to_input:
                    self._track_to_input[tid] = mem_inp
                    final_assign[ci] = mem_inp
                else:
                    track_inp = int(self._track_to_input[tid]) % n_in
                    if spatially_matched[ci] and track_inp != mem_inp:
                        self._track_to_input[tid] = mem_inp
                        final_assign[ci] = mem_inp
                    else:
                        final_assign[ci] = track_inp
            for ci, fi in enumerate(order):
                det_faces[fi]["_rr_input_idx"] = int(final_assign[ci]) % n_in
        else:
            for ci, fi in enumerate(order):
                det_faces[fi]["_rr_input_idx"] = int(base_assign[ci]) % n_in

        if no_track_mode:
            for fi in order:
                bb = np.asarray(det_faces[fi]["bbox"], dtype=np.float32).copy()
                inp = int(det_faces[fi]["_rr_input_idx"]) % n_in
                best_sid: int | None = None
                best_iou = self._NO_TRACK_IOU_FLOOR
                for sid, (sbb, _sinp, _ls) in self._spatial_slots.items():
                    iou_v = float(iou_fn(bb, sbb))
                    if iou_v > best_iou:
                        best_iou = iou_v
                        best_sid = sid
                if best_sid is not None:
                    self._spatial_slots[best_sid] = (bb, inp, frame_number)
                else:
                    sid = self._next_spatial_slot_id
                    self._next_spatial_slot_id += 1
                    self._spatial_slots[sid] = (bb, inp, frame_number)

        fresh_mem = [
            (
                np.asarray(det_faces[fi]["bbox"], dtype=np.float32).copy(),
                int(det_faces[fi]["_rr_input_idx"]),
                frame_number,
            )
            for fi in order
        ]
        if use_tracks:
            merged = list(fresh_mem)
            ttl = self._SLOT_TTL_FRAMES
            overlap_cap = 0.15
            for bb, ix, ls in self._memory_slots:
                if frame_number - int(ls) > ttl:
                    continue
                bba = np.asarray(bb, dtype=np.float64)
                if any(float(iou_fn(bba, cb)) > overlap_cap for cb in curr_boxes):
                    continue
                merged.append(
                    (np.asarray(bb, dtype=np.float32).copy(), int(ix), int(ls))
                )
            self._memory_slots = self._cap_memory_slots(merged)
        elif memory_without_tracking:
            self._memory_slots = self._cap_memory_slots(
                fresh_mem, max_slots=self._NO_TRACK_MEMORY_MAX
            )
        else:
            self._memory_slots = self._cap_memory_slots(fresh_mem)

        self._prev_frame_slots = [
            (
                np.asarray(det_faces[fi]["bbox"], dtype=np.float32).copy(),
                int(det_faces[fi]["_rr_input_idx"]),
            )
            for fi in order
        ]
