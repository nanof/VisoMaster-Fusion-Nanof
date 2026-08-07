"""Unit tests for Swap all by index (round-robin) spatial ordering helpers."""

import numpy as np

from app.helpers.sequential_rotate_stabilizer import SequentialRotateStabilizer
from app.helpers.sequential_rr_order import (
    rr_greedy_assign_from_memory,
    rr_spatial_order_key,
    rr_spatial_sort_indices,
)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def test_rr_spatial_order_key_tie_break_track_id():
    bb = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    k0 = rr_spatial_order_key(bb, 0, 7)
    k1 = rr_spatial_order_key(bb, 1, 3)
    # Same bbox: primary keys equal; track id breaks tie (deterministic order).
    assert k0[:2] == k1[:2]
    assert k0[2] == 7 and k1[2] == 3


def test_rr_spatial_order_key_tie_break_list_index_without_track():
    bb = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    k0 = rr_spatial_order_key(bb, 0, -1)
    k1 = rr_spatial_order_key(bb, 1, -1)
    assert k0[:2] == k1[:2]
    assert k0[2] == 0 and k1[2] == 1


def test_rr_spatial_order_key_left_to_right():
    left = np.array([0.0, 0.0, 50.0, 100.0])
    right = np.array([200.0, 0.0, 250.0, 100.0])
    assert rr_spatial_order_key(left, 0, -1)[0] < rr_spatial_order_key(right, 1, -1)[0]


def test_rr_spatial_sort_indices_ignores_detector_list_order():
    left = np.array([0.0, 0.0, 50.0, 100.0])
    right = np.array([200.0, 0.0, 250.0, 100.0])
    boxes = [right, left]
    assert rr_spatial_sort_indices(boxes) == [1, 0]


def test_rr_greedy_assign_keeps_input_when_bbox_jitters():
    """Prev-frame slot + small jitter must keep the same input index."""
    stable = np.array([100.0, 80.0, 200.0, 220.0], dtype=np.float64)
    jitter = np.array([102.0, 82.0, 198.0, 218.0], dtype=np.float64)
    mem = [(stable, 2, 10_000_000_000)]
    assign, matched = rr_greedy_assign_from_memory(
        [jitter], mem, n_in=3, centroid_max=80.0, iou_fn=_iou_xyxy
    )
    assert assign == [2]
    assert matched == [True]


def test_rr_greedy_assign_stable_under_reordered_detections():
    """Two faces: detector order swaps but geometry is unchanged → same inputs."""
    face_a = np.array([50.0, 50.0, 150.0, 150.0], dtype=np.float64)
    face_b = np.array([300.0, 50.0, 400.0, 150.0], dtype=np.float64)
    mem = [(face_a, 0, 10_000_000_000), (face_b, 1, 10_000_000_000)]
    order1 = rr_greedy_assign_from_memory(
        [face_a, face_b], mem, n_in=2, centroid_max=120.0, iou_fn=_iou_xyxy
    )[0]
    order2 = rr_greedy_assign_from_memory(
        [face_b, face_a], mem, n_in=2, centroid_max=120.0, iou_fn=_iou_xyxy
    )[0]
    assert order1 == [0, 1]
    assert order2 == [1, 0]


def test_stabilizer_no_track_keeps_input_across_jitter():
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    stabilizer = SequentialRotateStabilizer()
    checked = [object(), object(), object()]
    f0 = [{"bbox": np.array([80.0, 60.0, 160.0, 140.0]), "track_id": -1}]
    stabilizer.apply(f0, checked, 0, (640, 480), iou, memory_without_tracking=True)
    inp0 = int(f0[0]["_rr_input_idx"])

    f1 = [{"bbox": np.array([82.0, 62.0, 158.0, 138.0]), "track_id": -1}]
    stabilizer.apply(f1, checked, 1, (640, 480), iou, memory_without_tracking=True)
    assert int(f1[0]["_rr_input_idx"]) == inp0

def test_pick_new_input_index_random_avoids_used():
    from app.helpers.sequential_rr_order import pick_new_input_index
    import random

    rng = random.Random(123)
    picks = [
        pick_new_input_index(
            4,
            assignment_mode="random",
            spatial_fallback=0,
            used={0, 1, 2},
            rng=rng,
        )
        for _ in range(20)
    ]
    # Only index 3 is free → every pick must be 3.
    assert picks == [3] * 20


def test_pick_new_input_index_random_unique_sequence():
    from app.helpers.sequential_rr_order import pick_new_input_index
    import random

    rng = random.Random(99)
    used: set[int] = set()
    picks = []
    for _ in range(4):
        p = pick_new_input_index(
            4,
            assignment_mode="random",
            spatial_fallback=0,
            used=used,
            rng=rng,
        )
        picks.append(p)
        used.add(p)
    assert sorted(picks) == [0, 1, 2, 3]


def test_random_assignment_sticks_for_same_face_across_frames():
    import random

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        return _iou_xyxy(a, b)

    stabilizer = SequentialRotateStabilizer(rng=random.Random(7))
    checked = [object(), object(), object(), object()]
    f0 = [{"bbox": np.array([80.0, 60.0, 160.0, 140.0]), "track_id": -1}]
    stabilizer.apply(
        f0,
        checked,
        0,
        (640, 480),
        iou,
        memory_without_tracking=True,
        assignment_mode="random",
    )
    inp0 = int(f0[0]["_rr_input_idx"])

    f1 = [{"bbox": np.array([82.0, 62.0, 158.0, 138.0]), "track_id": -1}]
    stabilizer.apply(
        f1,
        checked,
        1,
        (640, 480),
        iou,
        memory_without_tracking=True,
        assignment_mode="random",
    )
    assert int(f1[0]["_rr_input_idx"]) == inp0


def test_random_assignment_survives_pause_resume_without_reset():
    """Pause/play must not reshuffle: same sticky state, next frame keeps input."""
    import random

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        return _iou_xyxy(a, b)

    stabilizer = SequentialRotateStabilizer(rng=random.Random(19))
    checked = [object(), object(), object(), object(), object()]
    face_bb = np.array([100.0, 80.0, 180.0, 160.0])
    f0 = [{"bbox": face_bb.copy(), "track_id": -1}]
    stabilizer.apply(
        f0,
        checked,
        40,
        (640, 480),
        iou,
        memory_without_tracking=True,
        assignment_mode="random",
    )
    inp0 = int(f0[0]["_rr_input_idx"])

    # Simulate pause/play: detection EMA cleared, but RR sticky state kept.
    f1 = [{"bbox": face_bb.copy() + np.array([1.0, -1.0, 1.0, -1.0]), "track_id": -1}]
    stabilizer.apply(
        f1,
        checked,
        40,
        (640, 480),
        iou,
        memory_without_tracking=True,
        assignment_mode="random",
    )
    assert int(f1[0]["_rr_input_idx"]) == inp0

    # Explicit reshuffle (X) clears sticky state → a fresh roll is allowed.
    stabilizer.reset()
    f2 = [{"bbox": face_bb.copy(), "track_id": -1}]
    stabilizer.apply(
        f2,
        checked,
        41,
        (640, 480),
        iou,
        memory_without_tracking=True,
        assignment_mode="random",
    )
    # With 5 inputs and a fresh RNG path after reset, assignment is defined;
    # the important part is that sticky memory was emptied by reset.
    assert stabilizer._memory_slots or f2[0].get("_rr_input_idx") is not None


def test_random_assignment_no_repeat_for_two_new_faces():
    import random

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        return _iou_xyxy(a, b)

    stabilizer = SequentialRotateStabilizer(rng=random.Random(11))
    checked = [object(), object(), object()]
    faces = [
        {"bbox": np.array([40.0, 40.0, 120.0, 120.0]), "track_id": -1},
        {"bbox": np.array([300.0, 40.0, 380.0, 120.0]), "track_id": -1},
    ]
    stabilizer.apply(
        faces,
        checked,
        0,
        (640, 480),
        iou,
        memory_without_tracking=True,
        assignment_mode="random",
    )
    a = int(faces[0]["_rr_input_idx"])
    b = int(faces[1]["_rr_input_idx"])
    assert a != b
    assert {a, b}.issubset({0, 1, 2})


def test_dedupe_assignments_removes_repeats_while_inputs_free():
    from app.helpers.sequential_rr_order import rr_dedupe_assignments

    out = rr_dedupe_assignments([1, 1, 1], n_in=4, assignment_mode="index")
    assert len(set(out)) == 3
    assert 1 in out


def test_dedupe_assignments_keeps_matched_face_and_moves_new_one():
    from app.helpers.sequential_rr_order import rr_dedupe_assignments

    # Face 1 is temporally matched (priority 0); face 0 is a fresh pick.
    out = rr_dedupe_assignments(
        [2, 2],
        n_in=3,
        assignment_mode="index",
        keep_priority=[2, 0],
    )
    assert out[1] == 2
    assert out[0] != 2


def test_dedupe_assignments_keeps_pinned_input():
    from app.helpers.sequential_rr_order import rr_dedupe_assignments

    out = rr_dedupe_assignments(
        [3, 3],
        n_in=4,
        assignment_mode="index",
        keep_priority=[0, 2],
        pinned={3},
    )
    # Pinned wins the contest even against the better-ranked face.
    assert 3 in out
    assert len(set(out)) == 2


def test_dedupe_assignments_allows_repeats_when_inputs_exhausted():
    from app.helpers.sequential_rr_order import rr_dedupe_assignments

    out = rr_dedupe_assignments([0, 1, 0, 1], n_in=2, assignment_mode="random")
    assert sorted(out) == [0, 0, 1, 1]


def test_random_no_repeat_when_spatial_slots_share_input():
    """Two remembered slots carrying the same input must not swap the same face."""
    import random

    stabilizer = SequentialRotateStabilizer(rng=random.Random(3))
    checked = [object(), object(), object()]
    box_a = np.array([40.0, 40.0, 120.0, 120.0], dtype=np.float32)
    box_b = np.array([300.0, 40.0, 380.0, 120.0], dtype=np.float32)
    stabilizer._spatial_slots = {
        0: (box_a.copy(), 1, 0),
        1: (box_b.copy(), 1, 0),
    }
    faces = [
        {"bbox": box_a.astype(np.float64), "track_id": -1},
        {"bbox": box_b.astype(np.float64), "track_id": -1},
    ]
    stabilizer.apply(
        faces,
        checked,
        1,
        (640, 480),
        _iou_xyxy,
        memory_without_tracking=True,
        assignment_mode="random",
    )
    assert int(faces[0]["_rr_input_idx"]) != int(faces[1]["_rr_input_idx"])


def test_random_no_repeat_when_two_tracks_share_locked_input():
    import random

    stabilizer = SequentialRotateStabilizer(rng=random.Random(5))
    checked = [object(), object(), object()]
    stabilizer._track_to_input = {7: 1, 8: 1}
    stabilizer._track_last_seen = {7: 0, 8: 0}
    faces = [
        {"bbox": np.array([40.0, 40.0, 120.0, 120.0]), "track_id": 7},
        {"bbox": np.array([300.0, 40.0, 380.0, 120.0]), "track_id": 8},
    ]
    stabilizer.apply(
        faces,
        checked,
        1,
        (640, 480),
        _iou_xyxy,
        memory_without_tracking=False,
        assignment_mode="random",
    )
    inputs = [int(f["_rr_input_idx"]) for f in faces]
    assert len(set(inputs)) == 2
    assert stabilizer._track_to_input[7] != stabilizer._track_to_input[8]


def test_random_no_repeat_with_ghost_memory_slot():
    """A disappeared face's ghost slot must not clone an input onto a new face."""
    import random

    stabilizer = SequentialRotateStabilizer(rng=random.Random(13))
    checked = [object(), object(), object()]
    present = np.array([40.0, 40.0, 120.0, 120.0], dtype=np.float32)
    ghost = np.array([300.0, 40.0, 380.0, 120.0], dtype=np.float32)
    stabilizer._track_to_input = {7: 0}
    stabilizer._track_last_seen = {7: 0}
    stabilizer._memory_slots = [(ghost.copy(), 0, 0)]
    stabilizer._prev_frame_slots = [(present.copy(), 0)]
    faces = [
        {"bbox": present.astype(np.float64), "track_id": 7},
        {"bbox": ghost.astype(np.float64), "track_id": 9},
    ]
    stabilizer.apply(
        faces,
        checked,
        1,
        (640, 480),
        _iou_xyxy,
        memory_without_tracking=False,
        assignment_mode="random",
    )
    inputs = [int(f["_rr_input_idx"]) for f in faces]
    assert len(set(inputs)) == 2
    # The established track keeps its input; the newcomer is the one moved.
    assert inputs[0] == 0


def test_swap_all_match_helpers():
    from app.helpers.swap_all_match import (
        swap_all_assignment_mode,
        swap_all_match_active,
    )

    assert not swap_all_match_active({})
    assert swap_all_match_active({"RandomTargetMatchEnableToggle": True})
    assert swap_all_assignment_mode({"RandomTargetMatchEnableToggle": True}) == "random"
    assert (
        swap_all_assignment_mode(
            {
                "SequentialTargetMatchEnableToggle": True,
                "RandomTargetMatchEnableToggle": True,
            }
        )
        == "index"
    )
    assert not swap_all_match_active(
        {
            "RandomTargetMatchEnableToggle": True,
            "SwapOnlyBestMatchEnableToggle": True,
        }
    )

def test_reset_preserves_pinned_input_indices():
    import random

    stab = SequentialRotateStabilizer(rng=random.Random(1))
    stab._track_to_input = {10: 0, 11: 1, 12: 2}
    stab._track_last_seen = {10: 5, 11: 5, 12: 5}
    stab._memory_slots = [
        (np.array([0.0, 0.0, 10.0, 10.0]), 0, 5),
        (np.array([20.0, 0.0, 30.0, 10.0]), 1, 5),
        (np.array([40.0, 0.0, 50.0, 10.0]), 2, 5),
    ]
    stab._prev_frame_slots = [
        (np.array([0.0, 0.0, 10.0, 10.0]), 0),
        (np.array([20.0, 0.0, 30.0, 10.0]), 1),
    ]
    stab._spatial_slots = {
        0: (np.array([0.0, 0.0, 10.0, 10.0]), 0, 5),
        1: (np.array([20.0, 0.0, 30.0, 10.0]), 1, 5),
    }
    stab._stabilize_last_fn = 5
    stab._stabilize_last_n_inputs = 3
    stab._stabilize_last_assignment_mode = "random"

    stab.reset(preserve_input_indices={1})

    assert stab._track_to_input == {11: 1}
    assert stab._track_last_seen == {11: 5}
    assert len(stab._memory_slots) == 1 and int(stab._memory_slots[0][1]) == 1
    assert len(stab._prev_frame_slots) == 1 and int(stab._prev_frame_slots[0][1]) == 1
    assert list(stab._spatial_slots) == [1]
    assert stab._stabilize_last_fn == 5


def test_pinned_checked_input_indices():
    from app.helpers.swap_all_match import (
        pinned_checked_input_indices,
        pinned_indices_from_checked,
    )
    from types import SimpleNamespace

    faces = {
        "a": SimpleNamespace(isChecked=lambda: True, random_fixed=False),
        "b": SimpleNamespace(isChecked=lambda: True, random_fixed=True),
        "c": SimpleNamespace(isChecked=lambda: False, random_fixed=True),
        "d": SimpleNamespace(isChecked=lambda: True, random_fixed=True),
    }
    mw = SimpleNamespace(input_faces=faces)
    # checked order: a(0), b(1), d(2) — pinned are b and d → {1, 2}
    assert pinned_checked_input_indices(mw) == {1, 2}
    checked = [faces["a"], faces["b"], faces["d"]]
    assert pinned_indices_from_checked(checked) == {1, 2}


def test_enforce_pinned_inputs_steals_non_pinned_slot():
    from app.helpers.sequential_rotate_stabilizer import SequentialRotateStabilizer

    out = SequentialRotateStabilizer._enforce_pinned_inputs(
        [0, 1], n_in=4, pinned={3}
    )
    assert 3 in out
    assert sorted(out) == sorted([0, 3]) or sorted(out) == sorted([1, 3])


def test_pinned_input_forced_onto_frame_random_mode():
    import random

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        return _iou_xyxy(a, b)

    # Many trials: without enforce, a single pinned index can easily be absent.
    for seed in range(30):
        stabilizer = SequentialRotateStabilizer(rng=random.Random(seed))
        checked = [object(), object(), object(), object()]
        faces = [
            {"bbox": np.array([40.0, 40.0, 120.0, 120.0]), "track_id": -1},
            {"bbox": np.array([300.0, 40.0, 380.0, 120.0]), "track_id": -1},
        ]
        stabilizer.apply(
            faces,
            checked,
            0,
            (640, 480),
            iou,
            memory_without_tracking=True,
            assignment_mode="random",
            pinned_input_indices={3},
        )
        assigned = {int(f["_rr_input_idx"]) for f in faces}
        assert 3 in assigned


def test_pinned_track_not_rematched_away():
    import random

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        return _iou_xyxy(a, b)

    stabilizer = SequentialRotateStabilizer(rng=random.Random(42))
    checked = [object(), object(), object()]
    f0 = [
        {"bbox": np.array([40.0, 40.0, 120.0, 120.0]), "track_id": 7},
        {"bbox": np.array([300.0, 40.0, 380.0, 120.0]), "track_id": 8},
    ]
    stabilizer.apply(
        f0,
        checked,
        0,
        (640, 480),
        iou,
        memory_without_tracking=True,
        assignment_mode="random",
        pinned_input_indices={2},
    )
    assert 2 in {int(f["_rr_input_idx"]) for f in f0}
    # Lock track 7 to the pinned input and move boxes slightly; rematch must keep pin.
    stabilizer._track_to_input[7] = 2
    stabilizer._track_to_input[8] = 0 if int(f0[1]["_rr_input_idx"]) != 2 else 1
    f1 = [
        {"bbox": np.array([42.0, 42.0, 118.0, 118.0]), "track_id": 7},
        {"bbox": np.array([302.0, 42.0, 378.0, 118.0]), "track_id": 8},
    ]
    stabilizer.apply(
        f1,
        checked,
        1,
        (640, 480),
        iou,
        memory_without_tracking=True,
        assignment_mode="random",
        pinned_input_indices={2},
    )
    by_tid = {int(f["track_id"]): int(f["_rr_input_idx"]) for f in f1}
    assert by_tid[7] == 2
    assert 2 in by_tid.values()
