"""
Pipeline profile overlay: merge feeder + worker timings, EMA / window aggregation, formatting.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, Dict, List, Tuple

# Display order for feeder keys (worker stages keep list order from the payload).
PIPELINE_PROFILE_FEEDER_ORDER: Tuple[str, ...] = (
    "read_frame_ms",
    "feeder_state_ms",
    "rgb_pack_ms",
    "feeder_params_lock_ms",
    "sequential_detect_ms",
)

# Display labels for overlay (fallback: raw stage id).
PIPELINE_STAGE_LABELS: Dict[str, str] = {
    "read_frame_ms": "Read frame",
    "feeder_state_ms": "Feeder state / markers",
    "rgb_pack_ms": "RGB pack",
    "feeder_params_lock_ms": "Params lock",
    "sequential_detect_ms": "Detect (feeder)",
    "prep_scaling_h2d": "Prep GPU (H2D)",
    "vr180": "VR180 pipeline",
    "std_upscale_rotate": "Scale / rotate",
    "std_detect_feeder_or_fallback": "Detect (worker)",
    "std_recognize": "Recognition",
    "std_swap_edit": "Swap / edit",
    "std_undo_resize": "Undo scale",
    "std_overlays_compare": "Overlays / compare",
    "frame_enhancer": "Frame enhancer",
    "d2h_numpy": "GPU to CPU",
    "pass_through": "Passthrough",
}

# Column widths for monospace overlay.
_OVERLAY_COL_LABEL = 22
_OVERLAY_COL_MS_THREAD = 8  # per-thread and Avg ms columns


def _overlay_fit_label(text: str) -> str:
    w = _OVERLAY_COL_LABEL
    if len(text) <= w:
        return text
    return text[: w - 1] + "…"


def _short_thread_column_title(name: str) -> str:
    if "FrameWorker-Pool-" in name:
        return "W" + name.split("FrameWorker-Pool-", 1)[-1]
    if "FrameWorker-Single-" in name:
        return "Single"
    return name[: _OVERLAY_COL_MS_THREAD] if len(name) > _OVERLAY_COL_MS_THREAD else name


def _thread_sort_key(name: str) -> tuple[int, int, str]:
    m = re.search(r"Pool-(\d+)", name)
    if m:
        return (0, int(m.group(1)), name)
    return (1, 0, name)


def _ordered_stage_keys_union(per_thread: Dict[str, Dict[str, float]]) -> List[str]:
    all_keys: set[str] = set()
    for d in per_thread.values():
        all_keys.update(d.keys())
    out: List[str] = []
    for k in PIPELINE_PROFILE_FEEDER_ORDER:
        if k in all_keys:
            out.append(k)
    for k in sorted(k for k in all_keys if k not in out):
        out.append(k)
    return out


def flatten_pipeline_profile_payload(
    payload: dict[str, Any] | None,
) -> List[Tuple[str, float]]:
    """Merge feeder dict + worker stage list into ordered (id, ms) rows."""
    if not payload:
        return []
    rows: List[Tuple[str, float]] = []
    fd = payload.get("feeder")
    if isinstance(fd, dict):
        for k in PIPELINE_PROFILE_FEEDER_ORDER:
            if k in fd:
                try:
                    rows.append((k, float(fd[k])))
                except (TypeError, ValueError):
                    pass
    wk = payload.get("worker")
    if isinstance(wk, list):
        for item in wk:
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[0], str)
            ):
                try:
                    rows.append((item[0], float(item[1])))
                except (TypeError, ValueError):
                    pass
    return rows


def total_ms_from_rows(rows: List[Tuple[str, float]]) -> float:
    return sum(ms for _, ms in rows if ms >= 0.0)


def update_ema_per_stage(
    ema_state: Dict[str, float],
    rows: List[Tuple[str, float]],
    alpha: float,
) -> Dict[str, float]:
    """In-place EMA per stage id. alpha in (0,1]: new = alpha*v + (1-alpha)*old."""
    a = max(0.001, min(1.0, float(alpha)))
    for k, v in rows:
        if k in ema_state:
            ema_state[k] = a * v + (1.0 - a) * ema_state[k]
        else:
            ema_state[k] = v
    return ema_state


def push_window_and_mean(
    history: Deque[List[Tuple[str, float]]],
    rows: List[Tuple[str, float]],
    window_n: int,
) -> Dict[str, float]:
    """Append snapshot; return mean ms per stage over last up to window_n frames."""
    n = max(1, min(120, int(window_n)))
    history.append(rows)
    while len(history) > n:
        history.popleft()
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for snap in history:
        seen: set[str] = set()
        for k, v in snap:
            if k in seen:
                continue
            seen.add(k)
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def format_profile_overlay_multithread(
    per_thread: Dict[str, Dict[str, float]],
) -> str:
    """One column per worker thread plus an Avg column (mean across threads per row)."""
    if not per_thread:
        return "Profile: —"
    threads_sorted = sorted(per_thread.keys(), key=_thread_sort_key)
    if not threads_sorted:
        return "Profile: —"
    stages = _ordered_stage_keys_union(per_thread)
    if not stages:
        return "Profile: —"

    cw_l = _OVERLAY_COL_LABEL
    cw = _OVERLAY_COL_MS_THREAD
    lines = ["Pipeline profile (ms)"]
    hdr = f"{'Stage':<{cw_l}}"
    for t in threads_sorted:
        hdr += f"  {_short_thread_column_title(t):>{cw}}"
    hdr += f"  {'Avg':>{cw}}"
    lines.append(hdr)

    for stage in stages:
        label = _overlay_fit_label(PIPELINE_STAGE_LABELS.get(stage, stage))
        row = f"{label:<{cw_l}}"
        vals: List[float] = []
        for t in threads_sorted:
            d = per_thread[t]
            v = d.get(stage)
            if v is not None:
                vals.append(float(v))
            row += (
                f"  {v:>{cw}.1f}"
                if v is not None
                else f"  {'—':>{cw}}"
            )
        avg_v = sum(vals) / len(vals) if vals else None
        row += (
            f"  {avg_v:>{cw}.1f}"
            if avg_v is not None
            else f"  {'—':>{cw}}"
        )
        lines.append(row)

    row = f"{'Total':<{cw_l}}"
    totals: List[float] = []
    for t in threads_sorted:
        d = per_thread[t]
        tot = sum(float(d.get(s, 0.0)) for s in stages)
        totals.append(tot)
        row += f"  {tot:>{cw}.1f}"
    avg_tot = sum(totals) / len(totals) if totals else 0.0
    row += f"  {avg_tot:>{cw}.1f}"
    lines.append(row)
    return "\n".join(lines)


def aggregate_rows_for_display(
    main_window: Any,
    rows: List[Tuple[str, float]],
    worker_thread: str | None,
) -> str:
    """EMA or window smoothing per thread; table with one column per thread + Avg."""
    display: Dict[str, Dict[str, float]] = getattr(
        main_window, "_pipeline_profile_display_by_thread", None
    )
    if display is None:
        display = {}
        main_window._pipeline_profile_display_by_thread = display

    wt = (worker_thread or "").strip() or "?"

    if not rows:
        return (
            format_profile_overlay_multithread(display)
            if display
            else "Profile: —"
        )

    ctrl = main_window.control
    mode = str(ctrl.get("PipelineProfileAggregationSelection", "EMA"))
    if mode in ("Ventana", "Window"):
        try:
            n = int(ctrl.get("PipelineProfileWindowFramesSlider", 30))
        except (TypeError, ValueError):
            n = 30
        deques: Dict[str, Deque[List[Tuple[str, float]]]] | None = getattr(
            main_window, "_pipeline_profile_window_deques", None
        )
        if deques is None:
            deques = {}
            main_window._pipeline_profile_window_deques = deques
        if wt not in deques:
            deques[wt] = deque()
        averaged = push_window_and_mean(deques[wt], rows, n)
        display[wt] = dict(averaged)
    else:
        try:
            alpha = float(ctrl.get("PipelineProfileEmaAlphaDecimalSlider", 0.25))
        except (TypeError, ValueError):
            alpha = 0.25
        ema_bt: Dict[str, Dict[str, float]] | None = getattr(
            main_window, "_pipeline_profile_ema_by_thread", None
        )
        if ema_bt is None:
            ema_bt = {}
            main_window._pipeline_profile_ema_by_thread = ema_bt
        if wt not in ema_bt:
            ema_bt[wt] = {}
        update_ema_per_stage(ema_bt[wt], rows, alpha)
        display[wt] = dict(ema_bt[wt])

    return format_profile_overlay_multithread(display)


def reset_pipeline_profile_state(main_window: Any) -> None:
    """Clear per-thread EMA/window history (e.g. on stop or new media)."""
    main_window._pipeline_profile_ema_by_thread = {}
    main_window._pipeline_profile_window_deques = {}
    main_window._pipeline_profile_display_by_thread = {}
