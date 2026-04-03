"""
Pipeline profile overlay: merge feeder + worker timings, EMA / window aggregation, formatting.
"""

from __future__ import annotations

import csv
import json
import os
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

PIPELINE_PROFILE_FEEDER_KEY_SET: frozenset[str] = frozenset(PIPELINE_PROFILE_FEEDER_ORDER)

# Cap stored samples per playback session (each displayed frame with overlay on).
_PIPELINE_PROFILE_SESSION_MAX = 8000
_PIPELINE_PROFILE_SESSION_REPORT_PREFIX = "[PIPELINE-PROFILE-SESSION]"

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
    "feeder_subtotal": "Sum Feeder (ms)",
    "worker_subtotal": "Sum Worker (ms)",
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
    header_lines: List[str] | None = None,
) -> str:
    """One column per worker thread plus an Avg column (mean across threads per row).

    Feeder-stage rows (read/detect in feeder thread) are grouped first, then worker
    stages (GPU pipeline after dequeue), with subtotal lines to compare both sides.
    """
    if not per_thread:
        return "Profile: —"
    threads_sorted = sorted(per_thread.keys(), key=_thread_sort_key)
    if not threads_sorted:
        return "Profile: —"
    stages = _ordered_stage_keys_union(per_thread)
    if not stages:
        return "Profile: —"

    feeder_stages = [s for s in stages if s in PIPELINE_PROFILE_FEEDER_KEY_SET]
    worker_stages = [s for s in stages if s not in PIPELINE_PROFILE_FEEDER_KEY_SET]

    cw_l = _OVERLAY_COL_LABEL
    cw = _OVERLAY_COL_MS_THREAD
    lines: List[str] = []
    if header_lines:
        lines.extend(header_lines)
    lines.append("Pipeline profile (ms) — Feeder | Worker")
    hdr = f"{'Stage':<{cw_l}}"
    for t in threads_sorted:
        hdr += f"  {_short_thread_column_title(t):>{cw}}"
    hdr += f"  {'Avg':>{cw}}"
    lines.append(hdr)

    def _append_stage_block(stage_list: List[str], title: str | None) -> None:
        if not stage_list:
            return
        if title:
            sep = f"{_overlay_fit_label(title):<{cw_l}}"
            for _t in threads_sorted:
                sep += f"  {'':>{cw}}"
            sep += f"  {'':>{cw}}"
            lines.append(sep)
        for stage in stage_list:
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

    def _append_subtotal_row(stage_key: str, stage_list: List[str]) -> None:
        if not stage_list:
            return
        label = _overlay_fit_label(PIPELINE_STAGE_LABELS.get(stage_key, stage_key))
        row = f"{label:<{cw_l}}"
        col_totals: List[float] = []
        for t in threads_sorted:
            d = per_thread[t]
            ssum = 0.0
            any_v = False
            for s in stage_list:
                v = d.get(s)
                if v is not None:
                    ssum += float(v)
                    any_v = True
            if any_v:
                col_totals.append(ssum)
                row += f"  {ssum:>{cw}.1f}"
            else:
                row += f"  {'—':>{cw}}"
        avg_v = sum(col_totals) / len(col_totals) if col_totals else None
        row += (
            f"  {avg_v:>{cw}.1f}"
            if avg_v is not None
            else f"  {'—':>{cw}}"
        )
        lines.append(row)

    _append_stage_block(feeder_stages, "— Feeder thread —" if feeder_stages else None)
    _append_subtotal_row("feeder_subtotal", feeder_stages)
    _append_stage_block(worker_stages, "— Worker thread —" if worker_stages else None)
    _append_subtotal_row("worker_subtotal", worker_stages)

    row = f"{'Total (all stages)':<{cw_l}}"
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
    header_lines: List[str] | None = None,
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
            format_profile_overlay_multithread(display, header_lines=header_lines)
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

    return format_profile_overlay_multithread(display, header_lines=header_lines)


def reset_pipeline_profile_state(main_window: Any) -> None:
    """Clear per-thread EMA/window history (e.g. on stop or new media)."""
    main_window._pipeline_profile_ema_by_thread = {}
    main_window._pipeline_profile_window_deques = {}
    main_window._pipeline_profile_display_by_thread = {}

def clear_pipeline_profile_session_samples(main_window: Any) -> None:
    """Empty session log at playback start (overlay samples for console report)."""
    main_window._pipeline_profile_session_samples = deque()


def append_pipeline_profile_session_sample(
    main_window: Any,
    profile_payload: dict[str, Any],
    rows: List[Tuple[str, float]],
) -> None:
    """Record one frame while the pipeline profile overlay is enabled."""
    if not main_window.control.get("PipelineProfileOverlayEnableToggle", False):
        return
    samples: Deque[dict[str, Any]] | None = getattr(
        main_window, "_pipeline_profile_session_samples", None
    )
    if samples is None:
        samples = deque()
        main_window._pipeline_profile_session_samples = samples
    stages_ms: dict[str, float] = {}
    for k, v in rows:
        try:
            stages_ms[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
    samples.append(
        {
            "frame_number": profile_payload.get("frame_number"),
            "worker_thread": profile_payload.get("worker_thread"),
            "queue_at_emit": profile_payload.get("frame_queue_depth_at_emit"),
            "queue_max": profile_payload.get("frame_queue_max"),
            "stages_ms": stages_ms,
        }
    )
    _append_pipeline_profile_csv_row(profile_payload, stages_ms)
    while len(samples) > _PIPELINE_PROFILE_SESSION_MAX:
        samples.popleft()


def _append_pipeline_profile_csv_row(
    profile_payload: dict[str, Any],
    stages_ms: dict[str, float],
) -> None:
    """Append one row when VISIOMASTER_PIPELINE_PROFILE_CSV points to a file path."""
    path = os.environ.get("VISIOMASTER_PIPELINE_PROFILE_CSV", "").strip()
    if not path:
        return
    try:
        fn = profile_payload.get("frame_number")
        wt = profile_payload.get("worker_thread")
        qe = profile_payload.get("frame_queue_depth_at_emit")
        qm = profile_payload.get("frame_queue_max")
        write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            if write_header:
                w.writerow(
                    [
                        "frame_number",
                        "worker_thread",
                        "queue_at_emit",
                        "queue_max",
                        "stages_ms_json",
                    ]
                )
            w.writerow(
                [
                    fn,
                    wt,
                    qe,
                    qm,
                    json.dumps(stages_ms, separators=(",", ":")),
                ]
            )
    except OSError as e:
        print(f"[WARN] Pipeline profile CSV append failed ({path}): {e}", flush=True)


def print_pipeline_profile_session_report(main_window: Any) -> None:
    """After stop: print min/avg/max per stage for A/B comparisons."""
    samples: Deque[dict[str, Any]] | None = getattr(
        main_window, "_pipeline_profile_session_samples", None
    )
    if not samples:
        return
    snap = list(samples)
    samples.clear()
    n = len(snap)
    pfx = _PIPELINE_PROFILE_SESSION_REPORT_PREFIX
    qvals: list[int] = []
    qmax_cap: int | None = None
    for s in snap:
        qe = s.get("queue_at_emit")
        if qe is not None:
            try:
                qvals.append(int(qe))
            except (TypeError, ValueError):
                pass
        qm = s.get("queue_max")
        if qm is not None and qmax_cap is None:
            try:
                qmax_cap = int(qm)
            except (TypeError, ValueError):
                pass
    keys: set[str] = set()
    for s in snap:
        keys.update(s["stages_ms"].keys())

    def _feeder_key_order(k: str) -> int:
        try:
            return PIPELINE_PROFILE_FEEDER_ORDER.index(k)
        except ValueError:
            return 999

    feeder_keys = sorted(keys & PIPELINE_PROFILE_FEEDER_KEY_SET, key=_feeder_key_order)
    worker_keys = sorted(keys - PIPELINE_PROFILE_FEEDER_KEY_SET)

    def stats_for_key(k: str) -> tuple[float, float, float, int]:
        vals = [s["stages_ms"][k] for s in snap if k in s["stages_ms"]]
        if not vals:
            return 0.0, 0.0, 0.0, 0
        return sum(vals) / len(vals), min(vals), max(vals), len(vals)

    lines: list[str] = []
    lines.append(f"{pfx} ========== Session summary (n={n} samples) ==========")
    if qvals:
        qa = sum(qvals) / len(qvals)
        lines.append(
            f"{pfx} Queue at emit: avg={qa:.2f} min={min(qvals)} max={max(qvals)}"
            + (f" (queue maxsize={qmax_cap})" if qmax_cap is not None else "")
        )
    f_sums: list[float] = []
    w_sums: list[float] = []
    for s in snap:
        sm = s["stages_ms"]
        f_sums.append(sum(sm[k] for k in sm if k in PIPELINE_PROFILE_FEEDER_KEY_SET))
        w_sums.append(sum(sm[k] for k in sm if k not in PIPELINE_PROFILE_FEEDER_KEY_SET))
    if f_sums:
        lines.append(
            f"{pfx} Sum Feeder ms/frame: avg={sum(f_sums)/len(f_sums):.2f} "
            f"min={min(f_sums):.2f} max={max(f_sums):.2f}"
        )
    if w_sums:
        lines.append(
            f"{pfx} Sum Worker ms/frame: avg={sum(w_sums)/len(w_sums):.2f} "
            f"min={min(w_sums):.2f} max={max(w_sums):.2f}"
        )

    def _block(title: str, ks: List[str]) -> None:
        if not ks:
            return
        lines.append(f"{pfx} --- {title} ---")
        for k in ks:
            avg, vmin, vmax, cnt = stats_for_key(k)
            if cnt == 0:
                continue
            lab = PIPELINE_STAGE_LABELS.get(k, k)
            lines.append(
                f"{pfx}   {lab}: avg={avg:.2f} ms  min={vmin:.2f}  max={vmax:.2f}  (n={cnt})"
            )

    _block("Feeder thread", feeder_keys)
    _block("Worker thread", worker_keys)
    lines.append(f"{pfx} ========== End session summary ==========")
    lines.append(
        f"{pfx} Baseline tools: VISIOMASTER_PERF_STAGES=1 (console per frame); "
        "VISIOMASTER_PIPELINE_PROFILE_CSV=path.csv (append rows during session); "
        "VISIOMASTER_NVTX=1 + NVIDIA Nsight Systems (GPU overlap)."
    )
    print("\n".join(lines), flush=True)

