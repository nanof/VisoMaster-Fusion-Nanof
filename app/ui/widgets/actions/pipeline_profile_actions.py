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

# Feeder ∑ + worker wall (outer _PerfStageCollector span). Swap sub-stages (sc_*)
# are summed inside std_swap_edit wall time; do not add both into one “physics” budget.
PIPELINE_PROFILE_FRAME_TOTAL_KEY = "frame_total_attributed_ms"

# Unified UI: ``PipelineProfileDisplayModeSelection`` (legacy overlay/dock bools stay in sync).
PIPELINE_PROFILE_DISPLAY_MODE_OFF = "Off"
PIPELINE_PROFILE_DISPLAY_MODE_OVERLAY = "Preview overlay"
PIPELINE_PROFILE_DISPLAY_MODE_DOCK = "Dock panel"
PIPELINE_PROFILE_DISPLAY_MODE_BOTH = "Overlay + dock"

PIPELINE_PROFILE_DISPLAY_MODES: Tuple[str, ...] = (
    PIPELINE_PROFILE_DISPLAY_MODE_OFF,
    PIPELINE_PROFILE_DISPLAY_MODE_OVERLAY,
    PIPELINE_PROFILE_DISPLAY_MODE_DOCK,
    PIPELINE_PROFILE_DISPLAY_MODE_BOTH,
)

PIPELINE_PROFILE_DISPLAY_MODES_ACTIVE: frozenset[str] = frozenset(
    (
        PIPELINE_PROFILE_DISPLAY_MODE_OVERLAY,
        PIPELINE_PROFILE_DISPLAY_MODE_DOCK,
        PIPELINE_PROFILE_DISPLAY_MODE_BOTH,
    )
)


def sync_legacy_profile_bools_from_mode(control: Dict[str, Any], mode: str) -> None:
    """Mirror combobox state in ``PipelineProfile*EnableToggle`` for existing call sites."""
    m = (
        mode
        if mode in PIPELINE_PROFILE_DISPLAY_MODES
        else PIPELINE_PROFILE_DISPLAY_MODE_OFF
    )
    control["PipelineProfileDisplayModeSelection"] = m
    control["PipelineProfileOverlayEnableToggle"] = m in (
        PIPELINE_PROFILE_DISPLAY_MODE_OVERLAY,
        PIPELINE_PROFILE_DISPLAY_MODE_BOTH,
    )
    control["PipelineProfileDockEnableToggle"] = m in (
        PIPELINE_PROFILE_DISPLAY_MODE_DOCK,
        PIPELINE_PROFILE_DISPLAY_MODE_BOTH,
    )


def migrate_pipeline_profile_display_mode(control: Dict[str, Any]) -> None:
    """After loading JSON: ensure mode exists and legacy bool keys match."""
    cur = control.get("PipelineProfileDisplayModeSelection")
    if isinstance(cur, str) and cur in PIPELINE_PROFILE_DISPLAY_MODES:
        sync_legacy_profile_bools_from_mode(control, cur)
        return
    ov = bool(control.get("PipelineProfileOverlayEnableToggle", False))
    dk = bool(control.get("PipelineProfileDockEnableToggle", False))
    if ov and dk:
        m = PIPELINE_PROFILE_DISPLAY_MODE_BOTH
    elif ov:
        m = PIPELINE_PROFILE_DISPLAY_MODE_OVERLAY
    elif dk:
        m = PIPELINE_PROFILE_DISPLAY_MODE_DOCK
    else:
        m = PIPELINE_PROFILE_DISPLAY_MODE_OFF
    sync_legacy_profile_bools_from_mode(control, m)


# Cap stored samples per playback session (each displayed frame with overlay on).
_PIPELINE_PROFILE_SESSION_MAX = 8000
_PIPELINE_PROFILE_SESSION_REPORT_PREFIX = "[PIPELINE-PROFILE-SESSION]"

# Display labels for overlay (fallback: raw stage id).
PIPELINE_STAGE_LABELS: Dict[str, str] = {
    "read_frame_ms": "Feeder: read frame",
    "feeder_state_ms": "Feeder: state / markers",
    "rgb_pack_ms": "Feeder: RGB pack",
    "feeder_params_lock_ms": "Feeder: params lock",
    "sequential_detect_ms": "Feeder: detect (sequential)",
    "prep_scaling_h2d": "Worker: frame to GPU (H2D)",
    "vr180": "Worker: VR180 pipeline",
    "std_upscale_rotate": "Worker: scale / rotate",
    "std_detect_feeder_or_fallback": "Worker: face detect",
    "std_recognize": "Worker: recognition / embeddings",
    "std_swap_edit": "Worker: swap+edits (outer span)",
    "std_undo_resize": "Worker: undo working resize",
    "std_overlays_compare": "Worker: compare / mask preview",
    "frame_enhancer": "Worker: global frame enhancer",
    "d2h_numpy": "Worker: result GPU→CPU (numpy)",
    "pass_through": "Passthrough",
    "feeder_subtotal": "Sum Feeder (ms)",
    "worker_subtotal": "Sum Worker (ms)",
    "sc_align_crop": "Swap: tform + multi-res face crops",
    "sc_swap_strength": "Swap: swapper infer + strength blend",
    "sc_border_mask_init": "Swap: side+border masks, buf init",
    "sc_swap_after_border_fx": "Swap: expr/editor/denoiser/mouth",
    "sc_face_restorer_primary": "Swap: face restorer (pass 1)",
    "sc_occluder_matting": "Swap: occluder + RVM + U2Net",
    "sc_parser_clip_restore_mouth": "Swap: parser + CLIP + eyes/mouth",
    "sc_dfl_xseg_calc_masks": "Swap: DFL XSeg + merge calc masks",
    "sc_restore_color_fx": "Swap: color + FX + later restorers",
    "sc_perspective_out": "Swap: perspective early exit",
    "sc_tail_view_maskpost": "Swap: final mask blur + view buf",
    "sc_warp_paste": "Swap: inverse warp + ROI paste",
    # Legacy swap_core keys (pre-split); still shown if old logs/CSV contain them.
    "sc_maskcalc_xseg": "Swap: occ+parser+XSeg+mask (legacy)",
    "sc_masks_occl_parser_xseg_merge": "Swap: occ+parser+XSeg (legacy)",
    PIPELINE_PROFILE_FRAME_TOTAL_KEY: "Frame total (feeder ∑ + worker wall)",
}

# Column widths for monospace overlay (wider labels after swap_core split).
_OVERLAY_COL_LABEL = 36
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
    for k in sorted(
        kk for kk in all_keys if kk not in out and kk != PIPELINE_PROFILE_FRAME_TOTAL_KEY
    ):
        out.append(k)
    if PIPELINE_PROFILE_FRAME_TOTAL_KEY in all_keys:
        out.append(PIPELINE_PROFILE_FRAME_TOTAL_KEY)
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
    fta = payload.get(PIPELINE_PROFILE_FRAME_TOTAL_KEY)
    if fta is not None:
        try:
            rows.append((PIPELINE_PROFILE_FRAME_TOTAL_KEY, float(fta)))
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
    *,
    global_mean_column: bool = False,
) -> str:
    """Format pipeline timings: per-thread columns + Avg, or a single Mean column.

    When ``global_mean_column`` is True, only ``Stage | Mean`` is shown (mean across
    worker threads for each row — same numbers as the former Avg column).

    Feeder-stage rows are grouped first, then worker stages, with subtotals.
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
    worker_stages = [
        s
        for s in stages
        if s not in PIPELINE_PROFILE_FEEDER_KEY_SET
        and s != PIPELINE_PROFILE_FRAME_TOTAL_KEY
    ]
    stages_for_breakdown_sum = [
        s for s in stages if s != PIPELINE_PROFILE_FRAME_TOTAL_KEY
    ]

    cw_l = _OVERLAY_COL_LABEL
    cw = _OVERLAY_COL_MS_THREAD
    lines: List[str] = []
    if header_lines:
        lines.extend(header_lines)
    if global_mean_column:
        lines.append("Pipeline profile (ms) — mean across workers")
        hdr = f"{'Stage':<{cw_l}}  {'Mean':>{cw}}"
    else:
        lines.append("Pipeline profile (ms) — Feeder | Worker")
        hdr = f"{'Stage':<{cw_l}}"
        for t in threads_sorted:
            hdr += f"  {_short_thread_column_title(t):>{cw}}"
        hdr += f"  {'Avg':>{cw}}"
    lines.append(hdr)

    def _rule_under_header() -> None:
        """Light rule aligned with table width (monospace)."""
        w = cw_l + (len(threads_sorted) + 1) * (cw + 2) if not global_mean_column else cw_l + cw + 2
        lines.append("." * min(max(w, 24), 72))

    def _section_gap() -> None:
        lines.append("")

    _rule_under_header()
    _section_gap()

    def _sep_row(title: str) -> None:
        sep = f"{_overlay_fit_label(title):<{cw_l}}"
        if global_mean_column:
            sep += f"  {'':>{cw}}"
        else:
            for _t in threads_sorted:
                sep += f"  {'':>{cw}}"
            sep += f"  {'':>{cw}}"
        lines.append(sep)

    def _append_stage_block(stage_list: List[str], title: str | None) -> None:
        if not stage_list:
            return
        if title:
            _sep_row(title)
        for stage in stage_list:
            label = _overlay_fit_label(PIPELINE_STAGE_LABELS.get(stage, stage))
            row = f"{label:<{cw_l}}"
            vals: List[float] = []
            for t in threads_sorted:
                d = per_thread[t]
                v = d.get(stage)
                if v is not None:
                    vals.append(float(v))
                if not global_mean_column:
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
                if not global_mean_column:
                    row += f"  {ssum:>{cw}.1f}"
            elif not global_mean_column:
                row += f"  {'—':>{cw}}"
        avg_v = sum(col_totals) / len(col_totals) if col_totals else None
        row += (
            f"  {avg_v:>{cw}.1f}"
            if avg_v is not None
            else f"  {'—':>{cw}}"
        )
        lines.append(row)

    _append_stage_block(feeder_stages, "── Feeder ──" if feeder_stages else None)
    _append_subtotal_row("feeder_subtotal", feeder_stages)
    _section_gap()
    _append_stage_block(worker_stages, "── Worker ──" if worker_stages else None)
    _append_subtotal_row("worker_subtotal", worker_stages)
    _section_gap()
    if PIPELINE_PROFILE_FRAME_TOTAL_KEY in stages:
        _sep_row("── Frame total ──")
        _append_stage_block([PIPELINE_PROFILE_FRAME_TOTAL_KEY], None)
        _section_gap()

    _sep_row("── Totals ──")
    row = f"{'Total (breakdown ∑)':<{cw_l}}"
    totals: List[float] = []
    for t in threads_sorted:
        d = per_thread[t]
        tot = sum(float(d.get(s, 0.0)) for s in stages_for_breakdown_sum)
        totals.append(tot)
        if not global_mean_column:
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
    """EMA or window smoothing per thread; overlay table per-thread + Avg or mean-only."""
    display: Dict[str, Dict[str, float]] = getattr(
        main_window, "_pipeline_profile_display_by_thread", None
    )
    if display is None:
        display = {}
        main_window._pipeline_profile_display_by_thread = display

    wt = (worker_thread or "").strip() or "?"

    ctrl = main_window.control
    global_mean = bool(
        ctrl.get("PipelineProfileOverlayGlobalMeanColumnToggle", False)
    )

    if not rows:
        return (
            format_profile_overlay_multithread(
                display,
                header_lines=header_lines,
                global_mean_column=global_mean,
            )
            if display
            else "Profile: —"
        )

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

    return format_profile_overlay_multithread(
        display,
        header_lines=header_lines,
        global_mean_column=global_mean,
    )


def reset_pipeline_profile_state(main_window: Any) -> None:
    """Clear per-thread EMA/window history (e.g. on stop or new media)."""
    main_window._pipeline_profile_ema_by_thread = {}
    main_window._pipeline_profile_window_deques = {}
    main_window._pipeline_profile_display_by_thread = {}
    main_window._pipeline_profile_last_overlay_headers = []

def clear_pipeline_profile_session_samples(main_window: Any) -> None:
    """Empty session log at playback start (overlay samples for console report)."""
    main_window._pipeline_profile_session_samples = deque()


def pipeline_profile_ui_timings_enabled(main_window: Any) -> bool:
    """True when feeder/worker should attach timing dicts for overlay or dock."""
    return bool(
        main_window.control.get("PipelineProfileOverlayEnableToggle", False)
    ) or bool(main_window.control.get("PipelineProfileDockEnableToggle", False))


def append_pipeline_profile_session_sample(
    main_window: Any,
    profile_payload: dict[str, Any],
    rows: List[Tuple[str, float]],
) -> None:
    """Record one frame while the pipeline profile overlay or dock is enabled."""
    if not pipeline_profile_ui_timings_enabled(main_window):
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

