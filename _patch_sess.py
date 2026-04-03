# Patch pipeline_profile_actions.py: add session report helpers before EOF (after reset_pipeline_profile_state block)
from pathlib import Path
p = Path("app/ui/widgets/actions/pipeline_profile_actions.py")
t = p.read_text(encoding="utf-8")
ins1 = '''PIPELINE_PROFILE_FEEDER_KEY_SET: frozenset[str] = frozenset(PIPELINE_PROFILE_FEEDER_ORDER)

# Display labels for overlay (fallback: raw stage id).'''
if ins1 not in t:
    raise SystemExit('insert1 not found')
t = t.replace(ins1, '''PIPELINE_PROFILE_FEEDER_KEY_SET: frozenset[str] = frozenset(PIPELINE_PROFILE_FEEDER_ORDER)

# Cap stored samples per playback session (each displayed frame with overlay on).
_PIPELINE_PROFILE_SESSION_MAX = 8000
_PIPELINE_PROFILE_SESSION_REPORT_PREFIX = "[PIPELINE-PROFILE-SESSION]"

# Display labels for overlay (fallback: raw stage id).''', 1)

old_end = '''def reset_pipeline_profile_state(main_window: Any) -> None:
    """Clear per-thread EMA/window history (e.g. on stop or new media)."""
    main_window._pipeline_profile_ema_by_thread = {}
    main_window._pipeline_profile_window_deques = {}
    main_window._pipeline_profile_display_by_thread = {}'''

new_end = old_end + '''


def clear_pipeline_profile_session_samples(main_window: Any) -> None:
    """Empty session log at playback start (overlay samples for console report)."""
    main_window._pipeline_profile_session_samples = deque()


def append_pipeline_profile_session_sample(
    main_window: Any,
    profile_payload: dict[str, Any],
    rows: List[Tuple[str, float]],
) -> None:
    """Record one frame's flattened timings while the pipeline profile overlay is enabled."""
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
    while len(samples) > _PIPELINE_PROFILE_SESSION_MAX:
        samples.popleft()


def print_pipeline_profile_session_report(main_window: Any) -> None:
    """After stop: print min/avg/max per stage to stdout for A/B comparisons (easy grep prefix)."""
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
    print("\\n".join(lines), flush=True)'''

if old_end not in t:
    raise SystemExit('old_end not found')
t = t.replace(old_end, new_end, 1)
p.write_text(t, encoding="utf-8')
print('pipeline_profile_actions patched')
