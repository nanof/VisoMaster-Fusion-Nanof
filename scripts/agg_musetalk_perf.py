"""Aggregate [MUSETALK-PERF] lines from a terminal/log file."""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "-")
    text = sys.stdin.read() if str(path) == "-" else path.read_text(encoding="utf-8", errors="replace")
    keys = [
        "crop",
        "batch_wait",
        "encode",
        "unet",
        "decode",
        "infer",
        "batch",
        "restore",
        "parse",
        "mouth_only",
        "blend",
        "total",
    ]
    vals: dict[str, list[float]] = defaultdict(list)
    n = 0
    for line in text.splitlines():
        if "[MUSETALK-PERF]" not in line:
            continue
        n += 1
        for k in keys:
            m = re.search(rf"{k}=([\d.]+)", line)
            if m:
                vals[k].append(float(m.group(1)))
    print(f"n={n}")
    if not n:
        return 1
    total_med = statistics.median(vals["total"])
    print(
        f"total median={total_med:.1f} mean={statistics.mean(vals['total']):.1f}"
    )
    print(
        f"batch size median={statistics.median(vals['batch']):.0f} "
        f"mean={statistics.mean(vals['batch']):.2f}"
    )
    over = [w - i for w, i in zip(vals["batch_wait"], vals["infer"])]
    print(
        f"batch_wait-infer (queue/gather/serialization) median={statistics.median(over):.1f}"
    )
    print("--- medians ms ---")
    for k in [
        "crop",
        "batch_wait",
        "encode",
        "unet",
        "decode",
        "infer",
        "restore",
        "parse",
        "mouth_only",
        "blend",
        "total",
    ]:
        med = statistics.median(vals[k])
        mean = statistics.mean(vals[k])
        pct = 100 * med / total_med if total_med else 0
        print(f"{k:12s} median={med:7.1f} mean={mean:7.1f} (~{pct:.0f}% of apply total)")
    bw = statistics.median(vals["batch_wait"])
    print("--- share of batch_wait ---")
    for k in ["encode", "unet", "decode", "infer"]:
        med = statistics.median(vals[k])
        print(f"{k:12s} {med:7.1f} ms  (~{100 * med / bw:.0f}% of batch_wait)")
    print("--- post-GPU worker work ---")
    for k in ["restore", "parse", "mouth_only", "blend"]:
        med = statistics.median(vals[k])
        print(f"{k:12s} {med:7.1f} ms  (~{100 * med / total_med:.0f}% of apply total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
