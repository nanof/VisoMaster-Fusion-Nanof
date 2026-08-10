"""Time a MuseTalk engine load, so the torch.compile warmup cost is measurable.

Run once per process: the Inductor cache is warm in-process, so a second load in
the same interpreter is not comparable to a cold app start.

    python -m scripts.time_musetalk_load          # honours the current env
    VISOFUSION_MUSETALK_COMPILE=1 python -m scripts.time_musetalk_load
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.processors.pytorch_extras.musetalk import (  # noqa: E402
    MuseTalkEngine,
    musetalk_compile_enabled,
)


def main() -> int:
    flag = os.environ.get("VISOFUSION_MUSETALK_COMPILE", "<unset>")
    print(f"VISOFUSION_MUSETALK_COMPILE={flag} -> compile={musetalk_compile_enabled()}")

    engine = MuseTalkEngine()
    t0 = time.perf_counter()
    ok = engine.load(use_float16=True)
    elapsed = time.perf_counter() - t0
    print(f"loaded={ok} compiled={engine._compiled} in {elapsed:.1f} s")
    if not ok:
        print(f"error: {engine._last_error}")
    engine.unload()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
