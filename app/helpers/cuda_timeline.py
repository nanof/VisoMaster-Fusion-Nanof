"""
Optional CUDA timeline markers for NVIDIA Nsight Systems.

Set VISIOMASTER_NVTX=1, then capture the process with Nsight Systems to see
feeder vs worker ranges on the GPU timeline.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator


def nvtx_enabled() -> bool:
    return os.environ.get("VISIOMASTER_NVTX", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    if not nvtx_enabled():
        yield
        return
    try:
        import torch

        if not torch.cuda.is_available():
            yield
            return
        nv = getattr(torch.cuda, "nvtx", None)
        if nv is None:
            yield
            return
        range_push = getattr(nv, "range_push", None)
        range_pop = getattr(nv, "range_pop", None)
        if range_push is None or range_pop is None:
            yield
            return
    except Exception:
        yield
        return
    try:
        range_push(name)
        yield
    finally:
        try:
            range_pop()
        except Exception:
            pass
