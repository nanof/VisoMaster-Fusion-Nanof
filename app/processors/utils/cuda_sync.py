"""CUDA host-sync helpers that avoid spin-waiting worker cores.

``cudaStreamSynchronize`` under CUDA's default schedule polls a flag, so every
worker blocked on inference can burn a core. An event created with
``cudaEventBlockingSync`` (``torch.cuda.Event(blocking=True)``) waits on an OS
primitive instead.

Set ``VISOMASTER_CUDA_SPIN_SYNC=1`` to restore the old spin-wait behaviour for
A/B FPS comparisons on the same build.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import cuda as torch_cuda

_SPIN_SYNC: bool = os.environ.get("VISOMASTER_CUDA_SPIN_SYNC", "0") == "1"
_TLS = threading.local()
_CUDA_OK: bool | None = None


def _cuda_ok() -> bool:
    """Cached ``torch.cuda.is_available()``. Called on the per-inference hot path."""
    global _CUDA_OK
    if _CUDA_OK is None:
        try:
            _CUDA_OK = bool(torch.cuda.is_available())
        except Exception:
            _CUDA_OK = False
    return _CUDA_OK


def spin_sync_enabled() -> bool:
    """True when the legacy spin-wait sync policy has been forced on."""
    return _SPIN_SYNC


def blocking_stream_sync(stream: "torch_cuda.Stream | None" = None) -> None:
    """Host-wait for a CUDA stream without spinning a core.

    Equivalent to ``stream.synchronize()`` — or
    ``torch.cuda.current_stream().synchronize()`` when *stream* is ``None`` —
    except that the calling thread sleeps instead of polling.

    No-op when CUDA is unavailable, so call sites do not need to guard.
    """
    if not _cuda_ok():
        return

    if _SPIN_SYNC:
        if stream is None:
            torch.cuda.current_stream().synchronize()
        else:
            stream.synchronize()
        return

    # The event is cached per thread: re-recording resets it, so one object per
    # worker serves every sync that worker performs and we avoid allocating a
    # CUDA event on the hot path.
    ev = getattr(_TLS, "sync_event", None)
    if ev is None:
        ev = torch.cuda.Event(blocking=True)
        _TLS.sync_event = ev
    ev.record(stream)
    ev.synchronize()
