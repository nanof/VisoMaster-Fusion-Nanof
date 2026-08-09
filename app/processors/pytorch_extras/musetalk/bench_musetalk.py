"""
Microbench MuseTalk hot path: VAE encode / UNet / VAE decode.

Priority is FPS (ms/frame and equiv. FPS). Run from repo root with CUDA + weights:

    python -m app.processors.pytorch_extras.musetalk.bench_musetalk

Env:
    VISOFUSION_MUSETALK_BATCH   batch sizes to time (default: 1,8)
    WARMUP / ITERS              iterations (defaults 5 / 20)
    VISOFUSION_MUSETALK_COMPILE forced on/off per mode by this script
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _ms(values: list[float]) -> float:
    return statistics.median(values) * 1000.0 if values else float("nan")


def _vram_mb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_stage(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    samples: list[float] = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return _ms(samples)


def _run_mode(
    *,
    compile_on: bool,
    batch_sizes: list[int],
    warmup: int,
    iters: int,
) -> None:
    os.environ["VISOFUSION_MUSETALK_COMPILE"] = "1" if compile_on else "0"

    from app.processors.pytorch_extras.musetalk import MuseTalkEngine
    from app.processors.pytorch_extras.musetalk.paths import musetalk_assets_ready

    if not musetalk_assets_ready():
        print("SKIP: MuseTalk weights missing under model_assets/musetalk/")
        return

    eng = MuseTalkEngine()
    if not eng.load(use_float16=True):
        print(f"SKIP: load failed: {eng._last_error}")
        return

    import torch

    vram = _vram_mb()
    label = "compile=on" if eng._compiled else (
        "compile=requested" if compile_on else "compile=off"
    )
    if compile_on and not eng._compiled:
        label = "compile=requested(fallback-eager)"
    print(f"\n=== MuseTalk bench ({label}) ===")
    print(f"  device={eng._device}  dtype={eng._dtype}  channels_last={eng._channels_last}")
    print(f"  VRAM after load: {vram:.0f} MiB")
    print(
        f"  {'batch':>5} | {'encode':>8} | {'unet':>8} | {'decode':>8} | "
        f"{'e2e':>8} | {'ms/frm':>8} | {'FPS~':>7}"
    )
    print(
        f"  {'-' * 5}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-"
        f"{'-' * 8}-+-{'-' * 8}-+-{'-' * 7}"
    )

    rng = np.random.default_rng(0)
    crops_cache = {
        n: [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(n)]
        for n in batch_sizes
    }
    # Synthetic audio windows matching Whisper-tiny MuseTalk layout (B, 50, 384).
    audio_cache = {
        n: torch.zeros((n, 50, 384), device=eng._device, dtype=eng._dtype)
        for n in batch_sizes
    }

    vae, unet, pe = eng._vae, eng._unet, eng._pe
    assert vae is not None and unet is not None and pe is not None

    for n in batch_sizes:
        crops = crops_cache[n]
        audio = audio_cache[n]

        def encode():
            return vae.get_latents_for_unet_batch(crops)

        latents_holder: dict[str, object] = {}

        def encode_keep():
            lat = encode()
            if eng._channels_last and lat.is_cuda:
                lat = lat.to(memory_format=torch.channels_last)
            latents_holder["lat"] = lat
            return lat

        encode_keep()
        lat0 = latents_holder["lat"]

        def unet_fwd():
            lat = latents_holder.get("lat", lat0)
            feat = pe(audio).to(unet.model.dtype)
            return unet.model(
                lat.to(device=eng._device, dtype=unet.model.dtype),
                eng._timesteps,
                encoder_hidden_states=feat,
            ).sample

        pred0 = unet_fwd()

        def decode():
            return vae.decode_latents(pred0.to(dtype=vae.vae.dtype))

        def e2e():
            lat = encode_keep()
            feat = pe(audio).to(unet.model.dtype)
            pred = unet.model(
                lat.to(device=eng._device, dtype=unet.model.dtype),
                eng._timesteps,
                encoder_hidden_states=feat,
            ).sample
            return vae.decode_latents(pred.to(dtype=vae.vae.dtype))

        enc_ms = _time_stage(encode_keep, warmup=warmup, iters=iters)
        unet_ms = _time_stage(unet_fwd, warmup=warmup, iters=iters)
        dec_ms = _time_stage(decode, warmup=warmup, iters=iters)
        e2e_ms = _time_stage(e2e, warmup=warmup, iters=iters)
        per = e2e_ms / n
        fps = 1000.0 / per if per > 0 else float("nan")
        print(
            f"  {n:5d} | {enc_ms:8.2f} | {unet_ms:8.2f} | {dec_ms:8.2f} | "
            f"{e2e_ms:8.2f} | {per:8.2f} | {fps:7.1f}"
        )

    eng.unload()


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device (MuseTalk FPS bench needs a GPU).")
        return 0

    warmup = int(os.environ.get("WARMUP", "5"))
    iters = int(os.environ.get("ITERS", "20"))
    raw_batches = os.environ.get("VISOFUSION_MUSETALK_BATCH", "1,8")
    batch_sizes = []
    for part in raw_batches.split(","):
        part = part.strip()
        if part:
            batch_sizes.append(max(1, int(part)))
    if not batch_sizes:
        batch_sizes = [1, 8]

    print(f"MuseTalk microbench  warmup={warmup} iters={iters} batches={batch_sizes}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Baseline first, then compile — so a warm GPU does not favour the second run.
    _run_mode(
        compile_on=False, batch_sizes=batch_sizes, warmup=warmup, iters=iters
    )
    _run_mode(
        compile_on=True, batch_sizes=batch_sizes, warmup=warmup, iters=iters
    )
    print(
        "\nPaste the tables into docs/model_viability/MUSETALK_TODO.md "
        "(Optimizations / VRAM profile)."
    )
    return 0


if __name__ == "__main__":
    # Ensure repo root is on sys.path when run as a file.
    root = Path(__file__).resolve().parents[4]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    raise SystemExit(main())
