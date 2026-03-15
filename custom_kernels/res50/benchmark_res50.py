"""
Benchmark — res50 (FaceLandmark5 / RetinaFace) custom kernels vs ORT baseline.

Tests the full RetinaFace pipeline:
  Input : (1, 3, 512, 512) float32
  Outputs: conf (1, 10752, 2), landmarks (1, 10752, 10) float32

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\res50\\benchmark_res50.py

Optional env vars:
    ONNX_DIR=model_assets          path to ONNX model files
    WARMUP=10                      warm-up iterations
    ITERS=50                       timed iterations
"""
from __future__ import annotations

import os
import sys
import time
import pathlib

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT      = pathlib.Path(__file__).parent.parent.parent
ONNX_DIR  = pathlib.Path(os.environ.get("ONNX_DIR", ROOT / "model_assets"))
WARMUP    = int(os.environ.get("WARMUP", 10))
ITERS     = int(os.environ.get("ITERS",  50))

R50_ONNX  = str(ONNX_DIR / "res50.onnx")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ms(elapsed_s: float, iters: int) -> float:
    return elapsed_s / iters * 1000.0


def _bench(fn, warmup=WARMUP, iters=ITERS) -> float:
    """Return ms/iter (GPU-synchronised)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return _ms(time.perf_counter() - t0, iters)


def _ort_session(onnx_path: str, provider: str = "CUDAExecutionProvider"):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=[provider])


def _print_row(tier, label, ms_val, ref_ms):
    speedup = ref_ms / ms_val if ms_val > 0 else float("nan")
    print(f"  Tier {tier} | {label:<44} | {ms_val:8.2f} ms | {speedup:6.2f}x")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def bench_res50():
    print("\n=== res50 / FaceLandmark5 (1,3,512,512) → conf + landmarks ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<44} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-'*6}-+-{'-'*44}-+-{'-'*8}-+-{'-'*7}")

    inp    = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")
    inp_np = inp.cpu().numpy()

    # ── Tier 0 — ORT FP32 CUDA EP ────────────────────────────────────────
    import numpy as np
    sess0    = _ort_session(R50_ONNX, "CUDAExecutionProvider")
    in_name  = sess0.get_inputs()[0].name      # "input"
    out_names = [o.name for o in sess0.get_outputs()]
    t0 = _bench(lambda: sess0.run(out_names, {in_name: inp_np}))
    _print_row("0", "ORT FP32 CUDA EP", t0, t0)

    # ── Tier 0b — ORT TensorRT EP ─────────────────────────────────────────
    t0b = t0
    try:
        import tensorrt  # registers nvinfer DLL path on Windows
    except Exception:
        pass
    import onnxruntime as _ort
    if "TensorrtExecutionProvider" not in _ort.get_available_providers():
        print(f"  Tier 0b | TensorRT EP — skipped (TensorrtExecutionProvider not available)")
    else:
        ctx = ROOT / "tensorrt-engines" / "res50_ctx.onnx"
        if not ctx.exists():
            print(f"  Tier 0b | TensorRT EP — skipped (no pre-built engine: {ctx.name})")
        else:
            import os as _os
            _prev_cwd = _os.getcwd()
            _os.chdir(str(ROOT))
            trt_opts = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "tensorrt-engines",
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": "tensorrt-engines",
                "trt_dump_ep_context_model": True,
                "trt_ep_context_file_path": "tensorrt-engines",
                "trt_layer_norm_fp32_fallback": True,
                "trt_max_workspace_size": 8589934592,
                "trt_builder_optimization_level": 5,
            }
            so = _ort.SessionOptions()
            so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess0b = _ort.InferenceSession(R50_ONNX, so, providers=[
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ])
            _os.chdir(_prev_cwd)
            t0b = _bench(lambda: sess0b.run(out_names, {in_name: inp_np}))
            _print_row("0b", "ORT TensorRT EP FP32 (app default)", t0b, t0)

    # ── Tier 1 — PyTorch FP32 ────────────────────────────────────────────
    sys.path.insert(0, str(ROOT))
    from custom_kernels.res50.res50_torch import Res50Torch, build_cuda_graph_runner
    r50_fp32 = Res50Torch.from_onnx(R50_ONNX, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: r50_fp32(inp))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # ── Tier 2 — PyTorch FP16 ─────────────────────────────────────────────
    r50_fp16 = Res50Torch.from_onnx(R50_ONNX, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: r50_fp16(inp))
    _print_row("2", "PyTorch FP16 (TensorCore conv dispatch)", t2, t0)

    # ── Tier 3 — FP16 + CUDA graph ────────────────────────────────────────
    try:
        runner = build_cuda_graph_runner(r50_fp16)
        with torch.no_grad():
            _ = runner(inp)  # ensure graph is ready
        t3 = _bench(lambda: runner(inp))
        _print_row("3", "FP16 + CUDA graph", t3, t0)
    except Exception as e:
        print(f"  Tier 3 | CUDA graph — skipped ({e})")

    print()
    return t0, t0b


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    print(f"\nDevice: {dev}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import onnxruntime as ort
        print(f"ORT:     {ort.__version__}")
    except ImportError:
        print("ORT:     not available (skipping Tier 0/0b)")
        sys.exit(1)

    if not pathlib.Path(R50_ONNX).exists():
        print(f"ERROR: res50 ONNX not found: {R50_ONNX}")
        sys.exit(1)

    bench_res50()
