"""
Benchmark — 1k3d68 custom kernel vs ORT baseline.

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\1k3d68\\benchmark_1k3d68.py

Optional env vars:
    ONNX_PATH=model_assets/1k3d68.onnx   path to ONNX model
    WARMUP=50                             warm-up iterations
    ITERS=500                             timed iterations
"""
from __future__ import annotations

import os
import sys
import time
import pathlib

import numpy as np
import torch

# ---------------------------------------------------------------------------
ROOT      = pathlib.Path(__file__).parent.parent.parent
ONNX_PATH = pathlib.Path(os.environ.get("ONNX_PATH", ROOT / "model_assets" / "1k3d68.onnx"))
WARMUP    = int(os.environ.get("WARMUP", 50))
ITERS     = int(os.environ.get("ITERS",  500))

sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------

def _bench(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return mean ms/iteration (GPU-synchronised wall-clock)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _ort_session(path: str, provider: str = "CUDAExecutionProvider"):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=[provider])


def _print_row(tier, label, ms_val, ref_ms):
    speedup = ref_ms / ms_val if ms_val > 0 else float("nan")
    print(f"  Tier {tier} | {label:<44} | {ms_val:8.3f} ms | {speedup:6.2f}×")


# ---------------------------------------------------------------------------

def bench():
    print(f"\n=== 1k3d68 face landmark detector — (1,3,192,192) → (1,3309) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<44} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-'*6}-+-{'-'*44}-+-{'-'*8}-+-{'-'*7}")

    inp    = torch.randn(1, 3, 192, 192, dtype=torch.float32, device="cuda")
    inp_np = inp.cpu().numpy()

    # ── Tier 0: ORT FP32 CUDA EP ─────────────────────────────────────────
    sess0    = _ort_session(ONNX_PATH, "CUDAExecutionProvider")
    in_name  = sess0.get_inputs()[0].name
    out_name = sess0.get_outputs()[0].name
    t0 = _bench(lambda: sess0.run([out_name], {in_name: inp_np}))
    _print_row("0", "ORT FP32 CUDA EP (baseline)", t0, t0)

    # ── Tier 0b: ORT TensorRT EP ─────────────────────────────────────────
    t0b = t0
    try:
        import tensorrt  # registers nvinfer DLL path on Windows
    except Exception:
        pass
    import onnxruntime as _ort
    if "TensorrtExecutionProvider" not in _ort.get_available_providers():
        print(f"  Tier 0b | TensorRT EP — skipped (TensorrtExecutionProvider not available)")
    else:
        ctx = ROOT / "tensorrt-engines" / "1k3d68_ctx.onnx"
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
            sess0b = _ort.InferenceSession(str(ONNX_PATH), so, providers=[
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ])
            _os.chdir(_prev_cwd)
            t0b = _bench(lambda: sess0b.run([out_name], {in_name: inp_np}))
            _print_row("0b", "ORT TensorRT EP FP32", t0b, t0)

    # ── Tier 1: PyTorch FP32 ─────────────────────────────────────────────
    from custom_kernels.landmark_1k3d68.landmark_1k3d68_torch import Landmark1k3d68Torch
    m_fp32 = Landmark1k3d68Torch.from_onnx(ONNX_PATH, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: m_fp32(inp))
    _print_row("1", "PyTorch FP32 eager", t1, t0)

    # ── Tier 2: PyTorch FP16 ─────────────────────────────────────────────
    m_fp16 = Landmark1k3d68Torch.from_onnx(ONNX_PATH, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: m_fp16(inp))
    _print_row("2", "PyTorch FP16 eager", t2, t0)

    # ── Tier 3: PyTorch FP16 + CUDA graph ────────────────────────────────
    from custom_kernels.landmark_1k3d68.landmark_1k3d68_torch import build_cuda_graph_runner
    runner = build_cuda_graph_runner(m_fp16)
    t3 = _bench(lambda: runner(inp))
    _print_row("3", "PyTorch FP16 + CUDA graph", t3, t0)

    print()

    # ── Numerical accuracy check ──────────────────────────────────────────
    print("=== Numerical accuracy (ORT FP32 vs PyTorch FP16 + CUDA graph) ===")
    ref_np = sess0.run([out_name], {in_name: inp_np})[0][0]        # (3309,)
    with torch.no_grad():
        pt_out = runner(inp)[0].cpu().numpy()                       # (3309,)
    max_err  = float(np.abs(ref_np - pt_out).max())
    mean_err = float(np.abs(ref_np - pt_out).mean())
    print(f"  max  |Δ| = {max_err:.4f}")
    print(f"  mean |Δ| = {mean_err:.6f}")
    if max_err < 0.5:
        print("  ✓ Accuracy OK (max error < 0.5)")
    else:
        print("  ✗ Accuracy WARNING — large deviation detected")
    print()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"\nDevice : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import onnxruntime as ort
        print(f"ORT    : {ort.__version__}")
    except ImportError:
        print("ORT    : not available — install onnxruntime-gpu")
        sys.exit(1)

    if not ONNX_PATH.exists():
        print(f"ERROR: ONNX not found: {ONNX_PATH}")
        sys.exit(1)

    bench()
