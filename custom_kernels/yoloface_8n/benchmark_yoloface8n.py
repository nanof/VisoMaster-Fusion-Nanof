"""
Benchmark — yoloface_8n (YOLOv8n-face detector) custom kernels vs ORT baseline.

Tests the full YOLOv8n-face pipeline:
  Input : (1, 3, 640, 640) float32 in [0, 1]
  Output: (1, 20, 8400) float32  — fully decoded [cx,cy,w,h, conf, kps×15]

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\yoloface_8n\\benchmark_yoloface8n.py

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

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT     = pathlib.Path(__file__).parent.parent.parent
ONNX_DIR = pathlib.Path(os.environ.get("ONNX_DIR", ROOT / "model_assets"))
WARMUP   = int(os.environ.get("WARMUP", 10))
ITERS    = int(os.environ.get("ITERS",  50))

YOLO_ONNX = str(ONNX_DIR / "yoloface_8n.onnx")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ms(elapsed_s: float, iters: int) -> float:
    return elapsed_s / iters * 1000.0


def _bench(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
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
    print(f"  Tier {tier} | {label:<46} | {ms_val:8.2f} ms | {speedup:6.2f}x")


# ---------------------------------------------------------------------------
# Numerical accuracy check
# ---------------------------------------------------------------------------
def _check_accuracy(ort_out: np.ndarray, pt_out: torch.Tensor) -> None:
    """Compare ORT FP32 output vs PyTorch FP16 output."""
    a = ort_out.astype(np.float32)                # (1, 20, 8400)
    b = pt_out.cpu().numpy().astype(np.float32)
    if a.shape != b.shape:
        print(f"  Shape mismatch: ORT={a.shape} vs PT={b.shape}")
        return

    # Per-row check: [bbox(4), cls(1), kps(15)] — report per group
    groups = [("bbox [0:4]", slice(0, 4)), ("cls  [4]",  slice(4, 5)),
              ("kps  [5:]",  slice(5, 20))]
    print("\n  Numerical accuracy (FP16 PyTorch vs ORT FP32):")
    for name, sl in groups:
        diff = np.abs(a[0, sl, :] - b[0, sl, :])
        print(f"    {name}: MAE={diff.mean():.2e}  MaxAbsErr={diff.max():.2e}")
    overall = np.abs(a - b).max()
    print(f"  Overall max abs error: {overall:.2e}  (target < 1e-2)")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def bench_yoloface8n():
    print(f"\n=== yoloface_8n / YOLOv8n-face  (1,3,640,640) → (1,20,8400) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<46} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-'*6}-+-{'-'*46}-+-{'-'*8}-+-{'-'*7}")

    inp_f32 = torch.rand(1, 3, 640, 640, dtype=torch.float32, device="cuda")
    inp_np  = inp_f32.cpu().numpy()

    # ── Tier 0 — ORT FP32 CUDA EP ────────────────────────────────────────
    sess0 = _ort_session(YOLO_ONNX, "CUDAExecutionProvider")
    in_name = sess0.get_inputs()[0].name   # "images"
    t0 = _bench(lambda: sess0.run(["output0"], {in_name: inp_np}))
    _print_row("0", "ORT FP32 CUDA EP", t0, t0)

    # ── Tier 0b — ORT TensorRT EP ─────────────────────────────────────────
    t0b = t0
    try:
        import tensorrt  # registers nvinfer DLL path on Windows
    except Exception:
        pass
    import onnxruntime as ort
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        print(f"  Tier 0b | TensorRT EP — skipped (TensorrtExecutionProvider not available)")
    else:
        ctx = ROOT / "tensorrt-engines" / "yoloface_8n_ctx.onnx"
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
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess0b = ort.InferenceSession(YOLO_ONNX, so, providers=[
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ])
            _os.chdir(_prev_cwd)
            t0b = _bench(lambda: sess0b.run(["output0"], {in_name: inp_np}))
            _print_row("0b", "ORT TensorRT EP FP32 (app default)", t0b, t0)

    # ── Tier 1 — PyTorch FP32 ────────────────────────────────────────────
    sys.path.insert(0, str(ROOT))
    from custom_kernels.yoloface_8n.yoloface8n_torch import (
        YoloFace8nTorch, build_cuda_graph_runner,
    )

    m_fp32 = YoloFace8nTorch.from_onnx(YOLO_ONNX, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: m_fp32(inp_f32))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # ── Tier 2 — PyTorch FP16 ─────────────────────────────────────────────
    m_fp16 = YoloFace8nTorch.from_onnx(YOLO_ONNX, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: m_fp16(inp_f32))
    _print_row("2", "PyTorch FP16 (TensorCore conv dispatch)", t2, t0)

    # ── Accuracy check ────────────────────────────────────────────────────
    ort_out = sess0.run(["output0"], {in_name: inp_np})[0]   # (1, 20, 8400)
    with torch.no_grad():
        pt_out = m_fp16(inp_f32)
    _check_accuracy(ort_out, pt_out)

    # ── Tier 3 — FP16 + CUDA graph ────────────────────────────────────────
    try:
        runner = build_cuda_graph_runner(m_fp16)
        with torch.no_grad():
            _ = runner(inp_f32)   # trigger first-call (already captured in constructor)
        t3 = _bench(lambda: runner(inp_f32))
        _print_row("3", "FP16 + CUDA graph (single captured graph)", t3, t0)
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

    if not pathlib.Path(YOLO_ONNX).exists():
        print(f"ERROR: yoloface_8n ONNX not found: {YOLO_ONNX}")
        sys.exit(1)

    bench_yoloface8n()
