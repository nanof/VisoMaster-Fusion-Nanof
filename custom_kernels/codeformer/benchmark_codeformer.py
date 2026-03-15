"""
Benchmark — CodeFormer custom kernels vs ORT baseline.

Tests the full CodeFormer pipeline:
  Input : (1, 3, 512, 512) float32
  Output: (1, 3, 512, 512) float32

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\codeformer\\benchmark_codeformer.py

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

CF_ONNX   = str(ONNX_DIR / "codeformer_fp16.onnx")

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
def bench_codeformer():
    print("\n=== CodeFormer (1,3,512,512) -> (1,3,512,512) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<44} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-'*6}-+-{'-'*44}-+-{'-'*8}-+-{'-'*7}")

    inp     = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")
    inp_np  = inp.cpu().numpy()
    w_val   = 0.5  # fidelity weight for all tiers

    # ── Tier 0 — ORT FP32 CUDA EP ────────────────────────────────────────
    import numpy as np
    sess0   = _ort_session(CF_ONNX, "CUDAExecutionProvider")
    in_name = sess0.get_inputs()[0].name     # "x"
    w_name  = sess0.get_inputs()[1].name     # "w"  (float64 scalar)
    out_name = sess0.get_outputs()[0].name
    w_np    = np.array([w_val], dtype=np.float64)
    t0 = _bench(lambda: sess0.run([out_name], {in_name: inp_np, w_name: w_np}))
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
        ctx = ROOT / "tensorrt-engines" / "codeformer_fp16_ctx.onnx"
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
            sess0b = _ort.InferenceSession(CF_ONNX, so, providers=[
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ])
            _os.chdir(_prev_cwd)
            t0b = _bench(lambda: sess0b.run([out_name], {in_name: inp_np, w_name: w_np}))
            _print_row("0b", "ORT TensorRT EP FP32 (app default)", t0b, t0)

    # ── Tier 1 — PyTorch FP32 ────────────────────────────────────────────
    sys.path.insert(0, str(ROOT))
    from custom_kernels.codeformer.codeformer_torch import (
        CodeFormerTorch, build_cuda_graph_runner,
    )
    cf_fp32 = CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: cf_fp32(inp, fidelity_weight=w_val))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # ── Tier 2 — PyTorch FP16 + Triton GroupNorm+SiLU ─────────────────────
    cf_fp16 = CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: cf_fp16(inp, fidelity_weight=w_val))
    _print_row("2", "PyTorch FP16 + Triton GroupNorm+SiLU", t2, t0)

    # ── Tier 3 — FP16 + Triton + CUDA graph (fixed w=0.5) ─────────────────
    t3 = t2
    try:
        runner3 = build_cuda_graph_runner(cf_fp16, inp_shape=(1, 3, 512, 512))
        with torch.no_grad():
            _ = runner3(inp)
        t3 = _bench(lambda: runner3(inp))
        _print_row("3", "FP16 + Triton + CUDA graph", t3, t0)
    except Exception as e:
        print(f"  Tier 3 | CUDA graph -- skipped ({e})")

    # ── Tier 4 — FP16 + Triton + SDPA 4D + GEMM + CUDA graph ──────────────
    t4 = t3
    try:
        cf_gemm = CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float16).cuda().eval()
        cf_gemm.to_gemm_mode()
        with torch.no_grad():
            _ = cf_gemm(inp, fidelity_weight=w_val)   # warm up JIT / cuDNN
        runner4 = build_cuda_graph_runner(cf_gemm, inp_shape=(1, 3, 512, 512))
        with torch.no_grad():
            _ = runner4(inp)
        t4 = _bench(lambda: runner4(inp))
        _print_row("4", "FP16 + Triton + SDPA4D + GEMM + CUDA graph", t4, t0)
    except Exception as e:
        print(f"  Tier 4 | SDPA+GEMM+CUDA graph -- skipped ({e})")

    # ── Tier 5 — FP16 + Triton + SDPA 4D + GEMM + NHWC + CUDA graph ───────
    t5 = t4
    try:
        cf_cl = CodeFormerTorch.from_onnx(CF_ONNX, compute_dtype=torch.float16).cuda().eval()
        cf_cl.to_channels_last().to_gemm_mode()
        with torch.no_grad():
            _ = cf_cl(inp, fidelity_weight=w_val)
        runner5 = build_cuda_graph_runner(cf_cl, inp_shape=(1, 3, 512, 512))
        with torch.no_grad():
            _ = runner5(inp)
        t5 = _bench(lambda: runner5(inp))
        _print_row("5", "FP16 + Triton + SDPA4D + GEMM + NHWC + CUDA graph", t5, t0)
    except Exception as e:
        print(f"  Tier 5 | SDPA+GEMM+NHWC+CUDA graph -- skipped ({e})")

    print()
    print("  Note: CUDA graph requires fixed fidelity_weight.")
    print("  App uses Tier 2 (direct call) for dynamic fidelity control.")
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
        import triton
        print(f"Triton:  {triton.__version__}")
    except ImportError:
        print("Triton:  not available (GroupNorm PyTorch fallback will be used)")
    try:
        import onnxruntime as ort
        print(f"ORT:     {ort.__version__}")
    except ImportError:
        print("ORT:     not available (skipping Tier 0/0b)")
        sys.exit(1)

    if not pathlib.Path(CF_ONNX).exists():
        print(f"ERROR: CodeFormer ONNX not found: {CF_ONNX}")
        sys.exit(1)

    bench_codeformer()
