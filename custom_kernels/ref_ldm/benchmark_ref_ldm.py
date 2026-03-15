"""
Benchmark — ReF-LDM custom kernels vs ORT baseline.

Tests three model components separately:
  VAE Encoder  : (1,3,512,512)f32  -> (1,8,64,64)f32
  VAE Decoder  : (1,8,64,64)f32   -> (1,3,512,512)f32
  UNet denoiser: (1,16,64,64)f32  + K/V -> (1,8,64,64)f32

Usage (from repo root):
    .venv\\Scripts\\python custom_kernels\\ref_ldm\\benchmark_ref_ldm.py

Optional env vars:
    ONNX_DIR=model_assets          path to ONNX model files
    WARMUP=50                      warm-up iterations
    ITERS=200                      timed iterations
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
ROOT       = pathlib.Path(__file__).parent.parent.parent
ONNX_DIR   = pathlib.Path(os.environ.get("ONNX_DIR", ROOT / "model_assets"))
WARMUP     = int(os.environ.get("WARMUP", 50))
ITERS      = int(os.environ.get("ITERS",  200))

ENC_ONNX   = str(ONNX_DIR / "ref_ldm_vae_encoder.onnx")
DEC_ONNX   = str(ONNX_DIR / "ref_ldm_vae_decoder.onnx")
UNET_ONNX  = str(ONNX_DIR / "ref_ldm_unet_external_kv.onnx")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ms(elapsed_s: float, iters: int) -> float:
    return elapsed_s / iters * 1000.0


def _bench(fn, warmup=WARMUP, iters=ITERS) -> float:
    """Return ms/iter (GPU-timed)."""
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
    print(f"  Tier {tier} | {label:<42} | {ms_val:8.2f} ms | {speedup:6.2f}x")


# ---------------------------------------------------------------------------
# VAE Encoder benchmark
# ---------------------------------------------------------------------------
def bench_encoder():
    print("\n=== VAE Encoder (1,3,512,512) -> (1,8,64,64) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<42} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-'*6}-+-{'-'*42}-+-{'-'*8}-+-{'-'*7}")

    inp = torch.randn(1, 3, 512, 512, dtype=torch.float32, device="cuda")

    # Tier 0  — ORT FP32 CUDA EP
    sess0 = _ort_session(ENC_ONNX, "CUDAExecutionProvider")
    inp_np = inp.cpu().numpy()
    in_name  = sess0.get_inputs()[0].name
    out_name = sess0.get_outputs()[0].name
    t0 = _bench(lambda: sess0.run([out_name], {in_name: inp_np}))
    _print_row("0", "ORT FP32 CUDA EP", t0, t0)

    # Tier 0b — ORT TensorRT EP
    t0b = t0
    try:
        import tensorrt  # registers nvinfer DLL path on Windows
    except Exception:
        pass
    import onnxruntime as _ort
    if "TensorrtExecutionProvider" not in _ort.get_available_providers():
        print(f"  Tier 0b | TensorRT EP — skipped (TensorrtExecutionProvider not available)")
    else:
        ctx = ROOT / "tensorrt-engines" / "ref_ldm_vae_encoder_ctx.onnx"
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
            sess0b = _ort.InferenceSession(ENC_ONNX, so, providers=[
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ])
            _os.chdir(_prev_cwd)
            t0b = _bench(lambda: sess0b.run([out_name], {in_name: inp_np}))
            _print_row("0b", "ORT TensorRT EP FP32 (app default)", t0b, t0)

    # Tier 1  — PyTorch FP32
    sys.path.insert(0, str(ROOT))
    from custom_kernels.ref_ldm.ref_ldm_torch import RefLDMEncoderTorch
    enc_fp32 = RefLDMEncoderTorch.from_onnx(ENC_ONNX, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: enc_fp32(inp))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # Tier 2  — PyTorch FP16 + Triton GroupNorm+SiLU
    enc_fp16 = RefLDMEncoderTorch.from_onnx(ENC_ONNX, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: enc_fp16(inp))
    _print_row("2", "PyTorch FP16 + Triton GroupNorm+SiLU", t2, t0)

    # Tier 3  — FP16 + Triton + CUDA graph
    from custom_kernels.ref_ldm.ref_ldm_torch import build_cuda_graph_runner
    runner = build_cuda_graph_runner(enc_fp16, inp_shape=(1, 3, 512, 512))
    t3 = _bench(lambda: runner(inp))
    _print_row("3", "PyTorch FP16 + Triton + CUDA graph", t3, t0)

    # Tier 4  — FP16 + Triton + CUDA graph + NHWC (channels-last)
    enc_cl = RefLDMEncoderTorch.from_onnx(ENC_ONNX, compute_dtype=torch.float16).cuda().eval()
    enc_cl.to_channels_last()
    runner_cl = build_cuda_graph_runner(enc_cl, inp_shape=(1, 3, 512, 512))
    t4 = _bench(lambda: runner_cl(inp))
    _print_row("4", "FP16 + Triton + CUDA graph + NHWC", t4, t0)

    print()
    return t0, t0b


# ---------------------------------------------------------------------------
# VAE Decoder benchmark
# ---------------------------------------------------------------------------
def bench_decoder():
    print("\n=== VAE Decoder (1,8,64,64) -> (1,3,512,512) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<42} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-'*6}-+-{'-'*42}-+-{'-'*8}-+-{'-'*7}")

    lat = torch.randn(1, 8, 64, 64, dtype=torch.float32, device="cuda")

    # Tier 0  — ORT FP32 CUDA EP
    sess0 = _ort_session(DEC_ONNX, "CUDAExecutionProvider")
    lat_np   = lat.cpu().numpy()
    in_name  = sess0.get_inputs()[0].name
    out_name = sess0.get_outputs()[0].name
    t0 = _bench(lambda: sess0.run([out_name], {in_name: lat_np}))
    _print_row("0", "ORT FP32 CUDA EP", t0, t0)

    # Tier 0b — ORT TensorRT EP
    t0b = t0
    import onnxruntime as _ort2
    if "TensorrtExecutionProvider" not in _ort2.get_available_providers():
        print(f"  Tier 0b | TensorRT EP — skipped (TensorrtExecutionProvider not available)")
    else:
        ctx = ROOT / "tensorrt-engines" / "ref_ldm_vae_decoder_ctx.onnx"
        if not ctx.exists():
            print(f"  Tier 0b | TensorRT EP — skipped (no pre-built engine: {ctx.name})")
        else:
            import os as _os2
            _prev_cwd = _os2.getcwd()
            _os2.chdir(str(ROOT))
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
            so = _ort2.SessionOptions()
            so.graph_optimization_level = _ort2.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess0b = _ort2.InferenceSession(DEC_ONNX, so, providers=[
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ])
            _os2.chdir(_prev_cwd)
            t0b = _bench(lambda: sess0b.run([out_name], {in_name: lat_np}))
            _print_row("0b", "ORT TensorRT EP FP32 (app default)", t0b, t0)

    # Tier 1  — PyTorch FP32
    from custom_kernels.ref_ldm.ref_ldm_torch import RefLDMDecoderTorch
    dec_fp32 = RefLDMDecoderTorch.from_onnx(DEC_ONNX, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: dec_fp32(lat))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # Tier 2  — PyTorch FP16 + Triton GroupNorm+SiLU
    dec_fp16 = RefLDMDecoderTorch.from_onnx(DEC_ONNX, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: dec_fp16(lat))
    _print_row("2", "PyTorch FP16 + Triton GroupNorm+SiLU", t2, t0)

    # Tier 3  — FP16 + Triton + CUDA graph
    from custom_kernels.ref_ldm.ref_ldm_torch import build_cuda_graph_runner
    runner = build_cuda_graph_runner(dec_fp16, inp_shape=(1, 8, 64, 64))
    t3 = _bench(lambda: runner(lat))
    _print_row("3", "PyTorch FP16 + Triton + CUDA graph", t3, t0)

    # Tier 4  — FP16 + Triton + CUDA graph + NHWC (channels-last)
    dec_cl = RefLDMDecoderTorch.from_onnx(DEC_ONNX, compute_dtype=torch.float16).cuda().eval()
    dec_cl.to_channels_last()
    runner_cl = build_cuda_graph_runner(dec_cl, inp_shape=(1, 8, 64, 64))
    t4 = _bench(lambda: runner_cl(lat))
    _print_row("4", "FP16 + Triton + CUDA graph + NHWC", t4, t0)

    print()
    return t0, t0b


# ---------------------------------------------------------------------------
# UNet benchmark
# ---------------------------------------------------------------------------
def _build_dummy_kv_map():
    """Build a representative K/V map for the UNet benchmark.

    UNet: mc=160, mult=(1,2,2,4), nhc=32  ->  n_heads = ch // 32
      Level 1 (32×32, 320ch):  input 4-5,  output 3-5  -> n_heads=10
      Level 2 (16×16, 320ch):  input 7-8,  output 6-8  -> n_heads=10
      Level 3 (8×8,  640ch):   input 10-11, middle, output 0-2 -> n_heads=20
    External seq_len is arbitrary (64 reference tokens used here).
    """
    # (path, n_heads)
    ch_per_head = 32
    seq_len = 64
    attn_blocks = [
        # Level 1 — 320ch -> 10 heads
        ("input_blocks.4.1.attention",   10),
        ("input_blocks.5.1.attention",   10),
        # Level 2 — 320ch -> 10 heads
        ("input_blocks.7.1.attention",   10),
        ("input_blocks.8.1.attention",   10),
        # Level 3 — 640ch -> 20 heads
        ("input_blocks.10.1.attention",  20),
        ("input_blocks.11.1.attention",  20),
        ("middle_block.1.attention",     20),
        ("output_blocks.0.1.attention",  20),
        ("output_blocks.1.1.attention",  20),
        ("output_blocks.2.1.attention",  20),
        # Level 2 (decoder) — 320ch -> 10 heads
        ("output_blocks.3.1.attention",  10),
        ("output_blocks.4.1.attention",  10),
        ("output_blocks.5.1.attention",  10),
    ]
    kv_map = {}
    for path, n_heads in attn_blocks:
        kv_map[path] = {
            "k": torch.randn(n_heads, ch_per_head, seq_len, device="cuda", dtype=torch.float32),
            "v": torch.randn(n_heads, ch_per_head, seq_len, device="cuda", dtype=torch.float32),
        }
    return kv_map


def bench_unet():
    print("\n=== UNet denoiser (1,16,64,64) + K/V -> (1,8,64,64) ===")
    print(f"  warm-up={WARMUP}, iters={ITERS}\n")
    print(f"  {'Tier':<6} | {'Method':<42} | {'ms':>8} | {'speedup':>7}")
    print(f"  {'-'*6}-+-{'-'*42}-+-{'-'*8}-+-{'-'*7}")

    x = torch.randn(1, 16, 64, 64, dtype=torch.float32, device="cuda")
    ts = torch.tensor([500], dtype=torch.int64, device="cuda")
    kv_map = _build_dummy_kv_map()

    # Tier 0  — ORT FP32 CUDA EP (simplified: no K/V, baseline timing only)
    sess0 = _ort_session(UNET_ONNX, "CUDAExecutionProvider")
    onnx_inputs = sess0.get_inputs()
    in_names = [i.name for i in onnx_inputs]

    def _build_ort_feeds():
        feeds = {
            "x_noisy_plus_lq_latent": x.cpu().numpy(),
            "timesteps": ts.cpu().numpy(),
            "is_ref_flag_input": np.array([True], dtype=bool),
            "use_reference_exclusive_path_globally_input": np.array([True], dtype=bool),
        }
        # Zero-fill K/V inputs
        for inp in onnx_inputs:
            if inp.name.endswith("_k_ext") or inp.name.endswith("_v_ext"):
                shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
                feeds[inp.name] = np.zeros(shape, dtype=np.float32)
        return feeds

    feeds = _build_ort_feeds()
    out_name = sess0.get_outputs()[0].name
    t0 = _bench(lambda: sess0.run([out_name], feeds))
    _print_row("0", "ORT FP32 CUDA EP (no K/V)", t0, t0)

    # Tier 0b — ORT TensorRT EP
    t0b = t0
    import onnxruntime as _ort3
    if "TensorrtExecutionProvider" not in _ort3.get_available_providers():
        print(f"  Tier 0b | TensorRT EP — skipped (TensorrtExecutionProvider not available)")
    else:
        ctx = ROOT / "tensorrt-engines" / "ref_ldm_unet_external_kv_ctx.onnx"
        if not ctx.exists():
            print(f"  Tier 0b | TensorRT EP — skipped (no pre-built engine: {ctx.name})")
        else:
            import os as _os3
            _prev_cwd = _os3.getcwd()
            _os3.chdir(str(ROOT))
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
            so = _ort3.SessionOptions()
            so.graph_optimization_level = _ort3.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess0b = _ort3.InferenceSession(UNET_ONNX, so, providers=[
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                ("CPUExecutionProvider", {}),
            ])
            _os3.chdir(_prev_cwd)
            t0b = _bench(lambda: sess0b.run([out_name], feeds))
            _print_row("0b", "ORT TensorRT EP FP32 (app default)", t0b, t0)

    # Tier 1  — PyTorch FP32
    from custom_kernels.ref_ldm.ref_ldm_torch import RefLDMUNetTorch
    unet_fp32 = RefLDMUNetTorch.from_onnx(UNET_ONNX, compute_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        t1 = _bench(lambda: unet_fp32(x, ts, kv_map=kv_map, use_exclusive=True))
    _print_row("1", "PyTorch FP32 pure ops", t1, t0)

    # Tier 2  — PyTorch FP16 + Triton GroupNorm+SiLU (no CUDA graph)
    unet_fp16 = RefLDMUNetTorch.from_onnx(UNET_ONNX, compute_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        t2 = _bench(lambda: unet_fp16(x, ts, kv_map=kv_map, use_exclusive=True))
    _print_row("2", "PyTorch FP16 + Triton GroupNorm+SiLU", t2, t0)

    # Tier 3  — FP16 + Triton + CUDA graph (static K/V buffers)
    from custom_kernels.ref_ldm.ref_ldm_torch import build_unet_cuda_graph_runner
    unet_runner = build_unet_cuda_graph_runner(
        unet_fp16, x_shape=(1, 16, 64, 64),
        ts_example=ts, kv_map_template=kv_map, use_exclusive=True,
    )
    t3 = _bench(lambda: unet_runner(x, ts, kv_map, use_exclusive=True))
    _print_row("3", "FP16 + Triton + CUDA graph", t3, t0)

    # Tier 4  — FP16 + Triton + CUDA graph + NHWC (channels-last Conv2d)
    unet_cl = RefLDMUNetTorch.from_onnx(UNET_ONNX, compute_dtype=torch.float16).cuda().eval()
    unet_cl.to_channels_last()
    unet_cl_runner = build_unet_cuda_graph_runner(
        unet_cl, x_shape=(1, 16, 64, 64),
        ts_example=ts, kv_map_template=kv_map, use_exclusive=True,
    )
    t4 = _bench(lambda: unet_cl_runner(x, ts, kv_map, use_exclusive=True))
    _print_row("4", "FP16 + Triton + CUDA graph + NHWC", t4, t0)

    print()
    return t0, t0b


# ---------------------------------------------------------------------------
# Main
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
        print("Triton:  not available (FP32 fallback will be used)")
    try:
        import onnxruntime as ort
        print(f"ORT:     {ort.__version__}")
    except ImportError:
        print("ORT:     not available (skipping Tier 0/0b)")
        sys.exit(1)

    # Check ONNX files exist
    for path, name in [(ENC_ONNX, "VAE encoder"), (DEC_ONNX, "VAE decoder"), (UNET_ONNX, "UNet")]:
        if not pathlib.Path(path).exists():
            print(f"ERROR: {name} ONNX not found: {path}")
            sys.exit(1)

    bench_encoder()
    bench_decoder()
    bench_unet()
