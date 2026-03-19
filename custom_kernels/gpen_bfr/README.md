# GPEN-BFR Custom Kernel

FP16 PyTorch reimplementation of all four **GPEN-BFR** face restoration models,
with CUDA-graph acceleration.  Used by VisoMaster-Fusion when the
*Custom* execution provider is selected.

## Models

| Model | Input | Output |
|-------|-------|--------|
| GPEN-BFR-256  | 256×256 | 256×256  |
| GPEN-BFR-512  | 512×512 | 512×512  |
| GPEN-BFR-1024 | 512×512 | 1024×1024 |
| GPEN-BFR-2048 | 512×512 | 2048×2048 |

## Benchmark Results (RTX-class GPU)

**Environment:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · Triton 3.6.0 · ORT 1.22.0 · 50 iterations

With Triton-accelerated kernels (demod + fused activation):

| Model | ORT CUDA EP | ORT TRT EP | PT FP32 | PT FP16 + Triton | PT FP16 + Triton + CUDAGraph |
|-------|-------------|------------|---------|------------------|------------------------------|
| 256   | 6.65 ms | 3.55 ms | 5.33 ms | 5.43 ms | **1.79 ms** (3.72× vs ORT) |
| 512   | 19.89 ms | 15.57 ms | 17.57 ms | 19.06 ms | **14.29 ms** (1.39× vs ORT) |
| 1024  | 73.16 ms | 45.46 ms | 38.54 ms | 12.21 ms | **18.29 ms** (4.00× vs ORT) |
| 2048  | 120.59 ms | 67.97 ms | 72.60 ms | 30.38 ms | **27.71 ms** (4.35× vs ORT) |

The CUDA-graph path (PT FP16 + Triton + CUDAGraph) is **1.39–4.35× faster** than ORT CUDA EP across model sizes.

> **Note on GPEN-BFR-1024:** PT FP16 eager (12.21 ms) is faster than CUDA graph (18.29 ms) for this model size.
> GPEN-BFR-512 shows the smallest speedup (1.39×) due to ConvTranspose upsampling at intermediate resolutions.
> Both GPEN-BFR-1024 and 2048 achieve strong speedups (4.00× and 4.35×) thanks to large feature maps where
> FP16 bandwidth savings dominate.

## Accuracy

Mean relative error vs ORT reference (GPEN-BFR-256):
- FP32: 1.37%
- FP16: 1.57%

## Architecture Notes

GPEN uses a **StyleGAN2 generator** with a strided-CNN encoder that replaces
the usual random noise injection with encoder features:

```
Input → ecd0 (1×1) → ecd1..N (FIR-blur + stride-2 3×3) → flatten
                                                         → FC → pixel-norm
                                                         → Style MLP (8×)
                                                         → latent [1,512]

gen_const [4×4] → conv1 (mod+demod, Concat enc[N]) →
→ (convs[2i] up + FIR, convs[2i+1] same, to_rgbs[i]) × N
→ skip-RGB sum → output
```

Key implementation details:
- **Encoder**: each stage uses FIR anti-alias (pad-2 + depthwise 4×4) then
  stride-2 3×3 conv with no padding.  Each activation is scaled by sqrt(2).
- **FC + Style MLP**: FC has LeakyRelu + sqrt(2) activation.  MLP input is
  pixel-normalized (RMS=1).  Each MLP layer has LeakyRelu + sqrt(2).
  MLP Gemm uses `transB=0` (weights stored row-major, need `.T` for F.linear).
- **Upsample styled conv**: ConvTranspose(stride=2, no-pad) then FIR blur
  (pad-1 + depthwise 4×4).  No bilinear interpolation.
- **Noise injection**: `cat([conv_out, noise_w * enc_feat], dim=1)` — doubles
  channels before the activation bias.

## Files

| File | Purpose |
|------|---------|
| `gpen_torch.py` | `GPENTorch` class + `build_cuda_graph_runner` |
| `benchmark_gpen.py` | 4-tier latency benchmark for all model sizes |
| `dump_ops.py` | ONNX initializer / node shape dump |
| `run_with_msvc.bat` | Windows helper: sets up MSVC env then runs Python |
| `trace_conv1.py` | ONNX graph tracer (generator section) |
| `trace_fc.py` | ONNX graph tracer (encoder / FC / MLP) |
| `trace_noise.py` | Traces noise injection sources per generator conv |
| `trace_convs0.py` | Traces generator convs.0 upsample path |
| `test_accuracy.py` | Numerical accuracy comparison vs ORT |

## Usage

```python
from custom_kernels.gpen_bfr.gpen_torch import GPENTorch, build_cuda_graph_runner
import torch

# Load model
model = GPENTorch.from_onnx("model_assets/GPEN-BFR-512.onnx",
                             compute_dtype=torch.float16).cuda().eval()

# Build CUDA graph runner (one-time, ~0.5s)
runner = build_cuda_graph_runner(model, inp_shape=(1, 3, 512, 512))

# Inference (repeated)
inp = torch.randn(1, 3, 512, 512, device="cuda")
with torch.no_grad():
    out = runner(inp)   # float32 output [1, 3, 512, 512]
```

## Running the Benchmark

```bat
custom_kernels/gpen_bfr/run_with_msvc.bat custom_kernels/gpen_bfr/benchmark_gpen.py
# or for a single size:
.venv/Scripts/python custom_kernels/gpen_bfr/benchmark_gpen.py 512
```
