# Custom Kernels — GFPGANv1.4

FP16 PyTorch reimplementation of the `GFPGANv1.4.onnx` face-restoration model,
with Triton and CUDA C++ fused kernels for StyleGAN2 weight demodulation and
activation fusion, plus CUDA-graph acceleration.  Used by VisoMaster-Fusion when
the *Custom* execution provider is selected.

## Benchmark Results (RTX 4090, CUDA 12.9, PyTorch 2.8+cu129, ORT 1.22.0, Triton 3.6.0)

50-iteration warm benchmark:

| Tier | Method | ms | vs ORT CUDA EP | vs ORT TRT EP |
|------|--------|----|---------------:|---------------:|
| 0 | ORT FP32 CUDA EP | 14.12 ms | 1.00x (baseline) | 0.79x |
| 0b | ORT TensorRT EP (app default) | 11.11 ms | 1.27x | 1.00x (baseline) |
| 1 | PyTorch FP32 pure ops | 12.88 ms | 1.10x | 0.86x |
| 2 | PyTorch FP16 + Triton demod + fused-act | 8.88 ms | **1.59x** | **1.25x** |
| 3 | PyTorch FP16 + Triton demod + CUDA graph (Custom) | **7.52 ms** | **1.88x** | **1.48x** |

The CUDA-graph path is **1.48x faster** than ORT TRT EP (app default) and **1.88x faster** than ORT CUDA EP.

Numerical accuracy: `mean|diff|` vs ORT ≈ 0.0096 (p99 ≈ 0.087), visually indistinguishable.

### Kernel Priority Chain

1. **Triton `triton_demod` + `triton_fused_gfpgan_act`** — Windows-friendly, no MSVC required (preferred)
2. **CUDA C++ `gfpgan_demod_ext`** — pre-built `.pyd` from `model_assets/custom_kernels/` or JIT via MSVC
3. **Pure PyTorch** — automatic fallback, still delivers Tier 2/3 class performance

---

## Architecture

GFPGANv1.4 (513 nodes, opset 11, all FP32):

```
Input [1,3,512,512]
  │
  ├── U-Net Encoder (conv_body_first + 7× ResBlock down + final_conv)
  │     512×512 → 4×4, channels: 32→64→128→256→256 (×5)
  │
  ├── Latent projection (final_linear [4096→8192]) → [1,16,512]
  │
  ├── U-Net Decoder (7× ResBlock up + 7× SFT condition nets)
  │     4×4 → 512×512, each step generates SFT (scale, shift) pairs
  │
  └── StyleGAN2 Decoder
        constant [1,512,4,4]
        → style_conv1 (4×4) + to_rgb1
        → 7× [upsample_conv + conv + to_rgb] at 8,16,32,64,128,256,512
        → RGB accumulation via skip connections
```

Key bottleneck profile: Conv 53%, Add 11%, Gemm 5.6%, Resize 5.4%.

**Why FP16 is faster:**
The model is entirely FP32 in ONNX.  Converting to FP16 for PyTorch inference:
- Halves memory bandwidth for all feature maps (~1.55× observed)
- Enables Tensor Core utilization on Ampere/Ada GPUs

**Triton fused demod kernel (`triton_demod`):**
Each modulated conv in ONNX expands to 7 nodes:
```
weight[C_out,C_in,kH,kW] * style[C_in]  →  w_mod
rsqrt(sum(w_mod^2, [C_in,kH,kW]) + eps) →  demod
w_demod = w_mod * demod
```
The Triton kernel fuses these into two passes:
- Pass 1: accumulate `(w * s)^2` per output channel with masked BLOCK loads
- Pass 2: apply `rsqrt(sum + eps)` scale, write FP16 output
- Grid: `(C_out,)` programs; BLOCK=256 (power-of-2 constexpr)

**Triton fused activation kernel (`triton_fused_gfpgan_act`):**
Fuses `cat([conv_bias, noise_term], dim=1)` + LeakyReLU + scale into one kernel.
Two variants: `_with_noise` and `_no_noise` — Python dispatcher selects at call time.

---

## Files

| File | Purpose |
|------|---------|
| `gfpgan_torch.py` | PyTorch FP16 model + Triton/CUDA kernels + CUDA graph runner |
| `benchmark_gfpgan.py` | 5-tier latency benchmark vs ORT baseline |
| `benchmark_results.txt` | Saved benchmark output |
| `dump_ops.py` | Dev tool: prints all ONNX node shapes |
| `ops_dump.txt` | Saved ops dump output |
| `profile_ort.py` | Dev tool: ORT profiler JSON analysis |
| `run_with_msvc.bat` | Windows build helper (sets up MSVC environment) |

Pre-built CUDA C++ binaries are stored in `model_assets/custom_kernels/gfpgan_demod_ext.pyd`
(multi-arch fat binary, sm_70–sm_90).  Triton JIT cache lives in
`model_assets/custom_kernels/triton_cache/`.

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

In Custom mode, `GFPGAN-v1.4` and `GFPGAN-1024` are executed via `GFPGANTorch`
(FP16 PyTorch + Triton kernels + CUDA graph).  All other ONNX models continue to
use TensorRT EP.

```python
# Internal call path (face_restorers.py):
from custom_kernels.gfpgan_v1_4.gfpgan_torch import GFPGANTorch
model = GFPGANTorch.from_onnx(onnx_path).cuda().eval()
output = model(input_f32_cuda)  # [1,3,512,512] float32 in/out
```

---

## Running the Benchmark

```bat
custom_kernels/gfpgan_v1_4/run_with_msvc.bat custom_kernels/gfpgan_v1_4/benchmark_gfpgan.py
# or directly:
.venv/Scripts/python custom_kernels/gfpgan_v1_4/benchmark_gfpgan.py
```

---

## Build Instructions (Windows — Developers Only)

> Regular users do **not** need to compile anything.  Triton kernels compile
> automatically on first use (no MSVC required).  The CUDA C++ kernel is optional
> and only provides marginal additional speedup over Triton.

### Build multi-arch fat binary (all supported GPUs)

```bat
.venv\Scripts\python custom_kernels\build_kernels.py
```

This compiles `gfpgan_demod_ext` and `adain_fp16_ext` for sm_70, sm_75, sm_80,
sm_86, sm_89, sm_90 (plus PTX forward compatibility) and saves the `.pyd` files
to `model_assets/custom_kernels/`.  Requires MSVC and CUDA Toolkit.

### Prerequisites

| Tool | Version |
|------|---------|
| Visual Studio Build Tools | 2019 or 2022 (MSVC v14.x) |
| CUDA Toolkit | 12.x |
| Python | 3.12 |
| PyTorch | 2.8+cu129 (must match runtime) |
| ninja | `uv pip install ninja` |
| triton-windows | `uv pip install triton-windows --extra-index-url https://download.pytorch.org/whl/cu129` |

---

## Shared with GFPGAN-1024

`gfpgan_torch.py` handles **both** variants (v1.4 = 512 output, 1024 = 1024 output).
Detection is automatic via `final_extend_linear` key presence in the ONNX model.
`custom_kernels/gfpgan_1024/` contains only thin wrappers and a separate benchmark.
