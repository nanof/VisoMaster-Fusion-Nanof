# ReF-LDM Custom Kernel

FP16 PyTorch reimplementation of the three **ReF-LDM** denoiser ONNX models,
with Triton fused GroupNorm+SiLU kernels and CUDA graph capture for all three
components.  Used by VisoMaster-Fusion when the *Custom* execution provider is
selected.

## Models

| Model | Input | Output |
|-------|-------|--------|
| VAE Encoder | (1,3,512,512) f32 | (1,8,64,64) f32 |
| VAE Decoder | (1,8,64,64) f32 | (1,3,512,512) f32 |
| UNet denoiser | (1,16,64,64) f32 + external K/V | (1,8,64,64) f32 |

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · Triton 3.6.0 · ORT 1.22.0 · TensorRT 10
**Conditions:** 200 iterations, 50 warm-up passes

### VAE Encoder (1,3,512,512) → (1,8,64,64)

| Tier | Method | Latency | vs ORT CUDA EP |
|------|--------|--------:|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 21.17 ms | 1.00× |
| 0b   | ORT TensorRT EP FP32 | — *(crashes: Windows access violation during engine build)* | — |
| 1    | PyTorch FP32 pure ops | 21.96 ms | 0.96× |
| 2    | PyTorch FP16 + Triton GroupNorm+SiLU | 10.81 ms | 1.96× |
| **3** | **FP16 + Triton + CUDA graph** | **10.58 ms** | **2.00×** |
| 4    | FP16 + Triton + CUDA graph + NHWC | 14.05 ms | 1.51× |

> NHWC (Tier 4) is slower for the encoder on cuDNN 9 — the small channel counts
> (128–512) don't benefit from NHWC at this spatial resolution. **Tier 3 is used by default.**

### VAE Decoder (1,8,64,64) → (1,3,512,512)

| Tier | Method | Latency | vs ORT CUDA EP |
|------|--------|--------:|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 71.21 ms | 1.00× |
| 0b   | ORT TensorRT EP FP32 | — *(crashes: Windows access violation during engine build)* | — |
| 1    | PyTorch FP32 pure ops | 36.42 ms | 1.96× |
| 2    | PyTorch FP16 + Triton GroupNorm+SiLU | 17.51 ms | 4.07× |
| **3** | **FP16 + Triton + CUDA graph** | **17.23 ms** | **4.13×** |
| 4    | FP16 + Triton + CUDA graph + NHWC | 23.96 ms | 2.97× |

> The `norm_out` GroupNorm fuses the SiLU activation directly inside the Triton kernel
> (eliminates a separate read + write of the 512×512 feature map), saving ~0.5 ms per call.
> NHWC (Tier 4) regresses for the decoder — small channel counts (ch=128) at 512×512 spatial
> resolution don't benefit from cuDNN NHWC layout. **Tier 3 (NCHW) is used by default.**

### UNet Denoiser (1,16,64,64) + K/V → (1,8,64,64)

| Tier | Method | Latency | vs ORT CUDA EP |
|------|--------|--------:|:--------------:|
| 0    | ORT FP32 CUDA EP — no K/V (baseline) | 10.25 ms | 1.00× |
| 0b   | ORT TensorRT EP FP32 | — *(crashes: Windows access violation during engine build)* | — |
| 1    | PyTorch FP32 pure ops | 15.63 ms | 0.66× |
| 2    | PyTorch FP16 + Triton GroupNorm+SiLU | 15.54 ms | 0.66× |
| 3    | FP16 + Triton + CUDA graph | 5.28 ms | 1.94× |
| **4** | **FP16 + Triton + CUDA graph + NHWC** | **5.25 ms** | **1.95×** |

> The UNet CUDA graph (Tier 3/4) works because K/V tensors are **static buffers** that are
> copied into the graph's pre-allocated memory each call — the graph sees constant shapes.
> ORT CUDA EP baseline uses zeroed K/V; PyTorch Tiers 1–2 run eager with real K/V (higher
> latency due to Python/kernel-launch overhead).  Once the CUDA graph is captured, kernel-launch
> overhead is eliminated and the UNet runs at **5.25 ms** — **1.95× faster than ORT CUDA EP**.

> **Note:** ORT TensorRT EP crashes with a Windows access violation (rc=3221225477) during engine
> build for all three ReF-LDM models. Custom kernels deliver 2.00×/4.13×/1.95× speedups over
> ORT CUDA EP baseline without TRT dependency.

> **Application uses:**
> - VAE Encoder → **Tier 3** (CUDA graph, NCHW — NHWC is slower for this model)
> - VAE Decoder → **Tier 3** (CUDA graph, NCHW — NHWC is slower for this model)
> - UNet → **Tier 4** (CUDA graph + NHWC — 1.95× faster than ORT CUDA EP)

### Kernel Priority Chain

1. **Triton `triton_group_norm_silu`** — Windows-friendly, no MSVC required (preferred)
2. **Pure PyTorch** — automatic fallback (still FP16)

---

## Architecture

### VAE (VQModelInterface)

Config from `configs/refldm.yaml`:
```
ch=128, ch_mult=[1,1,2,4], num_res_blocks=2
attn_resolutions=[]   (NO attention in encoder/decoder blocks)
z_channels=8, double_z=False
```

**Encoder** path: `conv_in → 4× DownBlock[2×ResBlock + optional Downsample] → mid[ResBlock+Attn+ResBlock] → norm_out+SiLU(fused) → conv_out → quant_conv`

**Decoder** path: `post_quant_conv → conv_in → mid → 4× UpBlock[2×ResBlock + optional Upsample] → norm_out+SiLU(fused) → conv_out`

> The exported ONNX decoder does **not** include a VQ lookup step — the upstream pipeline
> passes latents directly to `post_quant_conv`.

**norm_out SiLU fusion**: Both encoder and decoder fuse the final GroupNorm and SiLU
activation into a single Triton kernel pass.  This eliminates one read + write of the
output feature map (most significant for the decoder where spatial resolution is 512×512).

### UNet Denoiser

Config from `configs/refldm.yaml`:
```
model_channels=160, channel_mult=[1,2,2,4], num_res_blocks=2
attention_resolutions=[2,4,8]   (spatial resolutions 32, 16, 8)
num_head_channels=32
in_channels=16, out_channels=8
use_spatial_transformer=False   (uses AttentionBlock, not CrossAttn)
```

**External K/V mechanism**: Each `AttentionBlock` at attention resolutions
receives reference K/V tensors extracted by `KVExtractor` from a reference image.

- When `use_exclusive=True`: attention uses **only** reference K/V (no self-attention)
- When `use_exclusive=False`: self-K/V concatenated with reference K/V along sequence dim

K/V tensors are keyed by PyTorch module path (e.g. `"input_blocks.4.1.attention"`)
with shape `(n_heads, ch_per_head, seq_len)`.

**Triton fused GroupNorm+SiLU kernel (`triton_group_norm_silu`):**
Each ResBlock in both VAE and UNet contains `GroupNorm(32) → SiLU` pairs.
The Triton kernel fuses these into a single two-pass operation:
- Pass 1: compute per-group mean and variance (FP32 accumulators)
- Pass 2: normalize, apply weight/bias, optional SiLU fusion
- Grid: `(N*G,)` programs — one per (batch, group)
- Eliminates the `GroupNorm32` FP16→FP32→FP16 round-trip

**Flash attention**: All attention blocks use `F.scaled_dot_product_attention`
(PyTorch 2.0+ automatically dispatches to FlashAttention on supported hardware).

---

## Files

| File | Purpose |
|------|---------|
| `ref_ldm_torch.py` | `RefLDMEncoderTorch`, `RefLDMDecoderTorch`, `RefLDMUNetTorch` + Triton kernels + CUDA graph runner |
| `benchmark_ref_ldm.py` | 5-tier latency benchmark vs ORT baseline (Tier 5: `torch.compile`, Linux only) |
| `benchmark_results.txt` | Latest benchmark output |
| `__init__.py` | Package marker |

The Triton GroupNorm+SiLU kernel is defined in `custom_kernels/triton_ops.py`
as `triton_group_norm_silu` (Kernel 5).

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

All three ReF-LDM models are then executed via `RefLDMEncoderTorch`,
`RefLDMDecoderTorch`, and `RefLDMUNetTorch` respectively.

```python
# Internal call path (face_restorers.py):
from custom_kernels.ref_ldm.ref_ldm_torch import (
    RefLDMEncoderTorch, RefLDMDecoderTorch, RefLDMUNetTorch,
    build_cuda_graph_runner, build_unet_cuda_graph_runner,
)
enc  = RefLDMEncoderTorch.from_onnx(onnx_path).cuda().eval()
dec  = RefLDMDecoderTorch.from_onnx(onnx_path).cuda().eval()
unet = RefLDMUNetTorch.from_onnx(onnx_path).cuda().eval()

enc_runner  = build_cuda_graph_runner(enc,  inp_shape=(1, 3, 512, 512))
dec_runner  = build_cuda_graph_runner(dec,  inp_shape=(1, 8, 64,  64 ))
unet_runner = build_unet_cuda_graph_runner(
    unet, x_shape=(1, 16, 64, 64), ts_example=timestep,
    kv_map_template=kv_map, use_exclusive=True,
)

latent     = enc_runner(image_f32_cuda)              # (1,3,512,512)f32 → (1,8,64,64)f32
image_out  = dec_runner(latent)                      # (1,8,64,64)f32 → (1,3,512,512)f32
noise_pred = unet_runner(x_noisy, timestep, kv_map)  # denoising step
```

VAE models are CUDA-graph-captured on first use (3 warm-up passes + capture).
UNet uses `UNetCUDAGraphRunner` which pre-allocates static K/V buffers so the
variable reference K/V tensors are compatible with CUDA graph replay.

---

## Weight Loading

Weights are loaded from the ONNX initializers.  The `_load_from_onnx_weights`
helper tries multiple prefix strategies:

- `RefLDMEncoderTorch`: prefixes `["", "encoder.", "first_stage_model."]`
- `RefLDMDecoderTorch`: prefixes `["", "decoder.", "first_stage_model."]`
- `RefLDMUNetTorch`: prefixes `["unet_model.", "", "model.diffusion_model.", "diffusion_model."]`

If a prefix match fails, falls back to suffix matching (parameter name without
any top-level namespace).

---

## Numerical Accuracy

FP16 GroupNorm with FP32 accumulators in the Triton kernel provides
accuracy equivalent to the `GroupNorm32` FP32 upcast used in the original
LDM code, while avoiding redundant dtype conversions.  The fused SiLU path
computes `out * sigmoid(out)` inside the same Triton pass — numerically
identical to a separate SiLU call.

Expected mean relative error vs ORT FP32: `< 2%` for both VAE components.
UNet: `< 2%` mean error (K/V attention paths may show slightly higher variance
at very low sequence lengths).

## Running the Benchmark

```bat
.venv/Scripts/python custom_kernels/ref_ldm/benchmark_ref_ldm.py
```

Optional env vars:
```
WARMUP=10 ITERS=50 .venv/Scripts/python custom_kernels/ref_ldm/benchmark_ref_ldm.py
```

> **Note (Windows):** Tier 5 (`torch.compile`) is automatically skipped on Windows
> because `torch.inductor` + Triton causes a hard native segfault in `libtriton.pyd`
> during Inductor codegen.  Tier 5 will run correctly on Linux.
