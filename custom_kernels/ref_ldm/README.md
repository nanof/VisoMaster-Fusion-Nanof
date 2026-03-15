# ReF-LDM Custom Kernel

FP16 PyTorch reimplementation of the three **ReF-LDM** denoiser ONNX models,
with Triton fused GroupNorm+SiLU kernels and CUDA graph capture for the VAE components.
Used by VisoMaster-Fusion when the *Custom* execution provider is selected.

## Models

| Model | Input | Output |
|-------|-------|--------|
| VAE Encoder | (1,3,512,512) f32 | (1,8,64,64) f32 |
| VAE Decoder | (1,8,64,64) f32 | (1,3,512,512) f32 |
| UNet denoiser | (1,16,64,64) f32 + external K/V | (1,8,64,64) f32 |

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · Triton 3.6.0 · ORT 1.22.0
**Conditions:** 50 iterations, 10 warm-up passes

### VAE Encoder (1,3,512,512) → (1,8,64,64)

| Tier | Method | Latency | vs ORT CUDA EP | vs ORT TRT EP |
|------|--------|--------:|:--------------:|:-------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 20.99 ms | 1.00× | 0.59× |
| 0b   | ORT TensorRT EP | 12.28 ms | 1.71× | 1.00× (baseline) |
| 1    | PyTorch FP32 pure ops | 21.72 ms | 0.97× | — |
| 2    | PyTorch FP16 + Triton GroupNorm+SiLU | 10.80 ms | 1.94× | 0.88× |
| **3** | **FP16 + Triton + CUDA graph** | **10.45 ms** | **2.01×** | **1.17×** |
| 4    | FP16 + Triton + CUDA graph + NHWC | 13.94 ms | 1.51× | — |

> NHWC (Tier 4) is slower for the encoder on cuDNN 9 — the small channel counts
> (128–512) don't benefit from NHWC at this spatial resolution. **Tier 3 is used by default.**

### VAE Decoder (1,8,64,64) → (1,3,512,512)

| Tier | Method | Latency | vs ORT CUDA EP | vs ORT TRT EP |
|------|--------|--------:|:--------------:|:-------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 70.67 ms | 1.00× | 0.30× |
| 0b   | ORT TensorRT EP | 21.32 ms | 3.31× | 1.00× (baseline) |
| 1    | PyTorch FP32 pure ops | 52.65 ms | 1.34× | — |
| 2    | PyTorch FP16 + Triton GroupNorm+SiLU | 25.37 ms | 2.78× | 0.84× |
| **3** | **FP16 + Triton + CUDA graph** | **25.23 ms** | **2.80×** | **0.84×** |
| **4** | **FP16 + Triton + CUDA graph + NHWC** | **25.07 ms** | **2.82×** | **0.85×** |

> VAE Decoder Tier 4 (NHWC) provides a marginal improvement over Tier 3.
> **Tier 4 is used by default for the decoder.**

### UNet Denoiser (1,16,64,64) + K/V → (1,8,64,64)

| Tier | Method | Latency | vs ORT CUDA EP | vs ORT TRT EP |
|------|--------|--------:|:--------------:|:-------------:|
| 0    | ORT FP32 CUDA EP — no K/V (baseline) | 9.62 ms | 1.00× | 0.55× |
| 0b   | ORT TensorRT EP | 7.81 ms | 1.23× | 1.00× (baseline) |
| 1    | PyTorch FP32 pure ops | 13.87 ms | 0.69× | — |
| 2    | PyTorch FP16 + Triton GroupNorm+SiLU | 14.03 ms | 0.69× | — |
| **3** | **FP16 + Triton + CUDA graph** | **5.34 ms** | **1.80×** | **1.46×** |
| **4** | **FP16 + Triton + CUDA graph + NHWC** | **5.27 ms** | **1.83×** | **1.48×** |

> The UNet CUDA graph (Tier 3/4) works because K/V tensors are **static buffers** that are
> copied into the graph's pre-allocated memory each call — the graph sees constant shapes.
> ORT CUDA EP baseline uses zeroed K/V; PyTorch Tiers 1–2 include real K/V concat per block,
> explaining the higher eager latency. Once the CUDA graph is captured, kernel-launch overhead
> is eliminated and the UNet runs at **5.27–5.34 ms** — **1.46–1.48× faster than ORT TRT EP**.

> **Application uses:**
> - VAE Encoder → **Tier 3** (CUDA graph, NCHW — NHWC is slower for this model)
> - VAE Decoder → **Tier 4** (CUDA graph + NHWC)
> - UNet → **Tier 4** (CUDA graph + NHWC — 1.48× faster than ORT TRT EP)

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

**Encoder** path: `conv_in → 4× DownBlock[2×ResBlock + optional Downsample] → mid[ResBlock+Attn+ResBlock] → norm_out+SiLU → conv_out → quant_conv`

**Decoder** path: `quantize (VQ lookup) → post_quant_conv → conv_in → mid → 4× UpBlock[2×ResBlock + optional Upsample] → norm_out+SiLU → conv_out`

**VQ codebook**: nearest-neighbour lookup over 8192 codes of dimension 8.
Inference-only — no straight-through gradients needed.

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
| `benchmark_ref_ldm.py` | 4-tier latency benchmark vs ORT baseline |
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
    RefLDMEncoderTorch, RefLDMDecoderTorch, RefLDMUNetTorch, build_cuda_graph_runner
)
enc  = RefLDMEncoderTorch.from_onnx(onnx_path).cuda().eval()
dec  = RefLDMDecoderTorch.from_onnx(onnx_path).cuda().eval()
unet = RefLDMUNetTorch.from_onnx(onnx_path).cuda().eval()

enc_runner = build_cuda_graph_runner(enc, inp_shape=(1, 3, 512, 512))
dec_runner = build_cuda_graph_runner(dec, inp_shape=(1, 8, 64, 64))

latent    = enc_runner(image_f32_cuda)    # (1,3,512,512)f32 → (1,8,64,64)f32
image_out = dec_runner(latent)            # (1,8,64,64)f32 → (1,3,512,512)f32
noise_pred = unet(x_noisy_lq, timesteps, kv_map, use_exclusive=True)
```

VAE models are CUDA-graph-captured on first use (3 warm-up passes + capture).
UNet is called directly (no CUDA graph) due to dynamic K/V tensors.

---

## Weight Loading

Weights are loaded from the ONNX initializers.  The `_load_from_onnx_weights`
helper tries multiple prefix strategies:

- `RefLDMEncoderTorch`: prefixes `["", "encoder.", "first_stage_model."]`
- `RefLDMDecoderTorch`: prefixes `["", "decoder.", "first_stage_model."]`
- `RefLDMUNetTorch`: prefixes `["", "model.diffusion_model.", "diffusion_model."]`

If a prefix match fails, falls back to suffix matching (parameter name without
any top-level namespace).

---

## Numerical Accuracy

FP16 GroupNorm with FP32 accumulators in the Triton kernel provides
accuracy equivalent to the `GroupNorm32` FP32 upcast used in the original
LDM code, while avoiding redundant dtype conversions.

Expected mean relative error vs ORT FP32: `< 2%` for both VAE components.
UNet: `< 2%` mean error (K/V attention paths may show slightly higher variance
at very low sequence lengths).

## Running the Benchmark

```bat
.venv/Scripts/python custom_kernels/ref_ldm/benchmark_ref_ldm.py
```
