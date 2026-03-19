# Custom Kernels — vgg_combo (VGG Perceptual Feature Extractor)

FP16 PyTorch reimplementation of `vgg_combo_relu3_3_relu3_1.onnx` — a VGG
feature extractor used for perceptual difference masking.  Produces two sets
of VGG block-3 features concatenated along the channel dimension.  Used as
the **VGG combo** model in the application (face-compare mask generation).

Model: **vgg_combo**  `(N,3,512,512)f32 → (N,512,128,128)f32`
(supports batch=1 and batch=2; fixed spatial dims at inference)

---

## Benchmark Results (RTX 4090, CUDA 12.9, PyTorch 2.8+cu129, ORT 1.22.0, input 512×512)

50 iterations, 10 warm-up.

| Tier | Method | ms | vs CUDA EP |
|------|--------|---:|-----------:|
| 0 | ORT FP32 CUDA EP | 5.445 | 1.00x |
| 0b | ORT TRT EP FP32 | 6.488 | 0.84x ⚠ (slower than CUDA EP) |
| 1 | PyTorch FP32 | 3.454 | 1.58x |
| 2 | PyTorch FP16 | 1.415 | 3.85x |
| **3** | **PT FP16 + CUDA graph batch=1 (Custom)** | **1.846** | **2.95x** |
| 4 | PT FP16 + CUDA graph batch=2 | 4.687 | 2.32x per-pair |

> **Application uses Tier 3** (batch=1 CUDA graph; one call per face image).
>
> Note: PyTorch FP32 is already 1.58x faster than ORT CUDA EP for this model.
> VGG feature extraction with large 512×512 feature maps is heavily
> memory-bandwidth bound; FP16 halves the bandwidth requirement.
>
> ORT TRT EP is **slower** than ORT CUDA EP (6.488 ms vs 5.445 ms) — TRT does
> not benefit this all-conv architecture at 512×512.
>
> The batch=2 variant (Tier 4) combines both face inferences (swapped +
> original) into one call. At 4.687 ms total it delivers 2.32x vs two ORT
> CUDA EP calls (10.89 ms), equivalent to 2.32x for the two-image workload.

Run `benchmark_vgg_combo.py` to measure on your hardware.

### Speed-up Source

VGG is a **pure Conv2d / ReLU / MaxPool architecture** with no normalization.
Speed-up comes from:

1. **FP16** — cuDNN dispatches Conv2d on FP16 weights to TensorCore GEMM kernels
   (~2× throughput vs FP32 on Ampere/Ada GPUs; dominant gain here due to large
   512×512 feature maps being memory-bandwidth bound).
2. **CUDA graph** — eliminates Python/CUDA kernel-launch overhead (marginal
   additional gain at this scale; graph captured once per batch size).

No Triton kernels are used (no normalization layers to fuse).

---

## Architecture

Reverse-engineered from `model_assets/vgg_combo_relu3_3_relu3_1.onnx`:

```
Input:  (N, 3, 512, 512)   float32   — VGG-normalised image
Output: (N, 512, 128, 128) float32   — combo features
```

### Forward Pass

```
Block 1  Conv(  3→ 64, 3×3, p=1) → ReLU
         Conv( 64→ 64, 3×3, p=1) → ReLU
         MaxPool(2×2, s=2)                    → (N,  64, 256, 256)

Block 2  Conv( 64→128, 3×3, p=1) → ReLU
         Conv(128→128, 3×3, p=1) → ReLU
         MaxPool(2×2, s=2)                    → (N, 128, 128, 128)

Block 3  conv3_1: Conv(128→256, 3×3, p=1)    → pre_relu3_1  (N, 256, 128, 128)

  Branch A ──────────────────────────────────  feat_A = pre_relu3_1
  Branch B  ReLU(pre_relu3_1)  = relu3_1
            conv3_2: Conv(256→256, 3×3, p=1) → ReLU = relu3_2
            conv3_3: Conv(256→256, 3×3, p=1)          = pre_relu3_3
            ──────────────────────────────────  feat_B = pre_relu3_3

Output:   Concat([feat_A, feat_B], dim=1)    → (N, 512, 128, 128)
```

### Notes on ONNX Graph Details

**Resize nodes:** The ONNX graph contains two Resize nodes (bicubic,
`cubic_coeff_a = -0.75`) that dynamically upsample features to a fixed
`[N, 256, 128, 128]` target.  For the standard 512×512 input the feature maps
are already 128×128 after the two MaxPool layers, so the resize is always a
no-op.  This PyTorch implementation targets 512×512 input and omits the resize.

**Activations:** Despite the model name "relu3_3_relu3_1", the ONNX graph applies
the Resize to the **pre**-ReLU conv outputs (`model.10` and `model.14`).
Branch A contains `pre_relu3_1` and Branch B contains `pre_relu3_3`.

### Parameters

| Layer | Params |
|-------|--------|
| conv1_1 (3→64, 3×3)     |   1,792 |
| conv1_2 (64→64, 3×3)    |  36,928 |
| conv2_1 (64→128, 3×3)   |  73,856 |
| conv2_2 (128→128, 3×3)  | 147,584 |
| conv3_1 (128→256, 3×3)  | 295,168 |
| conv3_2 (256→256, 3×3)  | 590,080 |
| conv3_3 (256→256, 3×3)  | 590,080 |
| **Total**                | **~1.7 M** |

---

## Files

| File | Purpose |
|------|---------|
| `vgg_combo_torch.py` | FP16 PyTorch VGG combo + CUDA graph runner |
| `benchmark_vgg_combo.py` | 4-tier latency benchmark vs ORT baseline |
| `__init__.py` | Package marker |

---

## Application Integration

Select **"Custom"** in *Settings → General → Providers Priority*.

The VGG combo model is then executed via `VggComboTorch` (CUDA graph) in
`face_masks.py` for perceptual difference mask generation:

```python
# Internal call path (face_masks.py):
from custom_kernels.vgg_combo.vgg_combo_torch import VggComboTorch, build_cuda_graph_runner

model  = VggComboTorch.from_onnx(onnx_path).cuda().eval()
runner = build_cuda_graph_runner(model)   # captures CUDA graph once (batch=1)

# Per-call (called twice per frame — swapped face and original face):
image_normalised = ...    # torch.Tensor (1, 3, 512, 512) float32 CUDA
features = runner(image_normalised)       # (1, 512, 128, 128) float32
```

### Batch=2 Variant (combines both face inferences)

```python
runner2 = build_cuda_graph_runner(model, input_shape=(2, 3, 512, 512))
both_images = torch.stack([swapped_norm, original_norm])   # (2, 3, 512, 512)
both_feats  = runner2(both_images)                          # (2, 512, 128, 128)
swapped_feat  = both_feats[0:1]
original_feat = both_feats[1:2]
```

---

## Weight Loading

All 14 ONNX initialisers are named (`model.N.weight` / `model.N.bias`) and
loaded by name:

| ONNX name | PyTorch layer |
|-----------|---------------|
| `model.0.weight/bias`  | conv1_1 |
| `model.2.weight/bias`  | conv1_2 |
| `model.5.weight/bias`  | conv2_1 |
| `model.7.weight/bias`  | conv2_2 |
| `model.10.weight/bias` | conv3_1 |
| `model.12.weight/bias` | conv3_2 |
| `model.14.weight/bias` | conv3_3 |

---

## Numerical Accuracy

Measured FP16 + CUDA graph vs ORT FP32 (200 iterations, 20 warm-up):

| Metric | Value |
|--------|-------|
| Max \|Δ\| | 0.0749 |
| Feature range | [-91.16, 110.72] |
| Relative error | 0.037% of feature range |

The downstream computation averages the L1 difference across 512 channels,
so per-feature FP16 noise averages out effectively.

## Running the Benchmark

```bash
# from repo root
.venv/Scripts/python custom_kernels/vgg_combo/benchmark_vgg_combo.py
```

Optional env vars: `ONNX_PATH`, `WARMUP` (default 20), `ITERS` (default 200).
