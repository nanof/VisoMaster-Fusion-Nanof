# landmark_1k3d68 — Custom FP16 + CUDA-graph Kernel

FP16 PyTorch reimplementation of `model_assets/1k3d68.onnx` — the InsightFace
3D 68-point face landmark detector used for `FaceLandmark3d68`.
Used by VisoMaster-Fusion when the *Custom* execution provider is selected.

## Architecture

| Component | Detail |
|-----------|--------|
| Input     | `(1, 3, 192, 192)` float32 — ImageNet-normalised face crop |
| Output    | `(1, 3309)` float32 — raw regressed values; caller takes last `68×3` slice |
| Backbone  | Pre-activation ResNet-50 (MXNet / InsightFace style, depths 3-4-6-3) |
| BN        | **Not folded** — 18 live `BatchNorm2d` layers (`eps=2e-5`) |
| Head      | `Conv2d(2048→256, 3×3, s=2)` → `ReLU` → `Flatten` → `Linear(2304→3309)` |
| Params    | ~25.5 M |

### Pre-activation bottleneck

```
BN(in_ch) ──→ ReLU ──┬──→ Conv1×1(bias) → ReLU → Conv3×3(stride,bias) → ReLU → Conv1×1(no-bias) ──┐
                      │                                                                               Add → output
                      └──→ shortcut Conv1×1(stride, no-bias)  [first unit only; identity otherwise] ──┘
```

## Benchmark Results

GPU: NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · ORT 1.22.0
(50 iterations, 10 warm-up, batch=1)

| Tier | Method | Time | vs ORT CUDA EP |
|------|--------|------|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 1.855 ms | 1.00× |
| 0b   | ORT TensorRT EP FP32 | 1.653 ms | 1.12× |
| 1    | PyTorch FP32 eager | 6.229 ms | 0.30× |
| 2    | PyTorch FP16 eager | 7.269 ms | 0.26× |
| 3    | **PyTorch FP16 + CUDA graph (Custom)** | **0.819 ms** | **2.27×** |

The CUDA-graph path (tier 3) is **2.27× faster** than ORT CUDA EP.

### Speed-up breakdown (tier 3)

- FP16 cuDNN TensorCore convolutions (~2× on Ampere/Ada)
- CUDA graph elimination of per-kernel CPU launch overhead (~0.1–0.2 ms for
  a 54-conv network at 192 px)

## Accuracy

| Mode | Max |Δ| vs ORT FP32 |
|------|------------------------|
| FP16 + CUDA graph | 0.0188 ✓ |

## Files

| File | Purpose |
|------|---------|
| `landmark_1k3d68_torch.py` | `Landmark1k3d68Torch` model, `from_onnx()` loader, `build_cuda_graph_runner()` |
| `benchmark_1k3d68.py`      | Four-tier benchmark vs ORT + numerical accuracy check |

## Weight Loading

| Parameter type | Strategy |
|----------------|----------|
| All 54 `Conv2d` weights/biases | **Positional** — iterated in ONNX Conv-node topological order |
| 18 `BatchNorm2d` (γ, β, μ, σ²) | **Named** — MXNet-style initialiser names (`stage1_unit1_bn1_gamma`, etc.) |
| `Linear` (fc1)               | **Named** — `fc1_weight` / `fc1_bias` |

## Running the Benchmark

```bash
# from repo root
.venv\Scripts\python custom_kernels\landmark_1k3d68\benchmark_1k3d68.py

# override paths / iterations
ONNX_PATH=model_assets/1k3d68.onnx WARMUP=50 ITERS=500 \
    .venv\Scripts\python custom_kernels\landmark_1k3d68\benchmark_1k3d68.py
```

## Application Integration

When the inference provider is set to **Custom**, `detect_face_landmark_3d68()`
in `app/processors/face_landmark_detectors.py` calls `_get_1k3d68_runner()`,
which lazy-loads `Landmark1k3d68Torch` and wraps it in a CUDA graph runner.
The same pre-processing (ImageNet normalisation, 192-px arcface128 warp) and
post-processing (reshape → last 68 rows → inverse affine transform) are applied
as in the ORT path — the custom kernel is a pure drop-in replacement.
