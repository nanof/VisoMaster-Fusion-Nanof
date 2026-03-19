# landmark_203 — Custom FP16 + CUDA-graph Kernel

FP16 PyTorch + Triton reimplementation of `model_assets/landmark.onnx` — the
ConvNeXt-Tiny 203-point face landmark detector used for `FaceLandmark203`.
Used by VisoMaster-Fusion when the *Custom* execution provider is selected.

## Architecture

| Component | Detail |
|-----------|--------|
| Input     | `(1, 3, 224, 224)` float32 — face crop in [0, 1] |
| Outputs   | `(1, 214)` coeff · `(1, 262)` lmk · `(1, 406)` pts |
| Backbone  | ConvNeXt-Tiny — channels [96, 192, 384, 768], depths [3, 3, 9, 3] |
| Params    | ~27 M |

Only output index 2 — `(1, 406)` — is used by `detect_face_landmark_203()`.
It contains 203 (x, y) landmark coordinates normalised to [0, 1].

### ConvNeXt block

```
dwconv 7×7 (grouped) → NCHW→NHWC → LayerNorm(C) →
Linear(C→4C) → GELU → Linear(4C→C) → gamma (layer-scale) →
NHWC→NCHW → residual add
```

### Output heads

```
stage2 output (384-ch) → GAP → norm_s3  ──┐
                                            cat → fc_pts → (1, 406)
stage3 output (768-ch) → GAP → norm      ──┘
                                            └→ fc_coeff → (1, 214)
                                            └→ fc_lmk   → (1, 262)
```

## Benchmark Results

GPU: NVIDIA GeForce RTX 4090 · PyTorch 2.8.0+cu129 · CUDA 12.9 · Triton 3.6.0 · ORT 1.22.0
(50 iterations, 10 warm-up, batch=1)

| Tier | Method | Time | vs ORT CUDA EP |
|------|--------|------|:--------------:|
| 0    | ORT FP32 CUDA EP (baseline) | 4.677 ms | 1.00× |
| 0b   | ORT TensorRT EP FP32 | 1.750 ms | 2.67× |
| 1    | PyTorch FP32 eager | 3.364 ms | 1.39× |
| 2    | PyTorch FP16 eager | 4.266 ms | 1.10× |
| 3    | PyTorch FP16 + Triton LN (eager) | 4.841 ms | 0.97× |
| 4    | **PyTorch FP16 + Triton LN + CUDA graph (Custom)** | **0.767 ms** | **6.09×** |

The CUDA-graph path (tier 4) is **6.09× faster** than ORT CUDA EP.

Note: Triton LayerNorm adds overhead in eager mode (tier 3) but pays off significantly
once the CUDA graph eliminates per-kernel CPU launch overhead (tier 4).

### Speed-up breakdown (tier 4)

- FP16 TensorCore convolutions in the 22 Conv2d layers (~2× on Ampere/Ada)
- Triton LayerNorm: FP16 I/O with FP32 accumulators, single-pass for rows up to 768 elements
- CUDA graph elimination of per-kernel CPU launch overhead for this 40+ op network

## Accuracy

| Mode | Max |Δ| vs ORT FP32 |
|------|------------------------|
| FP16 + Triton LN + CUDA graph | 0.0005 (landmark coords) ✓ |

> (TRT EP: 2.67× faster than ORT CUDA EP. Custom kernel beats TRT EP by 2.28× at 0.767 ms vs 1.750 ms.)

## Files

| File | Purpose |
|------|---------|
| `landmark_203_torch.py` | `Landmark203Torch` model, `from_onnx()` loader, `build_cuda_graph_runner()` |
| `benchmark_landmark_203.py` | Five-tier benchmark vs ORT + numerical accuracy check |

## Weight loading

| Parameter type | Strategy |
|----------------|----------|
| 150 named initializers (dwconv, block norm, gamma, head norms, Gemm heads) | **Named** — loaded via `load_state_dict(strict=False)` |
| 4 × channel-first LN (scale + bias, shape `(C,1,1)`) | **Positional** — anonymous `onnx::Mul_NNN` / `onnx::Add_NNN` in topological order: stem-LN → dl1-LN → dl2-LN → dl3-LN |
| 36 anonymous MatMul weights (pwconv1/pwconv2 for each of 18 blocks) | **Positional** — anonymous `onnx::MatMul_NNN` in forward-pass order; each transposed `.T` for `nn.Linear.weight` |

## Running the Benchmark

```bash
# from repo root
.venv\Scripts\python custom_kernels\landmark_203\benchmark_landmark_203.py

# override paths / iterations
ONNX_PATH=model_assets/landmark.onnx WARMUP=50 ITERS=500 \
    .venv\Scripts\python custom_kernels\landmark_203\benchmark_landmark_203.py
```

## Application Integration

When the inference provider is set to **Custom**, `detect_face_landmark_203()`
in `app/processors/face_landmark_detectors.py` calls `_get_landmark203_runner()`,
which lazy-loads `Landmark203Torch` and wraps it in a CUDA graph runner.
The same pre-processing (face warp to 224 px, [0,1] normalisation) and
post-processing (reshape → scale by 224 → inverse affine) are applied as in
the ORT path — the custom kernel is a pure drop-in replacement.
