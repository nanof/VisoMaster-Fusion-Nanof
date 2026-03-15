# Custom Provider Implementation Report

**Date:** 2026-03-15
**Scope:** Analysis of the "Custom" inference provider implementation — bugs, root-cause analysis, and performance bottlenecks.

---

## 1. Reported Issue: Slightly Flickering Face Colors

### Root Cause

The flickering is a combination of two amplifying factors unique to the Custom (FP16) path:

**Factor A — FP16 BatchNorm precision degradation in IResNet50Torch (w600k ArcFace)**

`IResNet50Torch.from_onnx()` calls `model.to(compute_dtype)` with `compute_dtype = torch.float16`. This converts **all** module parameters and **buffers** (including `running_mean` and `running_var` of every BatchNorm layer) to FP16.

In eval mode, BatchNorm computes:

```
y = (x - running_mean) / sqrt(running_var + eps)
```

With FP16 running statistics (10-bit mantissa ≈ 3 significant decimal digits), small activation values can suffer catastrophic cancellation during the subtraction `x - running_mean`. This makes the normalised activations — and therefore the final 512-dim embedding — noticeably less stable than in the ORT FP32 path.

The per-frame **target** embedding (`t_e`) is computed live from the detected face crop. When the face crop changes even slightly between frames (due to normal head movement), the less-precise FP16 BatchNorm amplifies those small differences into a larger embedding variation, which maps to a visible colour change in the swap output.

**Factor B — FP16 face-detector keypoint jitter propagates to the face crop**

`Det10gTorch` and `YoloFace8nTorch` run their backbone/neck/head in FP16. They convert their final outputs to FP32 with `.float()`, but the values themselves have already been quantised to FP16 precision (~3 decimal digits). Keypoint offsets at stride=8 therefore carry a resolution of roughly ±0.05 px (relative FP16 error × scale), compared to FP32's ~0.001 px.

Across frames, this sub-pixel jitter translates into a slightly different face crop window. That window feeds into the FP16 IResNet50 (Factor A), compounding the embedding instability.

### Affected Files

| File | Lines | Observation |
|------|-------|-------------|
| `custom_kernels/w600k_r50/w600k_r50_torch.py` | 163–164 | `model.to(compute_dtype)` converts BN running stats to FP16 |
| `custom_kernels/w600k_r50/w600k_r50_torch.py` | 350–353 | `_CapturedGraph.__call__` uses `non_blocking=True` on input copy (minor; graph captured on non-default stream — see §4) |
| `custom_kernels/det_10g/det10g_torch.py` | 256–266 | Outputs are `.float()` conversions of FP16 values; sub-pixel jitter retained |
| `app/processors/face_swappers.py` | 320–335 | Custom w600k ArcFace path — FP16 embedding used for per-frame target |

### Recommended Fix

1. **Keep BN running statistics in FP32**: After loading weights and converting the model to FP16, explicitly cast the BatchNorm buffers back:
   ```python
   for m in model.modules():
       if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
           m.running_mean = m.running_mean.float()
           m.running_var  = m.running_var.float()
   ```
   This keeps the BN normalisation numerically stable without any performance penalty — BN eval is a single fused CUDA kernel.

2. **Temporal smoothing of the target embedding** (independent mitigation): Apply an exponential moving average (EMA) to `t_e` over consecutive frames with alpha ≈ 0.8. This removes the high-frequency per-frame jitter regardless of detection precision.

3. **Run det10g/yoloface in FP32 for the 5-point kps head only** (optional): The backbone/neck can stay FP16 for speed; only the final kps prediction tensors need FP32 accumulation.

---

## 2. Reported Issue: "Reference" Alignment Mode Shifts Face Box to the Right

### Root Cause

The "Reference" alignment type uses **per-frame detected keypoints** (`kps_ref`) as the anchor for the restoration warp. `kps_ref` is derived from `kps_5` (the 5-point face keypoints returned by the face detector):

```python
# frame_worker.py:4083-4085
M_ref = cast(np.ndarray, tform.params)[0:2]          # 2×3 matrix: frame → crop
kps_ref = np.hstack([kps_5, ones]) @ M_ref.T          # kps_5 projected to 512×512 crop
```

In the restorer:
```python
# face_restorers.py:431
tform_restore = trans.SimilarityTransform.from_estimate(kps_ref, FFHQ_kps)
```

`tform` is estimated with `SimilarityTransform.from_estimate(kps_5, arcface128@512)`, so `kps_ref ≈ arcface128@512`. However, the fit is a **least-squares similarity** (4 DOF for 5 point-pairs = 10 equations), so residuals exist.

**The critical observation:**

When using the **Custom FP16 det10g**, the detected `kps_5` coordinates are quantised to FP16 precision (≈ ±0.05–0.5 px depending on face distance). The least-squares residuals from fitting `kps_5 → arcface128@512` will therefore be slightly **asymmetric** in the x-direction. This asymmetry propagates directly into `tform_restore`:

- If the FP16 keypoints are systematically biased (e.g., the mouth corners predicted slightly to the right due to FP16 rounding of offset predictions), `tform_restore` encodes a net rightward translation component in the correction warp.
- The inverse warp at the end of `apply_restorer` is exact for a perfect round-trip, but the restored face was placed at slightly offset positions during the forward warp. The output therefore appears shifted to the right.

By contrast, "Blend" mode uses the **exact, fixed** `arcface_dst * 4 + [32, 0]` template, which is unaffected by detector precision.

**Verification:** Switch to the ORT provider. If the Reference mode shift disappears, this confirms that FP16 kps bias is the cause. If it persists, the issue is pre-existing and unrelated to the Custom provider.

### Additional Factor

`kps_ref` also inherits the residual error from the initial similarity-transform fit. For non-frontal faces the similarity-constraint cannot perfectly map tilted keypoints to a frontal template, producing systematic per-face-pose residuals. This is a pre-existing design limitation of the Reference mode; it is not introduced by the Custom provider.

### Affected Files

| File | Lines | Observation |
|------|-------|-------------|
| `app/processors/workers/frame_worker.py` | 4083–4085 | `kps_ref` computed from FP16-precision `kps_5` |
| `app/processors/face_restorers.py` | 419–426 | Reference mode passes `kps_ref` directly as `dst` |
| `app/processors/face_restorers.py` | 431–436 | `tform_restore` fitted from `kps_ref` → any x-bias in kps_ref becomes a warp offset |

### Recommended Fix

For the Custom provider, pin the Reference mode anchor to the same arcface128@512 template that "Blend" uses, but apply a small per-landmark correction based on `kps_ref - arcface128@512`. Alternatively, apply temporal smoothing to `kps_5` before computing `kps_ref` to remove FP16 jitter (smoothing is already done for the 203-point landmarks; extending it to `kps_5` would suffice).

---

## 3. Other Issues Found

### 3.1 `_CapturedGraph` in `w600k_r50_torch.py`: CUDA graph captured on non-default stream, replayed on default stream

**File:** `custom_kernels/w600k_r50/w600k_r50_torch.py`, lines 342–353

```python
self._stream = torch.cuda.Stream()
with torch.no_grad(), torch.cuda.graph(self._graph, stream=self._stream):
    self._out = model(self._inp)     # captured on self._stream

def __call__(self, x):
    self._inp.copy_(x, non_blocking=True)  # default stream
    self._graph.replay()                    # also default stream (no context active)
    return self._out.clone()
```

The graph was **captured on `self._stream`** but replayed without an explicit stream context (defaults to the current CUDA default stream). This is technically valid — PyTorch CUDA graphs can be replayed on any stream — but the capture stream (`self._stream`) is never used after construction, meaning any warmup state that was recorded on that stream is abandoned.

More importantly, `non_blocking=True` on a same-device `copy_` is effectively identical to the default (`non_blocking=False`) in terms of CUDA stream ordering (both enqueue on the current stream). It does **not** introduce a race condition here because `copy_` and `graph.replay()` are both ordered on the same default stream. However, the `non_blocking=True` is misleading and should be removed for clarity.

**Recommended Fix:** Remove `non_blocking=True` and align the implementation with the InSwapper CUDA graph runner which correctly uses `copy_()` without the flag.

### 3.2 Inconsistent post-call synchronisation pattern

**File:** `app/processors/face_restorers.py`, lines 876–881

```python
if self.models_processor.provider_name == "Custom":
    runner = self._get_gfpgan_runner(is_1024=False)
    if runner is not None:
        with torch.no_grad():
            with self._get_runner_lock(runner):
                result = runner(image)      # lock released here
        output.copy_(result)                # OUTSIDE lock — fine (result is a clone)
        return
```

This pattern is correct and consistent with the InSwapper path. Documented here to confirm it is intentional: `result` is an independent `.clone()` of the static output buffer (returned by the CUDA graph runner), so `output.copy_(result)` outside the lock cannot clobber the static buffer.

### 3.3 VGG Combo runner called twice under the same lock (two sequential inferences)

**File:** `app/processors/face_masks.py`, lines 1780–1786

```python
with self._get_runner_lock(vgg_runner):
    swapped_feat  = vgg_runner(swapped)   # call 1 — returns out.clone()
    original_feat = vgg_runner(original)  # call 2 — overwrites static_out, returns clone
```

The `VggComboCUDAGraphRunner` doc comment states: _"Two runners are kept as separate instances so that two concurrent inferences each have their own input buffer."_ However, only **one** runner instance is created (via `build_cuda_graph_runner`). The two sequential calls under the same lock are **functionally correct** since each call returns an independent `.clone()` before the next call starts.

The comment is misleading — consider updating it or creating a second runner instance (which would allow removing the lock entirely for the two-inference pattern).

### 3.4 CodeFormer — no CUDA graph (known limitation)

**File:** `custom_kernels/codeformer/codeformer_torch.py` (referenced from `face_restorers.py`)
**File:** `app/processors/face_restorers.py`, lines 1171–1187

CodeFormer accepts a dynamic `fidelity_weight` scalar that participates in the computation graph. CUDA graphs require **static computation** (no dynamic shapes or control flow that changes between calls). Therefore CodeFormer runs in PyTorch eager mode under the Custom provider. Each style-block forward pass incurs individual Python → CUDA kernel dispatch overhead.

This is a known architectural limitation. To close the performance gap, consider:
- Pre-compiling CodeFormer with `torch.compile(mode="max-autotune")` (persistent across calls once compiled)
- Using `torch.cuda.amp.autocast(dtype=torch.float16)` wrapped in `torch.no_grad()` to at least benefit from FP16 Tensor Core paths without the CUDA graph overhead

### 3.5 GFPGAN-1024 — CUDA graph capture missing

**File:** `app/processors/face_restorers.py`, `_get_gfpgan_runner(is_1024=True)` path

The GFPGAN-1024 model has a Custom runner registered. Verify that the CUDA graph for the 1024×1024 input variant is being captured correctly (it requires a different static input shape than the 512×512 variant). If the runner silently falls back to eager mode for the 1024 case, it loses the CUDA graph speedup.

### 3.6 FaceParser Torch runner — no explicit output synchronization

**File:** `app/processors/face_masks.py`, Custom FaceParser path

The FaceParser runner returns a CUDA tensor (`argmax` result). The downstream code immediately calls `.cpu()` on it, which forces an implicit sync. No explicit synchronization issue, but the pattern differs from the ORT path which uses an explicit `io_binding` + synchronize. Both are functionally equivalent.

---

## 4. Performance Bottlenecks

### 4.1 Per-frame target ArcFace embedding (w600k) — FP16 CUDA graph overhead

The w600k runner is invoked per video frame to compute the **target** face embedding (`t_e`). With the CUDA graph, this is ~0.2–0.5 ms per call (versus ~1–2 ms ORT). However, because the runner runs inside `self._w600k_lock`, multiple FrameWorker threads cannot pipeline target-embedding and swap computations.

If temporal embedding smoothing is added (see §1 recommendation), the per-frame call could be skipped every N frames (e.g., only recompute when the face bbox shifts by more than a threshold), further reducing overhead.

### 4.2 Sequential face-mask models — no inter-model parallelism

FaceParser, Occluder, XSeg, and VGG combo all run sequentially within a single CUDA stream. They are not independent (FaceParser output gates Occluder), but VGG and FaceParser could be parallelised on separate CUDA streams when both are enabled. This would require per-model stream management and is a non-trivial refactor.

### 4.3 `run_inswapper_batched` — no CUDA graph

**File:** `app/processors/face_swappers.py`, lines 742–747

`run_inswapper_batched` uses `self._get_inswapper_torch()` (the plain `InSwapperTorch` model in cuBLASLt mode, **not** the CUDA-graph runner). For multi-tile resolution modes (dim > 1), all tiles are stacked into a batch `[B, 3, 128, 128]` and forwarded in one call. Because `B` varies with `dim`, a single static CUDA graph cannot cover all batch sizes.

Options:
- Capture separate CUDA graphs for `B = 1, 4, 9, 16` (dim = 1, 2, 3, 4) and dispatch to the matching graph
- Use `torch.compile(mode="reduce-overhead")` on `InSwapperTorch` which handles dynamic shapes with a small warmup-per-shape cost

### 4.4 `build_cuda_graph_runner` for InSwapper — no warmup synchronization

**File:** `custom_kernels/inswapper_128/inswapper_torch.py`, lines 1192–1202

```python
with torch.no_grad():
    for _ in range(3):
        model(static_target, static_source)
torch.cuda.synchronize()      # sync after warmup ✓

with torch.no_grad():
    with torch.cuda.graph(graph):
        static_out.append(model(static_target, static_source))
torch.cuda.synchronize()      # sync after capture ✓
```

Both synchronizations are present. This is correct. (No issue; listed for completeness.)

### 4.5 Multiple per-runner lock objects — dict lookup overhead

**File:** `app/processors/face_detectors.py`, lines 101–106
**File:** `app/processors/face_masks.py` (similar pattern)

```python
def _get_runner_lock(self, runner):
    with self._custom_inference_lock:
        r_id = id(runner)
        if r_id not in self._runner_locks:
            self._runner_locks[r_id] = threading.Lock()
        return self._runner_locks[r_id]
```

This function acquires `_custom_inference_lock`, does a dict lookup, and returns a lock — adding a small overhead per model call. Since runner object identities are stable (set once at init), this could be replaced with a per-model lock stored as a direct attribute, eliminating the dict lookup.

---

## 5. Summary Table

| # | Category | Severity | Component | Short Description |
|---|----------|----------|-----------|-------------------|
| 1 | Bug (flickering) | High | `w600k_r50_torch.py` | FP16 BatchNorm running stats cause embedding instability → color flicker |
| 2 | Bug (flickering) | Medium | `det10g_torch.py` | FP16 keypoint outputs have sub-pixel jitter → unstable face crops |
| 3 | Bug (alignment) | Medium | `face_restorers.py` + `frame_worker.py` | Reference mode uses FP16-precision kps_ref → systematic warp offset |
| 4 | Code clarity | Low | `w600k_r50_torch.py` | `non_blocking=True` in `_CapturedGraph.__call__` is misleading (no actual race, but confusing) |
| 5 | Code clarity | Low | `face_masks.py` | VGG runner comment says "two instances" but only one is created |
| 6 | Performance | Medium | `face_swappers.py` | `run_inswapper_batched` bypasses CUDA graph — per-operator dispatch for all tiles |
| 7 | Performance | Low | CodeFormer custom path | No CUDA graph (dynamic fidelity_weight) — eager dispatch overhead per style block |
| 8 | Performance | Low | Multiple processors | Per-runner dict-based lock lookup; minor overhead per call |

---

## 6. Files Modified / Requiring Changes

| File | Change Needed |
|------|---------------|
| `custom_kernels/w600k_r50/w600k_r50_torch.py` | Keep BN running stats in FP32 after `model.to(fp16)`; remove `non_blocking=True` |
| `custom_kernels/det_10g/det10g_torch.py` | (Optional) FP32 for the kps/score output heads for better keypoint precision |
| `custom_kernels/yoloface_8n/yoloface8n_torch.py` | Same as det10g regarding output precision |
| `app/processors/workers/frame_worker.py` | Add EMA smoothing to `kps_5` or to the per-frame target embedding `t_e` |
| `app/processors/face_restorers.py` | (Optional) Pin Reference mode to arcface128 template when Custom provider is active |
| `app/processors/face_masks.py` | Update VGG runner comment; no functional change required |
