# Custom Kernels Implementation Findings & Optimization Plan

## 1. Executive Summary
The "Custom" provider delivers significant performance gains through PyTorch-native implementations, Triton kernels, and CUDA graphs. However, the current implementation contains several architectural bottlenecks that prevent full utilization of modern GPUs, especially when multiple worker threads are active. The primary issue is **over-serialization** due to global locks and shared workspaces.

## 2. Technical Findings

### 2.1. Serialization Bottlenecks (Global Locks)
Each processor class (`FaceMasks`, `FaceSwappers`, `FaceLandmarkDetectors`, `FaceRestorers`) employs a single `_custom_inference_lock`. This lock serializes ALL calls to any custom kernel within that class.
- **Problem**: If `FaceMasks` is running `FaceParser`, any other worker thread wanting to run `XSeg` or `Occluder` must wait, even though these models use entirely different static buffers and CUDA graphs.
- **Impact**: Parallelism is effectively disabled for the "Custom" provider, making it slower than ORT (which allows concurrent execution) in multi-threaded scenarios.

### 2.2. cuBLASLt Workspace Race Condition
The `InSwapperTorch` cuBLASLt extension (Phase 3) uses a shared 4 MiB workspace per `op_id` (one per style block).
- **Problem**: Simultaneous calls to the same `op_id` from different threads will use the same GPU workspace memory, leading to data corruption.
- **Current Fix**: A global `_batched_inference_lock` in `FaceSwappers` serializes all batched InSwapper calls.
- **Impact**: Significant performance hit during high-resolution processing (pixel-shift) where many tiles are processed.

### 2.3. Missing Fused Kernels (NHWC)
The `triton_adain` kernel has optimized paths for NCHW (including fused residual) and NHWC (B=1 and B>1), but lacks a **fused NHWC + residual** variant.
- **Current Behavior**: It falls back to a separate `y = y + residual` addition kernel.
- **Impact**: One extra memory pass over the activation volume per style block (12 times per InSwapper pass).

### 2.4. Ignored Model Outputs
In `app/processors/face_landmark_detectors.py`, the `FaceBlendShapes` model is called, but its output is completely ignored in both the "Custom" and ORT paths.
- **Status**: Likely a bug or incomplete feature implementation.

### 2.5. Redundant Memory Clones
Most CUDA graph runners return `.clone()` of the static output buffer to ensure thread safety outside the lock.
- **Opportunity**: If the caller immediately moves the data to CPU or performs a destructive operation (like `argmax`), we can optimize the memory path.

## 3. Implementation Plan

### Phase 1: Granular Locking (Immediate Priority)
- [ ] **Refactor Class Locks**: Replace global `_custom_inference_lock` with per-runner locks. Each `_CapturedGraph` or `Runner` instance should own its own `threading.Lock`.
- [ ] **InSwapper Lock Optimization**: Move `_custom_inference_lock` specifically to the `Inswapper128` and `W600K` runners respectively.

### Phase 2: Parallel Inference (High Impact)
- [ ] **Per-Worker Runners**: For the most compute-heavy models (`FaceParser`, `Inswapper128`), implement a small pool of runners (e.g., 2 instances). This allows two workers to run inference truly in parallel at the cost of ~50-100 MB VRAM.
- [ ] **Thread-Safe cuBLASLt**: Modify the C++ extension to support multiple workspaces or allocate a workspace from a pool per-call to remove the `_batched_inference_lock`.

### Phase 3: Fused Kernel Upgrades
- [ ] **Triton NHWC+Residual**: Implement `_adain_nhwc_batched_with_residual` in `triton_ops.py`.
- [ ] **Direct GPU-to-CPU Copy**: Add a `runner.copy_to_cpu(dst_numpy)` method to bypass the intermediate GPU `clone()` when data is destined for the CPU.

### Phase 4: Bug Fixes & Cleanup
- [ ] **FaceBlendShapes**: Investigate and fix the ignored output if it provides useful "score" data for landmark quality filtering.
- [ ] **CUDAGraphRunner Sync**: Ensure `self._inp.copy_(x, non_blocking=True)` is followed by proper synchronization or is captured within the graph to avoid race conditions on the input buffer.

## 4. Conclusion
By moving from class-level serialization to runner-level concurrency and fixing the cuBLASLt workspace sharing, we can achieve nearly linear scaling with the number of worker threads for the "Custom" provider. The VRAM overhead for doubling the static buffers is negligible (<200 MB) compared to the performance gains.
