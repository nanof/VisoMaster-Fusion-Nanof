# Cleanup Push TODO

Items deferred during rebase / PR 80 cherry-pick that should be cleaned up in a follow-up commit.

---

## ~~1. Dead attributes in `frame_worker.py` `__init__`~~ ✅ DONE

**File:** `app/processors/workers/frame_worker.py` lines 163–165

During the "GPU oriented changes" cherry-pick (commit `c6b67d0`), all usages of the lazy
per-call kernel creation (`_lap_kernel`, `_sobel_x`, `_sobel_y`) were replaced with the
pre-allocated `kernel_lap`, `kernel_sobel_x`, `kernel_sobel_y` (lines 181–188).

The three `None`-initialised attributes are now dead code:

```python
# REMOVE these three lines (no longer used):
self._lap_kernel: torch.Tensor | None = None
self._sobel_x: torch.Tensor | None = None
self._sobel_y: torch.Tensor | None = None
```

---

## ~~2. Dead `_blur_cache` and `_clip_transform` attributes in `face_masks.py` `__init__`~~ ✅ DONE

**File:** `app/processors/face_masks.py` line 31

During the "GPU oriented changes" cherry-pick, all three `_blur_cache` usages (FM-10) were
replaced with `v2.functional.gaussian_blur` direct calls. The cache dict init is now dead:

```python
# REMOVE this line (no longer used):
self._blur_cache: Dict[tuple, transforms.GaussianBlur] = {}
```

Also check if `Dict` from `typing` can be removed from the import list if it's no longer
used elsewhere in the file after this removal.

---

## 3. Deferred: STRATEGY_SEGMENTED audio in `video_processor.py`

**Context:** Upstream commit `7645464` (chyanbo/dev, merged into origin/dev) added segmented
audio extraction for skipped frames inside `_finalize_default_style_recording()`. This was
deferred during the complex rebase conflict resolution in favour of preserving our existing
try-block structure.

**What to add:** After the FFmpeg merge/copy step in `_finalize_default_style_recording()`,
incorporate the upstream `STRATEGY_SEGMENTED` path that extracts audio only from non-skipped
frame ranges using ffprobe + segment list, rather than the full-duration fallback.

Reference: `git show 7645464 -- app/processors/video_processor.py`

---

## 4. TensorRT lazy-init from PR 80 commit 2 (optional / low priority)

**File:** `app/processors/utils/tensorrt_predictor.py`

During cherry-pick of `c6b67d0` ("GPU oriented changes"), the PR's lazy context-pool
initialisation (create contexts on-demand via `_get_context()` instead of pre-allocating
`pool_size` contexts at startup) was skipped because it was a large structural change that
conflicted with our T-01–T-05 fixes.

The PR's approach saves startup VRAM by deferring context creation. Consider applying it in a
dedicated commit after verifying the T-01–T-05 fixes are compatible.

Reference: `git show elricfae/dev:app/processors/utils/tensorrt_predictor.py`
(or `git show 3a54739 -- app/processors/utils/tensorrt_predictor.py`)

---

## Notes

- All other conflicts in the rebase / cherry-pick were fully resolved; no other known deferred items.
- Items 1 and 2 are pure dead-code removal (safe, low risk).
- Item 3 is a functional improvement (audio quality for recordings with skipped frames).
- Item 4 is a performance optimisation (lower startup VRAM for TensorRT users).
