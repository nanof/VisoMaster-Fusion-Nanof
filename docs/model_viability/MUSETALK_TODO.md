# MuseTalk lip-sync — TODO tracker

Living checklist of what is done and what remains for the **MuseTalk 1.5** integration in VisoMaster Fusion.
Mark items with `[x]` / `[ ]` and add short notes under each section when status changes.

**Last updated:** 2026-08-10

---

## Done

### Core / engineering

- [x] Isolated package `app/processors/pytorch_extras/musetalk/` (engine, audio, blending, framing, parsing, models, paths)
- [x] Lazy load via `ModelsProcessor.ensure_musetalk_loaded()` / `unload_musetalk()`
- [x] Frame hooks in `FrameWorker` (before / after / hybrid swap order; gated by toggle + bypass)
- [x] Mouth-only local blend (union of original + generated mouth) to avoid double mouth/chin
- [x] GPU batcher (`VISOFUSION_MUSETALK_BATCH`, default 8) for concurrent workers
- [x] Audio prep: video track or external file (Whisper chunks)
- [x] Frame→audio mapping using source FPS (not the capped playback FPS)
- [x] Reload after VRAM flush (`reload_musetalk_if_enabled`)
- [x] Audio refresh when new media is loaded
- [x] Probe / debug (`VISOFUSION_MUSETALK_DEBUG`) without touching the ONNX hot path
- [x] Failures do not break the pipeline (unchanged frame + warn)



### UI controls (Common → MuseTalk Lip-Sync)

- [x] Toggle **Enable MuseTalk Lip-Sync** + load/unload
- [x] Bypass (A/B without unloading)
- [x] Pipeline order: Before the swap / After the swap / Hybrid (before + light after)
- [x] Hybrid re-sync amount slider
- [x] Audio source (Video track / External file) + path
- [x] Extra margin, face index, blend strength, bbox shift
- [x] Landmark crop + face parsing mask
- [x] Mouth-only repaint + padding
- [x] Repaint top, cheek width, lip colour match
- [x] Mouth width / height / centre
- [x] Restore mouth (model + strength) on the 256px crop
- [x] Required global settings adjustment on enable (68 landmarks, etc.)
- [x] Settings → MuseTalk **torch.compile** toggle (`MuseTalkCompileToggle`; env override)



### Models and dependencies

- [x] Entries in `musetalk_assets_list` (`models_data.py`)
- [x] Default download in `download_models.py` (~4 GB)
- [x] Opt-out `--skip-musetalk` / `VISOFUSION_SKIP_MUSETALK=1`
- [x] PyTorch extras deps in `requirements_cu13.txt` / `requirements-pytorch-extra.txt`
- [x] Embedded config `musetalk_v15.json`
- [x] Retalking stub redirected to MuseTalk (readiness messages)



### Tests

- [x] Unit: paths, blend, bbox, framing, landmarks, mask, parsing, lip chroma, mouth detail, mouth-only, pipeline order, audio windows, engine wiring, probe
- [x] Unit UI: toggle direction, slider refresh, required settings
- [x] Integration GPU: batching (audio index ↔ frame)



### Internal docs

- [x] Mention in `docs/model_viability/INTEGRATION_CHECKLIST.md` (status + how to enable)

---



## Pending / follow-up



### Quality and product

- [ ] End-to-end visual validation on real videos (preview + recording)
- [ ] External-audio dubbing check (sync, duration, seek)
- [ ] Multi-face behaviour (`MuseTalkFaceIndexSlider`) and small/distant faces
- [ ] Typical VRAM / FPS profile (note hardware + recommended batch size) — **partial:** RTX 5070 Ti ≈ 1.8 GiB; eager ~10 / ~28 FPS (b1/b8); compile ~14.4 / ~28.4 FPS. Recommend batch 8 when multi-worker; compile is off by default because it only helps at batch 1.
- [ ] Decide whether crop needs temporal stabilisation (landmark jitter)
- [ ] After finish recording try to not open another winfow explorer if there is one opened in the same folder 

### Optimizations

- [x] Optimize MuseTalk model (quantization?) — **FPS pass (2026-08-10):**
  - Already fp16; **not** INT8/TRT this round (quality risk; deferred).
  - `torch.compile` on the UNet, env `VISOFUSION_MUSETALK_COMPILE` (**default off**, opt-in). Falls back to eager if Inductor/Triton unavailable. `torch.compile(vae)` is a no-op — we call `vae.encode()`/`vae.decode()` and `OptimizedModule` delegates everything but `forward()` to the original module — so the **encoder/decoder submodules** are compiled instead (see the VAE hot path entry below; that is where the win turned out to be).
  - Compiled with `dynamic=False` on fixed batch specs (powers of two up to `max_batch`), and `_infer_batch` pads up to the next spec **before the encode**, so the UNet *and* the VAE both stay on warm shapes. Without the padding, playback batches of 3/5/6 triggered an Inductor recompile mid-stream and preview collapsed to **0.15 FPS** with `frame timed out waiting for the batcher`.
  - Hot path: `torch.inference_mode()` + `channels_last` on CUDA.
  - Whisper stays on **CPU** (VRAM only; not hot-path FPS).
  - Bench: `python -m app.processors.pytorch_extras.musetalk.bench_musetalk` (needs CUDA). Load timing: `python scripts/time_musetalk_load.py` (one load per process; the in-process cache makes a second load meaningless).
  - **Measured 2026-08-10** — RTX 5070 Ti, MuseTalk-5070-venv (torch 2.7.1+cu128 + `triton-windows` 3.3.1), WARMUP=3 ITERS=10, whisper=cpu, channels_last=on:
    - VRAM after load ≈ **1805–1813 MiB**; first compile load ≈ **215 s**
    - **compile=off**: b1 **100.4 ms (~10.0 FPS)**; b8 **36.0 ms/frm (~27.8 FPS)** — unet 73.0 / 64.4 ms
    - **compile=on** (`dynamic=False`, specs 1&8): b1 **69.2 ms (~14.4 FPS, 1.44×)**; b8 **35.3 ms/frm (~28.4 FPS)** — unet **43.3 / 33.5 ms** (~1.7–1.9× vs eager unet)
    - Note: `dynamic=True` hung; engine uses fixed batch specializations. Batch-8 E2E barely moves (VAE encode/decode dominate when amortized).
    - **Re-measured 2026-08-10, portable Python, WARMUP=3 ITERS=15, batches 1/4/8** (this is the table that settled it):

      | batch | mode | encode | unet | decode | e2e | ms/frm | FPS~ |
      |---|---|---|---|---|---|---|---|
      | 1 | off | 15.87 | 63.60 | 18.61 | 103.45 | 103.45 | 9.7 |
      | 4 | off | 61.33 | 61.49 | 58.24 | 152.80 | 38.20 | 26.2 |
      | 8 | off | 117.19 | 56.40 | 122.03 | 284.91 | 35.61 | 28.1 |
      | 1 | on | 19.01 | 49.76 | 23.67 | 125.50 | 125.50 | 8.0 |
      | 4 | on | 64.28 | 43.41 | 64.42 | 145.29 | 36.32 | 27.5 |
      | 8 | on | — | — | — | — | — | stalled recompiling >8 min |

    - **Why it shipped off at this point (superseded later the same day):** compile sped up the UNet (63.6 → 49.8 ms at b1) but that is the *minority* of the work. At b8, encode+decode were **239 of 285 ms (84%)** while the UNet was 20%; we had optimized the small half. End to end it was a wash at b4 and *worse* at b1, and b8 stalled recompiling for over 8 minutes. **Resolved by compiling the VAE submodules — see the VAE hot path entry, which supersedes this table.** It remains opt-in, now because of first-run warmup rather than lack of benefit.
    - In-app playback with lip-sync sits at ~**4–6.5 FPS** vs ~16–26 FPS without it, well below the bench's 35 ms/frm, so there is also per-frame cost outside the engine (BiSeNet parsing at 512×512 per frame, mouth restore, blending, GPU↔host round trips) worth profiling with `VISIOMASTER_PERF_STAGES=1`.
    - **Portable Python (2026-08-10):** embeddable builds lacked `Include/`+`libs/` so compile was skipped in the app. Fixed locally + script `scripts/ensure_portable_python_dev_headers.py` (also hooked from `Start_Portable.bat`). `compile_utils` auto-prepends MSVC `cl.exe` when missing from PATH. Needs VS Build Tools/Community once for first Triton host build; then cached.
- [x] MuseTalk VAE hot path — **done 2026-08-10, and it was the whole ballgame.** `_compile_vae()` now compiles `vae.vae.encoder` / `vae.vae.decoder` (the submodules `encode()`/`decode()` actually call) alongside the UNet, and `_infer_batch` pads **before the encode** so the VAE sees a warm shape too. RTX 5070 Ti, WARMUP=3 ITERS=10:

  | batch | mode | encode | unet | decode | e2e | ms/frm | FPS~ |
  |---|---|---|---|---|---|---|---|
  | 1 | off | 19.65 | 81.21 | 22.20 | 93.06 | 93.06 | 10.7 |
  | 4 | off | 64.40 | 61.31 | 64.02 | 165.57 | 41.39 | 24.2 |
  | 8 | off | 135.40 | 60.58 | 132.74 | 304.17 | 38.02 | 26.3 |
  | 1 | on | 14.60 | 43.62 | 14.76 | 72.47 | 72.47 | 13.8 |
  | 4 | on | 40.44 | 39.63 | 51.99 | 108.95 | 27.24 | 36.7 |
  | 8 | on | 78.04 | 35.03 | 85.00 | 191.33 | 23.92 | 41.8 |

  End to end **1.28× (b1) / 1.52× (b4) / 1.59× (b8)**, up from the ~2% the UNet-only compile gave. Batch-8 no longer stalls.
  - **Batch scaling, eager (why the UNet was never the target):** the UNet is a near-**constant** ~53–72 ms from b1 to b8 (launch-overhead bound, it never saturates the GPU), while encode/decode scale **linearly** at ~15 ms/frame each. At b8 the VAE is **88%** of the pass. Amortization saturates by b6: ms/frm 39.1 (b4) → 35.1 (b6) → 34.9 (b8), so raising the worker count from 4 to 8 would buy only ~11% — it is not a lever.
  - **`dynamic=True` rejected (measured):** wins less (encode 1.47× / decode 1.32× vs **1.68× / 1.53×** specialised), takes **186 s** to warm up, and *still* recompiles on the small shapes — first call at batch 1 took **86 s** and batch 2 **76 s**, exactly the mid-playback batcher stall we already fixed once.
  - **Cost is load time, all warmup:** **68–79 s** with a warm on-disk Inductor cache vs ~6–8 s eager, and **~477 s on the very first run** for a given GPU/shape set (portable Python first load with VAE compile: **~319 s**). That first-run cliff is why it stays opt-in via `VISOFUSION_MUSETALK_COMPILE=1`; for anything longer than a short preview it pays for itself quickly.
  - **In-app confirmation (2026-08-10, portable, `VISOFUSION_MUSETALK_PERF=1`, n=768, batch median 3):** apply total **411 ms** vs **530 ms** off (~**1.29×**); infer **212 ms** vs **302** (~**1.43×**); encode/unet/decode **51 / 103 / 50** vs **69 / 160 / 70**. Bench promised ~1.5×; app lands closer to 1.3× on apply because batch is smaller and ~100 ms of parse/mouth/blend is untouched. UNet still ~2.5× slower than isolated bench (GPU contention with BiSeNet).
  - Not pursued: caching the reference-branch latents. `get_latents_for_unet_batch` does encode **2N** images (masked + reference of the *same* crop), but the crop changes every frame in playback, so there is nothing to reuse across frames.
- [x] Profile in-app lip-sync with `VISIOMASTER_PERF_STAGES=1` — **2026-08-10, 670 pool frames, compile=off, load 5.9 s:** total median **616 ms/frame**; **`musetalk_preswap` 551 ms (~89%)**, `std_swap_edit` 47 ms (~8%), `std_recognize` 16 ms (~3%). Rest negligible.
- [x] Break down `apply_frame_bgr` with `VISOFUSION_MUSETALK_PERF=1` — **2026-08-10, n=815, batch median 4:** apply total **530 ms**. **`batch_wait` 420 ms (79%)** of which **`infer` 302 ms** (encode 69 / **unet 160** / decode 70) and **~116 ms** queue/gather beyond infer. Post-GPU: **mouth_only 44** + parse 29 + blend 27 ≈ **100 ms** (restore off). In-app UNet at b≈4 is **~2.6×** the isolated bench (160 vs 61 ms). Aggregator: `scripts/agg_musetalk_perf.py`.
- [ ] ~~Cut second BiSeNet in `mouth_only`~~ **Blocked / re-scoped (2026-08-10):** the parser (`face_masks._faceparser_labels`) is a **fixed 512×512, batch-1** ORT/TRT session (output bound to `(1,19,512,512)`), and the two parses run on **different images** (frame crop vs. generated recon), so there is no reuse or cheap shrink. It also **won't raise FPS directly**: both parses run on the *worker* thread after the GPU handoff and overlap with the next batch's infer. Their only FPS effect is **GPU contention** with the batch loop's UNet (see next item). Real fix would be a **dynamic-batch BiSeNet engine** to fuse both parses into one call (medium effort; needs TRT re-export).
- [ ] Investigate in-app UNet slowdown vs bench. Still open after VAE compile: in-app unet **~103 ms** (compile on, batch≈3) vs isolated compiled unet **~40 ms** at b4 — still ~2.5×. Working hypothesis unchanged: **GPU contention** from concurrent BiSeNet 512² parses while the batch loop runs. VAE compile cut infer 302→212 ms; next FPS lever is reducing that BiSeNet GPU load (or TRT for the VAE if compile load time is unacceptable).
- [x] Optimize Landmark detection — **Profile + fix (2026-08-10, RTX 5070 Ti)**
  - Script: `python scripts/bench_landmarks_profile.py`
  - **Root cause of ~810 ms FaceLandmark203:** `_cuda_ep_memory_options` used `cudnn_conv_algo_search=DEFAULT`, which on this ORT/cuDNN stack puts ConvNeXt (and 106) Convs into cuDNN *Fallback* (~150×). Switched default to **`HEURISTIC`** (still avoids EXHAUSTIVE VRAM growth). Override: `VISIOMASTER_CUDNN_CONV_ALGO_SEARCH`.
  - Also: bind only output `"856"` for 203; feeder/worker second 203 pass is landmark-only (no second RetinaFace); MuseTalk prefers `precomputed_kpss` (68) over `kpss_203`.
  - **A/B CUDA EP only** (`session.run`, same arena/`max_workspace=0`, DEFAULT vs HEURISTIC):
    | Model | DEFAULT | HEURISTIC | speedup |
    |---|---:|---:|---:|
    | FaceLandmark203 | 791.8 ms | **6.6 ms** | **119×** |
    | FaceLandmark106 | 294.3 ms | **1.4 ms** | **210×** |
    | FaceLandmark68 | 43.5 ms | 25.7 ms | 1.7× |
  - **App path after fix** (`run_detect_landmark`, median ms / face): detect 10.3 · lm68 36.2 · **lm203 9.9** · 203→68 46.3 · detect+68 37.0 · detect+203 16.8 (was 810 / 1188 / 985 for 203 paths).
  - Note: with UI provider **TensorRT-Engine**, FaceLandmark203 usually builds TRT and never hits the CUDA-EP Fallback path; the HEURISTIC fix is the safety net whenever a model falls back to CUDA EP (GPEN, TRT miss, or provider=CUDA).
  - 68 remains slower by design (2dfan4 heatmaps @ 256² vs ConvNeXt/106 regression). MuseTalk still needs iBUG-68 for exact crop.
  - Remaining levers: temporal subsample of dense landmarks; cable `SequentialDetector` target-only densos; Custom CUDA-graph path for 203.
- [x] Derive MuseTalk's 68 from the cheap 106 detector — **`MuseTalkFastLandmarksToggle` (default off)**
  - `framing.as_ibug68()` re-indexes 106 → iBUG-68 with the map already shared with DMDNet (`dmdnet_landmarks.landmarks106_to_68_xy`), so the crop keeps the **exact** bridge (index 29) instead of the interpolated one non-68 schemes fall back to.
  - When the toggle is on, enabling MuseTalk forces `LandmarkDetectModelSelection="106"` instead of `"68"` (`musetalk_required_control_settings`).
  - **App path, per face (RTX 5070 Ti, CUDA EP):** exact 68 (2dfan4) **34.5 ms** → fast 106 + reindex **5.1 ms** = **6.8×**, saves ~29.5 ms/face.
  - Trade-off: 106 samples the jaw contour more coarsely than 2dfan4, so the window can move a pixel or two; the bridge, the chin and the jaw-to-jaw width are unchanged. Tests in `test_musetalk_landmark_framing.py` assert 106 frames the same window as its own 68 mapping and ignores a misleading kps_5.
  - Not chosen: FaceFusion's `fan_68_5` (5→68) — cheaper still, but the chin/jaw it invents does not follow an open mouth, which is exactly what lip-sync moves.

### Robustness

- [x] Fill SHA256 hashes in `musetalk_assets_list` (currently `hash: ""`)
- [x] Seek / scrub with lip-sync on (correct chunks, no stalls) — **2026-08-10:** Whisper windows now preserve exact fractional container FPS (no long-clip drift), sought frames use absolute chunk indices instead of wrapping unrelated audio with modulo, and out-of-range audio leaves the frame untouched. Worker cancellation reaches queued MuseTalk requests within 25 ms; stale requests are discarded by the batcher, and pool workers are signalled before feeder/detection shutdown joins so timeline scrubs do not wait on the 20 s batch timeout.
- [x] Long recordings: audio memory, batcher, timeouts — **2026-08-10:** Audio prep streams the WAV in 30 s segments (no full-track `librosa.load` peak), encodes Whisper one segment at a time onto a **float16** host timeline, and keeps lazy `FrameFeatureWindows`. Preview still soft-times out at 20 s; **recording / segment export** pass `request_timeout_s=None` so a slow batch cannot punch lip-sync holes, while `stop_event` still cancels within 25 ms and the batcher drops cancelled requests.
- [x] VR / non-standard layouts behaviour (if applicable) — **2026-08-10:** MuseTalk now lip-syncs each VR180 **perspective crop** after swap / before P2E stitch (`_musetalk_apply_vr_crop`), using crop-local landmarks and a dummy inset bbox (same convention as VR landmark detection). The equirect post-swap hook is **disabled** under `VR180ModeEnableToggle` (it had no feeder detections → `no_bbox` skip, and equirect boxes would be geometrically wrong). Both Eyes / Single Eye reuse the existing E2P→stitch path; flat SBS/OU remains out of scope. Pipeline order Before/Hybrid still do not apply in VR (always after-swap on the crop). Tests: `test_musetalk_pipeline_order.py`.
- [ ] Clearer error messages when weights or deps are missing



### User documentation

- [ ] MuseTalk section in `docs/user_manual.md` / quickstart
- [ ] README note about ~4 GB download and opt-out



### Repo / release

- [x] Commit the integration (`dd1b1c6` on `main-nanof`)
- [ ] PR against the base branch
- [ ] CI checklist: unit green without weights; GPU integration marked `gpu`/`slow`



### Alternative lip-sync models (not MuseTalk)

Keep MuseTalk as the **preview / realtime** path. Explore higher-fidelity or next-gen models only where noted:

- [ ] **LatentSync 1.5/1.6** (ByteDance, Apache-2.0, weights on HF): spike as an **export-only** lip-sync path. Do **not** wire it into preview/playback — diffusion is too slow (~few fps). Goal: optional “high quality” pass at record/export time while MuseTalk stays on for live A/B. Reuse audio prep / face crop plumbing where possible; keep a separate engine package under `pytorch_extras/`.
- [ ] **FlashLips** (CVPR 2026): watch for public **code + weights**. Paper claims >100 FPS, mask-free, reconstruction (no diffusion) with better identity than typical lip-sync. If/when weights land, evaluate as a **realtime MuseTalk alternative or replacement** for preview. Blocked on release — no usable public weights as of 2026-08-09.



### Out of scope for MuseTalk (related)

- [ ] InstantID remains a stub (`instantid_stub.py`) — does not block MuseTalk

---



## Quick reference


| What          | Where / how                                                     |
| ------------- | --------------------------------------------------------------- |
| Enable        | UI Common → **Enable MuseTalk Lip-Sync**                        |
| Weights       | Launcher *Check / Update Models* or `python download_models.py` |
| Skip download | `--skip-musetalk` or `VISOFUSION_SKIP_MUSETALK=1`               |
| Debug         | `VISOFUSION_MUSETALK_DEBUG=1`                                   |
| Batch         | `VISOFUSION_MUSETALK_BATCH` (default 8)                         |
| Compile       | Settings → MuseTalk → **torch.compile** toggle, or `VISOFUSION_MUSETALK_COMPILE` (env wins when set; default **off**). UNet + VAE encoder/decoder. ~1.5× at b4 / ~1.6× at b8 isolated; ~1.3× apply in-app. Load ~70 s warm cache, several minutes first run. |
| Perf          | `VISOFUSION_MUSETALK_PERF=1` → `[MUSETALK-PERF]` crop/batch_wait/encode/unet/decode/parse/… |
| Bench         | `python -m app.processors.pytorch_extras.musetalk.bench_musetalk` |
| Load timing   | `python scripts/time_musetalk_load.py`                          |
| Code          | `app/processors/pytorch_extras/musetalk/`                       |
| Frame hook    | `frame_worker.py` (before / after / hybrid vs swap)             |
| Controls      | `common_layout_data.py` + Settings → MuseTalk (`MuseTalkCompileToggle`) + `control_actions.py` |


---



## Brief history


| Date       | Change                                                                                             |
| ---------- | -------------------------------------------------------------------------------------------------- |
| 2026-08-09 | Tracker created; core UI/engine/tests treated as done; pending QA, hashes, user docs, and release. |
| 2026-08-09 | Rewrote tracker in English.                                                                        |
| 2026-08-09 | Noted hybrid pipeline; LatentSync = export-only spike; FlashLips = wait for public weights.        |
| 2026-08-10 | FPS opts: torch.compile + channels_last + inference_mode; Whisper on CPU; bench script; INT8 deferred. |
| 2026-08-10 | Compile ships **off** by default: UNet-only win, VAE dominates (~84% at b8); pad to fixed batch specs; portable Python Include/libs + MSVC PATH for opt-in compile; load 6.3 s off vs 34.9 s on. Next FPS: VAE / in-app `PERF_STAGES`. |
| 2026-08-10 | **VAE encoder/decoder compiled** (the UNet is constant ~55 ms; the VAE was 88% of the pass and scales linearly). Padding moved before the encode. End to end **1.5× at b4 / 1.6× at b8**, b8 no longer stalls. `dynamic=True` measured and rejected (recompiles at b1/b2 with 76–86 s stalls). Still opt-in: first-ever compile ~477 s, ~70 s thereafter. |
| 2026-08-10 | In-app confirm (n=768): apply **530→411 ms (~1.29×)**, infer **302→212 (~1.43×)** with compile on; first portable load ~319 s. Next: BiSeNet GPU contention. |
| 2026-08-10 | Settings → MuseTalk **torch.compile** toggle (`MuseTalkCompileToggle`); env still overrides when set. |
| 2026-08-10 | Seek/scrub: fractional FPS Whisper windows, no modulo wrap, fast cancel + batcher drop of stale requests. |
| 2026-08-10 | Long recordings: stream WAV + incremental Whisper to fp16 host timeline; recording waits without soft timeout (preview keeps 20 s). |
| 2026-08-10 | VR180: MuseTalk on perspective crops (after swap, before stitch); equirect post-hook off. |



