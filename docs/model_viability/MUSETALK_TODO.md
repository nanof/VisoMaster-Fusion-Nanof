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
- [ ] Typical VRAM / FPS profile (note hardware + recommended batch size) — **partial:** RTX 5070 Ti ≈ 1.8 GiB; eager ~10 / ~28 FPS (b1/b8); compile ~14.4 / ~28.4 FPS. Recommend batch 8 when multi-worker; compile helps most at batch 1.
- [ ] Decide whether crop needs temporal stabilisation (landmark jitter)
- [ ] After finish recording try to not open another winfow explorer if there is one opened in the same folder 

### Optimizations

- [x] Optimize MuseTalk model (quantization?) — **FPS pass (2026-08-10):**
  - Already fp16; **not** INT8/TRT this round (quality risk; deferred).
  - `torch.compile` on UNet (+ VAE when stable), env `VISOFUSION_MUSETALK_COMPILE` (default on). Falls back to eager if Inductor/Triton unavailable.
  - Hot path: `torch.inference_mode()` + `channels_last` on CUDA.
  - Whisper stays on **CPU** (VRAM only; not hot-path FPS).
  - Bench: `python -m app.processors.pytorch_extras.musetalk.bench_musetalk` (needs CUDA).
  - **Measured 2026-08-10** — RTX 5070 Ti, MuseTalk-5070-venv (torch 2.7.1+cu128 + `triton-windows` 3.3.1), WARMUP=3 ITERS=10, whisper=cpu, channels_last=on:
    - VRAM after load ≈ **1805–1813 MiB**; first compile load ≈ **215 s**
    - **compile=off**: b1 **100.4 ms (~10.0 FPS)**; b8 **36.0 ms/frm (~27.8 FPS)** — unet 73.0 / 64.4 ms
    - **compile=on** (`dynamic=False`, specs 1&8): b1 **69.2 ms (~14.4 FPS, 1.44×)**; b8 **35.3 ms/frm (~28.4 FPS)** — unet **43.3 / 33.5 ms** (~1.7–1.9× vs eager unet)
    - Note: `dynamic=True` hung; engine uses fixed batch specializations. Batch-8 E2E barely moves (VAE encode/decode dominate when amortized).
- [ ] Optimize Landmark detection

### Robustness

- [ ] Fill SHA256 hashes in `musetalk_assets_list` (currently `hash: ""`)
- [ ] Seek / scrub with lip-sync on (correct chunks, no stalls)
- [ ] Long recordings: audio memory, batcher, timeouts
- [ ] VR / non-standard layouts behaviour (if applicable)
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
| Compile       | `VISOFUSION_MUSETALK_COMPILE` (default on; UNet/VAE torch.compile) |
| Bench         | `python -m app.processors.pytorch_extras.musetalk.bench_musetalk` |
| Code          | `app/processors/pytorch_extras/musetalk/`                       |
| Frame hook    | `frame_worker.py` (before / after / hybrid vs swap)             |
| Controls      | `common_layout_data.py` + `control_actions.py`                  |


---



## Brief history


| Date       | Change                                                                                             |
| ---------- | -------------------------------------------------------------------------------------------------- |
| 2026-08-09 | Tracker created; core UI/engine/tests treated as done; pending QA, hashes, user docs, and release. |
| 2026-08-09 | Rewrote tracker in English.                                                                        |
| 2026-08-09 | Noted hybrid pipeline; LatentSync = export-only spike; FlashLips = wait for public weights.        |
| 2026-08-10 | FPS opts: torch.compile + channels_last + inference_mode; Whisper on CPU; bench script; INT8 deferred. |



