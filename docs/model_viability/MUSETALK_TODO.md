# MuseTalk lip-sync — TODO tracker

Living checklist of what is done and what remains for the **MuseTalk 1.5** integration in VisoMaster Fusion.
Mark items with `[x]` / `[ ]` and add short notes under each section when status changes.

**Last updated:** 2026-08-09

---

## Done

### Core / engineering

- [x] Isolated package `app/processors/pytorch_extras/musetalk/` (engine, audio, blending, framing, parsing, models, paths)
- [x] Lazy load via `ModelsProcessor.ensure_musetalk_loaded()` / `unload_musetalk()`
- [x] Post-swap hook in `FrameWorker` (only when the toggle is on)
- [x] GPU batcher (`VISOFUSION_MUSETALK_BATCH`, default 8) for concurrent workers
- [x] Audio prep: video track or external file (Whisper chunks)
- [x] Frame→audio mapping using source FPS (not the capped playback FPS)
- [x] Reload after VRAM flush (`reload_musetalk_if_enabled`)
- [x] Audio refresh when new media is loaded
- [x] Probe / debug (`VISOFUSION_MUSETALK_DEBUG`) without touching the ONNX hot path
- [x] Failures do not break the pipeline (unchanged frame + warn)

### UI controls (Common → MuseTalk Lip-Sync)

- [x] Toggle **Enable MuseTalk Lip-Sync** + load/unload
- [x] Audio source (Video track / External file) + path
- [x] Extra margin, face index, blend strength, bbox shift
- [x] Landmark crop + face parsing mask
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

- [x] Unit: paths, blend, bbox, framing, landmarks, mask, parsing, lip chroma, mouth detail, audio windows, engine wiring, probe
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
- [ ] Typical VRAM / FPS profile (note hardware + recommended batch size)
- [ ] Decide whether crop needs temporal stabilisation (landmark jitter)

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

- [ ] Commit the integration (when requested)
- [ ] PR against the base branch
- [ ] CI checklist: unit green without weights; GPU integration marked `gpu`/`slow`

### Out of scope for MuseTalk (related)

- [ ] InstantID remains a stub (`instantid_stub.py`) — does not block MuseTalk

---

## Quick reference

| What | Where / how |
|------|-------------|
| Enable | UI Common → **Enable MuseTalk Lip-Sync** |
| Weights | Launcher *Check / Update Models* or `python download_models.py` |
| Skip download | `--skip-musetalk` or `VISOFUSION_SKIP_MUSETALK=1` |
| Debug | `VISOFUSION_MUSETALK_DEBUG=1` |
| Batch | `VISOFUSION_MUSETALK_BATCH` (default 8) |
| Code | `app/processors/pytorch_extras/musetalk/` |
| Frame hook | `frame_worker.py` (post swap/enhancers) |
| Controls | `common_layout_data.py` + `control_actions.py` |

---

## Brief history

| Date | Change |
|------|--------|
| 2026-08-09 | Tracker created; core UI/engine/tests treated as done; pending QA, hashes, user docs, and release. |
| 2026-08-09 | Rewrote tracker in English. |
