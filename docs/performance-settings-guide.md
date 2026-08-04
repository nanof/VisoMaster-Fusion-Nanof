# General / Detectors settings: performance and quality

This document summarizes the controls that most affect **FPS**, **latency**, and pipeline **quality** (face swap / recognition), with ballpark values for **maximum throughput** and guidance for **quality-oriented** profiles.

In-app help text aligns with the keys in `app/ui/widgets/settings_layout_data.py`.

### Performance impact legend

| Marker | Meaning |
|--------|---------|
| **High** | Usually dominates frame time or scales cost across the whole pipeline (resolution, detection rate, provider, swapper tier, ArcFace budget). |
| **Medium** | Meaningful when tuned; depends on scene (multi-face, queue depth, GPU saturation). |
| **Low** | Small overhead, situational gain, or mostly affects scrubbing / first frame / RAM—not steady-state FPS. |

**Highest-impact knobs (typical order of leverage):** **Resize input** / **performance preset**, **detection interval**, **detector input size** (incl. fast-detect cap), **execution provider** (TensorRT-Engine vs CPU), **swapper resolution path** (e.g. Inswapper128 vs heavier variants), then **ArcFace budget** (max per frame + lazy stride), **worker threads**, **inflight buffer**.

---

## 1. Hardware and core (General)

| Perf. impact | Setting | What it does | Quality | Performance |
|:------------:|--------|----------------|---------|-------------|
| **High** | **Providers Priority** | Backend order (CUDA / TensorRT / TensorRT-Engine / CPU). TensorRT-Engine is often fastest when engines are built. | TensorRT or CUDA if TRT fails; measure with `VISIOMASTER_PERF_BUNDLE=1`. | **TensorRT-Engine** when engines are healthy. |
| **Medium** | **Tune TensorRT dynamic batch profiles** (+ sliders) | When on, UI sets TRT min/opt/max **batch** for ORT TensorRT EP (swap, ArcFace, LP motion/stitch). When off, only `VISIOMASTER_TRT_MAX_BATCH_*` / `OPT_*` env vars apply. | Lower max/opt for fewer faces → faster engine builds / less VRAM; raise for heavy multi-face. | **On** to tune without env; **reload ONNX** after changes. See `docs/agent-architecture.md`. |
| **Medium** | **Use multiple GPUs for frame routing** | Spreads frames across GPUs. | Useful with 2+ GPUs for throughput; does not improve swap “quality” by itself. | **On** with multiple GPUs; **Off** with a single GPU. |
| **High** | **Primary GPU** | Main GPU for CUDA/TRT and cache paths. | Pick the most VRAM/fastest for heavy loads. | Pick the fastest device. |
| **Medium** | **Number of Threads** | Worker pool threads (playback/recording). | Try **4–6** if the queue backs up while the GPU is under 100%; more threads can increase contention. | **Increase** (e.g. 5–8) if the queue is full and the GPU is underused; **decrease** if you see hitching. |
| **Low** | **Keep Controls Active** | Keeps controls active while recording. | Convenient for live tweaking. | **Off** if you do not need it. |
| **Low** | **Track Markers on Video Seek** | When scrubbing, syncs controls to the timeline position. | Better for frame-accurate editing. | **Off** if you prioritize fast scrubbing. |
| **High** | **Resize Input Source** | Rescales input **before** the pipeline and sets output resolution (aspect ratio preserved). | **On** at **1080p** or native resolution for maximum detail (more compute). | **On** at **720p / 540p** or performance presets. |
| **Medium** | **Record output at different resolution** (sub-toggle under resize) | Preview/AI stay at **Input Resolution Target**; each recorded frame is Lanczos-upscaled (or downscaled) to **Record output resolution** before FFmpeg. Virtual cam / on-screen preview stay at preview size. | Set record height **≥** preview height for sharper files; same value = no extra resize. | **On** when you want fast preview (e.g. 540p) and a higher-res export (e.g. 1080p). See [`fps-optimization-backlog.md`](fps-optimization-backlog.md) PERF-018. |
| **Low** | **Frame Worker Delay** | Seconds to wait before AI work after a **seek**; reduces GPU overload. | **Low (0.1–0.2 s)** if the GPU can keep up. | **Higher (0.2–0.5 s)** if scrubbing causes hitches or VRAM spikes. |
| **High** | **Performance preset (video)** | Applies a coordinated bundle (resize, detection interval, detector, ArcFace, Inswapper 128, etc.). | **“Single face @ 1080p — quality”** or fine-grained **Custom**. | **“High FPS”**, **“Light”**, or **“Webcam / baja latencia — …”** (UI label). |
| **Low** | **Pipeline profile (timing)** | Shows per-stage timings (overlay/dock); diagnostic only. | **On** while tuning. | **Off** for long renders. |

---

## 2. Buffering and CUDA (General)

| Perf. impact | Setting | What it does | Quality | Performance |
|:------------:|--------|----------------|---------|-------------|
| **Medium** | **Inflight buffer (× preroll)** | Queue depth ≈ preroll × this value; more overlap between feeder and workers. | **Moderate (4–8)** if you see micro-stalls from empty queues. | **Higher (6–12)** if RAM allows and you want more overlap; **lower** for less latency/RAM. |
| **Medium** | **Separate CUDA streams** | Separate streams for feeder vs workers to overlap GPU work. | — | **On** usually helps throughput; **Off** if unstable. |
| **Medium** | **Batch ArcFace (2+ faces)** | With multiple faces in one frame, one batched ORT call (Inswapper128 path). | Usually neutral for quality. | **On** almost always when multiple faces appear. |

---

## 3. ArcFace and tracking (General)

| Perf. impact | Setting | What it does | Quality | Performance |
|:------------:|--------|----------------|---------|-------------|
| **Low** | **ArcFace cap: center bias** | When ArcFace per frame is capped, boosts faces near the **center** (0 = area only). | **Raise** if the main subject is centered. | **Lower / 0** for more even scheduling or many faces in frame. |
| **Low** | **ArcFace cap: boost recently matched tracks** | Increases priority for tracks that recently matched a target. | **Raise** so actively swapped faces keep their slot under a tight cap. | **Lower / 0** for less bias (may change who gets ArcFace). |
| **Low** | **Matched-track memory** | How long a track stays “recent” for boost and stride. | **Higher** = stickier identity across continuous scenes. | **Lower** = less state; may force more ArcFace work. |
| **Medium** | **Matched tracks: min frames between real ArcFace** | Reuses embeddings and skips ArcFace until N frames since last fresh run (**1** = feature off). | **1–3** for maximum fidelity and frequent refreshes. | **4–12** to save ArcFace when many faces share the same target. |

*ArcFace “cap” row settings matter most when **max ArcFace runs per frame** is limited (Detectors tab).*

---

## 4. Scene and motion — swapper auto-res (General)

| Perf. impact | Setting | What it does | Quality | Performance |
|:------------:|--------|----------------|---------|-------------|
| **Low** | **Scene cut → force fresh ArcFace** | On luminance histogram jump, forces one embedding refresh. | **On** to avoid dragging identities across shots. | Cost at hard cuts only; a **high** threshold reduces false refreshes. |
| **Low** | **Scene cut histogram L1 threshold** | **Higher** = fewer detected cuts; **lower** = more refreshes. | **Lower** if soft cuts are missed. | **Higher** for fewer false cuts and fewer extra refreshes. |
| **Low** | **Swapper auto-res: hysteresis** | Reduces flicker between automatic 128↔512 tiers per ByteTrack id. | **On** if auto-res flickers. | Usually worth it for stability. |
| **Low** | **Hysteresis motion EMA alpha** | **Low alpha** = stay on previous resolution longer; **high** = react sooner. | **Lower** for fewer abrupt tier changes. | **Higher** to avoid lingering on a low tier. |

---

## 5. Detection and memory (General)

| Perf. impact | Setting | What it does | Quality | Performance |
|:------------:|--------|----------------|---------|-------------|
| **Medium** | **Detection: track-guided ROI on skip frames** | With **Detection Interval > 1**, runs detection on a crop around the last bbox on skip frames. | **Off** if you lose fast-moving faces outside the ROI. | **On** + high interval for more FPS with few faces. |
| **Low** | **Track ROI pad (% of max(w,h))** | Padding around the bbox before cropping. | **Higher** if tracks are lost during motion. | **Lower** for slow motion and tighter crops (slightly cheaper). |
| **Low** | **Unload idle ONNX models** | After N idle minutes, models may unload (**0** = never unload for this reason). | **0** = no stall when switching back to a model. | **15–30 min** if VRAM is tight and you switch models often. |
| **Low** | **Warm up model after load** | Dummy inference after load to reduce first-frame hitch. | Smoother first playback. | Usually worth the one-time load cost. |
| **Low** | **Store recognition embeddings as FP16** | Smaller RAM footprint; promoted to FP32 where needed. | Nearly neutral in practice. | **On** with many faces/embeddings. |

---

## 6. Table — suggested values for **maximum performance**

Ballpark values for **maximum FPS / minimum time per frame**. Assumes **one GPU** and willingness to trade robustness in detection/ArcFace. If you lose faces or identities, raise only the relevant control.

### General

| Perf. impact | Setting | Suggested value (max. performance) | Short note |
|:------------:|--------|-------------------------------------|------------|
| **High** | **Providers Priority** | `TensorRT-Engine` (if engines exist; else `TensorRT`) | Lower latency when TRT is built correctly. |
| **Medium** | **Use multiple GPUs** | `On` if you have **2+ GPUs**; otherwise `Off` | Routing only helps with multiple GPUs. |
| **High** | **Primary GPU** | Your **fastest / most VRAM** device | — |
| **Medium** | **Number of Threads** | `6`–`8` (raise until the GPU is saturated without hitches) | If you see contention, drop to `4`–`5`. |
| **Low** | **Keep Controls Active** | `Off` | |
| **Low** | **Track Markers on Video Seek** | `Off` | |
| **High** | **Resize Input Source** | `On` → **540p** or **720p** | Lower resolution = less load end-to-end. |
| **Low** | **Frame Worker Delay** | `0.1` s (or the lowest stable value when seeking) | Matters little during continuous playback. |
| **High** | **Performance preset** | `Light — 540p, interval 4, tight ArcFace` or `Webcam / baja latencia — 480p, det 1, Inswapper 128, ArcFace mínimo` | Alternative: `High FPS — 720p input, interval 3, det 416, Inswapper 128`. |
| **Low** | **Pipeline profile** | `Off` | |
| **Medium** | **Inflight buffer** | `8`–`12` | If RAM is tight, use `4`–`6`. |
| **Medium** | **Separate CUDA streams** | `On` | If unstable, `Off`. |
| **Medium** | **Batch ArcFace** | `On` | |
| **Low** | **ArcFace cap: center bias** | `0` | |
| **Low** | **ArcFace cap: boost recently matched** | `50`–`70` | With a **max ArcFace/frame** cap, helps keep the active swap from starving. |
| **Low** | **Matched-track memory** | `32`–`48` | |
| **Medium** | **Matched tracks: min frames between real ArcFace** | `8`–`12` (slider max: `12`) | Fewer full ArcFace runs on already-matched tracks. |
| **Low** | **Scene cut → fresh ArcFace** | `On` + **high** threshold, or consider `Off` only if you prioritize FPS and accept cut risk | **High** threshold (`0.45`–`0.70`): fewer refreshes from false scene cuts. |
| **Low** | **Scene cut histogram threshold** | `0.45`–`0.70` | Fewer “scene cut” events. |
| **Low** | **Swapper auto-res hysteresis** | `On` if you use auto-res | Fewer erratic 128↔512 switches. |
| **Low** | **Hysteresis motion EMA alpha** | `0.40`–`0.55` | |
| **Medium** | **Track-guided ROI** | `On` **only if** Detection Interval **> 1** | |
| **Low** | **Track ROI pad** | `45`–`55` | Raise if you lose fast tracks. |
| **Low** | **Unload idle ONNX** | `0` | Avoids stalls when re-enabling models. |
| **Low** | **Warm up after load** | `On` | |
| **Low** | **Embeddings FP16** | `On` | |

### Detectors (typically **High** impact as a group)

| Perf. impact | Setting | Suggested value (max. performance) | Short note |
|:------------:|--------|--------------------------------------|------------|
| **High** | **Detection Interval (Frames)** | `4`–`6` | Higher risk with very fast motion. |
| **High** | **Detector internal size** | **416** or **320** (when applicable) | Smaller input = faster detector. |
| **High** | **Performance: cap detector internal size** | `On` | |
| **High** | **Fast-detect max side** | `416` or `320` | More aggressive = more FPS, more risk on small faces. |
| **High** | **Performance: max ArcFace runs per frame** | `4`–`6` | Hard cap with many faces (`0` = unlimited). |
| **High** | **Performance: min frames between real ArcFace (track)** | `4`–`8` | Global lazy ArcFace stride per track. |
| **Medium** | **Face Detect Model** | Whichever gives best **ms/frame** on your GPU (`VISIOMASTER_PERF_BUNDLE=1`) | |
| **High** | **Active swapper** | **Inswapper128** when preset/model allows | Lower cost than many 512 variants on typical setups. |

---

## 7. Quick summary

- **Fine quality (few faces, 1080p):** use preset *Single face @ 1080p — quality* or Custom; keep ArcFace stride low; conservative ROI or low detection interval.
- **Maximum FPS:** tune **High**-impact rows first: input resize / preset, **Detectors** tab (interval, detector size, ArcFace caps, lazy stride), **Providers**, **swapper tier**; then **Medium**: threads, inflight, streams, batch ArcFace, track-guided ROI when interval > 1.

Useful environment variables for measurement: `VISIOMASTER_PERF_BUNDLE=1`, and as needed `VISIOMASTER_PERF_STAGES`, `VISIOMASTER_PIPELINE_METRICS`, `VISIOMASTER_PIPELINE_PROFILE_CSV` (see the *Pipeline profile* control help and `docs/agent-architecture.md`).

**TensorRT (ORT EP) — perfiles de batch dinámico:** al cargar modelos con el proveedor TensorRT, Fusion puede fusionar `trt_profile_min_shapes` / `opt` / `max` para rutas que usan **batch variable** (LivePortrait motion/stitch/eye/lip, Inswapper128 batched, GhostFace/HyperSwap batched, ArcFace batched). Si la compilación TRT falla (nombre de input distinto en tu ONNX), usa `VISIOMASTER_TRT_NO_DYNAMIC_PROFILES=1` o ajusta los `VISIOMASTER_TRT_MAX_BATCH_*`. Detalle en `docs/agent-architecture.md` (tabla de variables).

**GhostFace / HyperSwap — Swap All (batch ORT 256):** el path multi-cara usa el mismo contrato I/O que la inferencia cara a cara: tensores `target`/`output` en **[-1, 1]** (no `[0, 1]` como Inswapper128 batched). Tras un fallo de batch HyperSwap, Fusion desactiva nuevos intentos batched en esa sesión y cae a secuencial.

---

## 8. Backlog de optimización (código / pipeline)

Las tablas de arriba cubren **controles ya expuestos en la app**. Para ideas que implican cambios de implementación (batch de restaurador, sync GPU, etc.), priorización por fases y **tabla de estado** editable en el repo, ver [`fps-optimization-backlog.md`](fps-optimization-backlog.md) (IDs **PERF-001** … **PERF-019**).
