# Added models and features: description and usage

Weight download: from the project root, `python download_models.py` (or the portable launcher flow). Hashes and URLs are in `app/processors/models_data.py`.

---

## New ONNX models in `models_list`

| Internal model | Brief description | Typical local file |
|----------------|-------------------|---------------------|
| **CrossFaceSimSwap** | Small network that maps an ArcFace embedding (512-D, w600k space) into the space SimSwap512 expects, to align identity between InsightFace and the SimSwap swapper. | `model_assets/crossface_simswap.onnx` |
| **FaceParsingBiSeNet18** | Semantic face parsing (19 classes aligned with the CelebA-style pipeline) with a BiSeNet ResNet-18 backbone; lighter than the default ResNet34 parser. | `model_assets/parsing/face_parsing_bisenet18.onnx` |
| **RvmPortraitMatting** | Robust Video Matting (MobileNetV3): estimates a foreground portrait alpha channel. In the app, recurrent state is reset every frame (full-video quality would keep state across frames). | `model_assets/matting/rvm_mobilenetv3_fp32.onnx` |
| **U2NetpSalientSeg** | u2netp-style network (rembg-like): saliency map at 320 px, rescaled to the crop; useful as an auxiliary “prominent object” mask, not SAM with prompts. | `model_assets/sam2/u2netp_human_seg.onnx` |
| **RestoreFormerFP16** | RestoreFormer face restoration in the base FP16 variant, alternative to RestoreFormer++ and GFPGAN/CodeFormer. | `model_assets/RestoreFormer.fp16.onnx` |
| **Yunet2023Mar** | YuNet variant (March 2023) with the same head logic as classic YuNet; replaces the older checkpoint when you pick detector **Yunet-2023**. | `model_assets/yunet_2023_mar.onnx` |
| **RifePreviewInterpAlt** | Second slot for the same RIFE graph (TensorStack) on a different path; lets you swap only `interp/rife_preview_interp_alt.onnx` for another compatible ONNX without touching the primary file. | `model_assets/interp/rife_preview_interp_alt.onnx` |

**Infra:** subfolders under `model_assets/` (`matting/`, `sam2/`, `parsing/`, `interp/`, etc.), `scripts/hash_model.py` for SHA256, and `pytorch_assets_list` in `models_data.py` (empty; `download_models.py` iterates it when it has entries).

---

## UI features and how to use them

### SimSwap512-CrossFace (swapper)

- **Description:** Combines **Inswapper128ArcFace (w600k)** recognition with the **SimSwap512** network: the embedding is first projected with **CrossFaceSimSwap**, then the usual 512×512 swap runs.
- **Where:** Swap tab → **Swapper Model** → **SimSwap512-CrossFace**.
- **When:** You want SimSwap but identity extracted with the same backbone as Inswapper/HyperSwap (w600k), without using **SimSwapArcFace** directly.

### Portrait matting (RVM)

- **Description:** Multiplies the swap mask by a foreground portrait alpha to soften edges and backgrounds; RVM is designed for video, but here state is not carried across frames.
- **Where:** Swap / masks tab → **Portrait matting (RVM)** (toggle).
- **Requirement:** **RvmPortraitMatting** ONNX downloaded.

### Salient mask (U2Net-p)

- **Description:** Adds a per-pixel confidence factor to the swap mask from saliency (object vs background), as a complement to XSeg/occluder/parser.
- **Where:** **Salient mask (U2Net-p)** (toggle).
- **Requirement:** **U2NetpSalientSeg** downloaded.

### Face parser backbone (ResNet34 / BiSeNet-18)

- **Description:** Chooses which ONNX feeds the Face Parser label pipeline: classic ResNet34 or lighter BiSeNet-18 (same class-mask style in the current flow).
- **Where:** With **Face Parser Mask** enabled → **Face parser backbone**.
- **Requirement:** **FaceParser** or **FaceParsingBiSeNet18**, depending on the option.

### RestoreFormer (restorer)

- **Description:** Restoration with the “base” RestoreFormer model in FP16; distinct from **RestoreFormer++** (heavier / different architecture in this project).
- **Where:** **Face Restorer** (and **Face Restorer 2** if applicable) → **Restorer Type** → **RestoreFormer**.
- **Requirement:** **RestoreFormerFP16**.

### Yunet-2023 (detector)

- **Description:** YuNet face detector with newer weights (same output family as standard YuNet in code).
- **Where:** **Settings** → detection model → **Yunet-2023**.
- **Requirement:** **Yunet2023Mar**.

### Neural ONNX checkpoint (RIFE x2)

- **Description:** In neural preview interpolation, chooses which ONNX session to use: default or the alternate under `interp/` (handy to experiment with another compatible file).
- **Where:** **Settings** → **Frame Interpolation** ON → method **Neural (ONNX)** → **Neural ONNX checkpoint**.
- **Requirement:** **RifePreviewInterp** and/or **RifePreviewInterpAlt** downloaded.

### PyTorch extras (InstantID / retalking — skeleton)

- **Description:** Optional dependencies and stubs for future PyTorch pipelines; they do not replace the hot ONNX swap path.
- **Files:** `requirements-pytorch-extra.txt`, `app/processors/pytorch_extras/`.
- **Enable:** `VISOFUSION_PYTORCH_EXTRAS=1` (or `true` / `yes` / `on`).
- **Status:** informational console messages; full integration pending design.

---

## Summary table: menu → action

| Goal | Control in the app |
|------|---------------------|
| SimSwap swap + w600k embedding + crossface | **Swapper Model → SimSwap512-CrossFace** |
| Refine edges with portrait matting | **Portrait matting (RVM)** |
| Saliency-based mask | **Salient mask (U2Net-p)** |
| Lighter parser | **Face parser backbone → BiSeNet-18** |
| Base RestoreFormer restorer | **Restorer Type → RestoreFormer** |
| Newer YuNet | **Detector → Yunet-2023** |
| Second RIFE slot | **Neural ONNX checkpoint → RifePreviewInterpAlt** |
