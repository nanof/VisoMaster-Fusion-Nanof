Vendored inference code from [ohayonguy/PMRF](https://github.com/ohayonguy/PMRF)
(MIT). Weights are downloaded on demand into `model_assets/pmrf/` from
`ohayonguy/PMRF_blind_face_image_restoration`. Training-only deps (`wandb`,
`torch_ema`, `natten`) are optional. Local patches: package imports under
`app.processors.external.pmrf`; SwinIR (MMSE prior) without `timm`; unfold-based
neighborhood attention when `natten` is missing; `torch.compile` off by default;
skip FID metric construction at init.
