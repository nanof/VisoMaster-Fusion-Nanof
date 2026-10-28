Vendored inference code from [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
(Apache-2.0, based on ByteDance SeedVR2). Weights are downloaded on demand into
`model_assets/seedvr2/`. ComfyUI node interfaces are unused; the app calls
`src.core.generation_phases` directly.
Local patches: do not stub `flash_attn` as an empty package (breaks HuggingFace
lazy imports); quieter setup logs.
