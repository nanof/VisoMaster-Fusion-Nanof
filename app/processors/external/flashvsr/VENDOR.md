Vendored inference code from [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
(Apache-2.0). Weights are downloaded on demand into `model_assets/flashvsr/`.
Local patches: optional dense-attention fallback when `block_sparse_attn` is
missing; Flash Attention is used only if `flash_attn_func` is actually callable
(SeedVR2 used to leave a None stub); no ASCII banner / tqdm on the TinyLong and
Full pipelines; TinyLong does not fetch Wan VAE; `diffsynth/__init__.py` and
`pipelines/__init__.py` only export FlashVSR pipelines; `model_config.py` keeps
Wan VSR hashes only; `modelscope` is optional.
