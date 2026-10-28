Vendored inference code from [XPixelGroup/HYPIR](https://github.com/XPixelGroup/HYPIR)
(HYPIR Software License Agreement — source is available; neural-network
weights remain proprietary to the authors). Weights are downloaded on demand
into `model_assets/hypir/` (`HYPIR_sd2.pth` from `lxq007/HYPIR` plus
Stable Diffusion 2.1 base). Only the enhancer + wavelet tiling helpers are
kept. Local patches: tiled VAE/UNet progress bars are off (Windows consoles);
unused `HYPIR.utils.tiled_vae` import dropped (that package was not vendored;
`enhance()` tiles via `make_tiled_fn`).
