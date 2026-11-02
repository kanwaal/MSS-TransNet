# NOTICE

MSS-Trans-Net
Copyright 2026 the MSS-Trans-Net authors

Licensed under the Apache License, Version 2.0; see `LICENSE`.

## Third-party software

This repository includes the following third-party source code. Original copyright
notices are retained in the files.

| Component | Location | Origin | License |
|---|---|---|---|
| R50–ViT-B/16 hybrid encoder and CNN decoder (unmodified) | `mss_transnet/_vendor/transunet/` | TransUNet, https://github.com/Beckschen/TransUNet, commit `02ef0010b36eb8328b5e689eadaf613602edf9b8` | Apache-2.0 (`LICENSE` in that folder) |
| Masked transformer decoder, adapted for 2D inputs (changes listed at the top of `decoder2d.py`) | `mss_transnet/_vendor/transunet2024/` | 3D-TransUNet, https://github.com/Beckschen/3D-TransUNet, commit `9f18182f5f7b26fc81e2f2d70fb5c40ee8a58908`; contains code derived from Mask2Former and DETR | Apache-2.0 (`LICENSE`), with the Mask2Former MIT notice (`LICENSE.Mask2Former`) and the DETR license (`LICENSE.DETR`) in that folder |
