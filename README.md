# Lensmith models

On-device AI models for [Lensmith](https://github.com/lensmith-app) filters, served from GitHub Pages:
`https://lensmith-app.github.io/lensmith-models/<file>`. The app downloads a model the first time a
filter uses it (`lens.stylize(...)` in a script) and checks it against the SHA-256 in `models.json`.

| id | Style | File |
|---|---|---|
| `face-paint` | Face Paint — soft painted cartoon, made for faces | face-paint.onnx |
| `paprika` | Paprika — bright anime-film colours | paprika.onnx |
| `celeba` | Anime Portrait — clean anime portrait look | celeba.onnx |
| `cartoon-3d` | 3D Cartoon — animated-movie 3D character (portrait) | dct-3d-head.onnx + dct-3d-bg.onnx |
| `cartoon-handdrawn` | Hand-drawn — soft illustrated character (portrait) | dct-handdrawn-head.onnx + dct-handdrawn-bg.onnx |
| `cartoon-anime` | Anime — full anime redraw (portrait) | dct-anime-head.onnx + dct-anime-bg.onnx |

**Style models** (`face-paint`, `paprika`, `celeba`) — format: ONNX (opset 17), input `image` float32 `[1, 3, H, W]` RGB scaled to -1..1, output `styled`
in the same form and size. Any H and W work; multiples of 4 are best.

**Portrait models** (`"type": "portrait"`, DCT-Net) — two networks plus a blend mask:
- `bg`: the whole picture, input `input_image:0` float32 `[H, W, 3]` **BGR 0–255** (H, W multiples of 16),
  output `output_image:0` uint8 `[H, W, 3]` BGR.
- `head`: the face, aligned with a similarity transform so the eyes, nose tip and mouth centre land on
  the reference points (the 112-px reference face of the authors' `align_trans.py`, square, `ratio`
  0.75, scaled to `box` = 288 px); same input/output format at 288×288. Blend it back with
  `dct-mask.png` (288×288 greyscale) through the inverse transform.

## Credits and licences

**Lensmith is non-commercial.** These models are only used under non-commercial terms.

- Architecture and training: **AnimeGANv2** by Xin Chen, Gang Liu et al.
  ([TachibanaYoshino/AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)) — free for
  non-commercial use; commercial use needs the author's permission.
- PyTorch port and the `face_paint_512_v2`, `paprika` and `celeba_distill` weights:
  **bryandlee** ([bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)), MIT
  licence — see `LICENSE-animegan2-pytorch.txt`.

- Portrait models: **DCT-Net** (Men et al., SIGGRAPH 2022, DAMO Academy / Alibaba Group,
  [menyifang/DCT-Net](https://github.com/menyifang/DCT-Net)), weights from ModelScope
  (`damo/cv_unet_person-image-cartoon*_compound-models`) — **Apache License 2.0**.

Converted to ONNX with `tools/export.py` and `tools/dct_export.py` (outputs checked against the originals).
