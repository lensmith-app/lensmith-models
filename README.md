# Lensmith models

On-device AI models for [Lensmith](https://github.com/lensmith-app) filters, served from GitHub Pages:
`https://lensmith-app.github.io/lensmith-models/<file>`. The app downloads a model the first time a
filter uses it (`lens.stylize(...)` in a script) and checks it against the SHA-256 in `models.json`.

| id | Style | File |
|---|---|---|
| `face-paint` | Face Paint — soft painted cartoon, made for faces | face-paint.onnx |
| `paprika` | Paprika — bright anime-film colours | paprika.onnx |
| `celeba` | Anime Portrait — clean anime portrait look | celeba.onnx |

Format: ONNX (opset 17), input `image` float32 `[1, 3, H, W]` RGB scaled to -1..1, output `styled`
in the same form and size. Any H and W work; multiples of 4 are best.

## Credits and licences

**Lensmith is non-commercial.** These models are only used under non-commercial terms.

- Architecture and training: **AnimeGANv2** by Xin Chen, Gang Liu et al.
  ([TachibanaYoshino/AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)) — free for
  non-commercial use; commercial use needs the author's permission.
- PyTorch port and the `face_paint_512_v2`, `paprika` and `celeba_distill` weights:
  **bryandlee** ([bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)), MIT
  licence — see `LICENSE-animegan2-pytorch.txt`.

Converted to ONNX with `tools/export.py` (outputs checked against PyTorch).
