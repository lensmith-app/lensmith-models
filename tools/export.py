"""Converts the AnimeGANv2 PyTorch weights to ONNX (dynamic size) and checks ONNX Runtime matches PyTorch.

    python export.py <download dir> <out dir>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

DL = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(DL))
from model import Generator  # noqa: E402  (bryandlee/animegan2-pytorch, MIT)

STYLES = {
    "face-paint": ("face_paint_512_v2.pt", "Face Paint", "Soft painted cartoon, made for faces"),
    "paprika": ("paprika.pt", "Paprika", "Bright anime-film colours"),
    "celeba": ("celeba_distill.pt", "Anime Portrait", "Clean anime portrait look"),
}

catalog = []
for sid, (file, name, desc) in STYLES.items():
    net = Generator().eval()
    # weights_only: never run code from the downloaded pickle.
    net.load_state_dict(torch.load(DL / file, map_location="cpu", weights_only=True))
    x = torch.rand(1, 3, 512, 384) * 2 - 1
    out = OUT / f"{sid}.onnx"
    torch.onnx.export(
        net, (x,), str(out), input_names=["image"], output_names=["styled"], opset_version=17,
        dynamic_axes={"image": {2: "h", 3: "w"}, "styled": {2: "h", 3: "w"}}, dynamo=False,
    )
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for shape in [(1, 3, 512, 384), (1, 3, 1024, 464)]:
        t = torch.rand(*shape) * 2 - 1
        with torch.no_grad():
            ref = net(t).numpy()
        t0 = time.time()
        got = sess.run(None, {"image": t.numpy()})[0]
        print(f"{sid} {shape}: max diff {np.abs(ref - got).max():.5f}, ort {time.time() - t0:.2f}s")
    data = out.read_bytes()
    catalog.append({"id": sid, "name": name, "description": desc, "file": out.name, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "multiple": 4})
(OUT / "models.json").write_text(json.dumps({"version": 1, "models": catalog}, indent=2) + "\n")
print(json.dumps(catalog, indent=1))
