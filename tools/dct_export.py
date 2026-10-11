"""DCT-Net trial: runs the official head/background models (TensorFlow) on a photo, reproducing
the authors' pipeline (5-point alignment, 288px head crop, alpha blend), and converts the graphs
to ONNX, checking ONNX Runtime gives the same picture.

    python dct_trial.py <dct_dl dir> <photo> <out dir> "lx,ly rx,ry nx,ny mlx,mly mrx,mry"
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

tf1 = tf.compat.v1
DL, PHOTO, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
POINTS = np.float32([[float(v) for v in p.split(",")] for p in sys.argv[4].split()])
OUT.mkdir(parents=True, exist_ok=True)
BOX = 288

# Reference 5 points (left eye, right eye, nose, mouth left, mouth right) for a square 112 crop,
# as in the authors' align_trans.py (default_square=True shifts x by 8), ratio 0.75, scaled to 288.
REF = np.float32([[30.29459953, 51.69630051], [65.53179932, 51.50139999], [48.02519989, 71.73660278],
                  [33.54930115, 92.3655014], [62.72990036, 92.20410156]]) + 1 + np.float32([8, 0])
REF = ((REF - 56) * 0.75 + 56) * BOX / 112.0


def load(pb):
    gd = tf1.GraphDef()
    gd.ParseFromString(Path(pb).read_bytes())
    return gd


def run_tf(gd, x):
    g = tf.Graph()
    with g.as_default():
        tf1.import_graph_def(gd, name="")
    with tf1.Session(graph=g) as s:
        return s.run("output_image:0", {"input_image:0": x})


def to_onnx(gd, path):
    model, _ = tf2onnx.convert.from_graph_def(gd, input_names=["input_image:0"], output_names=["output_image:0"],
                                              opset=17, output_path=str(path))
    return path


def similarity(src, dst):
    """Least-squares similarity transform (rotation, uniform scale, shift) mapping src -> dst."""
    sm, dm = src.mean(0), dst.mean(0)
    s, d = src - sm, dst - dm
    a = (s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]).sum() / (s ** 2).sum()
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum() / (s ** 2).sum()
    r = np.float32([[a, -b], [b, a]])
    return np.hstack([r, (dm - r @ sm)[:, None]]).astype(np.float32)


photo = cv2.cvtColor(cv2.imread(str(PHOTO)), cv2.COLOR_BGR2RGB)
h0, w0 = photo.shape[:2]
k = 720 / min(h0, w0)
img = cv2.resize(photo, (round(w0 * k), round(h0 * k)), interpolation=cv2.INTER_AREA)
pts = POINTS * k
H, W = img.shape[:2]

results = [img]
for style, prefix in [("3d", "cartoon"), ("handdrawn", "cartoon"), ("anime", "cartoon_anime")]:
    d = DL / style
    head_gd, bg_gd = load(d / f"{prefix}_h.pb"), load(d / f"{prefix}_bg.pb")
    mask = cv2.cvtColor(cv2.resize(cv2.imread(str(d / "alpha.jpg")), (BOX, BOX), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

    bgr = img[:, :, ::-1]
    ph, pw = (H + 15) // 16 * 16, (W + 15) // 16 * 16
    pad = np.full((ph, pw, 3), 255, np.uint8)
    pad[:H, :W] = bgr
    t = time.time()
    bg = run_tf(bg_gd, pad.astype(np.float32))[:H, :W]
    tm = similarity(pts, REF)
    head_in = cv2.warpAffine(img, tm, (BOX, BOX), borderValue=(255, 255, 255))[:, :, ::-1].astype(np.float32)
    head = run_tf(head_gd, head_in)
    cv2.imwrite(str(OUT / f"head_{style}.jpg"), np.concatenate([head_in, head.astype(np.float32)], 1).astype(np.uint8))
    print("RESULT tm", tm.round(3).tolist())
    tf_secs = time.time() - t
    inv = cv2.invertAffineTransform(tm)
    head_full = cv2.warpAffine(head, inv, (W, H), borderValue=(0, 0, 0))
    m = cv2.warpAffine(mask, inv, (W, H), borderValue=0)[:, :, None]
    res = m * head_full + (1 - m) * bg
    results.append(np.clip(res, 0, 255).astype(np.uint8)[:, :, ::-1])

    # ONNX versions for the phone: same input gives the same output?
    t = time.time()
    oh = ort.InferenceSession(str(to_onnx(head_gd, OUT / f"dct-{style}-head.onnx")))
    ob = ort.InferenceSession(str(to_onnx(bg_gd, OUT / f"dct-{style}-bg.onnx")))
    conv = time.time() - t
    t = time.time()
    head_o = oh.run(None, {oh.get_inputs()[0].name: head_in})[0]
    bg_o = ob.run(None, {ob.get_inputs()[0].name: pad.astype(np.float32)})[0]
    ort_secs = time.time() - t
    print(f"RESULT {style}: tf {tf_secs:.2f}s, onnx {ort_secs:.2f}s, head diff {np.abs(head_o.astype(np.float32) - head.astype(np.float32)).max():.3f}, "
          f"bg diff {np.abs(bg_o[:H, :W].astype(np.float32) - bg.astype(np.float32)).max():.3f}, head in/out {head_in.shape}->{head.shape}, convert {conv:.1f}s")

w = 400
rows = [cv2.resize(r, (w, round(r.shape[0] * w / r.shape[1]))) for r in results]
sheet = np.concatenate(rows, axis=1)
cv2.imwrite(str(OUT / "dct_compare.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("ok")
