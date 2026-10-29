"""Fit the photo sanity gate from the cached leaf features and measure it.

Run after train_disease.py (it needs data/features_cache.npz):
    python scripts/train_gate.py

Reports how many real leaf photos would be wrongly rejected, on the training
images and on separate images the models never saw (tests/samples, if present),
and how many synthetic non-leaf inputs are caught.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import gate  # noqa: E402
from core.features import extract, extract_path  # noqa: E402

X = np.load(os.path.join(ROOT, "data", "features_cache.npz"), allow_pickle=True)["X"]
g = gate.fit(X)
json.dump(g, open(os.path.join(ROOT, "models", "leaf_gate.json"), "w"), indent=1)


def rejected(F):
    return np.array([not gate.check(f, g)[0] for f in F])


out = {"thresholds": g, "train_images": int(len(X)), "train_wrongly_rejected": int(rejected(X).sum())}

samples = os.path.join(ROOT, "tests", "samples")
if os.path.isdir(samples):
    paths = [os.path.join(r, f) for r, _, fs in os.walk(samples) for f in fs]
    Xf = np.stack([extract_path(p) for p in paths])
    out["unseen_leaf_images"] = int(len(Xf))
    out["unseen_wrongly_rejected"] = int(rejected(Xf).sum())

rng = np.random.default_rng(0)
junk = {
    "flat green": np.full((128, 128, 3), (40, 150, 60), np.uint8),
    "white page": np.full((128, 128, 3), 255, np.uint8),
    "black": np.zeros((128, 128, 3), np.uint8),
    "random noise": rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
    "blue gradient": np.stack([np.full((128, 128), 60), np.tile(np.linspace(120, 220, 128), (128, 1)),
                               np.full((128, 128), 235)], -1).astype(np.uint8),
}
try:  # real non-leaf photos bundled with scikit-image
    from skimage import data
    for name in ("astronaut", "coffee", "chelsea", "rocket", "camera", "page", "text", "moon", "clock",
                 "coins", "brick", "grass", "gravel", "horse"):
        try:
            im = getattr(data, name)()
            if im.ndim == 2:
                im = np.stack([im] * 3, -1)
            junk["photo: " + name] = np.asarray(Image.fromarray(im[..., :3].astype(np.uint8)).resize((128, 128)))
        except Exception:
            pass
except Exception:
    pass
Xj = np.stack([extract(a) for a in junk.values()])
rj = rejected(Xj)
out["non_leaf_inputs"] = int(len(rj))
out["non_leaf_rejected"] = int(rj.sum())
out["non_leaf_passed"] = [k for k, r in zip(junk, rj) if not r]
json.dump(out, open(os.path.join(ROOT, "reports", "gate_metrics.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
