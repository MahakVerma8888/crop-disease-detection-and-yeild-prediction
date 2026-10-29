"""Photo sanity gate: reject inputs that cannot be a usable leaf photo.

The disease classifier always picks one of its 38 classes, even for a blank
page, so confidence alone cannot catch non-leaf inputs (a flat green square
scored 100%). This gate looks at three simple, explainable properties of the
photo, using thresholds learned from the real leaf photos:

  * edge strength too LOW   -> blank, flat colour or smooth gradient
  * edge strength too HIGH  -> noise or very busy texture
  * plant-coloured area too SMALL -> no green/yellow/brown leaf in the frame

It is a sanity check, not a leaf detector: a colourful photo of something
else (a cat, a person) can still pass.
"""
import json
import os

import numpy as np

from core.features import FEATURE_DIM

# Positions inside the feature vector (see core/features.py):
#   34 HSV histogram values + 18 channel mean/std values = 52, then
#   [brown, yellow, dark, green, foreground] fractions, and the last three
#   values are edge statistics (mean edge strength first).
_BROWN, _YELLOW, _GREEN, _FG = 52, 53, 55, 56
_EDGE_MEAN = FEATURE_DIM - 3

REASONS = {
    "flat": "The photo is almost blank, one flat colour or a smooth gradient.",
    "busy": "The photo is too noisy or too busy to read as a leaf.",
    "no_leaf": "The photo has almost no leaf-coloured area (green, yellow or brown).",
}


def signals(feats: np.ndarray) -> dict:
    """feats: (n, FEATURE_DIM) -> dict of 1-D arrays."""
    f = np.atleast_2d(feats)
    plant = (f[:, _BROWN] + f[:, _YELLOW] + f[:, _GREEN]) * f[:, _FG]
    return {"edge": f[:, _EDGE_MEAN], "plant": plant}


def fit(train_feats: np.ndarray) -> dict:
    """Learn thresholds from real leaf photos: half the 0.5th percentile of the
    low-side signals, 1.5x the 99.5th percentile of the high-side one."""
    s = signals(train_feats)
    return {
        "edge_min": round(float(np.percentile(s["edge"], 0.5)) * 0.5, 5),
        "edge_max": round(float(np.percentile(s["edge"], 99.5)) * 1.5, 5),
        "plant_min": round(float(np.percentile(s["plant"], 0.5)) * 0.5, 5),
    }


def check(feats: np.ndarray, gate: dict):
    """Return (ok, reason_text_or_None) for ONE feature vector."""
    s = signals(feats)
    edge, plant = float(s["edge"][0]), float(s["plant"][0])
    if edge < gate["edge_min"]:
        return False, REASONS["flat"]
    if edge > gate["edge_max"]:
        return False, REASONS["busy"]
    if plant < gate["plant_min"]:
        return False, REASONS["no_leaf"]
    return True, None


def load(models_dir: str):
    p = os.path.join(models_dir, "leaf_gate.json")
    return json.load(open(p)) if os.path.exists(p) else None
