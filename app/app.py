"""Smart Crop - leaf disease detection and yield prediction (Flask).

Run:  python app/app.py        (serves on http://127.0.0.1:5000)
"""
import io
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import gate as leaf_gate  # noqa: E402
from core.advice import lookup  # noqa: E402
from core.features import extract, load_rgb  # noqa: E402

CONF_THRESHOLD = 0.50  # below this the app asks for a clearer photo
ALLOWED = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB upload cap


def _load(name):
    p = os.path.join(ROOT, "models", name)
    return joblib.load(p) if os.path.exists(p) else None


DISEASE = _load("disease_model.joblib")
YIELD = _load("yield_model.joblib")
YMETA = json.load(open(os.path.join(ROOT, "models", "yield_meta.json"))) if YIELD else None
GATE = leaf_gate.load(os.path.join(ROOT, "models"))  # photo sanity check thresholds


def _report(name):
    p = os.path.join(ROOT, "reports", name)
    return json.load(open(p)) if os.path.exists(p) else None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok", disease_model=DISEASE is not None, yield_model=YIELD is not None,
                   photo_check=GATE is not None)


@app.route("/api/meta")
def meta():
    """Dropdown options and per-crop defaults for the yield form."""
    if YMETA is None:
        return jsonify(error="Yield model not trained yet. Run scripts/train_yield.py"), 503
    return jsonify(crops=YIELD["cat_levels"]["Crop"], states=YIELD["cat_levels"]["State"],
                   seasons=YIELD["cat_levels"]["Season"], defaults=YMETA["defaults"],
                   state_rain=YMETA["state_rain"], combos=YMETA["combos"],
                   year_min=YMETA["year_min"], year_max=YMETA["year_max"])


@app.route("/api/report")
def report():
    return jsonify(disease=_report("disease_metrics.json"), yield_=_report("yield_metrics.json"))


@app.route("/reports/<path:fn>")
def report_file(fn):
    return send_from_directory(os.path.join(ROOT, "reports"), fn)


@app.route("/api/disease", methods=["POST"])
def disease():
    if DISEASE is None:
        return jsonify(error="Disease model not trained yet. Run scripts/train_disease.py"), 503
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="No image received. Choose a leaf photo (JPG or PNG)."), 400
    if f.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return jsonify(error="Unsupported file type. Use a JPG or PNG photo."), 400
    data = f.read()
    try:
        Image.open(io.BytesIO(data)).verify()  # reject corrupt / non-image uploads
        rgb = load_rgb(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="That file could not be read as an image. Try another photo."), 400

    x = extract(rgb).reshape(1, -1)
    if GATE is not None:
        usable, why = leaf_gate.check(x, GATE)
        if not usable:
            return jsonify(error=f"This does not look like a usable leaf photo. {why} "
                                 "Use one leaf on a plain background in even light."), 422
    proba = DISEASE["model"].predict_proba(x)[0]
    classes = DISEASE["classes"]
    order = np.argsort(proba)[::-1][:3]
    top = []
    for i in order:
        info = lookup(classes[i])
        top.append({"label": classes[i], "crop": info["crop"], "condition": info["condition"],
                    "probability": round(float(proba[i]), 4)})
    best = lookup(classes[order[0]])
    conf = float(proba[order[0]])
    return jsonify(prediction=best | {"label": classes[order[0]], "confidence": round(conf, 4)},
                   top3=top, confident=conf >= CONF_THRESHOLD, threshold=CONF_THRESHOLD)


_LABELS = {"Crop_Year": "Year", "Area": "Area", "Annual_Rainfall": "Annual rainfall",
           "Fert_per_ha": "Fertiliser", "Pest_per_ha": "Pesticide"}


def _num(d, key, lo, hi):
    name = _LABELS.get(key, key)
    try:
        v = float(d.get(key))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.")
    if np.isnan(v) or not (lo <= v <= hi):
        raise ValueError(f"{name} must be between {lo:,g} and {hi:,.0f}.")
    return v


@app.route("/api/yield", methods=["POST"])
def predict_yield():
    if YIELD is None:
        return jsonify(error="Yield model not trained yet. Run scripts/train_yield.py"), 503
    d = request.get_json(silent=True) or {}
    lv = YIELD["cat_levels"]
    try:
        for k in ("Crop", "State", "Season"):
            if d.get(k) not in lv[k]:
                raise ValueError(f"Choose a valid {k.lower()}")
        year = int(_num(d, "Crop_Year", 1990, 2050))
        area = _num(d, "Area", 0.01, 100000)
        rain = _num(d, "Annual_Rainfall", 50, 7000)
        fert = _num(d, "Fert_per_ha", 0, 2000)
        pest = _num(d, "Pest_per_ha", 0, 50)
    except ValueError as e:
        return jsonify(error=str(e)), 400

    row = {"Crop": d["Crop"], "State": d["State"], "Season": d["Season"], "Crop_Year": year,
           "Annual_Rainfall": rain, "Fert_per_ha": fert, "Pest_per_ha": pest}  # area only scales production
    X = pd.DataFrame([row])[YIELD["feats"]]
    for c in YIELD["cats"]:
        X[c] = pd.Categorical(X[c], categories=lv[c])
    y = float(np.expm1(YIELD["model"].predict(X)[0]))

    ref = next((r for r in YMETA["defaults"] if r["Crop"] == d["Crop"]), None)
    notes = []
    if d["State"] not in YMETA["combos"].get(d["Crop"], []):
        notes.append(f"{d['Crop']} has no records for {d['State']} in the training data, so this estimate is less reliable.")
    if year > YMETA["year_max"]:
        notes.append(f"The model was trained on {YMETA['year_min']}-{YMETA['year_max']}; later years are extrapolated.")
    coconut = d["Crop"].lower().startswith("coconut")
    return jsonify(yield_per_ha=round(y, 3), production=round(y * area, 1),
                   unit="nuts per hectare" if coconut else "tonnes per hectare",
                   production_unit="nuts" if coconut else "tonnes",
                   typical_low=ref and round(ref["y_p10"], 3), typical_median=ref and round(ref["y_median"], 3),
                   typical_high=ref and round(ref["y_p90"], 3), notes=notes)


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="That image is larger than 6 MB. Use a smaller photo."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
