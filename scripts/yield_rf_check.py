"""Like-for-like check: random forest vs gradient boosting on the TEMPORAL hold-out.

train_yield.py reports a random-split comparison of three models but the
temporal hold-out only for the chosen model. This script closes that gap so
the model choice can be defended on the test that matters (predicting later
years). Run:  python scripts/yield_rf_check.py
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "crop_yield.csv"))
for c in ("Season", "Crop", "State"):
    df[c] = df[c].str.strip()
df = df[(df.Area > 0) & (df.Production >= 0)].copy()
df["Fert_per_ha"], df["Pest_per_ha"] = df.Fertilizer / df.Area, df.Pesticide / df.Area
df["y"] = np.log1p(df.Yield)
CATS = ["Crop", "State", "Season"]
NUMS = ["Crop_Year", "Annual_Rainfall", "Fert_per_ha", "Pest_per_ha"]
tr, te = df[df.Crop_Year <= 2014], df[df.Crop_Year > 2014]
ohe = pd.get_dummies(df[CATS + NUMS], columns=CATS)
res = {}


def score(name, model, Xtr, Xte):
    t = time.time()
    model.fit(Xtr, tr["y"])
    p = model.predict(Xte)
    pred, raw = np.expm1(p), te["Yield"].values
    res[name] = {"R2_log": round(r2_score(te["y"], p), 4), "R2_yield": round(r2_score(raw, pred), 4),
                 "MAE_yield": round(mean_absolute_error(raw, pred), 3),
                 "median_abs_pct_error": round(float(np.median(np.abs(pred - raw) / np.maximum(raw, 1e-6)) * 100), 1),
                 "fit_seconds": round(time.time() - t)}
    return model


rf = score("RandomForest (120 trees)", RandomForestRegressor(n_estimators=120, min_samples_leaf=2, n_jobs=2, random_state=42),
           ohe.loc[tr.index], ohe.loc[te.index])
joblib.dump(rf, "/tmp/rf_check.joblib")
res["RandomForest (120 trees)"]["model_file_MB"] = round(os.path.getsize("/tmp/rf_check.joblib") / 1e6, 1)
os.remove("/tmp/rf_check.joblib")

d2 = df.copy()
for c in CATS:
    d2[c] = pd.Categorical(d2[c])
feats = CATS + NUMS
hgb = HistGradientBoostingRegressor(categorical_features=[feats.index(c) for c in CATS], max_iter=400, learning_rate=0.08,
                                    max_leaf_nodes=48, min_samples_leaf=15, l2_regularization=1.0, random_state=42)
score("HistGradientBoosting (chosen)", hgb, d2.loc[tr.index, feats], d2.loc[te.index, feats])
joblib.dump(hgb, "/tmp/hgb_check.joblib")
res["HistGradientBoosting (chosen)"]["model_file_MB"] = round(os.path.getsize("/tmp/hgb_check.joblib") / 1e6, 1)
os.remove("/tmp/hgb_check.joblib")

json.dump(res, open(os.path.join(ROOT, "reports", "yield_temporal_comparison.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
