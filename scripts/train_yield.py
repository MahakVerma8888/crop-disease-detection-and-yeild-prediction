"""Train the crop-yield regressor.

Data: Indian state-level crop statistics 1997-2020 (data/crop_yield.csv).
Model: HistGradientBoostingRegressor on log1p(Yield) with native categorical
handling. Fertilizer and pesticide in the raw file are state totals (kg), so
they are converted to per-hectare rates, which is what a farmer would enter.

Two evaluations are reported:
  * random 80/20 split
  * temporal hold-out: train on years <= 2014, test on 2015-2020 (the honest
    "predict a future year" check)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "crop_yield.csv"))
df["Season"] = df["Season"].str.strip()
df["Crop"] = df["Crop"].str.strip()
df["State"] = df["State"].str.strip()

# --- cleaning -------------------------------------------------------------
df = df[(df.Area > 0) & (df.Production >= 0)].copy()
df["Fert_per_ha"] = df.Fertilizer / df.Area
df["Pest_per_ha"] = df.Pesticide / df.Area
# Coconut is recorded in nuts (not tonnes), which makes its "yield" ~100x larger;
# keep it, but the model sees Crop so it learns the different scale.
df["y"] = np.log1p(df.Yield)

CATS = ["Crop", "State", "Season"]
# NOTE: "Area" is deliberately NOT a feature. In this dataset it is the state's
# total area under the crop (millions of ha), so it only acts as a proxy for
# "which big state" and breaks for a single farm. The app multiplies the
# predicted yield per hectare by the farmer's own area instead.
NUMS = ["Crop_Year", "Annual_Rainfall", "Fert_per_ha", "Pest_per_ha"]
FEATS = CATS + NUMS

cat_levels = {c: sorted(df[c].unique().tolist()) for c in CATS}
for c in CATS:
    df[c] = pd.Categorical(df[c], categories=cat_levels[c])


def make_hgb():
    return HistGradientBoostingRegressor(
        categorical_features=[FEATS.index(c) for c in CATS],
        max_iter=400, learning_rate=0.08, max_leaf_nodes=48,
        min_samples_leaf=15, l2_regularization=1.0, random_state=42,
    )


def report(name, model, Xtr, ytr, Xte, yte, raw_te):
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    pred = np.expm1(p)
    mape = np.median(np.abs(pred - raw_te) / np.maximum(raw_te, 1e-6)) * 100
    return {
        "model": name,
        "R2_log": round(r2_score(yte, p), 4),
        "R2_yield": round(r2_score(raw_te, pred), 4),
        "MAE_yield": round(mean_absolute_error(raw_te, pred), 3),
        "median_abs_pct_error": round(float(mape), 1),
    }


results = {"random_split": [], "temporal_split": []}

# --- baselines on random split -------------------------------------------
X, y, raw = df[FEATS], df["y"], df["Yield"]
Xtr, Xte, ytr, yte, _, rte = train_test_split(X, y, raw, test_size=0.2, random_state=42)

ohe = pd.get_dummies(df[FEATS], columns=CATS)
otr, ote = ohe.loc[Xtr.index], ohe.loc[Xte.index]
results["random_split"].append(report("Ridge (baseline)", Ridge(alpha=1.0), otr, ytr, ote, yte, rte))
results["random_split"].append(report(
    "RandomForest (120 trees)",
    RandomForestRegressor(n_estimators=120, min_samples_leaf=2, n_jobs=2, random_state=42),
    otr, ytr, ote, yte, rte))
results["random_split"].append(report("HistGradientBoosting (chosen)", make_hgb(), Xtr, ytr, Xte, yte, rte))

# --- temporal hold-out -----------------------------------------------------
tr = df[df.Crop_Year <= 2014]
te = df[df.Crop_Year > 2014]
results["temporal_split"].append(report(
    "HistGradientBoosting (chosen)", make_hgb(), tr[FEATS], tr["y"], te[FEATS], te["y"], te["Yield"]))
results["temporal_split"][0]["train_rows"] = int(len(tr))
results["temporal_split"][0]["test_rows"] = int(len(te))

# --- permutation importance (on the random-split model) ---------------------
from sklearn.inspection import permutation_importance

final = make_hgb().fit(Xtr, ytr)
pi = permutation_importance(final, Xte, yte, n_repeats=3, random_state=42, n_jobs=1)
imp = sorted(zip(FEATS, pi.importances_mean.round(4).tolist()), key=lambda t: -t[1])
results["permutation_importance"] = imp

# --- fit final model on all data and save ----------------------------------
full = make_hgb().fit(df[FEATS], df["y"])

# typical values per crop (used by the web UI to pre-fill sensible defaults)
defaults = (
    df.groupby("Crop", observed=True)
    .agg(rain=("Annual_Rainfall", "median"), fert=("Fert_per_ha", "median"),
         pest=("Pest_per_ha", "median"), area=("Area", "median"),
         y_median=("Yield", "median"), y_p10=("Yield", lambda s: s.quantile(0.1)),
         y_p90=("Yield", lambda s: s.quantile(0.9)), n=("Yield", "size"))
    .round(3).reset_index().to_dict(orient="records")
)
state_rain = df.groupby("State", observed=True).Annual_Rainfall.median().round(1).to_dict()
# which states grow which crop (so the UI can warn about unseen combinations)
combos = df.groupby("Crop", observed=True).State.unique().apply(lambda a: sorted(a.tolist())).to_dict()

os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
joblib.dump({"model": full, "feats": FEATS, "cats": CATS, "cat_levels": cat_levels},
            os.path.join(ROOT, "models", "yield_model.joblib"))
json.dump({"defaults": defaults, "state_rain": state_rain, "combos": combos,
           "year_min": int(df.Crop_Year.min()), "year_max": int(df.Crop_Year.max())},
          open(os.path.join(ROOT, "models", "yield_meta.json"), "w"), indent=1)

results["dataset"] = {"rows_used": int(len(df)), "crops": len(cat_levels["Crop"]),
                      "states": len(cat_levels["State"]), "years": [int(df.Crop_Year.min()), int(df.Crop_Year.max())]}
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
json.dump(results, open(os.path.join(ROOT, "reports", "yield_metrics.json"), "w"), indent=1)
print(json.dumps(results, indent=1))
