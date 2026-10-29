"""Train the leaf-disease classifier.

usage: python scripts/train_disease.py [IMAGE_DIR]

IMAGE_DIR contains one sub-folder per class (PlantVillage layout,
e.g. Tomato___Late_blight/*.JPG). Features come from core/features.py.

Protocol: stratified 70/15/15 train/validation/test split.
Three candidate models are compared on the validation set; the winner is
re-fitted on train+validation and reported ONCE on the untouched test set.
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.features import FEATURE_DIM, extract_path  # noqa: E402

IMG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "images")
CACHE = os.path.join(ROOT, "data", "features_cache.npz")


def collect():
    paths, labels = [], []
    for cls_dir in sorted(glob.glob(os.path.join(IMG_DIR, "*"))):
        if not os.path.isdir(cls_dir):
            continue
        for p in sorted(glob.glob(os.path.join(cls_dir, "*"))):
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(p)
                labels.append(os.path.basename(cls_dir))
    return paths, np.array(labels)


def safe_extract(p):
    try:
        return extract_path(p)
    except Exception as e:  # corrupt image
        print("skip", p, e)
        return None


if __name__ == "__main__":
    paths, labels = collect()
    print(f"{len(paths)} images, {len(set(labels))} classes, feature dim {FEATURE_DIM}")

    t0 = time.time()
    with Pool(2) as pool:
        feats = pool.map(safe_extract, paths, chunksize=64)
    keep = [i for i, f in enumerate(feats) if f is not None]
    X = np.stack([feats[i] for i in keep])
    y = labels[keep]
    print(f"feature extraction: {time.time() - t0:.0f}s, X={X.shape}")
    np.savez_compressed(CACHE, X=X, y=y)

    classes = sorted(set(y))
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.15 / 0.85, stratify=ytv, random_state=42)
    print("split sizes train/val/test:", len(ytr), len(yva), len(yte))

    candidates = {
        "ExtraTrees (400)": lambda: ExtraTreesClassifier(n_estimators=400, n_jobs=2, random_state=42),
        "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=42)),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.1, max_leaf_nodes=24, early_stopping=False, random_state=42),
    }
    val_scores = {}
    for name, mk in candidates.items():
        t = time.time()
        m = mk().fit(Xtr, ytr)
        acc = accuracy_score(yva, m.predict(Xva))
        val_scores[name] = round(float(acc), 4)
        print(f"  {name:24s} val acc {acc:.4f}  ({time.time() - t:.0f}s)")

    best = max(val_scores, key=val_scores.get)
    print("chosen:", best)
    final = candidates[best]().fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    pred = final.predict(Xte)
    proba = final.predict_proba(Xte)
    conf = proba.max(1)
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro")
    print(f"TEST accuracy {acc:.4f}  macro-F1 {f1m:.4f}")

    # how accuracy changes if we only answer when confident (the app uses 0.50)
    cov = {}
    for thr in (0.3, 0.5, 0.7, 0.9):
        m_ = conf >= thr
        cov[str(thr)] = {"coverage": round(float(m_.mean()), 4),
                         "accuracy_when_answered": round(float(accuracy_score(yte[m_], pred[m_])), 4) if m_.any() else None}

    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=classes)

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    joblib.dump({"model": final, "classes": classes, "feature_dim": FEATURE_DIM},
                os.path.join(ROOT, "models", "disease_model.joblib"), compress=3)

    out = {
        "n_images": int(len(y)), "n_classes": len(classes),
        "split": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))},
        "validation_accuracy": val_scores, "chosen_model": best,
        "test_accuracy": round(float(acc), 4), "test_macro_f1": round(float(f1m), 4),
        "confidence_threshold_tradeoff": cov,
        "per_class": {c: {k: round(v, 3) for k, v in rep[c].items() if k != "support"} | {"support": int(rep[c]["support"])}
                      for c in classes},
    }
    json.dump(out, open(os.path.join(ROOT, "reports", "disease_metrics.json"), "w"), indent=1)

    # confusion matrix figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    short = [c.replace("___", " - ").replace("_", " ")[:34] for c in classes]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalised confusion matrix (test set, accuracy {acc:.1%})")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", "disease_confusion_matrix.png"), dpi=130)
    print("saved model + reports")
