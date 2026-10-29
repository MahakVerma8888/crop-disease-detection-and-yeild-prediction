# Smart Crop: Disease Detection and Yield Prediction

A web app with two tools:

1. **Check a leaf.** Upload a leaf photo and get the crop, the disease (or "healthy"), a confidence score, and general crop-protection steps. If the model is unsure it says so instead of guessing.
2. **Estimate yield.** Enter crop, state, season, year, rainfall, fertiliser and pesticide per hectare, and your field area. It returns expected yield per hectare, total production, and how that compares with historical records for the crop.

A third tab, **How well it works**, shows the measured accuracy, a confusion matrix and the known limits.

Everything runs locally and offline (no CDN, no external API), so it works at an exhibition without Wi-Fi.

## Run it

```bash
pip install flask scikit-learn scikit-image pandas numpy pillow joblib matplotlib
python app/app.py          # open http://127.0.0.1:5000
```

Trained models are already in `models/`. To retrain:

```bash
python scripts/train_yield.py                 # uses data/crop_yield.csv
python scripts/train_disease.py data/images   # folder of class sub-folders (PlantVillage layout)
python scripts/train_gate.py                  # photo-check thresholds
python -m pytest tests                        # 12 tests
```

## Project layout

| Path | What it is |
|---|---|
| `app/app.py` | Flask server and JSON API (`/api/disease`, `/api/yield`, `/api/meta`, `/api/report`, `/health`) |
| `app/templates`, `app/static` | The web interface (plain HTML, CSS, JS) |
| `core/features.py` | Leaf-image feature extraction (colour, texture, edges), shared by training and the app |
| `core/advice.py` | General crop-protection guidance per disease |
| `core/gate.py` | Photo check: rejects blank, noisy or plant-free images before classification |
| `scripts/train_disease.py` | Feature extraction, model comparison, evaluation, confusion matrix |
| `scripts/train_yield.py` | Yield model, baselines, temporal hold-out, feature importance |
| `scripts/train_gate.py` | Learns the photo-check thresholds from the training features |
| `scripts/yield_rf_check.py` | Random forest vs gradient boosting on the temporal hold-out |
| `scripts/fetch_images.py`, `fetch_all.sh` | Helper used to pull the PlantVillage subset from GitHub |
| `models/` | Saved models and yield-form metadata |
| `reports/` | Metrics (JSON) and the confusion matrix image |
| `data/crop_yield.csv` | Yield dataset |
| `tests/test_app.py` | Automated checks for the API |

## Data

| Purpose | Dataset | Notes |
|---|---|---|
| Leaf disease | **PlantVillage** (Hughes and Salathé, 2015; Mohanty et al., 2016). 54,305 images, 14 crops, 38 classes. Copy: github.com/spMohanty/PlantVillage-Dataset | This project trains on up to 300 images per class (11,227 images) to keep training fast. |
| Yield | **Indian crop yield by state, 1997–2020**: crop, season, state, area, production, rainfall, fertiliser, pesticide, yield (19,689 rows, 55 crops, 30 states). Mendeley Data DOI 10.17632/ncw2vbcgnk (CC BY 4.0); also on Kaggle. | Fertiliser and pesticide are state totals, so they are converted to per-hectare rates. |

## Method in brief

- **Disease model.** Each photo is resized to 128×128 and described by 399 numbers: colour histograms and statistics, brown/yellow/dark/green pixel fractions, local binary patterns, GLCM texture statistics and HOG edge features. Three classifiers (ExtraTrees, RBF-SVM, gradient boosting) are compared on a validation split, and the winner is scored once on a held-out test split.
- **Photo check.** Before classifying, the app rejects images with too little or too much edge detail or almost no plant colour (HTTP 422). It catches 15 of 19 non-leaf test inputs; colourful photos can still pass.
- **Yield model.** Gradient boosting on log(yield) with crop, state and season as native categories. Compared with ridge regression and random forest (the forest scores higher on raw-yield R² but is 67.5 MB vs 2.4 MB; typical error is about the same), and checked on a temporal hold-out (train to 2014, test 2015–2020).
- **Area is not a model input.** In the dataset, area is a state-wide total, so using it would make the model fail for a single farm. The app multiplies yield per hectare by your area.

See `reports/` and the in-app results tab for numbers.

## Limits

- Leaf photos in PlantVillage are lab-style. Field photos will be less accurate; the app asks for a retake when confidence is below 50%.
- Images of the same physical leaf can land in both train and test, so the accuracy figure is an upper bound.
- The disease model is classical machine learning, not a deep network. The deep-learning environment (PyTorch/TensorFlow) could not be installed in the build sandbox. Swapping in a MobileNet is the natural upgrade.
- The yield model captures typical outcomes for a crop, state and season. It does not see soil, variety, irrigation or in-season weather.
- Disease advice is general guidance. Confirm with a local agriculture office before spraying.

## Training images (not in this repo)

The 11,227 PlantVillage training images (198 MB) are not stored here. To retrain the disease model, download PlantVillage (github.com/spMohanty/PlantVillage-Dataset, colour images), use `scripts/fetch_images.py` / `scripts/fetch_all.sh` to place up to 300 images per class under `data/images/<class>/`, then run `python scripts/train_disease.py data/images`. The trained models in `models/` already work without them.
