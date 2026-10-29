"""Hand-crafted leaf-image features (colour + texture + shape/edge).

Used identically by scripts/train_disease.py and the web app, so training and
inference can never drift apart.

Feature groups (total FEATURE_DIM):
  * colour   - HSV histograms, per-channel mean/std in RGB, HSV and Lab,
               fraction of brown/yellow/dark pixels (typical lesion colours)
  * texture  - uniform LBP histogram, GLCM statistics
  * shape    - HOG on a 64x64 grayscale thumbnail, edge density
"""
import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

SIZE = 128  # working resolution


def load_rgb(path_or_file, size=SIZE):
    im = Image.open(path_or_file).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im)


def _hist(x, bins, rng):
    h, _ = np.histogram(x, bins=bins, range=rng)
    return h / max(h.sum(), 1)


def extract(rgb: np.ndarray) -> np.ndarray:
    """rgb: uint8 array (SIZE, SIZE, 3) -> 1-D float32 feature vector."""
    f = rgb.astype(np.float32) / 255.0
    hsv = rgb2hsv(f)
    lab = rgb2lab(f)
    gray = rgb2gray(f)
    feats = []

    # --- colour -----------------------------------------------------------
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    feats += [_hist(h, 18, (0, 1)), _hist(s, 8, (0, 1)), _hist(v, 8, (0, 1))]
    for ch in (f[..., 0], f[..., 1], f[..., 2], h, s, v, lab[..., 0], lab[..., 1], lab[..., 2]):
        feats.append(np.array([ch.mean(), ch.std()]))
    # lesion-like colour fractions (leaf-ish pixels only: ignore very dark background)
    fg = v > 0.15
    n = max(fg.sum(), 1)
    brown = fg & (h > 0.04) & (h < 0.14) & (s > 0.25)
    yellow = fg & (h >= 0.14) & (h < 0.21) & (s > 0.3)
    dark = fg & (v < 0.30)
    green = fg & (h >= 0.21) & (h < 0.45) & (s > 0.2)
    feats.append(np.array([brown.sum() / n, yellow.sum() / n, dark.sum() / n, green.sum() / n, fg.mean()]))

    # --- texture ----------------------------------------------------------
    g8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(g8, P=8, R=1, method="uniform")
    feats.append(_hist(lbp, 10, (0, 10)))
    g32 = (g8 // 8).astype(np.uint8)  # 32 grey levels
    glcm = graycomatrix(g32, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    feats.append(np.array([graycoprops(glcm, p).mean() for p in
                           ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")]))

    # --- shape / edges ----------------------------------------------------
    thumb = np.asarray(Image.fromarray(g8).resize((64, 64), Image.BILINEAR)).astype(np.float32) / 255.0
    feats.append(hog(thumb, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                     feature_vector=True))
    gx = np.abs(np.diff(gray, axis=1))[:-1, :]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    mag = gx + gy
    feats.append(np.array([mag.mean(), mag.std(), (mag > 0.1).mean()]))

    return np.concatenate(feats).astype(np.float32)


def extract_path(path):
    return extract(load_rgb(path))


FEATURE_DIM = extract(np.zeros((SIZE, SIZE, 3), dtype=np.uint8)).shape[0]
