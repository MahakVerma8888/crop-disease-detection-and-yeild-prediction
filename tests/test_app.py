"""API tests. Run from the project root:  python -m unittest discover -s tests -v"""
import io
import os
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))
from app import app  # noqa: E402

GOOD = {"Crop": "Rice", "State": "Punjab", "Season": "Kharif", "Crop_Year": 2019,
        "Area": 2, "Annual_Rainfall": 800, "Fert_per_ha": 150, "Pest_per_ha": 0.4}


def png_bytes(color=(40, 140, 60), size=(200, 160)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    buf.seek(0)
    return buf


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    # ---- basics
    def test_health_and_page(self):
        r = self.c.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json["disease_model"] and r.json["yield_model"])
        page = self.c.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Smart Crop", page.data)

    def test_meta(self):
        j = self.c.get("/api/meta").json
        self.assertEqual(len(j["crops"]), 55)
        self.assertIn("Kharif", j["seasons"])
        self.assertIn("Rice", j["combos"])

    # ---- yield
    def test_yield_ok_and_scales_with_area(self):
        a = self.c.post("/api/yield", json=GOOD).json
        b = self.c.post("/api/yield", json={**GOOD, "Area": 20}).json
        self.assertGreater(a["yield_per_ha"], 0)
        self.assertAlmostEqual(a["yield_per_ha"], b["yield_per_ha"], places=6)  # area must not change per-ha yield
        self.assertAlmostEqual(b["production"], a["production"] * 10, delta=0.5)
        self.assertEqual(a["unit"], "tonnes per hectare")

    def test_yield_sane_for_known_crops(self):
        wheat = self.c.post("/api/yield", json={**GOOD, "Crop": "Wheat", "Season": "Rabi"}).json["yield_per_ha"]
        self.assertTrue(2.5 < wheat < 6.5, wheat)          # Punjab wheat is ~4-5 t/ha
        rice = self.c.post("/api/yield", json=GOOD).json["yield_per_ha"]
        self.assertTrue(2.0 < rice < 6.5, rice)            # Punjab rice is ~3-4 t/ha

    def test_yield_coconut_unit(self):
        j = self.c.post("/api/yield", json={**GOOD, "Crop": "Coconut", "State": "Kerala", "Season": "Whole Year"}).json
        self.assertEqual(j["unit"], "nuts per hectare")

    def test_yield_warns_on_unseen_combo_and_future_year(self):
        j = self.c.post("/api/yield", json={**GOOD, "State": "Goa", "Crop_Year": 2030}).json
        self.assertTrue(j["notes"])

    def test_yield_rejects_bad_input(self):
        for bad in ({"Crop": "Unobtainium"}, {"State": "Atlantis"}, {"Area": -1}, {"Area": "abc"},
                    {"Annual_Rainfall": 1e6}, {"Crop_Year": 1800}):
            r = self.c.post("/api/yield", json={**GOOD, **bad})
            self.assertEqual(r.status_code, 400, bad)
            self.assertIn("error", r.json)
        self.assertEqual(self.c.post("/api/yield", data="not json").status_code, 400)

    # ---- disease
    def test_disease_rejects_missing_wrong_and_corrupt(self):
        self.assertEqual(self.c.post("/api/disease").status_code, 400)
        r = self.c.post("/api/disease", data={"image": (io.BytesIO(b"hello"), "x.txt")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        r = self.c.post("/api/disease", data={"image": (io.BytesIO(b"not an image"), "x.jpg")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_disease_rejects_oversize(self):
        big = io.BytesIO(b"0" * (7 * 1024 * 1024))
        r = self.c.post("/api/disease", data={"image": (big, "big.jpg")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 413)

    def test_photo_gate_rejects_non_leaf_inputs(self):
        """Blank, flat-colour, noisy and plant-free images must not get a diagnosis."""
        rng = np.random.default_rng(0)
        cases = {
            "flat green": np.full((160, 200, 3), (40, 150, 60), np.uint8),
            "white page": np.full((160, 200, 3), 255, np.uint8),
            "black": np.zeros((160, 200, 3), np.uint8),
            "noise": rng.integers(0, 255, (160, 200, 3), dtype=np.uint8),
        }
        for name, arr in cases.items():
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "PNG")
            buf.seek(0)
            r = self.c.post("/api/disease", data={"image": (buf, "x.png")}, content_type="multipart/form-data")
            self.assertEqual(r.status_code, 422, name)
            self.assertIn("usable leaf photo", r.json["error"])

    def _sample_leaf(self):
        d = os.path.join(ROOT, "tests", "samples")
        if not os.path.isdir(d):
            self.skipTest("no tests/samples folder")
        cls = sorted(os.listdir(d))[0]
        fn = sorted(os.listdir(os.path.join(d, cls)))[0]
        return open(os.path.join(d, cls, fn), "rb"), fn

    def test_disease_returns_structured_answer(self):
        fh, fn = self._sample_leaf()
        r = self.c.post("/api/disease", data={"image": (fh, fn)}, content_type="multipart/form-data")
        fh.close()
        self.assertEqual(r.status_code, 200)
        j = r.json
        self.assertEqual(len(j["top3"]), 3)
        probs = [t["probability"] for t in j["top3"]]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertTrue(0 <= j["prediction"]["confidence"] <= 1)
        self.assertIn("confident", j)

    def test_disease_on_real_sample_images(self):
        """If sample PlantVillage images exist locally, check the app labels most of them correctly."""
        d = os.path.join(ROOT, "tests", "samples")
        if not os.path.isdir(d):
            self.skipTest("no tests/samples folder")
        ok = n = 0
        for cls in sorted(os.listdir(d)):
            for fn in sorted(os.listdir(os.path.join(d, cls)))[:5]:
                with open(os.path.join(d, cls, fn), "rb") as fh:
                    r = self.c.post("/api/disease", data={"image": (fh, fn)}, content_type="multipart/form-data")
                n += 1
                ok += r.json["prediction"]["label"] == cls
        self.assertGreater(ok / n, 0.8, f"{ok}/{n}")


if __name__ == "__main__":
    unittest.main()
