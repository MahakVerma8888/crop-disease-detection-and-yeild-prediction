"""Fetch selected PlantVillage images from a partial git clone, in parallel.

usage: fetch_images.py REPO TSV OUTDIR START END WORKERS

TSV rows are "<blob oid>\t<path in repo>". Each image is written to
OUTDIR/<class folder>/<file>. Already-downloaded files are skipped, so the
script can be re-run safely.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

repo, tsv, out = sys.argv[1], sys.argv[2], sys.argv[3]
a, b, workers = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
rows = [line.rstrip("\n").split("\t", 1) for line in open(tsv)][a:b]


def get(row):
    oid, path = row
    dst = os.path.join(out, path.split("/", 2)[2])
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for _ in range(3):
        p = subprocess.run(["git", "-C", repo, "cat-file", "blob", oid], capture_output=True)
        if p.returncode == 0 and p.stdout:
            with open(dst, "wb") as f:
                f.write(p.stdout)
            return 1
    return 0


with ThreadPoolExecutor(workers) as ex:
    ok = sum(ex.map(get, rows))
print("ok", ok, "of", len(rows))
