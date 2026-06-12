"""Regeneruje TYLKO zbiór lowres (po zmianie degradacji), nie ruszając clean/multiface.

Czyści LOWRES_DIR, usuwa stare wpisy lowres z manifestu, generuje nowe (łagodniejsza
degradacja z build_test_sets.degrade_cctv) i zapisuje manifest. Szybkie (bez detekcji).
"""
import argparse
import json
import random

import cv2

from config import DATA_DIR, LOWRES_DIR, SEED
from pipeline import detect_align
import build_test_sets as B


def main(lowres_per):
    rng = random.Random(SEED)
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())

    # usuń stare pliki i wpisy lowres
    for p in LOWRES_DIR.glob("*.png"):
        p.unlink()
    manifest = [m for m in manifest if m["difficulty"] != "lowres"]

    clean_probes = [m for m in manifest if m["difficulty"] == "clean" and "identity" in m]
    print(f"regeneruję lowres: {lowres_per}/źródło z {len(clean_probes)} źródeł")

    n = 0
    for m in clean_probes:
        src = detect_align.imread(m["source_clean"])
        for _ in range(lowres_per):
            deg, meta = B.degrade_cctv(src, rng)
            name = f"{m['identity'].replace(':', '_')}_{n:06d}.png"
            path = str(LOWRES_DIR / name)
            cv2.imwrite(path, deg)
            manifest.append({"path": path, "identity": m["identity"],
                             "difficulty": "lowres", "source_clean": m["source_clean"],
                             **meta})
            n += 1

    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"lowres={n}; manifest={len(manifest)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowres-per", type=int, default=3)
    args = ap.parse_args()
    main(args.lowres_per)
