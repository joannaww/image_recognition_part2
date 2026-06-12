"""Dogenerowuje multiface (+N na źródło) do balansu z lowres, dopisując do manifestu.

Uruchom raz, gdy multiface < lowres. Nie rusza istniejących plików/clean/lowres.
"""
import argparse
import json
import random

import cv2

from config import SPLIT_PATH, MULTIFACE_DIR, DATA_DIR, SEED
from pipeline import detect_align
import build_test_sets as B


def main(extra_per):
    rng = random.Random(SEED + 1)   # inny seed niż pierwotny przebieg
    split = json.loads(SPLIT_PATH.read_text())
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    distractor_ids = list(split["distractors"].values())

    clean_probes = [m for m in manifest if m["difficulty"] == "clean" and "identity" in m]
    start = sum(1 for m in manifest if m["difficulty"] == "multiface")
    print(f"multiface start={start}, dogeneruję {extra_per}/źródło z {len(clean_probes)} źródeł")

    n = start
    added = 0
    for m in clean_probes:
        subj = detect_align.imread(m["source_clean"])
        for _ in range(extra_per):
            d_imgs = [detect_align.imread(rng.choice(d_list))
                      for d_list in random.Random(rng.random()).sample(
                          distractor_ids, rng.randint(1, 3))]
            scene = B.build_multiface_scene(subj, d_imgs, rng)
            if scene is None:
                continue
            name = f"{m['identity'].replace(':', '_')}_{n:06d}.png"
            path = str(MULTIFACE_DIR / name)
            cv2.imwrite(path, scene)
            manifest.append({"path": path, "identity": m["identity"],
                             "difficulty": "multiface", "source_clean": m["source_clean"],
                             "n_distractors": len(d_imgs)})
            n += 1
            added += 1

    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_mf = sum(1 for m in manifest if m["difficulty"] == "multiface")
    print(f"dodano {added}; multiface łącznie={total_mf}; manifest={len(manifest)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-per", type=int, default=1)
    args = ap.parse_args()
    main(args.extra_per)
