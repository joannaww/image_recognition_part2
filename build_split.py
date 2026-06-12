"""Podział tożsamości na role (deterministyczny, seed=42) -> artifacts/data_split.json.

Role:
  enrolled        — wdrożeni użytkownicy (>=100, w tym członkowie grupy z own_faces)
  impostors       — tożsamości NIE wdrożone, do par impostor / open-set
  distractors     — twarze doklejane do scen "wiele twarzy" (#1)

Źródła: faces_training (FaceCard, 427 os.), new_faces (100 os.), own_faces (grupa).
Każda tożsamość ma >= MIN_IMAGES zdjęć, żeby starczyło na enroll + test.
"""
import json
import random

from config import (FACES_TRAINING, NEW_FACES, OWN_FACES, SPLIT_PATH, SEED,
                    IMG_EXTS)

MIN_IMAGES = 4
N_ENROLLED = 100
N_IMPOSTORS = 120
N_DISTRACTORS = 150


def list_identities(root, source):
    out = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if len(imgs) >= MIN_IMAGES:
            out[f"{source}:{d.name}"] = imgs
    return out


def main():
    rng = random.Random(SEED)

    facecard = list_identities(FACES_TRAINING, "facecard")
    newf = list_identities(NEW_FACES, "newface")
    own = list_identities(OWN_FACES, "own")

    print(f"facecard={len(facecard)}  new_faces={len(newf)}  own={len(own)}")

    # Członkowie grupy ZAWSZE wdrożeni.
    enrolled = dict(own)

    pool = list(facecard.items())
    rng.shuffle(pool)
    for k, v in pool:
        if len(enrolled) >= N_ENROLLED:
            break
        enrolled[k] = v

    used = set(enrolled)
    rest = [(k, v) for k, v in facecard.items() if k not in used]
    rest += list(newf.items())
    rng.shuffle(rest)

    impostors = dict(rest[:N_IMPOSTORS])
    distractors = dict(rest[N_IMPOSTORS:N_IMPOSTORS + N_DISTRACTORS])

    split = {
        "seed": SEED,
        "enrolled": {k: v for k, v in enrolled.items()},
        "impostors": impostors,
        "distractors": distractors,
        "counts": {"enrolled": len(enrolled), "impostors": len(impostors),
                   "distractors": len(distractors)},
    }
    SPLIT_PATH.write_text(json.dumps(split, indent=2))
    print(f"Zapisano {SPLIT_PATH}  ->  {split['counts']}")


if __name__ == "__main__":
    main()
